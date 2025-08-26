
import math
import gc # 引入垃圾回收模块
from cam_pos.read_cam_pos import get_vid_cam_from_bin, get_first_vid_cam_from_json
from gaussian_renderer import render
from scene import Scene, GaussianModel
from pathlib import Path
import sys
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from point_render_pipeline import zip_point_cloud, build_projection_matrix, get_video_cam_info_from_bin
from utils.general_utils import safe_state

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, rasterize_gaussians, _C
from scipy.spatial.transform import Rotation as R
import pyautogui
import pygetwindow as gw
import time

# 动态获取项目根目录（假设 point_render.py 在 glsl/ 子目录下）
ROOT_DIR = Path(__file__).parent  # 向上两级到 GSSample/
sys.path.append(str(ROOT_DIR))  # 将根目录加入 Python 路径
mode = "GPU"
FPS = 10
NUM_POINTS_TO_KEEP = 3_000_000
NUM_SAMPLES_PER_POINT = 1024

wh_idx = 3
scene_name = "drums"
scenes_name = ["drums", "ficus", "hotdog", "lego", "materials", "mic", "ship", "chair"]  # 在这里设置需要渲染的场景
data_dir = r"H:\800_DataSet\NeRF_Synthesis\3DGS_nerf_synthesis" + "\\" + scene_name
start_checkpoint = r"H:\800_DataSet\NeRF_Synthesis\3DGS_res_7000" + "\\" + scene_name + "\\" + "chkpnt7000.pth"

# 导入log系统，设置日志输出位置
import logging
# 1. 获取一个logger对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # 设置logger的级别

# 2. 避免重复添加handler
if not logger.handlers:
    # 3. 创建一个文件handler，明确指定编码
    log_file_path = f'point_render_{mode}.log'
    # 'w' 表示每次运行都覆盖旧日志, 如果想追加, 可以用 'a'
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8') 
    file_handler.setLevel(logging.INFO) # 设置handler的级别

    # 4. 创建一个日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 5. 将handler添加到logger
    logger.addHandler(file_handler)


def normalize(v):
    """对向量进行归一化。"""
    if np.linalg.norm(v) == 0:
        return v
    return v / np.linalg.norm(v)

def look_at(camera_position, target_position, world_up):
    """
    计算并返回一个从相机到世界坐标系的4x4变换矩阵 (Camera-to-World)。
    """
    z_axis = normalize(camera_position - target_position)
    x_axis = normalize(np.cross(world_up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    cam_to_world = np.eye(4)
    cam_to_world[:3, 0] = x_axis
    cam_to_world[:3, 1] = y_axis
    cam_to_world[:3, 2] = z_axis
    cam_to_world[:3, 3] = camera_position
    
    return cam_to_world

# ==============================================================================
# == 全新实现：基于 look_at 的显式相机控制器
# ==============================================================================
class ExplicitCameraController:
    def __init__(self, initial_R, initial_T, move_speed=0.1, rot_speed=0.0005):
        self.move_speed = move_speed
        self.rot_speed = rot_speed
        
        # --- 1. 初始化相机状态参数 (全部在Blender/OpenGL坐标系下) ---
        
        w2c = np.eye(4)
        w2c[:3, :3] = initial_R.T
        w2c[:3, 3] = initial_T
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1
        cam_center = c2w[:3, 3]
        
        # 从Blender c2w矩阵中提取初始状态
        self.position = cam_center
        initial_forward_vec = c2w[:3, 2] # Blender中-Z是向前

        # 核心参数：这些是我们将通过键鼠直接控制的
        self.yaw = np.arctan2(initial_forward_vec[1], initial_forward_vec[0])
        self.pitch = np.arcsin(initial_forward_vec[2])
        
        # 场景定义 (Blender坐标系)
        self.world_up = np.array([0.0, 0.0, 1.0]) # Blender中Y轴是向上的！

        # 鼠标控制状态
        # self.last_mouse_pos = None
        # self.first_mouse_move = True
        self.window_center_x = 0
        self.window_center_y = 0
        
        self.is_resetting_mouse = False

        print("\n[Explicit Controller] Initialized for Blender Coordinate System.")
        self.print_state()

    def print_state(self):
        """打印当前相机的核心状态"""
        pos = self.position
        print(f"[State] Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
              f"Yaw: {np.rad2deg(self.yaw):.1f}°, Pitch: {np.rad2deg(self.pitch):.1f}°")

    def update(self):
        """根据当前状态计算R和T，这是每帧都要调用的核心函数"""
        # 1. 根据 yaw 和 pitch 计算出当前的前向向量 (在Z-up系统中)
        x = np.cos(self.yaw) * np.cos(self.pitch)
        y = np.sin(self.yaw) * np.cos(self.pitch)
        z = np.sin(self.pitch)
        forward_vec = normalize(np.array([x, y, z]))

        # --- 2. 计算目标点 ---
        # 这里的look_at期望的forward是 pos -> target，但我们算的是相机朝向
        # 我们需要重新构造一个符合Blender风格的look_at矩阵
        forward = normalize(-forward_vec) # Blender -Z is forward
        right = normalize(np.cross(self.world_up, forward))
        up = normalize(np.cross(forward, right))

        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward # Z轴指向后方
        c2w[:3, 3] = self.position

        # --- 3. 将Blender c2w转换为渲染器所需的 R 和 T ---
        c2w_for_colmap = c2w.copy()
        c2w_for_colmap[:3, 1:3] *= -1 # 应用“魔法咒语”，转换到COLMAP坐标系
        
        w2c = np.linalg.inv(c2w_for_colmap)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        return T, R

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # 如果是自动归位事件，则忽略它，并重置标志
            if self.is_resetting_mouse:
                self.is_resetting_mouse = False
                return

            # 如果窗口中心点尚未设置，则不执行任何操作
            if self.window_center_x == 0 or self.window_center_y == 0:
                return

            # 计算鼠标当前位置相对于窗口中心的偏移量
            dx = x - self.window_center_x
            dy = y - self.window_center_y

            # 只有在鼠标真的偏离中心时才更新视角（防止重置时光标的微小抖动触发旋转）
            if dx != 0 or dy != 0:
                self.yaw += dx * self.rot_speed
                self.pitch += dy * self.rot_speed # dy为正表示鼠标向下，俯仰角减小
                PITCH_LIMIT_DEGREES = 30.0
                pitch_limit_radians = np.radians(PITCH_LIMIT_DEGREES)
                self.pitch = np.clip(self.pitch, -pitch_limit_radians, pitch_limit_radians)
            
            self.print_state()

    def key_handler(self, key):
        # 计算当前朝向的移动向量 (在Blender坐标系下)
        forward_dir = np.array([np.cos(self.yaw) * np.cos(self.pitch), 
                                np.sin(self.yaw) * np.cos(self.pitch),
                                np.sin(self.pitch)])
        right_dir = normalize(np.cross(forward_dir, self.world_up))

        if key == ord('w') or key == ord('W'): self.position -= forward_dir * self.move_speed
        elif key == ord('s') or key == ord('S'): self.position += forward_dir * self.move_speed
        elif key == ord('a') or key == ord('A'): self.position -= right_dir * self.move_speed
        elif key == ord('d') or key == ord('D'): self.position += right_dir * self.move_speed
        elif key == 32 or key == ord('q'):  # Space
            self.position += self.world_up * self.move_speed
        elif key == 16 or key == 0 or key == ord('z'):  # Shift
            self.position -= self.world_up * self.move_speed
        
        self.print_state()


def render_by_3DGS_rt(
    dataset, opt, pipe, checkpoint,
    data_dir=".", 
    output_dir=".", 
    num_points_to_keep=1_000_000, 
    opacity_threshold=0.01,
    img_width=100, 
    img_height=100,
    fps=10,
    delete_images=True,
    point_size=2,
    logger=None
):
    with torch.no_grad():
        first_iter = 0
        gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians)
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        gaussians.training_setup(opt)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        initial_cam = get_first_vid_cam_from_json(tar_width=img_width, tar_height=img_height)
        
        # 更新相机宽高比（如果窗口大小可变，这里会很有用）
        initial_cam.update_wh(img_width, img_height)

        # ==============================================================================
        # == MODIFIED: 设置交互式渲染
        # ==============================================================================
        
# 2. 实例化我们全新的控制器
        # 注意：初始的R,T可能是Tensor，需要转为numpy
        camera_controller = ExplicitCameraController(initial_cam.R, initial_cam.T)

        # 3. 将渲染循环中使用的相机对象命名为 render_cam
        render_cam = initial_cam

        
        # 2. 创建OpenCV窗口并绑定鼠标回调函数
        window_name = "Real-time 3DGS Renderer | Controls: WASDQE, Mouse, Scroll | Exit: ESC"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, camera_controller.mouse_callback)

        print("\n--- Interactive Controls ---")
        print("Orbit: Left Mouse Drag")
        print("Pan: Middle Mouse Drag OR Shift + Left Mouse Drag")
        print("Zoom: Mouse Scroll Wheel")
        print("Fly: WASD for left/right/forward/backward, Q/E for down/up")
        print("Exit: Press ESC key")
        print("--------------------------\n")

        # ==============================================================================
        # == MODIFIED: 主渲染循环
        # ==============================================================================



        # 找到窗口在屏幕上的位置 (只需要做一次)
        # 我们需要这个来将窗口中心坐标转换为屏幕坐标以用于 pyautogui
        render_window_screen_pos = None
        try:
            time.sleep(0.5)
            # 使用一个临时变量来查找窗口
            _win = gw.getWindowsWithTitle(window_name)[0]
            _win.activate()
            render_window_screen_pos = (_win.left, _win.top)
            print(f"成功找到渲染窗口，屏幕位置: {render_window_screen_pos}")
        except Exception as e:
            print(f"警告：无法找到渲染窗口来精确锁定鼠标: {e}")
            print("鼠标锁定功能可能不准确。")
        
        # 禁用pyautogui的故障安全功能（当鼠标移动到屏幕左上角时中断程序）
        pyautogui.FAILSAFE = False

        # 计算窗口的中心点 (在窗口坐标系下)
        win_center_x_relative = img_width // 2
        win_center_y_relative = img_height // 2
        
        # 将窗口中心点信息传递给控制器
        camera_controller.window_center_x = win_center_x_relative
        camera_controller.window_center_y = win_center_y_relative

        # 如果我们成功找到了窗口的屏幕位置，计算出中心点的绝对屏幕坐标
        screen_center_x, screen_center_y = 0, 0
        if render_window_screen_pos:
            screen_center_x = render_window_screen_pos[0] + win_center_x_relative
            screen_center_y = render_window_screen_pos[1] + win_center_y_relative

        while True:
            # 1. 从控制器获取最新的 R 和 T
            new_T_np, new_R_np = camera_controller.update()

            # 2. 调用您提供的 update_cam_pos 方法来更新渲染相机
            render_cam.update_cam_pos(
                torch.tensor(new_T_np, dtype=torch.float32, device="cuda"),
                torch.tensor(new_R_np, dtype=torch.float32, device="cuda")
            )
            # 您的硬性要求：通过此函数获取图像
            image = render(render_cam, gaussians, pipe, background)["render"].detach()

            # 将Tensor转换为OpenCV格式的图像
            image_cpu = torch.clamp(image, 0.0, 1.0).cpu().numpy()
            image_cv = (image_cpu * 255).astype(np.uint8)
            image_cv = np.transpose(image_cv, (1, 2, 0))  # CHW -> HWC
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # RGB -> BGR
            # 注意：如果您的渲染结果是上下颠倒的，请保留cv2.flip
            # image_cv = cv2.flip(image_cv, 0) 
            
            # 显示图像
            cv2.imshow(window_name, image_cv)

            # 等待1ms并捕获键盘输入
            key = cv2.waitKey(1) & 0xFF

            # 按下ESC键 (ASCII码 27) 退出循环
            if key == 27:
                break
            
            # 将按键传递给控制器处理
            if key != 255: # 255表示没有按键
                camera_controller.key_handler(key)
            # 【关键】将鼠标光标重置到窗口中心的绝对屏幕坐标
            # 只有在我们成功计算出屏幕中心点时才执行
            if screen_center_x > 0 and screen_center_y > 0:
                # 告诉控制器，我们即将要自动移动鼠标了
                camera_controller.is_resetting_mouse = True
                # 执行自动移动
                pyautogui.moveTo(screen_center_x, screen_center_y)

        # 循环结束后销毁所有窗口
        cv2.destroyAllWindows()
        print("Interactive session ended.")


if __name__ == "__main__":
    dir = ROOT_DIR
    wh = [[100, 100], [616, 409], [1920, 1080], [800, 800]]

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = start_checkpoint)# r"G:\Lab\GSSample\output\92825354-f\chkpnt7000.pth")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_by_3DGS_rt(
        lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint,
        data_dir=data_dir,
        output_dir=dir,
        num_points_to_keep=NUM_POINTS_TO_KEEP, # 10_000_000
        opacity_threshold=None,
        img_width=wh[wh_idx][0],
        img_height=wh[wh_idx][1],
        fps=FPS,
        delete_images=True,
        point_size=1,
        logger=logger
    )