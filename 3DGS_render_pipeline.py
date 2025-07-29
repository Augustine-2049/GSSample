
import math
import gc # 引入垃圾回收模块
from video_cam import get_vid_cam_from_bin
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
from point_render_pipeline import zip_point_cloud, build_projection_matrix, get_video_cam_info
from utils.general_utils import safe_state

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, rasterize_gaussians, _C
# from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh

# 动态获取项目根目录（假设 point_render.py 在 glsl/ 子目录下）
ROOT_DIR = Path(__file__).parent  # 向上两级到 GSSample/
sys.path.append(str(ROOT_DIR))  # 将根目录加入 Python 路径
mode = "GPU"
FPS = 10
NUM_POINTS_TO_KEEP = 20_000_000
NUM_SAMPLES_PER_POINT = 1024
USE_HALF_PRECISION = False
version = 0.4

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



def render_by_3DGS(
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
    # --- 日志和文件夹设置 ---
    def log_info(msg):
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
    def log_error(msg):
        if logger is None:
            print(msg)
        else:
            logger.error(msg)
    log_info("--- step 1: load and process point cloud ---")
    img_dir = os.path.join(output_dir, "render_img")
    os.makedirs(img_dir, exist_ok=True)
    # ... (清空图片) ...
    del_cnt = 0
    for file in os.listdir(img_dir):
        if file.endswith(".png"):
            os.remove(os.path.join(img_dir, file))
            del_cnt += 1
    log_info(f"del {del_cnt} pics")

    # --- step 4: 加载和预处理点云数据 ---
    log_info("--- step 4: load point cloud data ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        points = zip_point_cloud(data_dir, num_points_to_keep, opacity_threshold=opacity_threshold).elements[0].data
        tensor_dtype = torch.float16 if USE_HALF_PRECISION else torch.float32
        # **处理 Colors **
        # 注意：PlyFile 读取的颜色是 uint8，需要转为 float32 进行归一化
        colors = torch.stack([
            torch.from_numpy(points['red'].copy()),
            torch.from_numpy(points['green'].copy()),
            torch.from_numpy(points['blue'].copy())
        ], dim=1).to(device, dtype=torch.float32) / 255.0
        shs = None
        # 0 阶球谐函数的常数基函数
        SH_C0 = 0.28209479177387814
        
        # 应用逆向公式: f_dc = (Color - 0.5) / C0
        # 0 阶 SH 只有 1 个系数，所以我们需要在系数维度上增加一个维度
        # 以匹配 (N, K, 3) 的标准格式，其中 K=1
        shs = ((colors - 0.5) / SH_C0).unsqueeze(1).to(tensor_dtype)
        del colors # .unsqueeze(1) 将 (N, 3) -> (N, 1, 3)
        # **处理 Scales **
        # 3DGS 的 scale 和 rotate 本身就是 float32，可以直接转换
        scales = torch.stack([
            torch.from_numpy(points['scale_0'].copy()),
            torch.from_numpy(points['scale_1'].copy()),
            torch.from_numpy(points['scale_2'].copy())
        ], dim=1).to(device)

        # **处理 Rotates **
        # 3DGS 的四元数通常是 w,x,y,z 顺序，这里假设 rot_0 是 w
        rotations = torch.stack([
            torch.from_numpy(points['rot_0'].copy()), # w
            torch.from_numpy(points['rot_1'].copy()), # x
            torch.from_numpy(points['rot_2'].copy()), # y
            torch.from_numpy(points['rot_3'].copy())  # z
        ], dim=1).to(device)
        cov3D_precomp = None
        xyzs = torch.stack([
            torch.from_numpy(points['x'].copy()),
            torch.from_numpy(points['y'].copy()),
            torch.from_numpy(points['z'].copy()),
        ], dim=1).to(device)
        opacity = torch.from_numpy(points['opacity'].copy()).to(device)
        del points
        gc.collect()
        log_info("Released CPU memory for raw point cloud data.")
        # --- 数据加载作用域结束 ---
        screenspace_points = torch.zeros_like(xyzs, dtype=xyzs.dtype, requires_grad=False, device="cuda") + 0
        means3D = xyzs
        means2D = screenspace_points

        # --- step 8: 加载相机数据 ---
        log_info("--- step 8: load camera data ---")
        fov_rad, cam_poses = get_video_cam_info(data_dir)
        aspect_ratio = img_width / img_height
        tanfovx = np.tan(fov_rad * 0.5)
        tanfovy = tanfovx / aspect_ratio
    
        znear, zfar = 0.01, 100.0
        f = 1.0 / np.tan(fov_rad / 2.0)
        projection_matrix = torch.from_numpy(build_projection_matrix(fov_rad, aspect_ratio, znear, zfar)).to(device, dtype=torch.float32)

        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        # --- step 9: 准备视频写入器 ---
        log_info("--- step 9: start render ---")
        video_path = os.path.join(output_dir, "render_video_gs.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        # 注意：cv2的尺寸是 (宽, 高)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height)) # 30是帧率
        for idx, pose_c2w in enumerate(tqdm(cam_poses, desc="GPU render progress")):
            # 1. 首先，正常计算出 OpenCV 约定下的视图矩阵 (W2C_cv)
            w2c_cv_np = np.linalg.inv(pose_c2w)
            # 2. 定义一个修正矩阵，用于从 OpenCV 相机空间转换到 OpenGL 相机空间
            #    这个矩阵的作用是翻转 Y 轴和 Z 轴
            cam_fix_matrix = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            # 3. 计算最终的、符合 OpenGL 约定的视图矩阵
            #    View_gl = Correction_Matrix @ W2C_cv
            view_matrix_gl_np = cam_fix_matrix @ w2c_cv_np
            
            # 4. 将最终正确的视图矩阵转换为 PyTorch 张量
            view_matrix = torch.from_numpy(view_matrix_gl_np).to(device, dtype=torch.float32)

            campos = torch.from_numpy(pose_c2w[:3, 3]).to(device, dtype=torch.float32)
            proj_matrix = projection_matrix @ view_matrix


            empty_tensor = torch.tensor([], device=device)
            args = (
                background,
                means3D,
                empty_tensor,
                opacity,
                scales,
                rotations,
                1.0,
                empty_tensor,
                view_matrix.T,
                proj_matrix.T,
                tanfovx,
                tanfovy,
                img_height,
                img_width,
                shs,
                0,
                campos,
                False,
                False,
            )
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

            if color.shape[0] == 3:
                rendered_image = color.permute(1, 2, 0).detach().cpu().numpy() # (C,H,W) -> (H,W,C)
                    
            #    这是解决 bpp 错误的关键
            image_numpy_uint8 = (rendered_image * 255).astype(np.uint8)
            image_numpy_uint8 = np.flipud(image_numpy_uint8)
            image_numpy_uint8 = np.fliplr(image_numpy_uint8)

            # --- 4. 将颜色通道从 RGB 转换为 BGR ---
            #    这是 OpenCV 的标准要求
            image_bgr = cv2.cvtColor(image_numpy_uint8, cv2.COLOR_RGB2BGR)

            # --- 5. 现在可以安全地显示图像了 ---
            # cv2.imshow("rendered_image", image_bgr)
            # cv2.waitKey(0)
            video_writer.write(np.array(image_bgr))
            torch.cuda.empty_cache()  # 重点，没这个卡死
        video_writer.release()

def render_by_3DGS_video(
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
        vid_cam_infos = get_vid_cam_from_bin(tar_width=img_width, tar_height=img_height)
        video_path = os.path.join(output_dir, "render_video_gs.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        # 注意：cv2的尺寸是 (宽, 高)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height)) # 30是帧率
        for idx, cam in enumerate(tqdm(vid_cam_infos, desc="GPU render progress")):
            cam.update_wh(img_width, img_height)
            image = render(cam, gaussians, pipe, background)["render"].detach()
            image_cpu = torch.clamp(image, 0.0, 1.0).cpu().numpy()
            image_cv = (image_cpu * 255).astype(np.uint8)
            image_cv = np.transpose(image_cv, (1, 2, 0))  # CHW -> HWC
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # RGB -> BGR
            video_writer.write(np.array(image_cv))
            # cv2.imwrite(f"render_img/{idx}.png", image_cv)
            # if idx % 5 == 0:
            torch.cuda.empty_cache()
        video_writer.release()

if __name__ == "__main__":
    dir = ROOT_DIR
    wh = [[100, 100], [616, 409], [1920, 1080]]
    wh_idx = 2

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
    parser.add_argument("--start_checkpoint", type=str, default = r"G:\Lab\GSSample\output\92825354-f\chkpnt7000.pth")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_by_3DGS(
    #     data_dir=dir,
    #     output_dir=dir,
    #     num_points_to_keep=NUM_POINTS_TO_KEEP, # 10_000_000
    #     opacity_threshold=None,
    #     img_width=wh[wh_idx][0],
    #     img_height=wh[wh_idx][1],
    #     fps=FPS,
    #     delete_images=True,
    #     point_size=1,
    #     logger=logger
    # )
    render_by_3DGS_video(
        lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint,
        data_dir=dir,
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


