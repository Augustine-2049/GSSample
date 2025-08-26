'''
v0.5:
    1. 基于v0.5的RT渲染

1. zip_point_cloud:
    读取3DGS点云，并压缩，存储在point_cloud_zip.ply
2. get_video_cam_info_from_json:
    读取transforms_video.json，获取相机信息
3. get_video_cam_info_from_bin:
    读取video.bin，获取相机信息，注意这个放在场景中
4. build_projection_matrix:
    构建透视投影矩阵
5. rasterize_points:
    光栅化和着色阶段
6. render_pipeline:
    从点云到视频的渲染流程

'''
from plyfile import PlyData, PlyElement
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import moderngl
import sys
from pathlib import Path
import glob
import cv2
import time
import glfw
from camera_controller import InteractiveCamera # 导入我们创建的类
from collections import deque
from cam_pos.read_cam_pos import get_video_cam_info_from_json, get_video_cam_info_from_bin

# 动态获取项目根目录（假设 point_render.py 在 glsl/ 子目录下）
ROOT_DIR = Path(__file__).parent  # 向上两级到 GSSample/
sys.path.append(str(ROOT_DIR))  # 将根目录加入 Python 路径
mode = "GPU"
FPS = 10
# 200(k) 1000 = 30%black，28fps
# 200 1000 = 27%black, 12fps, no filter
# 2000 100 = 37%black, 14fps
# 2000 10 = 75%black 29fps
# 2000 10 = 50%black 29fps, no filter
NUM_POINTS_TO_KEEP = 1_000_000  # 1_000_000
NUM_SAMPLES_PER_POINT = 256# 256
SCALE_CEIL = 5E-5
OPACITY_CEIL = 0.5
# SCALE_CEIL = 5E-3
# OPACITY_CEIL = 1
USE_HALF_PRECISION = False
version = 0.5
wh_idx = 3
scene_name = "drums" # op : ["drums", "ficus", "hotdog", "lego", "materials", "mic", "ship", "chair"] 
point_cloud_dir = r"H:\800_DataSet\NeRF_Synthesis\3DGS_res_7000" + "\\" + scene_name + "\\" + "point_cloud" + "\\" + "iteration_7000"

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






def zip_point_cloud(src_path = ".", num_to_keep: int = 100000, opacity_threshold=0.01, tar_path = "."):
    # 1. 读取点云文件
    src_path = os.path.join(src_path, "point_cloud.ply")
    if not os.path.exists(src_path):
        print(f"原生3DGS点云文件不存在: {src_path}")
    if not os.path.exists(tar_path):
        print(f"压缩后的目标位置非法: {tar_path}")
        return
    tar_path = os.path.join(tar_path, "point_cloud_zip.ply")

    plydata = PlyData.read(src_path)
    element = plydata['vertex'] # 等价于plydata.elements[0]

    # 2. 选取排序后前 N 个点
    # 如果 num_to_keep 大于总点数，切片会自动处理，只返回所有点
    sorted_data = np.sort(element.data, order='opacity')[::-1]
    if opacity_threshold is not None:
        
        sorted_data = sorted_data # sorted_data[sorted_data['opacity'] > opacity_threshold]
    if num_to_keep is not None:
        sorted_data = sorted_data[:num_to_keep].copy()
    # 随机取点
    # random_indices = np.random.choice(len(sorted_data), num_to_keep, replace=False)
    # sorted_data = sorted_data[random_indices]
    element = PlyElement.describe(sorted_data, 'vertex')

    # 3. 将数据转换为最终的格式
    num_points = len(element.data)
    final_dtype = np.dtype([
        ('x', 'f4'), 
        ('y', 'f4'), 
        ('z', 'f4'),
        ('opacity', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('scale_0', 'f4'),
        ('scale_1', 'f4'),
        ('scale_2', 'f4'),
        ('rot_0', 'f4'),
        ('rot_1', 'f4'),
        ('rot_2', 'f4'),
        ('rot_3', 'f4')
    ])
    # 创建目标：一个空的、具有目标结构的 NumPy 数组，它的大小和原始点云的点的数量相同
    final_data = np.empty(num_points, dtype=final_dtype)
    final_data['x'] = element.data['x']
    final_data['y'] = element.data['y']
    final_data['z'] = element.data['z']
    final_data['opacity'] = torch.sigmoid(torch.from_numpy(element.data['opacity'])).numpy()
    final_data['scale_0'] = np.exp(element.data['scale_0']) * 3
    final_data['scale_1'] = np.exp(element.data['scale_1']) * 3
    final_data['scale_2'] = np.exp(element.data['scale_2']) * 3
    final_data['rot_0'] = element.data['rot_0']
    final_data['rot_1'] = element.data['rot_1']
    final_data['rot_2'] = element.data['rot_2']
    final_data['rot_3'] = element.data['rot_3']
    rotate_len = np.linalg.norm(np.column_stack([final_data['rot_0'], final_data['rot_1'], final_data['rot_2'], final_data['rot_3']]), axis=1)
    final_data['rot_0'] = final_data['rot_0'] / rotate_len
    final_data['rot_1'] = final_data['rot_1'] / rotate_len
    final_data['rot_2'] = final_data['rot_2'] / rotate_len
    final_data['rot_3'] = final_data['rot_3'] / rotate_len
    SH_C0 = 0.28209479177387814
    final_data['red']   = (element.data['f_dc_0'] * SH_C0 + 0.5).clip(0, 1) * 255
    final_data['green'] = (element.data['f_dc_1'] * SH_C0 + 0.5).clip(0, 1) * 255
    final_data['blue']  = (element.data['f_dc_2'] * SH_C0 + 0.5).clip(0, 1) * 255
    final_data = final_data.astype(final_dtype)

    # 4. 保存到ply文件
    new_element = PlyElement.describe(final_data, 'vertex')
    new_plydata = PlyData([new_element])
    new_plydata.write(tar_path)
    print(f"点云已保存到 {tar_path}")
    return new_plydata




def build_projection_matrix(fov_x, fov_y, aspect_ratio=1, znear=0.01, zfar=100):
    """
    构建一个标准的透视投影矩阵。
    znear
    """
    fx = 1.0 / np.tan(fov_x / 2.0)
    fy = 1.0 / np.tan(fov_y / 2.0)
    return np.array([
        [fx, 0, 0, 0],
        [0, fy, 0, 0],
        [0, 0, (zfar + znear) / (znear - zfar), (2 * zfar * znear) / (znear - zfar)],
        [0, 0, -1, 0]
    ], dtype=np.float32)


def render_pipeline(
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
    # --- step 0: 日志和文件夹设置 ---
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

    # --- step 1: 初始化 GLFW 窗口 ---
    log_info("--- Initializing GLFW Window ---")
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    # 创建一个可见窗口
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(img_width, img_height, "Interactive 3DGS Renderer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)

    # --- step 2: 初始化 ModernGL (现在使用窗口上下文) ---
    log_info("--- step 2: init ModernGL context ---")
    ctx = moderngl.create_context()

    # --- 设置键盘回调 ---
    motion_state = {} # 存储按键状态的字典
    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_W: motion_state['forward'] = True
            if key == glfw.KEY_S: motion_state['backward'] = True
            if key == glfw.KEY_A: motion_state['left'] = True
            if key == glfw.KEY_D: motion_state['right'] = True
            if key == glfw.KEY_SPACE: motion_state['down'] = True
            if key == glfw.KEY_LEFT_CONTROL: motion_state['up'] = True
            if key == glfw.KEY_Q: motion_state['rot_left'] = True
            if key == glfw.KEY_E: motion_state['rot_right'] = True
        elif action == glfw.RELEASE:
            if key == glfw.KEY_W: motion_state['forward'] = False
            if key == glfw.KEY_S: motion_state['backward'] = False
            if key == glfw.KEY_A: motion_state['left'] = False
            if key == glfw.KEY_D: motion_state['right'] = False
            if key == glfw.KEY_SPACE: motion_state['down'] = False
            if key == glfw.KEY_LEFT_CONTROL: motion_state['up'] = False
            if key == glfw.KEY_Q: motion_state['rot_left'] = False
            if key == glfw.KEY_E: motion_state['rot_right'] = False
            
    glfw.set_key_callback(window, key_callback)



    
    # --- step 2: 初始化 ModernGL ---
    log_info("--- step 2: init ModernGL context ---")
    # 初始化ModernGL离屏上下文
    # ctx = moderngl.create_standalone_context()
    ctx.enable(moderngl.DEPTH_TEST) # 全局启用深度测试
    
    # --- step 3: 编译着色器程序 ---
    log_info("--- step 3: compile shader program ---")
    global version
    log_info(f"version: {version}")
    path = Path(__file__).parent
    with open(os.path.join(path, 'glsl', f'vs_v{version:.1f}.glsl'), 'r', encoding="utf-8") as f:
        vert_shader = f.read()
        log_info(f"vert_shader: {vert_shader}")

    with open(os.path.join(path, 'glsl', f'fs_v{version:.1f}.glsl'), 'r', encoding="utf-8") as f:
        frag_shader = f.read()
        log_info(f"frag_shader: {frag_shader}")
    geom_shader = None

    # 在渲染循环中设置 point_size
    # 这个值需要非常小，因为它是在-1到1的裁剪空间中
    # 比如 0.01 对应屏幕尺寸的 1%
    with open(os.path.join(path, 'glsl', f'gs_v{version:.1f}.glsl'), 'r', encoding="utf-8") as f:
        geom_shader = f.read()
        log_info(f"geom_shader: {geom_shader}")
    prog = ctx.program(vertex_shader=vert_shader, 
                    geometry_shader=geom_shader,
                    fragment_shader=frag_shader,
                    # varyings=['captured_gl_position']
    )


    # --- step 4: 加载和预处理点云数据 ---
    log_info("--- step 4: load point cloud data ---")
    points = zip_point_cloud(data_dir, num_points_to_keep, opacity_threshold=opacity_threshold).elements[0].data
    colors = np.column_stack([points['red'], points['green'], points['blue']])
    scales = np.column_stack([points['scale_0'], points['scale_1'], points['scale_2']])
    rotates = np.column_stack([points['rot_0'], points['rot_1'], points['rot_2'], points['rot_3']])
    points_world = np.column_stack([points['x'], points['y'], points['z']])
    opacity = points['opacity']
    scale = points['scale_0'] * points['scale_1'] * points['scale_2']
    scale_max = np.max(scale)
    scale_min = np.min(scale)
    log_info(f"scale_max: {scale_max}, scale_min: {scale_min}")
    opacity_max = np.max(opacity)
    opacity_min = np.min(opacity)
    log_info(f"opacity_max: {opacity_max}, opacity_min: {opacity_min}")
    point_num = len(points_world)
    # 统计scale > scale_max 和 opacity > opacity_max 的点数
    scale_max_cnt = np.sum(scale > SCALE_CEIL)
    opacity_max_cnt = np.sum(opacity > OPACITY_CEIL)
    log_info(f"scale > scale_max: {scale_max_cnt}, opacity > opacity_max: {opacity_max_cnt}")

    scaled_ratio = np.clip(scale / SCALE_CEIL, 0, 1)
    opacity_ratio = np.clip(opacity / OPACITY_CEIL, 0, 1)
    cnt = (scaled_ratio * opacity_ratio * NUM_SAMPLES_PER_POINT).astype(np.int32)
    
    # 统计边界值情况
    cnt_zero = np.sum(cnt == 0)
    cnt_max = np.sum(cnt >= NUM_SAMPLES_PER_POINT)
    cnt_total = len(cnt)

    cnt[cnt == 0] = 1



    # --- step 5 & 6: 创建并上传 VBOs ---
    log_info("--- step 5: create and upload data to GPU buffer ---")
    # 将颜色规范化到 [0, 1] for GLSL
    colors_gl = colors.astype('f4') / 255.0
    vbo_points = ctx.buffer(points_world.astype('f4').tobytes())
    vbo_colors = ctx.buffer(colors_gl.tobytes())
    vbo_scales = ctx.buffer(scales.astype('f4').tobytes())
    vbo_rotates = ctx.buffer(rotates.astype('f4').tobytes())
    vbo_opacity = ctx.buffer(opacity.astype('f4').tobytes())
    vbo_cnt = ctx.buffer(cnt.astype('i4').tobytes())
    # ssbo_cnt = ctx.buffer(cnt.astype('i4').tobytes()) # 创建一个 SSBO
    # --- step 6: 创建 VAO ---
    log_info("--- step 6: bind buffer to shader input ---")
    # 将缓冲区绑定到着色器的输入
    vao_content = [
        (vbo_points, '3f', 'in_vert'),
        (vbo_colors, '3f', 'in_color'),
        (vbo_scales, '3f', 'in_scale'),
        (vbo_rotates, '4f', 'in_rotate'),
        # (vbo_opacity, '1f', 'in_opacity'),
        (vbo_cnt, '1i', 'in_cnt'),
    ]   
    vbo_random_seeds = ctx.buffer(np.random.rand(len(points_world), 2).astype('f4').tobytes())
    vao_content.append((vbo_random_seeds, '2f', 'in_rand_seed'))

    vao_main = ctx.vertex_array(prog, vao_content)




    # --- step 8 : 创建渲染目标 (FBOs) ---
    log_info("--- step 8: create frame buffer object (FBO) ---")


    # Pass 1 的渲染目标：一个带 RGBA 纹理的 FBO
    # 这个纹理将保存带有空洞的原始渲染结果
    texture_pass1 = ctx.texture((img_width, img_height), 4, dtype='f1') # f1 = 8-bit, f4 = 32-bit float
    depth_pass1 = ctx.depth_renderbuffer((img_width, img_height))
    fbo_pass1 = ctx.framebuffer(color_attachments=[texture_pass1], depth_attachment=depth_pass1)
    # Pass 2 的渲染目标：一个只带 RGB Renderbuffer 的 FBO
    # 这是我们的最终输出目标，我们直接从这里读取最终图像
    # 用 Renderbuffer 比用 Texture 更高效，因为我们不需要再对它进行采样
    # 此时实时需要texture
    # **将 Pass 2 的 FBO 的颜色附件也改为一个 Texture**
    # 这样它的结果才能被后续的 Blit Pass 读取
    texture_pass2 = ctx.texture((img_width, img_height), 4, dtype='f1') # 使用 RGBA
    fbo_pass2 = ctx.framebuffer(color_attachments=[texture_pass2])        
    with open(os.path.join(path, 'glsl', f'postproc_vs_v{version:.1f}.glsl'), 'r', encoding="utf-8") as f:
        postproc_vert_shader = f.read()
        log_info(f"postproc_vert_shader: {postproc_vert_shader}")
    with open(os.path.join(path, 'glsl', f'postproc_fs_v{version:.1f}.glsl'), 'r', encoding="utf-8") as f:
        postproc_frag_shader = f.read()
        log_info(f"postproc_frag_shader: {postproc_frag_shader}")
    postproc_prog = ctx.program(vertex_shader=postproc_vert_shader,
                                fragment_shader=postproc_frag_shader)
    postproc_vao = ctx.vertex_array(postproc_prog, [
        (ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32).tobytes()), '2f', 'in_vert')
    ])

    # --- step 9: 加载相机数据 ---
    log_info("--- step 9: load camera data ---")
    camera = InteractiveCamera() # 创建我们的交互相机实例
    scene_dir = r"G:\Lab\NGSassist\data\bicycle"
    fov_x, fov_y, cam_poses = get_video_cam_info_from_bin(scene_dir, img_width, img_height)
    
    # --- 鼠标控制相关变量 ---
    last_mouse_x = img_width / 2
    last_mouse_y = img_height / 2
    first_mouse = True
    mouse_sensitivity = 0.1
    
    # 鼠标回调函数
    def mouse_callback(window, xpos, ypos):
        nonlocal last_mouse_x, last_mouse_y, first_mouse
        
        if first_mouse:
            last_mouse_x = xpos
            last_mouse_y = ypos
            first_mouse = False
            
        xoffset = xpos - last_mouse_x
        yoffset = ypos - last_mouse_y  # 修复Y轴方向
        last_mouse_x = xpos
        last_mouse_y = ypos
        
        xoffset *= mouse_sensitivity
        yoffset *= mouse_sensitivity
        
        camera.yaw += xoffset
        camera.pitch += yoffset  # 修复pitch方向
        
        # 限制pitch角度
        camera.pitch = np.clip(camera.pitch, -89.0, 89.0)
        camera.update_vectors()
    
    # 设置鼠标回调
    glfw.set_cursor_pos_callback(window, mouse_callback)
    # 隐藏鼠标光标
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    
    # --- 键盘控制相关变量 ---
    motion_state = {
        'forward': False,
        'backward': False,
        'left': False,
        'right': False,
        'up': False,
        'down': False,
        'rot_left': False,
        'rot_right': False
    }
    
    # 键盘回调函数
    def key_callback(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_W and action == glfw.PRESS:
            motion_state['forward'] = True
        elif key == glfw.KEY_W and action == glfw.RELEASE:
            motion_state['forward'] = False
        elif key == glfw.KEY_S and action == glfw.PRESS:
            motion_state['backward'] = True
        elif key == glfw.KEY_S and action == glfw.RELEASE:
            motion_state['backward'] = False
        elif key == glfw.KEY_A and action == glfw.PRESS:
            motion_state['left'] = True
        elif key == glfw.KEY_A and action == glfw.RELEASE:
            motion_state['left'] = False
        elif key == glfw.KEY_D and action == glfw.PRESS:
            motion_state['right'] = True
        elif key == glfw.KEY_D and action == glfw.RELEASE:
            motion_state['right'] = False
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            motion_state['down'] = True
        elif key == glfw.KEY_SPACE and action == glfw.RELEASE:
            motion_state['down'] = False
        elif key == glfw.KEY_LEFT_SHIFT and action == glfw.PRESS:
            motion_state['up'] = True
        elif key == glfw.KEY_LEFT_SHIFT and action == glfw.RELEASE:
            motion_state['up'] = False
        elif key == glfw.KEY_Q and action == glfw.PRESS:
            motion_state['rot_left'] = True
        elif key == glfw.KEY_Q and action == glfw.RELEASE:
            motion_state['rot_left'] = False
        elif key == glfw.KEY_E and action == glfw.PRESS:
            motion_state['rot_right'] = True
        elif key == glfw.KEY_E and action == glfw.RELEASE:
            motion_state['rot_right'] = False
    
    # 设置键盘回调
    glfw.set_key_callback(window, key_callback)
    aspect_ratio = img_width / img_height
    znear, zfar = 0.01, 100.0
    projection_matrix = build_projection_matrix(fov_x, fov_y, aspect_ratio, znear, zfar)
     # --- 创建一个用于将最终纹理画到屏幕的简单 Program ---
    blit_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_vert;
            out vec2 v_texCoord;
            void main() {
                v_texCoord = in_vert * 0.5 + 0.5;
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            uniform sampler2D u_texture;
            in vec2 v_texCoord;
            out vec4 f_color;
            void main() {
                vec2 flipped_texCoord = vec2(v_texCoord.x, 1.0 - v_texCoord.y);
                f_color = texture(u_texture, flipped_texCoord);
            }
        """
    )
    blit_vao = ctx.vertex_array(blit_prog, [(ctx.buffer(np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4')), '2f', 'in_vert')])
   
    ###############################################################################################
    start_time = time.time()
    # 设置时间uniform（如果存在）
    if 'u_time' in prog:
        prog['u_time'].value = start_time
    if 'scale_max' in prog:
        prog['scale_max'].value = SCALE_CEIL
    if 'opacity_max' in prog:
        prog['opacity_max'].value = OPACITY_CEIL

    # --- 在进入循环之前，创建Query对象 ---
    query_pass1 = ctx.query(time=True)
    query_pass2 = ctx.query(time=True)

    # 创建一个列表来存储耗时结果
    pass1_times = []
    pass2_times = []
    black_pixels = 0.0
    total_pixels = img_width * img_height
    view_matrixs = []
    for pose_c2w in cam_poses:
        view_matrix = np.linalg.inv(pose_c2w)
        view_matrixs.append(view_matrix)
    proj_matrix = projection_matrix

    # FPS计算相关变量
    frame_times = deque(maxlen=60)  # 存储最近60帧的时间
    last_time = glfw.get_time()
    frame_count = 0
    last_fps_output_time = last_time  # 记录上次FPS输出的时间
    
    # 打印控制说明
    print("\n=== 3D Gaussian Splatting 实时渲染控制 ===")
    print("鼠标: 控制视角旋转")
    print("W/S: 前进/后退")
    print("A/D: 左移/右移")
    print("空格/Shift: 下降/上升")
    print("Q/E: 左转/右转")
    print("ESC: 退出程序")
    print("==========================================\n")
    while not glfw.window_should_close(window):
        # --- 1. 事件和时间处理 ---
        glfw.poll_events()
        
        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time
        
        # FPS计算
        frame_times.append(delta_time)
        frame_count += 1
        if len(frame_times) > 1:
            avg_frame_time = sum(frame_times) / len(frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            current_fps = 0
            
        # 每3秒输出一次FPS信息到控制台（固定一行）
        if current_time - last_fps_output_time >= 3.0:
            print(f"\rCurrent FPS: {current_fps:.1f}", end='', flush=True)
            last_fps_output_time = current_time
        
        # --- 2. 更新相机 ---
        camera.update(motion_state, delta_time)
        view_matrix = camera.get_view_matrix() # 获取最新的视图矩阵


        # --- 4. Pass 1 & 2: 渲染和后处理 (逻辑不变) ---
        # ... (设置主渲染 prog 的 uniforms, view=view_matrix.T) ...
        # --- pass1 start ---
        with query_pass1:
            
            # 告诉主渲染着色器，可见索引在哪里
            # 设置 uniforms
            # 注意：GLSL中矩阵乘法是右乘，所以需要转置
            prog['view'].write(view_matrix.T.astype('f4').tobytes())
            prog['projection'].write(proj_matrix.T.astype('f4').tobytes())
            fbo_pass1.use()  # 激活离屏FBO
            fbo_pass1.clear(0.0, 0.0, 0.0, 0.0)  # 透明黑清空
            # 启用 Alpha 混合，让片元着色器可以输出透明度
            ctx.enable(moderngl.BLEND) 
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            # 设置ctx.enable
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)  # 启用 PROGRAM_POINT_SIZE 以便在 VS 中控制大小
            ctx.point_size = 1
            # prog['elem_size'].value = 0.01
            prog['MAX_INSTANCES'].value = NUM_SAMPLES_PER_POINT
            vao_main.render(moderngl.POINTS, vertices=point_num, instances=NUM_SAMPLES_PER_POINT)
            # vao_main.render_indirect(indirect_cmd_buffer, mode=moderngl.POINTS, count=num_visible_commands)


        # --- Pass 2: 后处理 (空洞填充) ---
        with query_pass2:
            fbo_pass2.use() # 切换回默认的屏幕帧缓冲
            fbo_pass2.clear(0.0, 0.0, 0.0, 1.0) # 用不透明黑清空屏幕
            ctx.disable(moderngl.DEPTH_TEST) # 后处理不需要深度测试
            ctx.disable(moderngl.BLEND)
            # 激活后处理着色器
            
            # 绑定 Pass 1 的输出纹理到纹理单元 0
            texture_pass1.use(location=0)
            postproc_prog['u_render_texture'].value = 0 # 告诉GLSL去纹理单元0找

            # 渲染全屏四边形来触发填充逻辑
            postproc_vao.render(moderngl.TRIANGLE_STRIP)
        # --- pass2 end ---
        
        # === 结果读取和计时 ===
        # 在当前帧的最后，把上一帧的耗时结果存起来
        # query.elapsed 返回的是纳秒 (nanoseconds)
        pass1_times.append(query_pass1.elapsed / 1_000_000) # 转换为毫秒
        pass2_times.append(query_pass2.elapsed / 1_000_000) # 转换为毫秒
        
        # --- 5. 将最终结果画到屏幕上 ---
        ctx.screen.use() # 激活窗口的默认帧缓冲
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        ################## 基元分析改成了pass1，正常是pass2 #########################################
        texture_pass1.use(location=0) # 假设 fbo_pass2 的输出是 texture_pass2
        blit_prog['u_texture'].value = 0
        blit_vao.render(moderngl.TRIANGLE_STRIP)
        
        # --- 6. 交换缓冲区 ---
        glfw.swap_buffers(window)
        
    # --- 清理 ---
    glfw.terminate()
    
    black_pixels = black_pixels / len(cam_poses)
    log_info("--- step 11: render done! ---")

    # --- 循环结束后，计算并打印平均耗时 ---
    # 忽略第一帧可能因JIT编译等产生的抖动
    avg_pass1_ms = np.mean(pass1_times[1:]) 
    avg_pass2_ms = np.mean(pass2_times[1:])
    
    log_info(f"cnt == 0: {cnt_zero} ({cnt_zero/cnt_total*100:.2f}%)")
    log_info(f"cnt >= {NUM_SAMPLES_PER_POINT}: {cnt_max} ({cnt_max/cnt_total*100:.2f}%)")
    log_info(f"cnt in (0, {NUM_SAMPLES_PER_POINT}): {cnt_total - cnt_zero - cnt_max} ({(cnt_total - cnt_zero - cnt_max)/cnt_total*100:.2f}%)")
    log_info(f"scene point num: {point_num}")
    log_info(f"black pixels: {black_pixels * 100:.2f}%")
    log_info(f"Average Pass 1 time: {avg_pass1_ms:.4f} ms")
    log_info(f"Average Pass 2 time: {avg_pass2_ms:.4f} ms")    



if __name__ == "__main__":
    dir = ROOT_DIR
    wh = [[100, 100], [616, 409], [1920, 1080], [800, 800]]
    render_pipeline(
        data_dir=point_cloud_dir,
        output_dir=dir,
        num_points_to_keep=NUM_POINTS_TO_KEEP, # 10_000_000
        opacity_threshold=None,
        img_width=wh[wh_idx][0],
        img_height=wh[wh_idx][1],
        fps=FPS,
        delete_images=True,
        point_size=1,
        logger=logger)

