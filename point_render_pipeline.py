'''
1. zip_point_cloud:
    读取3DGS点云，并压缩，存储在point_cloud_zip.ply
2. get_video_cam_info:
    读取transforms_video.json，获取相机信息
3. build_projection_matrix:
    构建透视投影矩阵
4. rasterize_points:
    光栅化和着色阶段
5. render_pipeline:
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
import math
import gc # 引入垃圾回收模块


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
    final_data['scale_0'] = np.exp(element.data['scale_0'])
    final_data['scale_1'] = np.exp(element.data['scale_1'])
    final_data['scale_2'] = np.exp(element.data['scale_2'])
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


def get_video_cam_info(json_path = os.getcwd()):
    '''
    返回一个列表，每个元素是一个numpy数组
    '''
    cam_path = os.path.join(json_path, "transforms_video.json")
    if not os.path.exists(cam_path):
        print(f"相机位姿文件不存在: {cam_path}")
        return None, []
    with open(cam_path, 'r') as f:
        cam_data = json.load(f)
    fov = cam_data['camera_angle_x']
    frames = []
    for frame in cam_data['frames']:
        # 将列表转换为numpy数组
        transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
        frames.append(transform_matrix)
    return fov, frames

def build_projection_matrix(fov_rad, aspect_ratio=1, znear=0.1, zfar=100):
    """构建一个标准的透视投影矩阵。"""
    f = 1.0 / np.tan(fov_rad / 2.0)
    return np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
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
    
    # --- step 2: 初始化 ModernGL ---
    log_info("--- step 2: init ModernGL context ---")
    # 初始化ModernGL离屏上下文
    ctx = moderngl.create_standalone_context()
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

    if version < 0.2:
        prog = ctx.program(vertex_shader=vert_shader, 
                        fragment_shader=frag_shader)
    elif version >= 0.2:
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
    log_info(f"point num: {len(points_world)}")

    # --- step 5 & 6: 创建并上传 VBOs ---
    log_info("--- step 5: create and upload data to GPU buffer ---")
    # 将颜色规范化到 [0, 1] for GLSL
    colors_gl = colors.astype('f4') / 255.0
    vbo_points = ctx.buffer(points_world.astype('f4').tobytes())
    vbo_colors = ctx.buffer(colors_gl.tobytes())
    vbo_scales = ctx.buffer(scales.astype('f4').tobytes())
    vbo_rotates = ctx.buffer(rotates.astype('f4').tobytes())
    vbo_opacity = ctx.buffer(opacity.astype('f4').tobytes())
    # --- step 6: 创建 VAO ---
    log_info("--- step 6: bind buffer to shader input ---")
    # 将缓冲区绑定到着色器的输入
    vao_content = [
        (vbo_points, '3f', 'in_vert'),
        (vbo_colors, '3f', 'in_color'),
        (vbo_scales, '3f', 'in_scale'),
        (vbo_rotates, '4f', 'in_rotate'),
        (vbo_opacity, '1f', 'in_opacity'),
    ]   
    if version >= 0.3:
        vbo_random_seeds = ctx.buffer(np.random.rand(len(points_world), 2).astype('f4').tobytes())
        vao_content.append((vbo_random_seeds, '2f', 'in_rand_seed'))

    vao_main = ctx.vertex_array(prog, vao_content)

    # --- step 7: 创建渲染目标 (FBOs) ---
    log_info("--- step 7: create frame buffer object (FBO) ---")
    if version <= 0.3:
        fbo = ctx.framebuffer(
            ctx.renderbuffer((img_width, img_height)),
            ctx.depth_renderbuffer((img_width, img_height))
        )
        fbo.use()
    else:
        # Pass 1 的渲染目标：一个带 RGBA 纹理的 FBO
        # 这个纹理将保存带有空洞的原始渲染结果
        texture_pass1 = ctx.texture((img_width, img_height), 4, dtype='f1') # f1 = 8-bit, f4 = 32-bit float
        depth_pass1 = ctx.depth_renderbuffer((img_width, img_height))
        fbo_pass1 = ctx.framebuffer(color_attachments=[texture_pass1], depth_attachment=depth_pass1)
        # Pass 2 的渲染目标：一个只带 RGB Renderbuffer 的 FBO
        # 这是我们的最终输出目标，我们直接从这里读取最终图像
        # 用 Renderbuffer 比用 Texture 更高效，因为我们不需要再对它进行采样
        color_pass2 = ctx.renderbuffer((img_width, img_height), 3) # RGB
        fbo_pass2 = ctx.framebuffer(color_attachments=[color_pass2])        
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

    # --- step 8: 加载相机数据 ---
    log_info("--- step 8: load camera data ---")
    fov_rad, cam_poses = get_video_cam_info(data_dir)
    aspect_ratio = img_width / img_height
    znear, zfar = 0.1, 100.0
    projection_matrix = build_projection_matrix(fov_rad, aspect_ratio, znear, zfar)

    # --- step 9: 准备视频写入器 ---
    log_info("--- step 9: start render ---")
    video_path = os.path.join(output_dir, "render_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # 注意：cv2的尺寸是 (宽, 高)
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height)) # 30是帧率
    start_time = time.time()
    # 设置时间uniform（如果存在）
    if 'u_time' in prog:
        prog['u_time'].value = start_time
    if 'scale_max' in prog:
        prog['scale_max'].value = scale_max
    for idx, pose_c2w in enumerate(tqdm(cam_poses, desc="GPU render progress")):
        if version >= 0.4:
            fbo_pass1.use()  # 激活离屏FBO
            fbo_pass1.clear(0.0, 0.0, 0.0, 0.0)  # 透明黑清空
            # 启用 Alpha 混合，让片元着色器可以输出透明度
            ctx.enable(moderngl.BLEND) 
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ctx.enable(moderngl.DEPTH_TEST)
        else:
            fbo.use() # 单通道直接渲染到屏幕
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            ctx.enable(moderngl.DEPTH_TEST)

        # 设置 uniforms
        view_matrix = np.linalg.inv(pose_c2w)
        proj_matrix = projection_matrix
        # 注意：GLSL中矩阵乘法是右乘，所以需要转置
        prog['view'].write(view_matrix.T.astype('f4').tobytes())
        prog['projection'].write(proj_matrix.T.astype('f4').tobytes())
        

        # 根据版本执行渲染
        if version == 0.1:
            ctx.point_size = 3
            vao_main.render(moderngl.POINTS)
        elif version == 0.2:
            # 设置点大小为最小
            ctx.point_size = 10
            prog['elem_size'].value = 0.01
            # num_samples_per_point = 100
            vao_main.render(moderngl.POINTS)
        elif version >= 0.3:
            # 启用 PROGRAM_POINT_SIZE 以便在 VS 中控制大小
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            ctx.point_size = 1
            # prog['elem_size'].value = 0.01
            num_samples_per_point = NUM_SAMPLES_PER_POINT
            prog['MAX_INSTANCES'].value = num_samples_per_point
            vao_main.render(moderngl.POINTS, instances=num_samples_per_point)

        # --- Pass 2: 后处理 (空洞填充) ---
        if version >= 0.4:
            fbo_pass2.use() # 切换回默认的屏幕帧缓冲
            fbo_pass2.clear(0.0, 0.0, 0.0, 1.0) # 用不透明黑清空屏幕
            ctx.disable(moderngl.DEPTH_TEST) # 后处理不需要深度测试
            # 激活后处理着色器
            
            # 绑定 Pass 1 的输出纹理到纹理单元 0
            texture_pass1.use(location=0)
            postproc_prog['u_render_texture'].value = 0 # 告诉GLSL去纹理单元0找

            # 渲染全屏四边形来触发填充逻辑
            postproc_vao.render(moderngl.TRIANGLE_STRIP)
        # --- 最终结果已经在屏幕上了 ---
        # 现在可以从屏幕帧缓冲读取数据来保存为图片或视频帧
        if version >= 0.4:
            image_bytes = fbo_pass2.read(components=3)
            image_np_rgb = np.frombuffer(image_bytes, dtype=np.uint8).reshape((img_height, img_width, 3))
            image_np_bgr = cv2.cvtColor(np.flipud(image_np_rgb), cv2.COLOR_RGB2BGR) # **翻转并转为BGR**
            video_writer.write(image_np_bgr)            
        else:
            # 从FBO读取渲染好的图像
            image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'BGR', 0, -1)
            # image.save(os.path.join(img_dir, f"frame_{idx:05d}.png"))
            video_writer.write(np.array(image))
    video_writer.release()
    log_info("save video to: " + video_path)
    log_info("--- step 11: render done! ---")
    

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, rasterize_gaussians, _C
# from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh

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

            # raster_settings = GaussianRasterizationSettings(
            #     image_height=int(img_height),
            #     image_width=int(img_width),
            #     tanfovx=tanfovx,
            #     tanfovy=tanfovy,
            #     bg=background,
            #     scale_modifier=1.0,
            #     viewmatrix=view_matrix.T,
            #     projmatrix=proj_matrix.T,
            #     sh_degree=0,
            #     campos=campos,
            #     prefiltered=False,
            #     debug=False
            # )  
            # rasterizer = GaussianRasterizer(raster_settings=raster_settings)


            # rendered_image, radii = rasterizer(
            #     means3D = means3D,
            #     means2D = means2D,
            #     shs = shs,
            #     colors_precomp = None,
            #     opacities = opacity,
            #     scales = scales,
            #     rotations = rotations,
            #     cov3D_precomp = None)
            # 显示渲染的图片
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




if __name__ == "__main__":
    dir = ROOT_DIR
    wh = [[100, 100], [616, 409], [1920, 1080]]
    wh_idx = 2
    render_pipeline(
        data_dir=dir, 
        output_dir=dir, 
        num_points_to_keep=NUM_POINTS_TO_KEEP, # 10_000_000
        opacity_threshold=None,
        img_width=wh[wh_idx][0], 
        img_height=wh[wh_idx][1],
        fps=FPS,
        delete_images=True,
        point_size=1,
        logger=logger)
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
