'''
1. zip_point_cloud:
    读取3DGS点云，并压缩，存储在point_cloud_zip.ply
2. get_video_cam_info:
    读取transforms_video.json，获取相机信息
3. render_with_GPU:
    使用GPU渲染，存储在render_img文件夹中
4. render_with_CPU:
    使用CPU渲染，存储在render_img文件夹中
5. create_video_from_images:
    将render_img文件夹中的图片合成视频
6. render_pipeline:
    从点云到视频的渲染流程（render_with_GPU + create_video_from_images）

'''
from plyfile import PlyData, PlyElement
import os
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



# 动态获取项目根目录（假设 point_render.py 在 glsl/ 子目录下）
ROOT_DIR = Path(__file__).parent  # 向上两级到 GSSample/
sys.path.append(str(ROOT_DIR))  # 将根目录加入 Python 路径
mode = "GPU"
version = 0.2

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






def zip_point_cloud(num_to_keep: int = 100000,src_path = ".", tar_path = "."):
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
    num_to_keep = min(len(sorted_data), num_to_keep) # 确保不越界
    # filtered_data = sorted_data[:num_to_keep]
    # 随机取点
    random_indices = np.random.choice(len(sorted_data), num_to_keep, replace=False)
    filtered_data = sorted_data[random_indices]
    element = PlyElement.describe(filtered_data, 'vertex')

    # 3. 将数据转换为最终的格式
    num_points = len(element.data)
    final_dtype = np.dtype([
        ('x', 'f4'), 
        ('y', 'f4'), 
        ('z', 'f4'),
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
    final_data['scale_0'] = element.data['scale_0']
    final_data['scale_1'] = element.data['scale_1']
    final_data['scale_2'] = element.data['scale_2']
    final_data['rot_0'] = element.data['rot_0']
    final_data['rot_1'] = element.data['rot_1']
    final_data['rot_2'] = element.data['rot_2']
    final_data['rot_3'] = element.data['rot_3']
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
    ])

def rasterize_points(projected_points, colors, width, height, point_size=1):
    """
    光栅化和着色阶段 (未来可替换和优化的'Shader')
    
    参数:
    - projected_points: (N, 3) NumPy 数组，包含 (x, y, depth) 的屏幕坐标。
    - colors: (N, 3) NumPy 数组，包含 (r, g, b) 颜色。
    
    返回:
    - 一个 (height, width, 3) 的图像数组。
    """
    # 初始化图像缓冲区和深度缓冲区
    image_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), float('inf'), dtype=np.float32)

    # 为了绘制大于1像素的点，计算偏移量
    half_size = point_size // 2
    
    # 按照深度从远到近排序点，这样可以减少深度缓冲区的写入次数 (这是一个优化)
    sorted_indices = np.argsort(projected_points[:, 2])[::-1]
    
    for i in sorted_indices:
        x, y, depth = projected_points[i]
        
        # 将浮点坐标转换为整数像素中心
        ix, iy = int(round(x)), int(round(y))

        # 遍历点覆盖的像素区域
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                px, py = ix + dx, iy + dy
                
                # 边界检查
                if not (0 <= px < width and 0 <= py < height):
                    continue

                # 深度测试 (Z-buffering)
                if depth < depth_buffer[py, px]:
                    # 测试通过，更新深度并为像素着色
                    depth_buffer[py, px] = depth
                    image_buffer[py, px] = colors[i]
                    
    return image_buffer



def render_with_CPU(
    data_dir=".", 
    output_dir=".", 
    num_points_to_keep=100000, 
    img_width=100, 
    img_height=100,
    point_size=2,
    logger=None
):
    """
    使用纯NumPy和Pillow实现的渲染主函数。
    """
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
    output_dir = os.path.join(output_dir, "render_img")
    os.makedirs(output_dir, exist_ok=True)
    # 清空原本的图片
    del_cnt = 0
    for file in os.listdir(output_dir):
        if file.endswith(".png"):
            del_cnt += 1
            os.remove(os.path.join(output_dir, file))
    log_info(f"del {del_cnt} pics")
    log_info("--- step 1: load and process point cloud ---")
    points = zip_point_cloud(num_points_to_keep, data_dir).elements[0].data
    # 将结构化数组转换为普通numpy数组
    points_world = np.column_stack([points['x'], points['y'], points['z']])
    colors = np.column_stack([points['red'], points['green'], points['blue']])

    log_info(f"success load {len(points_world)} points.")

    log_info("\n--- step 2: load camera info ---")
    fov_rad, cam_poses = get_video_cam_info(data_dir)
    log_info(f"success load {len(cam_poses)} camera poses.")
    
    # 调试信息
    log_info(f"point cloud data shape: {points_world.shape}")
    log_info(f"color data shape: {colors.shape}")
    log_info(f"camera FOV: {fov_rad:.6f} rad ({fov_rad*180/np.pi:.1f} deg)")
    if len(cam_poses) > 0:
        log_info(f"first camera pose matrix shape: {cam_poses[0].shape}")

    # 准备用于矩阵乘法的齐次坐标 (N, 4)
    points_homogeneous = np.hstack([points_world, np.ones((len(points_world), 1))])
    log_info(f"homogeneous coordinate shape: {points_homogeneous.shape}")
    
    # 渲染参数
    aspect_ratio = img_width / img_height
    znear, zfar = 0.1, 100.0
    
    # 构建投影矩阵 (在循环外完成，因为它不随相机位姿改变)
    projection_matrix = build_projection_matrix(fov_rad, aspect_ratio, znear, zfar)
    
    log_info(f"\n--- step 3: start render {len(cam_poses)} frames ---")
    for idx, pose_c2w in enumerate(tqdm(cam_poses, desc="render progress")):
        # --- 视图变换 (World -> Camera) ---
        view_matrix = np.linalg.inv(pose_c2w)
        if idx % 100 == 0:
            log_info(f"{idx} frame:")
            log_info(f"cam pose: {pose_c2w}")
            log_info(f"view matrix: {view_matrix}")
            log_info(f"proj matrix: {projection_matrix}")
        points_camera = points_homogeneous @ view_matrix.T

        # --- 投影变换 (Camera -> Clip Space) ---
        points_clip = points_camera @ projection_matrix.T
        if idx % 100 == 0:
            log_info(f"points_clip[:5]: {points_clip[:5]}")
        # --- 视锥体裁剪 & 透视除法 (Clip -> NDC) ---
        # w分量用于透视
        w = points_clip[:, 3]
        
        # 裁剪掉相机后方(w<0)或太近(z>w)或太远(z<-w)的点
        # 这是一个简化的裁剪，仅保留相机前方的点
        visible_mask = (w > znear)
        
        if not np.any(visible_mask):
            # 如果没有可见点，直接保存黑色图片
            Image.new('RGB', (img_width, img_height), (0, 0, 0)).save(os.path.join(output_dir, f"frame_{idx:05d}.png"))
            continue

        # 只处理可见的点
        points_clip_visible = points_clip[visible_mask]
        colors_visible = colors[visible_mask]
        w_visible = w[visible_mask, np.newaxis]
        
        # 透视除法，得到[-1, 1]范围的NDC坐标
        points_ndc = points_clip_visible[:, :3] / w_visible
        
        # --- 视口变换 (NDC -> Screen) ---
        screen_x = (points_ndc[:, 0] + 1) * 0.5 * img_width
        screen_y = (1 - points_ndc[:, 1]) * 0.5 * img_height # Y轴翻转
        screen_z = points_ndc[:, 2] # 深度值在[-1, 1]

        projected_points = np.vstack([screen_x, screen_y, screen_z]).T

        # --- 光栅化/着色 ---
        image_array = rasterize_points(projected_points, colors_visible, img_width, img_height, point_size)

        # --- 保存图像 ---
        image = Image.fromarray(image_array, 'RGB')
        image.save(os.path.join(output_dir, f"frame_{idx:05d}.png"))
        
    log_info(f"\nrender done! all images saved to: {output_dir}")



def render_with_GPU(
    data_dir=".", 
    output_dir=".", 
    num_points_to_keep=100000, 
    img_width=100, 
    img_height=100,
    point_size=2,
    logger=None
):
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
    output_dir = os.path.join(output_dir, "render_img")
    os.makedirs(output_dir, exist_ok=True)
    # 清空原本的图片
    del_cnt = 0
    for file in os.listdir(output_dir):
        if file.endswith(".png"):
            os.remove(os.path.join(output_dir, file))
            del_cnt += 1
    log_info(f"del {del_cnt} pics")
    
    log_info("--- step 2: init ModernGL context ---")
    # 初始化ModernGL离屏上下文
    ctx = moderngl.create_standalone_context()
    
    log_info("--- step 3: compile shader program ---")
    # 编译着色器程序
    version = 0.1
    # 1. 加载着色器代码
    path = Path(__file__).parent
    with open(os.path.join(path, 'glsl', f'vs_v{version:.1f}.glsl'), 'r', encoding="utf-8") as f:
        vert_shader = f.read()
        log_info(f"vert_shader: {vert_shader}")
    with open(os.path.join(path, 'glsl', f'fs_v{version:.1f}.glsl'), 'r', encoding="utf-8") as f:
        frag_shader = f.read()
        log_info(f"frag_shader: {frag_shader}")
    prog = ctx.program(vertex_shader=vert_shader, 
                       fragment_shader=frag_shader)

    log_info("--- step 4: load point cloud data ---")
    # 加载点云数据
    points = zip_point_cloud(num_points_to_keep, data_dir).elements[0].data
    colors = np.column_stack([points['red'], points['green'], points['blue']])
    points_world = np.column_stack([points['x'], points['y'], points['z']])
    log_info(f"point num: {len(points_world)}")

    log_info("--- step 5: normalize colors ---")
    # 将颜色规范化到 [0, 1] for GLSL
    colors_gl = colors.astype('f4') / 255.0

    log_info("--- step 6: create and upload data to GPU buffer ---")
    # 创建并上传数据到GPU缓冲区
    vbo_points = ctx.buffer(points_world.astype('f4').tobytes())
    vbo_colors = ctx.buffer(colors_gl.tobytes())

    log_info("--- step 7: bind buffer to shader input ---")
    # 将缓冲区绑定到着色器的输入
    vao_content = [
        (vbo_points, '3f', 'in_vert'),
        (vbo_colors, '3f', 'in_color')
    ]
    vao = ctx.vertex_array(prog, vao_content)

    log_info("--- step 8: create frame buffer object (FBO) ---")
    # 创建一个用于接收渲染结果的帧缓冲对象 (FBO)
    fbo = ctx.framebuffer(
        ctx.renderbuffer((img_width, img_height)),
        ctx.depth_renderbuffer((img_width, img_height))
    )
    fbo.use()
    
    log_info("--- step 9: load camera data ---")
    # 加载相机数据
    fov_rad, cam_poses = get_video_cam_info(data_dir)
    aspect_ratio = img_width / img_height
    znear, zfar = 0.1, 100.0
    projection_matrix = build_projection_matrix(fov_rad, aspect_ratio, znear, zfar)

    log_info("--- step 10: start render ---")
    for idx, pose_c2w in enumerate(tqdm(cam_poses, desc="GPU render progress")):
        # 构建视图和投影矩阵 (和之前一样)
        view_matrix = np.linalg.inv(pose_c2w)
        proj_matrix = projection_matrix
        if idx % 100 == 0:
            log_info(f"{idx} frame:")
            log_info(f"cam pose: {pose_c2w}")
            log_info(f"view matrix: {view_matrix}")
            log_info(f"proj matrix: {proj_matrix}")
        
        # 发送矩阵到Shader的Uniform变量
        # 注意：GLSL中矩阵乘法是右乘，所以需要转置
        prog['view'].write(view_matrix.T.astype('f4').tobytes())
        prog['projection'].write(proj_matrix.T.astype('f4').tobytes())

        # 清理屏幕和深度缓冲
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        
        # 激活深度测试和点大小设置
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.point_size = point_size
        
        # 执行渲染！
        vao.render(moderngl.POINTS)
        
        # 从FBO读取渲染好的图像
        image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
        image.save(os.path.join(output_dir, f"frame_{idx:05d}.png"))
    log_info("--- step 11: render done! ---")


def create_video_from_images(
    output_folder: str, 
    output_video_path: str, 
    fps: int = 30, 
    delete_images: bool = True,
    logger=None # 日志记录器
):
    """
    从一个文件夹中的图片序列合成为一个视频文件。

    参数:
    - image_folder (str): 包含图片序列的文件夹路径。
    - output_video_path (str): 输出视频的完整路径 (例如 'output/final_video.mp4')。
    - fps (int): 生成视频的帧率 (Frames Per Second)。
    - delete_images (bool): 如果为True，则在视频成功创建后删除原始图片。
                            **为安全起见，请谨慎使用此选项。**
    """
    def log_info(msg):
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
    image_folder = os.path.join(output_folder, "render_img")
    log_info(f"--- start create video from folder '{image_folder}' ---")

    # 1. 查找所有图片并进行排序
    # 使用glob查找所有png图片，这比os.listdir更方便
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    
    if not image_files:
        log_info(f"error: no .png image found in folder '{image_folder}'")
        return

    # 2. 读取第一张图片以获取视频尺寸
    try:
        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            raise IOError(f"无法读取第一张图片: {image_files[0]}")
        height, width, layers = first_frame.shape
    except Exception as e:
        log_info(f"error: failed to read image size. {e}")
        return

    # 3. 初始化视频写入器 (VideoWriter)
    # FOURCC是一种用于指定视频编解码器的4字节代码。
    # 'mp4v' 是用于.mp4格式的常用编解码器。
    # 其他选项如 'XVID' 用于.avi格式。
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 确保输出视频的目录存在
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        log_info(f"error: failed to initialize video writer, please check path and codec.")
        return

    # 4. 逐帧写入视频
    log_info(f"start to merge {len(image_files)} frames into video...")
    for image_path in tqdm(image_files, desc="视频合成进度"):
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 5. 释放资源
    # 这是至关重要的一步，忘记它会导致视频文件损坏或为空
    video_writer.release()
    log_info(f"\nvideo saved to: {output_video_path}")

    # 6. (可选) 删除原始图片
    if delete_images:
        log_info("\n--- start to delete original images ---")
        try:
            for image_path in tqdm(image_files, desc="删除图片进度"):
                os.remove(image_path)
            log_info("\nall original images deleted.")
        except Exception as e:
            log_info(f"\ndelete image error: {e}")
            log_info(f"please delete images in folder: {image_folder}")

def render_pipeline(
    data_dir=".", 
    output_dir=".", 
    num_points_to_keep=1_000_000, 
    img_width=100, 
    img_height=100,
    fps=10,
    delete_images=True,
    point_size=2,
    logger=None
):
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
    # 清空原本的图片
    del_cnt = 0
    for file in os.listdir(img_dir):
        if file.endswith(".png"):
            os.remove(os.path.join(img_dir, file))
            del_cnt += 1
    log_info(f"del {del_cnt} pics")
    
    log_info("--- step 2: init ModernGL context ---")
    # 初始化ModernGL离屏上下文
    ctx = moderngl.create_standalone_context()
    
    log_info("--- step 3: compile shader program ---")
    # 编译着色器程序
    # 1. 加载着色器代码
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

    log_info("--- step 4: load point cloud data ---")
    # 加载点云数据
    points = zip_point_cloud(num_points_to_keep, data_dir).elements[0].data
    colors = np.column_stack([points['red'], points['green'], points['blue']])
    scales = np.column_stack([points['scale_0'], points['scale_1'], points['scale_2']])
    rotates = np.column_stack([points['rot_0'], points['rot_1'], points['rot_2'], points['rot_3']])
    points_world = np.column_stack([points['x'], points['y'], points['z']])
    log_info(f"point num: {len(points_world)}")

    log_info("--- step 5: normalize colors ---")
    # 将颜色规范化到 [0, 1] for GLSL
    colors_gl = colors.astype('f4') / 255.0

    log_info("--- step 6: create and upload data to GPU buffer ---")
    # 创建并上传数据到GPU缓冲区
    vbo_points = ctx.buffer(points_world.astype('f4').tobytes())
    vbo_colors = ctx.buffer(colors_gl.tobytes())
    vbo_scales = ctx.buffer(scales.astype('f4').tobytes())
    vbo_rotates = ctx.buffer(rotates.astype('f4').tobytes())
    if version >= 0.2:
        vbo_random_seeds = ctx.buffer(np.random.rand(len(points_world), 2).astype('f4').tobytes())
    log_info("--- step 7: bind buffer to shader input ---")
    # 将缓冲区绑定到着色器的输入
    vao_content = [
        (vbo_points, '3f', 'in_vert'),
        (vbo_colors, '3f', 'in_color'),
        (vbo_scales, '3f', 'in_scale'),
        (vbo_rotates, '4f', 'in_rotate'),
    ]   
    if version >= 0.2:
        vao_content.append((vbo_random_seeds, '2f', 'in_rand_seed'))

    vao = ctx.vertex_array(prog, vao_content)

    log_info("--- step 8: create frame buffer object (FBO) ---")
    # 创建一个用于接收渲染结果的帧缓冲对象 (FBO)
    fbo = ctx.framebuffer(
        ctx.renderbuffer((img_width, img_height)),
        ctx.depth_renderbuffer((img_width, img_height))
    )
    fbo.use()
    
    log_info("--- step 9: load camera data ---")
    # 加载相机数据
    fov_rad, cam_poses = get_video_cam_info(data_dir)
    aspect_ratio = img_width / img_height
    znear, zfar = 0.1, 100.0
    projection_matrix = build_projection_matrix(fov_rad, aspect_ratio, znear, zfar)

    log_info("--- step 10: start render ---")
    # 在循环开始前
    video_path = os.path.join(output_dir, "render_video1.mp4")
    # 使用 mp4v 编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # 注意：cv2的尺寸是 (宽, 高)
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height)) # 30是帧率
    start_time = time.time()
    # 设置时间uniform（如果存在）
    if 'u_time' in prog:
        prog['u_time'].value = start_time
    for idx, pose_c2w in enumerate(tqdm(cam_poses, desc="GPU render progress")):
        # 构建视图和投影矩阵 (和之前一样)
        view_matrix = np.linalg.inv(pose_c2w)
        proj_matrix = projection_matrix
        # if idx % 100 == 0:
        #     log_info(f"{idx} frame:")
        #     log_info(f"cam pose: {pose_c2w}")
        #     log_info(f"view matrix: {view_matrix}")
        #     log_info(f"proj matrix: {proj_matrix}")
        
        # 发送矩阵到Shader的Uniform变量
        # 注意：GLSL中矩阵乘法是右乘，所以需要转置
        prog['view'].write(view_matrix.T.astype('f4').tobytes())
        prog['projection'].write(proj_matrix.T.astype('f4').tobytes())

        # 清理屏幕和深度缓冲
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        
        # 激活深度测试和点大小设置
        # 执行渲染
        ctx.enable(moderngl.DEPTH_TEST)
        if version == 0.1:
            ctx.point_size = 3
            vao.render(moderngl.POINTS)
        elif version == 0.2:
            # 设置点大小为最小
            ctx.point_size = 0.01
            num_samples_per_point = 100
            vao.render(moderngl.POINTS, instances=num_samples_per_point)
        elif version == 0.3:
            ctx.point_size = 0.01
            num_samples_per_point = 1
            vao.render(moderngl.POINTS, instances=num_samples_per_point)

        
        # 从FBO读取渲染好的图像
        image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'BGR', 0, -1)
        # image.save(os.path.join(img_dir, f"frame_{idx:05d}.png"))
        video_writer.write(np.array(image))
    video_writer.release()
    log_info("save video to: " + video_path)
    log_info("--- step 11: render done! ---")
    



if __name__ == "__main__":
    dir = ROOT_DIR
    wh = [[100, 100], [616, 409], [1920, 1080]]
    wh_idx = 2
    if mode == "GPU":
        # render_with_GPU(
        #     data_dir=dir, 
        #     output_dir=dir, 
        #     num_points_to_keep=1000000, 
        #     img_width=wh[wh_idx][0], 
        #     img_height=wh[wh_idx][1],
        #     point_size=2,
        #     logger=logger)
        
        # create_video_from_images(
        #     output_folder=dir, 
        #     output_video_path=os.path.join(dir, "render_video.mp4"),
        #     fps=10,
        #     delete_images=True,
        #     logger=logger
        #     )
        render_pipeline(
            data_dir=dir, 
            output_dir=dir, 
            num_points_to_keep=1_000_000, # 10_000_000
            img_width=wh[wh_idx][0], 
            img_height=wh[wh_idx][1],
            fps=10,
            delete_images=True,
            point_size=1,
            logger=logger)
    elif mode == "CPU":
        render_with_CPU(
            data_dir=dir, 
            output_dir=dir, 
            num_points_to_keep=10_000_000, 
            img_width=wh[wh_idx][0], 
            img_height=wh[wh_idx][1],
            point_size=2,
            logger=logger)
        create_video_from_images(
            output_folder=dir, 
            output_video_path=os.path.join(dir, "render_video.mp4"),
            fps=10,
            delete_images=True,
            logger=logger
            )
