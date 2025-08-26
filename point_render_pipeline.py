'''
v0.4:
    1. 这里开始只运行0.4以上版本

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
NUM_POINTS_TO_KEEP = 1_000_000  # 3_000_000
NUM_SAMPLES_PER_POINT = 512
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


    # # --- step 3.5: 编译视锥体裁剪的计算着色器 ---
    # with open(os.path.join(path, 'glsl', f'cs_v{version:.1f}.glsl'), 'r', encoding="utf-8") as f:
    #     cull_shader_code = f.read()
    #     log_info(f"cull_shader_code: {cull_shader_code}")
    # prog_cull = ctx.compute_shader(cull_shader_code)

    
    # # 准备用于裁剪的 SSBO (需要 vec4)
    # points_world_vec4 = np.hstack([points_world, np.ones((point_num, 1))]).astype('f4')
    # ssbo_all_positions = ctx.buffer(points_world_vec4.tobytes())
    # ssbo_cnt = ctx.buffer(cnt.astype('i4').tobytes())
    
    # # 创建用于存储裁剪结果的 SSBO
    # # 命令缓冲区现在的大小是 point_num (最坏情况)
    # indirect_cmd_buffer = ctx.buffer(reserve=point_num * 5 * 4) 
    # ssbo_atomic_counter = ctx.buffer(data=np.array([0], dtype='u4').tobytes()) # 初始化为0



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




    # --- step7，创建计算着色器和间接绘制缓冲区 ---
    # 计算着色器，用于生成绘制命令
    # with open(os.path.join(path, 'glsl', f'cs_v{version:.1f}.glsl'), 'r', encoding="utf-8") as f:
    #     compute_shader_code = f.read()
    # compute_prog = ctx.compute_shader(compute_shader_code)
    # # 间接绘制命令缓冲区
    # # 每个命令是 5 * 4 = 20 字节 (count, instanceCount, first, baseInstance, baseVertex)
    # # 我们使用 MultiDrawArraysIndirect，所以需要 N 个命令
    # indirect_cmd_buffer = ctx.buffer(reserve=point_num * 5 * 4)
    # log_info("--- Running Pass 0 (Compute Shader) to generate indirect commands ONCE ---")
    
    
    # # 绑定 SSBOs
    # ssbo_cnt.bind_to_storage_buffer(0)
    # indirect_cmd_buffer.bind_to_storage_buffer(1)
    
    # # 设置 uniforms 并运行计算着色器
    # compute_prog['u_point_count'].value = point_num
    # work_groups_x = (point_num + 1023) // 1024
    # compute_prog.run(group_x=work_groups_x)
    
    # # **确保计算着色器执行完毕**
    # ctx.finish() 
    # log_info("--- Indirect command buffer generated. ---")




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

    # --- step 9: 加载相机数据 ---
    log_info("--- step 9: load camera data ---")
    scene_dir = r"G:\Lab\GSSample"# r"G:\Lab\NGSassist\data\bicycle"
    fov_x, fov_y, cam_poses = get_video_cam_info_from_json(scene_dir, img_width, img_height)
    aspect_ratio = img_width / img_height
    znear, zfar = 0.01, 100.0
    projection_matrix = build_projection_matrix(fov_x, fov_y, aspect_ratio, znear, zfar)

    # --- step 10: 准备视频写入器 ---
    log_info("--- step 10: start render ---")
    # video_path = os.path.join(output_dir, "render_video.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # # 注意：cv2的尺寸是 (宽, 高)
    # video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height)) # 30是帧率

    # --- step 10.b: 尝试消费者线程 ---
    from queue import Queue
    from threading import Thread
    # 创建一个线程安全的队列，设置一个最大尺寸以防止内存爆炸
    video_path = os.path.join(output_dir, "render_video.mp4")
    frame_queue = Queue(maxsize=360) 
    
    def video_writer_worker(queue, video_path, width, height, fps):
        """一个在独立线程中运行的消费者函数"""
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        while True:
            # 从队列中获取数据，如果队列为空，会阻塞等待
            item = queue.get()
            # 收到结束信号 (None)
            if item is None:
                break
            
            image_bytes, img_h, img_w = item
            
            # --- 在这个线程中执行所有耗时的 CPU 操作 ---
            image_np_rgb = np.frombuffer(image_bytes, dtype=np.uint8).reshape((img_h, img_w, 3))
            image_np_bgr = cv2.cvtColor(np.flipud(image_np_rgb), cv2.COLOR_RGB2BGR)
            video_writer.write(image_np_bgr)
            
            # 通知队列，一个任务已完成
            queue.task_done()
            
        video_writer.release()
        print("Video writer thread finished.")
    # 启动消费者线程
    writer_thread = Thread(target=video_writer_worker, 
                           args=(frame_queue, video_path, img_width, img_height, fps))
    writer_thread.start()

    # 创建两个 CPU 端的字节缓冲区
    cpu_buffers = [bytearray(img_width * img_height * 3) for _ in range(2)]
    current_buffer_idx = 0
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


    for idx, pose_c2w in enumerate(tqdm(cam_poses, desc="GPU render progress")):

        # # --- pass0 视锥体裁剪 start ---
        # # 1. 重置原子计数器
        # ssbo_atomic_counter.write(np.array([0], dtype='u4'))
        # # 3. 绑定 SSBOs
        # ssbo_all_positions.bind_to_storage_buffer(0)
        # ssbo_cnt.bind_to_storage_buffer(1)
        # indirect_cmd_buffer.bind_to_storage_buffer(2)
        # ssbo_atomic_counter.bind_to_storage_buffer(3)
        # # 4. 设置 Uniforms
        # prog_cull['u_proj_view'].write((proj_matrix @ view_matrixs[idx]).T.astype('f4').tobytes())
        # prog_cull['u_total_points'].value = point_num
        
        # # 5. 运行计算着色器
        # work_groups_x = (point_num + 1023) // 1024
        # prog_cull.run(group_x=work_groups_x)
        # # 6. 插入内存屏障，确保后续的渲染可以安全地读取 SSBO
        # ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        # # (可选，用于调试) 读回可见点数量
        # num_visible_commands = np.frombuffer(ssbo_atomic_counter.read(), dtype='u4')[0]
        # if idx % 50 == 0:
        #     log_info(f"Frame {idx}: Visible commands: {num_visible_commands}")



        # --- pass1 start ---
        with query_pass1:
            
            # 告诉主渲染着色器，可见索引在哪里
            # 设置 uniforms
            # 注意：GLSL中矩阵乘法是右乘，所以需要转置
            prog['view'].write(view_matrixs[idx].T.astype('f4').tobytes())
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
        # **执行间接绘制**
        # vao_main.render_indirect(indirect_cmd_buffer, mode=moderngl.POINTS)        
        # --- pass1 end ---

        # 统计纯黑像素数
        # image_bytes = fbo_pass1.read(components=4, alignment=1)
        # image_np_rgba = np.frombuffer(image_bytes, dtype=np.uint8).reshape((img_height, img_width, 4))
        # black_pixels += np.sum(image_np_rgba[:, :, 3] == 0) / total_pixels


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

        # --- 最终结果已经在屏幕上了 ---
        # 现在可以从屏幕帧缓冲读取数据来保存为图片或视频帧
        # image_bytes = fbo_pass2.read(components=3)
        # image_np_rgb = np.frombuffer(image_bytes, dtype=np.uint8).reshape((img_height, img_width, 3))
        # image_np_bgr = cv2.cvtColor(np.flipud(image_np_rgb), cv2.COLOR_RGB2BGR) # **翻转并转为BGR**
        # video_writer.write(image_np_bgr)  

        # --- step 10.b: 尝试消费者线程：从 GPU 读取数据并放入队列 ---
        # fbo_pass2.read() 仍然会同步，但它只等待 GPU 的 ~9.3ms，而不是等待上一帧的视频写入
        # image_bytes = fbo_pass2.read(components=3)
        fbo_pass1.read_into(cpu_buffers[current_buffer_idx])
        # image_bytes_copy = bytes(image_bytes)  # 深拷贝
        
        # 将原始数据快速放入队列，让主线程解放出来
        # 如果队列满了，put() 会阻塞，这形成了一个自然的“背压”机制
        # frame_queue.put((image_bytes, img_height, img_width)) 
        frame_queue.put((bytes(cpu_buffers[current_buffer_idx]), img_height, img_width))
        
        # 3. 切换到另一个缓冲区，为下一帧做准备
        current_buffer_idx = 1 - current_buffer_idx
                
    # 发送结束信号到队列
    frame_queue.put(None)
    # 等待消费者线程处理完队列中的所有剩余帧
    writer_thread.join()
    
    #video_writer.release()
    log_info("save video to: " + video_path)
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

