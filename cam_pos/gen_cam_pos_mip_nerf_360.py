'''
1. create_loop_vid_cam_bin()
    读取COLMAP模型，按ID排序位姿，并插值生成一条首尾相连的平滑路径，写回video.bin

'''

import os
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat, write_images_binary
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov, fov2focal
import torch

def create_loop_vid_cam_bin(
    colmap_model_path: str,
    output_path: str,
    num_frames: int = 360
):
    """
    读取COLMAP模型，按ID排序位姿，并插值生成一条首尾相连的平滑路径。

    参数:
    - colmap_model_path (str): 包含 cameras.bin 和 images.bin 的目录。
    - output_path (str): 保存新的 images.bin 的完整文件路径。
    - num_frames (int): 最终生成的平滑路径的帧数。
    """
    print("--- 步骤 1: 读取COLMAP模型 ---")
    try:
        images = read_images_binary(os.path.join(colmap_model_path, "images.bin"))
        images = sorted(images.items(), key=lambda x: x[1].id)
        interpolation_steps = 10
        num_keyframes = max(2, num_frames // interpolation_steps)
        keyframe_indices = np.linspace(0, len(images) - 1, num_keyframes, dtype=int)
        keyframes = [images[i] for i in keyframe_indices]
        print(f"成功读取 {len(images)} 个原始图像位姿。")
    except Exception as e:
        print(f"读取模型失败: {e}")
        return
    images = keyframes
    print(images[0])
    # --- 步骤 2: 按ID排序并提取位姿数据 ---
    print("--- 步骤 2: 按Image ID排序位姿 ---")
    sorted_ids = [image[0] for image in images]
    
    positions = []
    rotations = []

    for image in images:
        img_id = image[0]
        image = image[1]
        R_w2c = Rotation.from_quat([image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0]]).as_matrix()
        t_w2c = image.tvec
        
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        
        positions.append(t_c2w)
        rotations.append(Rotation.from_matrix(R_c2w))

    # --- 步骤 3: 创建闭合的B-样条插值器 ---
    print(f"--- 步骤 3: 使用闭合B-样条插值生成 {num_frames} 帧平滑路径 ---")
    
    # a. 位置插值 (使用周期性B-样条)
    # 为了创建闭合曲线，我们在数据点末尾添加起始点
    closed_positions = np.vstack([positions, positions[0]])
    # 创建时间轴，同样首尾相连
    t = np.linspace(0, 1, len(closed_positions))
    
    # 创建B-样条插值器 (k=3 表示三次样条)
    # 注意：scipy的周期性边界条件有时不稳定，手动闭合数据点更可靠
    spline = make_interp_spline(t, closed_positions, k=min(4, len(t)-1))
    
    # 在新的时间点上进行采样 (不包括最后一个点，因为它和第一个点重合)
    t_new = np.linspace(0, 1, num_frames + 1)[:-1]
    interpolated_positions = spline(t_new)

    # b. 旋转插值 (Slerp)
    # 为了闭合旋转，同样在末尾添加起始旋转
    closed_rotations = rotations + [rotations[0]]
    slerp = Slerp(t, Rotation.concatenate(closed_rotations))
    interpolated_rotations = slerp(t_new)

    # --- 步骤 4: 创建新的Image对象并写回.bin文件 ---
    print(f"--- 步骤 4: 将 {num_frames} 个新位姿写回到 images.bin ---")
    new_images = {}
    # 我们需要一个相机内参模型，就用原始的第一个吧
    camera_id_to_use = images[sorted_ids[0]][1].camera_id
    from read_write_model import Image
    for i in range(num_frames):
        new_id = i + 1
        
        R_c2w = interpolated_rotations[i].as_matrix()
        t_c2w = interpolated_positions[i]
        
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        
        q_xyzw = Rotation.from_matrix(R_w2c).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        new_image = Image(
            id=new_id, qvec=q_wxyz, tvec=t_w2c,
            camera_id=camera_id_to_use, name=f"interp_{i:05d}.png",
            xys=np.array([]), point3D_ids=np.array([])
        )
        new_images[new_id] = new_image

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_images_binary(new_images, output_path)
        print(f"成功写入新的 images.bin 文件到: {output_path}")
    except Exception as e:
        print(f"写入文件失败: {e}")


if __name__ == '__main__':
    # --- 请配置你的路径 ---
    
    # 包含 cameras.bin, images.bin 的原始COLMAP模型目录
    COLMAP_MODEL_DIR = r"G:\Lab\NGSassist\data\bicycle\sparse\0"
    
    # 你希望保存新的 images.bin 的完整路径
    OUTPUT_IMAGES_BIN_PATH = r"G:\Lab\NGSassist\data\bicycle\sparse\0\video.bin"
    
    # 你希望最终视频有多少帧
    NUM_OUTPUT_FRAMES = 360
    cam_gen_func = create_loop_vid_cam_bin
    # create_closed_loop_path_by_id
    # create_smooth_camera_path
    # --- 配置结束 ---
    
    cam_gen_func(
        colmap_model_path=COLMAP_MODEL_DIR,
        output_path=OUTPUT_IMAGES_BIN_PATH,
        num_frames=NUM_OUTPUT_FRAMES)