'''
仅仅针对nerf_synthetic数据集
- 生成一个俯视30°的环形相机位姿组合


'''

import json
import numpy as np
import os

def normalize(v):
    """对向量进行归一化。"""
    if np.linalg.norm(v) == 0:
        return v
    return v / np.linalg.norm(v)

def look_at(camera_position, target_position, world_up):
    """
    计算并返回一个从相机到世界坐标系的4x4变换矩阵 (Camera-to-World)。
    
    Args:
        camera_position (np.array): 相机在世界坐标系中的位置 (3,)。
        target_position (np.array): 相机朝向的目标点 (3,)。
        world_up (np.array): 世界坐标系的上方向向量, e.g., (0, 0, 1) for Z-up。
    
    Returns:
        np.array: (4, 4) 的相机到世界变换矩阵。
    """
    # 1. 计算相机的Z轴（朝向目标的相反方向）
    z_axis = normalize(camera_position - target_position)
    # 2. 计算相机的X轴（右方向）
    x_axis = normalize(np.cross(world_up, z_axis))
    # 3. 计算相机的Y轴（上方向）
    y_axis = normalize(np.cross(z_axis, x_axis))

    # 构建从相机到世界的变换矩阵 (Camera-to-World)
    cam_to_world = np.eye(4)
    cam_to_world[:3, 0] = x_axis
    cam_to_world[:3, 1] = y_axis
    cam_to_world[:3, 2] = z_axis
    cam_to_world[:3, 3] = camera_position
    print(camera_position)
    return cam_to_world

def generate_circular_transform_nerf_syn(
    radius: float = 4.0,
    elevation_deg: float = 30.0,
    duration_sec: float = 6.0,
    fps: int = 30,
    output_dir: str = r"H:\800_DataSet\NeRF_Synthesis\3DGS_nerf_synthesis"
):
    """
    生成一个环绕原点(0,0,0)的平滑圆形相机路径，并将其保存为NeRF/3DGS兼容的JSON文件。

    Args:
        radius (float): 相机到场景中心的距离（半径）。
        elevation_deg (float): 俯视角度（从水平面向下看），单位是度。
        duration_sec (float): 视频的总时长，单位是秒。
        fps (int): 视频的帧率 (Frames Per Second)。
        fov_deg (float): 相机的水平视场角 (Horizontal Field of View)，单位是度。
        output_path (str): 输出的JSON文件的完整路径（包括文件名）。
    """
    total_frames = int(duration_sec * fps)
    
    print(f"开始生成相机路径: {total_frames} 帧, {duration_sec}s @ {fps}fps")

    # 准备输出数据结构
    camera_angle_x_radians = 0.6911112070083618 # np.radians(fov_deg)
    out_data = {
        'camera_angle_x': camera_angle_x_radians,
        'frames': []
    }

    # 定义场景中心和全局“上”方向
    target = np.array([0.0, 0.0, 0.0])
    world_up = np.array([0.0, 0.0, 1.0]) # Z轴朝上，与Blender/NeRF合成数据一致

    # 将俯仰角转换为弧度
    elevation_rad = np.radians(elevation_deg)

    for i in range(total_frames):
        # 计算当前帧的水平旋转角度（azimuth）
        azimuth_rad = i * (2 * np.pi / total_frames)
        
        # 使用球坐标计算相机位置
        x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = radius * np.sin(elevation_rad)
        
        camera_pos = np.array([x, y, z])
        
        # 计算相机到世界的变换矩阵
        transform_matrix = look_at(camera_pos, target, world_up)
        
        # 将NumPy矩阵转换为Python列表
        transform_matrix_list = transform_matrix.tolist()
        
        # 构造这一帧的数据
        frame_data = {
            # 文件路径只是一个占位符，因为我们不渲染
            'file_path': f'./video/r_{i:04d}', 
            'transform_matrix': transform_matrix_list
        }
        out_data['frames'].append(frame_data)

    print("相机路径生成完毕。")

    json_filename = "transforms_video.json"
    copied_count = 0
    print(f"\n正在扫描 '{output_dir}' 中的子目录...")
    for item_name in os.listdir(output_dir):
        # 构造完整的子目录路径
        subdir_path = os.path.join(output_dir, item_name)
        
        # 检查这是否是一个目录
        if os.path.isdir(subdir_path):
            # 构造最终的JSON文件输出路径
            dest_json_path = os.path.join(subdir_path, json_filename)
            
            try:
                print(f"  -> 正在写入到: {dest_json_path}")
                with open(dest_json_path, 'w') as out_file:
                    json.dump(out_data, out_file, indent=4)
                copied_count += 1
            except Exception as e:
                print(f"    写入失败! 错误: {e}")
    
    print(f"\n任务完成！成功将 '{json_filename}' 写入到 {copied_count} 个子目录中。")


if __name__ == "__main__":
    # --- 使用示例 ---
    # 定义你想要的相机路径参数
    
    # 示例1：创建一个标准的、10秒、30fps的视频路径
    print("--- 生成示例路径 1 ---")
    generate_circular_transform_nerf_syn(
        radius=4.5,
        elevation_deg=30.0,
        duration_sec=10.0,
        fps=10,
        output_dir=r"H:\800_DataSet\NeRF_Synthesis\3DGS_nerf_synthesis"
    )
    