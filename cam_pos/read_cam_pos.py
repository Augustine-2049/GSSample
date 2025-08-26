'''
读取相机位姿文件
- get_video_cam_info_from_json/get_video_cam_info_from_bin
    - 从json、bin文件读取相机位姿
        - 具体来说包含fov、相机位姿列表
    - 针对点云渲染
- get_vid_cam_from_bin
    - 读取bin并转换成3DGS的相机外参信息，返回list

'''
import os
import json
import numpy as np
from pathlib import Path

def get_video_cam_info_from_json(
    json_path=os.getcwd(), 
    tar_width=1920, 
    tar_height=1080
    ):
    """
    从 NeRF 风格的 transforms_video.json 文件读取相机信息。

    为了与COLMAP加载器统一接口，此函数现在接受目标宽高并计算FovY。

    Args:
        json_path (str): 包含 transforms_video.json 文件的目录路径。
        tar_width (int): 目标渲染宽度，用于计算 FovY。
        tar_height (int): 目标渲染高度，用于计算 FovY。

    Returns:
        tuple: (FovX, FovY, frames)
            - FovX (float): 水平视场角（弧度）。
            - FovY (float): 根据目标宽高计算出的垂直视场角（弧度）。
            - frames (list): 每个元素是一个 (4, 4) 的 numpy 数组，代表相机到世界的变换矩阵。
    """
    cam_path = os.path.join(json_path, "transforms_video.json")
    if not os.path.exists(cam_path):
        print(f"相机位姿文件不存在: {cam_path}")
        # 返回与成功时一致的结构，但值为None或空
        return None, None, []

    with open(cam_path, 'r') as f:
        cam_data = json.load(f)

    # 1. 获取 FovX (水平视场角)
    # 'camera_angle_x' 已经是弧度制的 FovX
    FovX = cam_data['camera_angle_x']
    
    # 2. 根据 FovX 和目标宽高计算 FovY (垂直视场角)
    # 公式源于: tan(FovY / 2) = (height / width) * tan(FovX / 2)
    FovY = 2 * np.arctan( (tar_height / tar_width) * np.tan(FovX / 2) )

    # 3. 读取并转换相机外参矩阵
    frames = []
    for frame in cam_data['frames']:
        # 将列表转换为numpy数组
        transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
        frames.append(transform_matrix)
        
    return FovX, FovY, frames


def get_video_cam_info_from_bin(
    scene_path, tar_width = 1920, tar_height = 1080,
    ):
    '''
    从COLMAP二进制文件读取相机信息
    返回fov和相机位姿列表
    fov记录在cameras.bin中
    frames记录在video.bin中
    '''

    import struct
    vid_path = os.path.join(scene_path, "sparse", "0", "video.bin")
    if not os.path.exists(vid_path):
        print(f"相机位姿文件不存在: {vid_path}")
        return None, []
    
    # 读取cameras.bin获取相机内参
    cameras_path = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    if not os.path.exists(cameras_path):
        print(f"相机内参文件不存在: {cameras_path}")
        return None, []
    from read_write_model import read_cameras_binary, read_images_binary
    cameras = read_cameras_binary(cameras_path)  # 相机内参
    images = read_images_binary(vid_path)  # 相机外参
    
    # 计算FOV
    # 获取第一个相机的内参
    first_camera = list(cameras.values())[0]
    width, height = first_camera.width, first_camera.height
    
    # PINHOLE
    fx, fy, cx, cy = first_camera.params
    focal_length_x = fx * tar_width / width
    focal_length_y = fy * tar_width / width
    from utils.graphics_utils import focal2fov
    FovY = focal2fov(focal_length_x, tar_height)
    FovX = focal2fov(focal_length_x, tar_width)

    # fov = 2 * np.arctan(width / (2 * fx))  # 水平FOV
    
    # 构建相机位姿列表
    frames = []
    # 相机外参实际就是七个数字
    for image_id, image in images.items():
        # 获取四元数和平移向量
        qvec = image.qvec  # (w, x, y, z) 从世界坐标系变换到相机坐标系的刚体变换
        tvec = image.tvec  # (x, y, z) 从世界坐标系变换到相机坐标系的刚体变换
        
        # 转换为旋转矩阵
        from scipy.spatial.transform import Rotation as R
        rot_w2c = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
        R_wc = rot_w2c.as_matrix()
        R_cw = R_wc.T
        t_cw = -R_cw @ tvec
        
        c2w_cv = np.identity(4)
        c2w_cv[:3, :3] = R_cw
        c2w_cv[:3, 3] = t_cw
        
        
        # --- 步骤 2: 定义从 OpenCV 到 OpenGL 相机坐标系的转换矩阵 ---
        # 这个转换在相机的局部空间中进行，翻转 Y 和 Z 轴
        opencv_to_opengl_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        # --- 步骤 3: 将 C2W_cv 转换为 C2W_gl ---
        # 公式: C2W_gl = C2W_cv @ OpencvToOpengl
        transform_matrix = c2w_cv @ opencv_to_opengl_transform
        frames.append(transform_matrix)

    return FovX, FovY, frames


import os
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat, write_images_binary
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov, fov2focal
import torch


def get_vid_cam_from_bin(
    path = r"G:\Lab\NGSassist\data\bicycle",
    tar_width = None,
    tar_height = None,
    ):
    '''
    这个函数根据场景已经有的cameras.bin（内参）和images.bin（位姿），
    生成3DGS的相机外参信息，返回list

    参数：
    path: 场景路径
    tar_width: 目标宽度
    tar_height: 目标高度

    '''
    cam_path = os.path.join(path, r"sparse\0\cameras.bin")
    cameras = read_cameras_binary(cam_path)
    img_path = os.path.join(path, r"sparse\0\video.bin")  # images / video
    tmp_path = os.path.join(path, r"images\_DSC8679.JPG")
    from PIL import Image
    tmp_image_pil = Image.new('RGB', (1, 1)) # 创建一个占位符，注意3DGS在Camera中写死锚定这个。
    images = read_images_binary(img_path)
    # print(cameras[1].model)
    vid_cam_infos = []
    for idx, key in enumerate(images):
        extr = images[key]
        intr = cameras[extr.camera_id]
        height = intr.height
        width = intr.width
        if idx == 0:
            print(width, height)
            print(intr.model)
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            # FovY = focal2fov(focal_length_x, tar_height)
            # FovX = focal2fov(focal_length_x, tar_width)
        elif intr.model=="PINHOLE":  # MipNeRF = PINHOLE
            # colmap给了在(616, 409)下的焦距，我们需要以此推Fov
            # 如果需要进行像素放缩，则以x为基准，计算新分辨率下的focal_lenght_x = intr.params[0] * tar_width / width
                # focal_length_y不可以直接乘以因子，否则只是在原画幅上进行拉伸，而是 先与focal_x一同拉伸。后面直接用【tar视野】计算fov
                # focal_length_y = focal_length_y * tar_width / width * (tar_)
            focal_length_x = intr.params[0] * tar_width / width
            focal_length_y = intr.params[1] * tar_width / width
            # FovY = focal2fov(focal_length_y, height)
            # FovX = focal2fov(focal_length_x, width)
            FovY = focal2fov(focal_length_x, tar_height)
            FovX = focal2fov(focal_length_x, tar_width)            
            image_path = None
            image_name = None
        image = Image.open(tmp_path)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=tmp_image_pil,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        vid_cam_infos.append(cam_info)

    camera_list = []
    from scene.cameras import Camera
    for id, cam_info in enumerate(vid_cam_infos):
        # print(id)
        # [3, 1080, 1920]
        camera_list.append(
            Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=torch.tensor(np.array(cam_info.image)).permute(2, 0, 1), gt_alpha_mask=None,
                    image_name=cam_info.image_name, uid=id, data_device="cuda"))


    return camera_list


def get_vid_cam_from_json(
    path = r"H:\800_DataSet\NeRF_Synthesis\3DGS_nerf_synthesis\chair",
    transformsfile = "transforms_video.json",
    tar_width = None,
    tar_height = None,
    ):
    '''
    默认以训练集图片大小为窗口
    默认video信息在目标数据集内部
    
    '''
    from PIL import Image
    vid_cam_infos = []
    for filename in sorted(os.listdir(path)): # sorted()保证了顺序，例如r_0.png在r_10.png之前
        # 检查文件名是否以.png结尾（不区分大小写）
        if filename.lower().endswith(".png"):
            # 如果是，构建完整的绝对路径
            first_png_path = os.path.join(path, filename)
            # 找到第一个后立即退出循环
            break
    print(first_png_path)
    image = Image.open(first_png_path)
    height = image.size[1]
    width = image.size[0]
    # 截取image为长宽到1
    image = image.crop((0, 0, 1, 1))
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            # 长宽比不变时，fov可以不变，一般控制fovx不变，fovy根据长宽比计算
            FovX = fovx
            FovY = 2 * np.arctan( (tar_height / tar_width) * np.tan(fovx / 2) )
            image_path = None
            image_name = None
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height)
            vid_cam_infos.append(cam_info)
            
    camera_list = []
    from scene.cameras import Camera
    for id, cam_info in enumerate(vid_cam_infos):
        # print(id)
        # [3, 1080, 1920]
        # 初始化时image不可以为空，因为源代码写死用image的后两个维度作为WH，所以如下初始化
        camera_list.append(
            Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=torch.zeros((1, 1, 1)), gt_alpha_mask=None,
                    image_name=cam_info.image_name, uid=id, data_device="cuda"))


    return camera_list


def get_first_vid_cam_from_json(
    path = r"H:\800_DataSet\NeRF_Synthesis\3DGS_nerf_synthesis\chair",
    transformsfile = "transforms_video.json",
    tar_width = None,
    tar_height = None,
    ):
    '''
    默认以训练集图片大小为窗口
    默认video信息在目标数据集内部
    
    '''
    from PIL import Image
    vid_cam_infos = []
    for filename in sorted(os.listdir(path)): # sorted()保证了顺序，例如r_0.png在r_10.png之前
        # 检查文件名是否以.png结尾（不区分大小写）
        if filename.lower().endswith(".png"):
            # 如果是，构建完整的绝对路径
            first_png_path = os.path.join(path, filename)
            # 找到第一个后立即退出循环
            break
    image = Image.open(first_png_path)
    height = image.size[1]
    width = image.size[0]
    # 截取image为长宽到1
    image = image.crop((0, 0, 1, 1))
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            # 长宽比不变时，fov可以不变，一般控制fovx不变，fovy根据长宽比计算
            FovX = fovx
            FovY = 2 * np.arctan( (tar_height / tar_width) * np.tan(fovx / 2) )
            image_path = None
            image_name = None
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height)
            vid_cam_infos.append(cam_info)
            break
            
    camera_list = []
    from scene.cameras import Camera
    for id, cam_info in enumerate(vid_cam_infos):
        # print(id)
        # [3, 1080, 1920]
        # 初始化时image不可以为空，因为源代码写死用image的后两个维度作为WH，所以如下初始化
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=torch.zeros((1, 1, 1)), gt_alpha_mask=None,
                    image_name=cam_info.image_name, uid=0, data_device="cuda")



        
