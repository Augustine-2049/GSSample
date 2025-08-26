'''
1. create_loop_vid_cam_bin()
    读取COLMAP模型，按ID排序位姿，并插值生成一条首尾相连的平滑路径，写回video.bin
2. get_vid_cam_from_bin()
    读取bin并转换成3DGS的相机外参信息，返回list

'''

import os
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov, fov2focal
import torch

# 确保 read_write_model.py 在你的Python路径中
from read_write_model import read_cameras_binary, read_images_binary, write_images_binary, Image



