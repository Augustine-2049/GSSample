import numpy as np
from scipy.spatial.transform import Rotation as R

class InteractiveCamera:
    def __init__(self, position=np.array([0.0, 0.0, 5.0]), yaw= -90.0, pitch=0.0):
        self.position = position.astype(np.float32)
        self.yaw = float(yaw)
        self.pitch = float(pitch)

        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.update_vectors()

    def update_vectors(self):
        # 根据 yaw 和 pitch 计算新的 front, right, up 向量
        yaw_rad = np.deg2rad(self.yaw)
        pitch_rad = np.deg2rad(self.pitch)
        
        self.front[0] = np.cos(yaw_rad) * np.cos(pitch_rad)
        self.front[1] = np.sin(pitch_rad)
        self.front[2] = np.sin(yaw_rad) * np.cos(pitch_rad)
        self.front = self.front / np.linalg.norm(self.front)
        
        self.right = np.cross(self.front, self.world_up)
        self.right = self.right / np.linalg.norm(self.right)
        
        self.up = np.cross(self.right, self.front)
        self.up = self.up / np.linalg.norm(self.up)

    def get_view_matrix(self):
        # LookAt 矩阵的构建
        target = self.position + self.front
        
        # 使用 NumPy 创建 LookAt 矩阵
        z_axis = (self.position - target) / np.linalg.norm(self.position - target)
        x_axis = np.cross(self.world_up, z_axis) / np.linalg.norm(np.cross(self.world_up, z_axis))
        y_axis = np.cross(z_axis, x_axis)

        translation = np.identity(4)
        translation[0, 3] = -self.position[0]
        translation[1, 3] = -self.position[1]
        translation[2, 3] = -self.position[2]

        rotation = np.identity(4)
        rotation[0, :3] = x_axis
        rotation[1, :3] = y_axis
        rotation[2, :3] = z_axis
        
        return (rotation @ translation).astype(np.float32)

    def update(self, motion_state, delta_time):
        move_speed = 5.0 * delta_time
        rotation_speed = 100.0 * delta_time
        
        if motion_state.get('forward'):
            self.position += self.front * move_speed
        if motion_state.get('backward'):
            self.position -= self.front * move_speed
        if motion_state.get('left'):
            self.position -= self.right * move_speed
        if motion_state.get('right'):
            self.position += self.right * move_speed
        if motion_state.get('up'):
            self.position += self.world_up * move_speed
        if motion_state.get('down'):
            self.position -= self.world_up * move_speed
            
        if motion_state.get('rot_left'):
            self.yaw -= rotation_speed
        if motion_state.get('rot_right'):
            self.yaw += rotation_speed
            
        # 限制 pitch 角度防止万向节死锁
        self.pitch = np.clip(self.pitch, -89.0, 89.0)
        self.update_vectors()