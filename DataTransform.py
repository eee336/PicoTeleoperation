import numpy as np
from scipy.spatial.transform import Rotation
import math


class DataTransform:

    def __init__(self):
        self.euler = [0, 0, 0]
        self.base_pos = np.array([0, 0, 0, 0, 0, 0, 1])
        self.data_pos = np.array([0, 0, 0, 0, 0, 0, 1])

    #四元数---->旋转矩阵
    def quat2matrix(self,quat):
        matrix = Rotation.from_quat(quat).as_matrix()
        return matrix

    #四元数---->欧拉角
    def quat2euler(self,quat):
        euler = Rotation.from_quat(quat).as_euler('XYZ', degrees=True)
        return euler

    #欧拉角---->四元数
    def euler2quat(self,euler):
        quat = Rotation.from_euler("XYZ", euler, degrees=True).as_quat()
        return quat

    #欧拉角---->旋转矩阵
    def euler2matrix(self,euler):
        matrix = Rotation.from_euler("XYZ", euler, degrees=True).as_matrix()
        return matrix

    #旋转矩阵--->欧拉角
    def matrix2euler(self,matrix):
        euler = Rotation.from_matrix(matrix).as_euler('XYZ', degrees=True)
        return euler

    #旋转矩阵--->四元数
    def matrix2quat(self,matrix):
        quat = Rotation.from_matrix(matrix).as_quat()
        return quat

    def create_transform_matrix(self,position,matrix):
        transform_matirx = np.eye(4)
        transform_matirx[:3, :3] = matrix
        transform_matirx[:3, 3] = position
        transform_matirx[3, :3] = 0
        return transform_matirx

    #偏移矩阵转xyzrpy
    def transform_matrix2xyzrpy(self,matrix):
        """
        将旋转矩阵转换为XYZ欧拉角（Roll, Pitch, Yaw）
        对应旋转顺序：绕X轴（Roll）→ 绕Y轴（Pitch）→ 绕Z轴（Yaw）

        参数:
            matrix (np.ndarray): 3x3或4x4旋转矩阵

        返回:
            list: [x, y, z, roll, pitch, yaw]，其中x,y,z是平移量，roll,pitch,yaw是旋转角度（弧度）
        """
        # 提取平移部分
        x = matrix[0, 3]
        y = matrix[1, 3]
        z = matrix[2, 3]

        # 提取旋转矩阵部分
        R = matrix[:3, :3]

        quat=self.matrix2quat(R)

        return [x, y, z, quat]

    def adjustment_matrix(self, transform_matrix):
        if transform_matrix.shape != (4, 4):
            raise ValueError("Input transform must be a 4x4 numpy array.")

        #手柄坐标系映射到mujoco坐标系
        adj_mat = np.array([
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        matrix2=self.euler2matrix([-90, 0, 90])
        #机械臂末端坐标系到机械臂底座的映射
        r_adj = self.create_transform_matrix([0, 0, 0],matrix2)
        transform = adj_mat @ transform_matrix
        transform = np.dot(transform, r_adj)
        return transform

    def calc_pose_incre(self, base_pos, now_pos):
        #记录手柄的基准位姿
        base_matrix = self.quat2matrix(base_pos[3:])
        base_position = self.base_pos[:3]
        begin_matrix = self.create_transform_matrix(base_position, base_matrix)

        #机械臂工作基准位置/末端执行器
        zero_position = [0, 0, 0]
        zero_matrix = self.quat2matrix([0, 0, 0, 1])
        zero_matrix = self.create_transform_matrix(zero_position, zero_matrix)

        #当前位姿
        end_position = now_pos[:3]
        end_matrix = self.quat2matrix(now_pos[3:])
        end_matrix = self.create_transform_matrix(end_position, end_matrix)

        #计算当前位姿相对于基准位姿的变换矩阵
        result_matrix = np.dot(zero_matrix, np.dot(np.linalg.inv(begin_matrix), end_matrix))
        xyzrpy = self.transform_matrix2xyzrpy(result_matrix)
        return xyzrpy

# if __name__ == '__main__':
#     t = DataTransform()
#     pos=[0.34861052,0.85787028,-0.4099614 ]
#     x=-0.05216405913233757
#     y=-0.19402047991752625
#     z=0.48760128021240234
#     w=0.8496351838111877
#
#     print(t.euler2quat([0,0,0]))

