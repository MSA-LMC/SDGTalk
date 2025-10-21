"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Modified from smplx code for FLAME by Xuangeng Chu (xg.chu@outlook.com)
"""
import os

import torch
import pickle
import numpy as np
import torch.nn as nn

from .lbs import lbs, batch_rodrigues, vertices2landmarks

class FLAMEModel(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, n_shape, n_exp, scale=1.0, no_lmks=False):
        super().__init__()
        self.scale = scale
        self.no_lmks = no_lmks
        # print("creating the FLAME Model")
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        self.flame_path = os.path.join(_abs_path, '../../assets/flame')
        self.flame_ckpt = torch.load(
            os.path.join(self.flame_path, 'FLAME_with_eye.pt'), map_location='cpu', weights_only=True
        )
        flame_model = self.flame_ckpt['flame_model']
        flame_lmk = self.flame_ckpt['lmk_embeddings']
        
        self.dtype = torch.float32
        self.register_buffer('faces_tensor', flame_model['f']) # 代表连接相邻顶点所拼成的面，每个面都是三角形 torch.Size([9976, 3])

        # The vertices of the template model
        self.register_buffer('v_template', flame_model['v_template']) # torch.Size([5023, 3])

        # The shape components
        shapedirs = flame_model['shapedirs'] # torch.Size([5023, 3, 400]) 是所有旋转矩阵的姿势参数的主成成分（PCA）系数，这里的400代表400个基元，这个shapedirs也是从数据集里学习出来的
        self.register_buffer('shapedirs', torch.cat([shapedirs[:, :, :n_shape], shapedirs[:, :, 300:300 + n_exp]], 2))

        # Pose blend shape basis
        num_pose_basis = flame_model['posedirs'].shape[-1] # flame_model['posedirs'].shape torch.Size([5023, 3, 36]) 姿态参数的主成分分析（PCA）系数，用来和FLAME的pose相乘，对顶点位置产生影响
        self.register_buffer('posedirs', flame_model['posedirs'].reshape(-1, num_pose_basis).T) # flame_model['posedirs'].reshape(-1, num_pose_basis).T # torch.Size([36, 5023*3])

        self.register_buffer('J_regressor', flame_model['J_regressor']) # torch.Size([5, 5023]) 关节回归矩阵，将5023个顶点回归到关节点上，用5023个点组成的矩阵乘上J_regressor矩阵，能得到5个点的坐标([5, 3] = [5, 5023] * [5023, 3])


        # indices of parents for each joints
        parents = flame_model['kintree_table'][0] # flame_model['kintree_table'].shape =  torch.Size([2, 5]) 简单理解为5个关节点的序号即可，第二个向量内容为0-4，第一个向量内容为4294967296，0-4，这个很大的数字我也不知道哪来的，这个字段不太重要
        parents[0] = -1 # [-1,0,1,2,3,4] 代表第一个关节没有父节点，第二个关节的父节点是第一个关节，第三个关节的父节点是第二个关节，以此类推
        self.register_buffer('parents', parents)


        self.register_buffer('lbs_weights', flame_model['weights']) # flame_model['weights'].shape =  torch.Size([5023, 5]) 即lbs_weights，蒙皮权重，代表每个顶点（5023个）受每个关节（5个）影响的权重

        # Fixing Eyeball and neck rotation
        self.register_buffer('eye_pose', torch.zeros([1, 6], dtype=torch.float32))
        self.register_buffer('neck_pose', torch.zeros([1, 3], dtype=torch.float32))

        # Static and Dynamic Landmark embeddings for FLAME
        self.register_buffer('lmk_faces_idx', flame_lmk['static_lmk_faces_idx']) # flame_ckpt['lmk_embeddings']['static_lmk_faces_idx'].shape = torch.Size([51]) 面（三角形）的索引 FLAME 面部 landmarks 的面片索引
        self.register_buffer('lmk_bary_coords', flame_lmk['static_lmk_bary_coords'].to(dtype=self.dtype)) # shape = torch.Size([51, 3]) FLAME 面部 landmarks 的重心坐标
        self.register_buffer('dynamic_lmk_faces_idx', flame_lmk['dynamic_lmk_faces_idx'].to(dtype=torch.long)) # shape = torch.Size([79, 17]) 脖子的旋转角度对应的 FLAME 脸部轮廓的面片 脸部轮廓的 landmarks 总共有 17 个，脖子的可以旋转 79 度
        self.register_buffer('dynamic_lmk_bary_coords', flame_lmk['dynamic_lmk_bary_coords'].to(dtype=self.dtype)) # shape = torch.Size([79, 17, 3]) FLAME 脖子的旋转角度对应的 FLAME 脸部轮廓 landmarks 的重心坐标
        self.register_buffer('full_lmk_faces_idx', flame_lmk['full_lmk_faces_idx_with_eye'].to(dtype=torch.long)) # torch.Size([1, 70]) FLAME 脸部 landmarks 的 面索引(注意不是顶点索引)，包含眼睛的 70 个 landmarks
        self.register_buffer('full_lmk_bary_coords', flame_lmk['full_lmk_bary_coords_with_eye'].to(dtype=self.dtype)) # torch.Size([1, 70, 3]) 脖子的旋转角度对应的FLAME 脸部 landmarks 的重心坐标权重，包含眼睛的 70 个 landmarks 每个地标对应三角面中的重心坐标权重

        neck_kin_chain = []
        NECK_IDX = 1 # 索引1专门用于表示颈部关节。选择颈部作为起点是有意义的，因为头部运动通常以颈部为支点
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        # 从颈部关节开始，沿着骨骼层级向上追踪父节点，直到到达根节点（通常以-1表示没有父节点） 从颈部到根部的完整路径，这是计算头部姿态变换所必需的
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))
        # print("FLAME Model Done.")

    def get_faces(self, ):
        return self.faces_tensor.long()

    def _find_dynamic_lmk_idx_and_bcoords(
            self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords,
            neck_kin_chain, dtype=torch.float32
        ):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    # @torch.no_grad()
    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None, verts_sclae=None):
        """
            Input:
                shape_params: N X number of shape parameters 实际是N X 300
                expression_params: N X number of expression parameters 实际是N X 100
                pose_params: N X number of pose parameters (6) 实际是N X 6
            return:d
                vertices: N X V X 3 实际是 N X 5023 X 3
                landmarks: N X number of landmarks X 3 实际是 N X 70 X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1) # N X 6
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1) # N X 6
        if expression_params is None:
            expression_params = torch.zeros(batch_size, self.cfg.n_exp).to(shape_params.device) # N X 100

        betas = torch.cat([shape_params, expression_params], dim=1) # N X 400
        full_pose = torch.cat([
                pose_params[:, :3], self.neck_pose.expand(batch_size, -1), 
                pose_params[:, 3:], eye_pose_params
            ], dim=1
        ) # N X 6 + 3 + 6 = N X 15
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1) # N X 5023 X 3
        vertices, _ = lbs(
            betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs, self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype, detach_pose_correctives=False
        ) # [N, 5023, 3] # 5023个顶点的坐标
        

        if self.no_lmks:
            # vertices = vertices * self.scale # 5023个顶点的坐标乘以缩放系数
            return vertices * self.scale
        
        landmarks3d = vertices2landmarks(
            vertices, 
            self.faces_tensor, # [9976, 3]
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1), # [N, 70]
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1) # [N, 70, 3]
        )
        landmark_3d = reselect_eyes(vertices, landmarks3d)
        if verts_sclae is not None:
            return vertices * verts_sclae, landmark_3d * verts_sclae
        return vertices * self.scale, landmarks3d * self.scale

    def _vertices2landmarks(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices, self.faces_tensor, 
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1)
        )
        landmark_3d = reselect_eyes(vertices, landmarks3d)
        return landmark_3d


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def reselect_eyes(vertices, lmks70):
    """
    vertices: N X 5023 X 3
    lmks70: N X 70 X 3
    重新选择和调整3D人脸模型的眼部关键点标记。在人脸建模和动画中，眼部区域的精确定位对于表达和视觉效果至关重要
    """
    lmks70 = lmks70.clone()
    eye_in_shape = [2422,2422, 2452, 2454, 2471, 3638, 2276, 2360, 3835, 1292, 1217, 1146, 1146, 999, 827, ] # 眼睛的索引
    eye_in_shape_reduce = [0,2,4,5,6,7,8,9,10,11,13,14] # 眼睛的索引，去掉了眼睛的内侧和外侧的点
    cur_eye = vertices[:, eye_in_shape]
    cur_eye[:, 0] = (cur_eye[:, 0] + cur_eye[:, 1]) * 0.5
    cur_eye[:, 2] = (cur_eye[:, 2] + cur_eye[:, 3]) * 0.5
    cur_eye[:, 11] = (cur_eye[:, 11] + cur_eye[:, 12]) * 0.5
    cur_eye = cur_eye[:, eye_in_shape_reduce]
    lmks70[:, [37,38,40,41,43,44,46,47]] = cur_eye[:, [1,2,4,5,7,8,10,11]] # 函数最后将处理过的眼部顶点分配给特定的关键点索引（37,38,40,41,43,44,46,47），这些索引可能对应于标准70点人脸模型中的眼睛轮廓点
    return lmks70
