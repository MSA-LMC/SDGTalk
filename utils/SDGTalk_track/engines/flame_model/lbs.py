# THIS FILE HAS BEEN COPIED FROM THE EMOCA TRAINING REPOSITORY

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn.functional as F

def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def find_dynamic_lmk_idx_and_bcoords(vertices, pose, dynamic_lmk_faces_idx,
                                     dynamic_lmk_b_coords,
                                     neck_kin_chain, dtype=torch.float32):
    ''' Compute the faces, barycentric coordinates for the dynamic landmarks


        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.

        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.
        dtype: torch.dtype, optional

        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    '''

    batch_size = vertices.shape[0]

    aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                 neck_kin_chain)
    rot_mats = batch_rodrigues(
        aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

    rel_rot_mat = torch.eye(3, device=vertices.device,
                            dtype=dtype).unsqueeze_(dim=0)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
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


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' 
        Calculates landmarks by barycentric interpolation
        通过重心坐标插值(barycentric interpolation)计算三维网格上的特定地标点(landmarks)
        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices 输入顶点的张量
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh 网格的面
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks. 具有人脸索引的张量，用于计算 地标。
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks用于插值地标的重心坐标张量

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3) # lmk_faces.shape = [N,70,3] 其中N是batchsize，70是面的index，3是每个面有3个顶点
    """
    由于 PyTorch 的广播机制，偏移量会被应用到每个批次中的所有面索引，确保第一个批次的面索引指向第一个批次的顶点(0 到 num_verts-1)，第二个批次的面索引指向第二个批次的顶点(num_verts 到 2*num_verts-1)，以此类推
    """
    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts # lmk_faces.shape = [N,70,3] 其中N是batchsize，70是面数，3是每个面有3个顶点(为什么要* num_verts,可以直接看下面的代码)
    """
    import torch
    data_in = torch.randn(8, 5023, 3).view(-1, 3)
    index_in = torch.randint(1, 20, (8,70,3))
    data_out = data_in[index_in]
    print(data_out.shape) torch.Size([8, 70, 3, 3])

    data_in 是一个二维张量，形状为 (N, D)，其中 N = 8 * 5023 = 40184，D = 3。
    index_in 是一个三维张量，形状为 (A, B, C)，其中 A = 8, B = 70, C = 3。
    对于 index_in 中的每个元素 i，都会执行 data_in[i]，得到一个长度为 3 的向量。
    所有这些向量会被组合成一个新的张量，其形状为 (A, B, C, D)，即 (8, 70, 3, 3)
    """
    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3) # lmk_vertices.shape = [N,70,3,3] 其中N是batchsize，70是面数，3是每个面有3个顶点,3是每个顶点的坐标
    """
    对于每个地标，取其对应三角面的三个顶点坐标，分别乘以预先计算好的重心坐标权重，然后将结果相加。这正是基于重心坐标进行点插值的标准方法，它确保插值点位于三角形内部或边界上。
    """
    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32, detach_pose_correctives=True):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor B x (J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs) # blend_shapes就是一个乘法操作

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped) # J.shape = (N, J, 3),其中 J = 5 ,关节点坐标

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        """
        pose.view(-1, 3) = [N*5,3] = [N*J,3]
        """
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3]) # N*Jx3x3 -> NxJx3x3

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]) # [N,J-1,3,3] -> [N,(J-1)*9] = [N,P= 36]
        # (N x P) x (P, V * 3) -> N x V x 3 # [N,36] x [36,5023*3] -> [N,5023*3] -> [N,5023,3]
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    if detach_pose_correctives:
        pose_offsets = pose_offsets.detach()

    v_posed = pose_offsets + v_shaped # v_posed.shape = [N,5023,3] 5023是顶点的数量

    # 4. Get the global joint location 沿着关节链，返回关节链上的旋转矩阵，即相对父节点的旋转，最后沿着关节链得到世界坐标系下关节的位置，最终得到世界坐标系下 关节顶点 的位置
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1) 每个顶点受哪些关节的影响及其权重
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1]) #W.shape = [N,5023,5] 5023是顶点的数量，5是关节的数量

    # (N x V x J) x (N x J x 16) -> (N x V x 16) -> (N x V x 4 x 4)
    num_joints = J_regressor.shape[0] # J_regressor.shape = [5, 5023]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4) # A.view(batch_size, num_joints, 16).shape = [N,J,16], T.shape = [N,5023,4,4]

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],dtype=dtype, device=device) # [N,5023,1] 
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2) # v_posed_homo.shape = [N,5023,4]

    # [N,5023,4,4]*[N,5023,4,1] -> [N,5023,4,1]
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0] # verts.shape = [N,V,3] = [N.5023,3]

    """
    verts: torch.tensor BxVx3 顶点坐标位置
        The vertices of the mesh after applying the shape and pose
        displacements.
    joints: torch.tensor BxJx3 关节顶点坐标位置
        The joints of the model
    """
    return verts, J_transformed


def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints

    vertices 为batch_size x V x 3的矩阵为当前batch中顶点的xyz坐标,其中V=5023
    与J_regressor矩阵做相乘后,得到batch_size x J x3的矩阵,其中J是FLAME中关节的数量,J=5
    这里的乘法可以当做一个回归操作,所以J_regressor被称为回归矩阵
    意思就是从顶点坐标回归出关节点坐标
    '''
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    前面提到pose是轴角旋转的形式，首先用Rodrigues公式将其转为旋转矩阵的形式，轴角旋转的维度是Nx3，转为旋转矩阵后的维度是Nx3x3，这里维度的变化是因为轴角表示法中每个关节的旋转用3维向量表示，将这个向量用Rodrigues公式转为旋转矩阵后，每个旋转矩阵的维度就是3x3（详细可自行学习Rodrigues公式）
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
            
            [ R11 R12 R13 t1 ]
            [ R21 R22 R23 t2 ]
            [ R31 R33 R33 t3 ]
            [ 0 0 0 1 ]
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    rot_mats是形状为BxNx3x3的旋转矩阵，joints是BxNx3的关节位置，parents是BxN的父节点信息。返回的是posed_joints和rel_transforms。目标是对关节应用刚体变换，得到姿态变换后的关节位置和相对变换矩阵

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    沿着关节链，返回关节链上的旋转矩阵，即相对父节点的旋转，最后沿着关节链得到世界坐标系下关节的位置，最终得到世界坐标系下顶点的位置
    """

    joints = torch.unsqueeze(joints, dim=-1) # joints.shape = BxNx3x1 ,N=J=5 关节位置

    rel_joints = joints.clone()
    """
    rel_joints被创建为joints的克隆，但第二个及之后的关节位置减去其父节点的位置。这一步应该是计算相对于父节点的位移，也就是将每个子关节的位置表示为相对于父关节的局部坐标。例如，对于关节i（i>0），其位置是相对于父关节parent[i]的，因此需要减去父关节的位置，得到局部坐标
    """
    rel_joints[:, 1:] -= joints[:, parents[1:]]# rel_joints.shape = BxNx3x1 ,N=J=5

    # transforms_mat = transform_mat(
    #     rot_mats.view(-1, 3, 3),
    #     rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)
    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3), # [N,J,3,3] -> [N*J,3,3]
        rel_joints.reshape(-1, 3, 1) # [N,J,3,1] -> [N*J,3,1]
        ).reshape(-1, joints.shape[1], 4, 4) # [N*J,4,4] -> [N,J,4,4] 每个关节对应一个变换矩阵

    transform_chain = [transforms_mat[:, 0]] # transforms_mat[:, 0].shape = [N,4,4]
    for i in range(1, parents.shape[0]): # parents.shape[0] = 5
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        # 将transform_chain中父节点的变换矩阵与当前关节的变换矩阵相乘，得到该关节的世界变换
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1) # [B,J,4,4]

    # The last column of the transformations contains the posed joints 提取变换后的关节位置（取平移分量）
    posed_joints = transforms[:, :, :3, 3] # posed_joints.shape = [B,J,3] 关节位置

    # 计算相对变换矩阵（relative transforms）
    joints_homogen = F.pad(joints, [0, 0, 0, 1]) # 转换为齐次坐标（BxNx4x1）
    # rel_transforms.shape = [B,N,4,4] 关节的刚性变换矩阵 
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen),  # 计算绝对位置
        [3, 0, 0, 0, 0, 0, 0, 0]) # 对齐矩阵维度

    return posed_joints, rel_transforms
