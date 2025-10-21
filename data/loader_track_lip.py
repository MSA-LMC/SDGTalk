import os
import json
import torch
import pickle
import random
import numpy as np
import torchvision
from copy import deepcopy

from utils.utils_lmdb import LMDBEngine
from utils.flame_model import FLAMEModel

FOCAL_LENGTH = 12.0

class FlameDriveData(torch.utils.data.Dataset):
    def __init__(self, data_cfg, split):
        super().__init__()
        # build path
        self._split = split
        assert self._split in ['train', 'val', 'test'], f'Invalid split: {self._split}'
        # meta data
        self._data_path = data_cfg.PATH
        self._point_plane_size = data_cfg.POINT_PLANE_SIZE # POINT_PLANE_SIZE: 296
        
        # build records
        with open(os.path.join(self._data_path, 'optim.pkl'), 'rb') as f:
            self._data = pickle.load(f)
        
        with open(os.path.join(self._data_path, 'dataset.json'), 'r') as f:
            self._frames = json.load(f)[self._split]

        self._video_info = build_video_info(self._frames) 

        if self._split in ['val', 'test']:
            first_frame = [self._video_info[v][0] for v in self._video_info.keys()] 
            self._frames = [f for f in self._frames if f not in first_frame] 
        # build model
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=data_cfg.FLAME_SCALE, no_lmks=True)

    def slice(self, slice):
        self._frames = self._frames[:slice]

    def __getitem__(self, index):
        frame_key = self._frames[index]
        return self._load_one_record(frame_key)

    def __len__(self, ):
        return len(self._frames)

    def _init_lmdb_database(self):
        self._lmdb_engine = LMDBEngine(os.path.join(self._data_path, 'img_lmdb'), write=False)

    def _choose_feature_image(self, frame_key, number=1):
        video_id = get_video_id(frame_key)
        frame_index = get_frame_index(frame_key)

        if self._split == 'train': 
            candidate_key = [key for key in self._video_info[video_id] if key != frame_key] 
            feature_key = random.sample(candidate_key, k=number)[0]
        else:
            feature_key = self._video_info[video_id][0] 
        
        f_image = self._lmdb_engine[feature_key].float() / 255.0
        # resize feature image
        f_image_resize = torchvision.transforms.functional.resize(f_image, (518, 518), antialias=True)

        # feature points
        f_record = {}
        for key in [ 'posecode', 'shapecode', 'expcode', 'eyecode']:
            f_record[key] = torch.tensor(self._data[feature_key][key]).float()
        f_points = self.flame_model(
            shape_params=f_record['shapecode'][None], pose_params=f_record['posecode'][None],
            expression_params=f_record['expcode'][None], eye_pose_params=f_record['eyecode'][None],
        )[0].float()

        f_transform = torch.tensor(self._data[feature_key]['transform_matrix']).float() 
        f_planes = build_points_planes(self._point_plane_size, f_transform)
        return feature_key, f_image, f_image_resize, f_points, f_planes

    def _load_one_record(self, frame_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        # feature image
        f_key, f_image, f_image_resize, f_points, f_planes = self._choose_feature_image(frame_key)

        # driven image 
        t_image = self._lmdb_engine[frame_key].float() / 255.0
        t_bbox = torch.tensor(self._data[frame_key]['bbox']).float()
        t_transform = torch.tensor(self._data[frame_key]['transform_matrix']).float()

        t_record = {}
        for key in ['posecode', 'shapecode', 'expcode', 'eyecode']:
            t_record[key] = torch.tensor(self._data[frame_key][key]).float()
        t_points = self.flame_model(
            shape_params=t_record['shapecode'][None], pose_params=t_record['posecode'][None],
            expression_params=t_record['expcode'][None], eye_pose_params=t_record['eyecode'][None],
        )[0].float()

        one_record = {
            'f_image': f_image,
            'f_image_resize': f_image_resize, 
            'f_planes': f_planes, 
            'f_points': f_points, 

            't_image': t_image, 
            't_bbox': t_bbox, 
            't_transform': t_transform, 
            't_points': t_points, 
            'infos': {'f_key':f_key, 't_key':frame_key},
        }
        return one_record


class AudioDriveData(torch.utils.data.Dataset):
    def __init__(self, data_cfg, split):
        super().__init__()
        # build path
        self._split = split
        assert self._split in ['train', 'val', 'test'], f'Invalid split: {self._split}'
        # meta data
        self._data_path = data_cfg.PATH
        self._point_plane_size = data_cfg.POINT_PLANE_SIZE # POINT_PLANE_SIZE: 296
        
        # build records
        with open(os.path.join(self._data_path, 'optim.pkl'), 'rb') as f:
            self._data = pickle.load(f)

        # build audio_emb
        audio_emb_path = os.path.join(self._data_path, f'audio_{data_cfg.AUDIO_EMBEDDING_TYPE}.pkl')
        assert os.path.exists(audio_emb_path), f'Audio embedding file not found: {audio_emb_path}'
        with open(audio_emb_path, 'rb') as f:
            self._audio_emb = pickle.load(f)

        # build face landmark
        with open(os.path.join(self._data_path, 'face_landmark.pkl'), 'rb') as f:
            self._landmark = pickle.load(f)
        
        # build frames
        with open(os.path.join(self._data_path, 'dataset.json'), 'r') as f:
            self._frames = json.load(f)[self._split]

        """
        video_info = {
            video_id: [frame_key1, frame_key2, ...]
        }
        """
        self._video_info = build_video_info(self._frames) 

        if self._split in ['val', 'test']:
            first_frame = [self._video_info[v][0] for v in self._video_info.keys()] 
            self._frames = [f for f in self._frames if f not in first_frame] 
        # build model
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=data_cfg.FLAME_SCALE, no_lmks=True)

    def slice(self, slice):
        self._frames = self._frames[:slice]

    def __getitem__(self, index):
        frame_key = self._frames[index]
        return self._load_one_record(frame_key)

    def __len__(self, ):
        return len(self._frames)

    def _init_lmdb_database(self):
        self._lmdb_engine = LMDBEngine(os.path.join(self._data_path, 'img_lmdb'), write=False)

    def _choose_feature_image(self, frame_key, number=1):
        video_id = get_video_id(frame_key)

        if self._split == 'train': 
            candidate_key = [key for key in self._video_info[video_id] if key != frame_key] 
            feature_key = random.sample(candidate_key, k=number)[0]
        else:
            feature_key = self._video_info[video_id][0] 
        # feature image
        f_image = self._lmdb_engine[feature_key].float() / 255.0
        # resize feature image
        f_image_resize = torchvision.transforms.functional.resize(f_image, (518, 518), antialias=True)

        # feature points
        f_record = {}
        for key in [ 'posecode', 'shapecode', 'expcode', 'eyecode']:
            f_record[key] = torch.tensor(self._data[feature_key][key]).float()
            
        f_points = self.flame_model(
            shape_params=f_record['shapecode'][None], pose_params=f_record['posecode'][None],
            expression_params=f_record['expcode'][None], eye_pose_params=f_record['eyecode'][None],
        )[0].float()

        f_transform = torch.tensor(self._data[feature_key]['transform_matrix']).float() # 4x4矩阵
        # {'plane_points': plane_points, 'plane_dirs': cam_dirs[0]}
        f_planes = build_points_planes(self._point_plane_size, f_transform)
        return feature_key, f_image, f_image_resize, f_points, f_planes

    def _load_one_record(self, frame_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        # feature image
        f_key, f_image, f_image_resize, f_points, f_planes = self._choose_feature_image(frame_key)

        # driven image
        video_id = get_video_id(frame_key)
        frame_index = get_frame_index(frame_key)
        
        t_image = self._lmdb_engine[frame_key].float() / 255.0
        t_bbox = torch.tensor(self._data[frame_key]['bbox']).float()
        t_transform = torch.tensor(self._data[frame_key]['transform_matrix']).float()
        t_audioemb = torch.tensor(self._audio_emb[video_id][frame_index]).float()    # [time_window 16, feature_dim 29/768/1024]
        t_facelandmark = torch.tensor(self._landmark[video_id][frame_index]) # [68, 2] 

        # lip box
        lips = slice(48, 60) # 
        xmin, xmax = int(t_facelandmark[lips, 1].min()), int(t_facelandmark[lips, 1].max())
        ymin, ymax = int(t_facelandmark[lips, 0].min()), int(t_facelandmark[lips, 0].max())
        cx = (xmin + xmax) // 2 
        cy = (ymin + ymax) // 2 
        l = max(xmax - xmin, ymax - ymin) // 2
        xmin = cx - l
        xmax = cx + l
        ymin = cy - l
        ymax = cy + l
        t_lipbox = torch.tensor([xmin, xmax, ymin, ymax], dtype=torch.int32) # [xmin, ymin, xmax, ymax]

        # eye landmark
        t_eyelandmark = torch.tensor([int(t_facelandmark[41][1] - t_facelandmark[37][1]),int(t_facelandmark[40][1] - t_facelandmark[38][1]), int(t_facelandmark[39][0] - t_facelandmark[36][0]), int(t_facelandmark[47][1] - t_facelandmark[43][1]), int(t_facelandmark[46][1] - t_facelandmark[44][1]), int(t_facelandmark[45][0] - t_facelandmark[42][0])]).float()

        t_record = {}
        for key in ['posecode', 'shapecode', 'expcode', 'eyecode']:
            t_record[key] = torch.tensor(self._data[frame_key][key]).float()
        t_points = self.flame_model(
            shape_params=t_record['shapecode'][None], pose_params=t_record['posecode'][None],
            expression_params=t_record['expcode'][None], eye_pose_params=t_record['eyecode'][None],
        )[0].float()

        one_record = {
            'f_image': f_image,
            'f_image_resize': f_image_resize, 
            'f_planes': f_planes, 
            'f_points': f_points,

            't_image': t_image, 
            't_bbox': t_bbox, 
            't_lipbox': t_lipbox,
            't_eyelandmark': t_eyelandmark,
            't_points': t_points, 
            't_transform': t_transform, 
            't_audioemb': t_audioemb,
            'infos': {'f_key':f_key, 't_key':frame_key},
        }
        return one_record

class InferTestData(torch.utils.data.Dataset):
    def __init__(self, driver_path, feature_data=None, point_plane_size=296):
        super().__init__()
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        
        # ------------------------------------------------------
        self.driver_path = driver_path
        self.point_plane_size = point_plane_size

        with open(os.path.join(self.driver_path, 'optim.pkl'), 'rb') as f:
            self._data = pickle.load(f)
            self._frames = sorted(list(self._data.keys()), key=lambda x:int(x.split('_')[-1]))


        pkls_dir = os.path.join(_abs_path, '../demo/raw/pkls')
        # build audio_emb (deepspeech)
        with open(os.path.join(pkls_dir, 'audio_hubert.pkl'), 'rb') as f:
            self._audio_emb = pickle.load(f)

        # build face landmark
        with open(os.path.join(pkls_dir, 'face_landmark.pkl'), 'rb') as f:
            self._landmark = pickle.load(f)
        # ---------------------------------------------------------

        # build model
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True)
        # meta data ------------------------------------------------------
        self.feature_data = feature_data
        self.f_image_resize = torchvision.transforms.functional.resize(self.feature_data['image'].cpu(), (518, 518), antialias=True)

        # feature planes
        f_transform = self.feature_data['transform_matrix'].float().cpu()
        self.f_planes = build_points_planes(self.point_plane_size, f_transform)

        # feature points
        f_record = {}
        for key in [ 'posecode', 'shapecode', 'expcode', 'eyecode']:
            f_record[key] = torch.tensor(self.feature_data[key]).float()
        self.f_points = self.flame_model(
            shape_params=f_record['shapecode'][None], pose_params=f_record['posecode'][None],
            expression_params=f_record['expcode'][None], eye_pose_params=f_record['eyecode'][None],
        )[0].float().cpu()
        # --------------------------------------------------------------------

    def slice(self, slice):
        self._frames = self._frames[:slice]


    def __getitem__(self, index):
        frame_key = self._frames[index] 
        return self._load_one_record(frame_key)

    def __len__(self, ):
        return len(self._frames) - 8

    def _init_lmdb_database(self):
        # print('Init the LMDB Database!')
        self._lmdb_engine = LMDBEngine(os.path.join(self.driver_path, 'img_lmdb'), write=False)

    def _load_one_record(self, frame_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        video_id = get_video_id(frame_key)
        frame_index = get_frame_index(frame_key)

        t_image = self._lmdb_engine[frame_key].float() / 255.0
        t_transform = torch.tensor(self._data[frame_key]['transform_matrix']).float()
        t_audioemb = torch.tensor(self._audio_emb[video_id][frame_index]).float()
        t_facelandmark = torch.tensor(self._landmark[video_id][frame_index])
        # eye landmark
        t_eyelandmark = torch.tensor([int(t_facelandmark[41][1] - t_facelandmark[37][1]),int(t_facelandmark[40][1] - t_facelandmark[38][1]), int(t_facelandmark[39][0] - t_facelandmark[36][0]), int(t_facelandmark[47][1] - t_facelandmark[43][1]), int(t_facelandmark[46][1] - t_facelandmark[44][1]), int(t_facelandmark[45][0] - t_facelandmark[42][0])]).float()


        one_data = {
            'f_image_resize': deepcopy(self.f_image_resize), 
            'f_planes': deepcopy(self.f_planes), 
            'f_points': deepcopy(self.f_points), 

            't_image': t_image, 
            't_eyelandmark': t_eyelandmark,
            't_transform': t_transform, 
            't_audioemb': t_audioemb,
            'infos': {'t_key':frame_key},
        }
        return one_data



def build_points_planes(plane_size, transforms):
    x, y = torch.meshgrid(
        torch.linspace(1, -1, plane_size, dtype=torch.float32), 
        torch.linspace(1, -1, plane_size, dtype=torch.float32), 
        indexing="xy",
    ) # x: (plane_size, plane_size), y: (plane_size, plane_size)
    R = transforms[:3, :3] 
    T = transforms[:3, 3:] 

    cam_dirs = torch.tensor([[0., 0., 1.]], dtype=torch.float32) 

    ray_dirs = torch.nn.functional.pad(
        torch.stack([x/FOCAL_LENGTH, y/FOCAL_LENGTH], dim=-1), (0, 1), value=1.0
    ) # ray_dirs.shape = (plane_size, plane_size, 3)
    cam_dirs = torch.matmul(R, cam_dirs.reshape(-1, 3)[:, :, None])[..., 0] 
    ray_dirs = torch.matmul(R, ray_dirs.reshape(-1, 3)[:, :, None])[..., 0] 

    origins = (-torch.matmul(R, T)[..., 0]).broadcast_to(ray_dirs.shape).squeeze() 
    distance = ((origins[0] * cam_dirs[0]).sum()).abs()
    plane_points = origins + distance * ray_dirs # (plane_size*plane_size, 3)
    # cam_dirs[0].shape = (3,)
    return {'plane_points': plane_points, 'plane_dirs': cam_dirs[0]} 

def build_video_info(frames):
    """
    video_info = {
        video_id: [frame_key1, frame_key2, ...]
    }
    """
    video_info = {}

    for key in frames:
        video_id = get_video_id(key)
        if video_id not in video_info.keys():
            video_info[video_id] = []
        video_info[video_id].append(key)
    for video_id in video_info.keys():
        video_info[video_id] = sorted(
            video_info[video_id], key=lambda x:int(x.split('_')[-1])
        )

    return video_info


def get_video_id(frame_key):
    if frame_key.split('_')[0] in ['img']: 
        video_id = frame_key.split('_')[1]
    else:
        video_id = frame_key.split('_')[0] 
    return video_id

def get_frame_index(frame_key):
    frame_index = int(frame_key.split('_')[-1])
    return frame_index