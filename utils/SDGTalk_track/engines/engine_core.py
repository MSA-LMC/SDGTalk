import os
import sys
import torch
from torch import nn
import pickle
import random
import shutil
import numpy as np
import torchvision
from tqdm.rich import tqdm
from PIL import Image
from .utils_lmdb import LMDBEngine
from .engine_optim import OptimEngine
from .vgghead_detector import VGGHeadDetector
from .flame_model import FLAMEModel, RenderMesh
from .emica_encoder import ImageEngine, EmicaEncoder
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import face_alignment

class CoreEngine:
    def __init__(self, focal_length, device='cuda'):
        random.seed(42)
        self._device = device
        # paths and data engine
        self.emica_encoder = EmicaEncoder(device=device)
        self.emica_data_engine = ImageEngine(device=device)
        self.vgghead_encoder = VGGHeadDetector(device=device)

        # 加载人脸分割模型
        self.seg_image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.seg_model.to(device)
        
        calibration_results = {'focal_length':focal_length, 'verts_scale': 5.0}
        self.calibration_results = calibration_results
        self.optim_engine = OptimEngine(self.calibration_results, device=device)
    
    def seg_face(self, crop_image):
        """使用分割模型生成人脸掩码"""
        # run inference on image
        image_array = crop_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # 转换为 HWC 格式
        image = Image.fromarray(image_array)
        inputs = self.seg_image_processor(images=image, return_tensors="pt").to(self._device)
        outputs = self.seg_model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

        # resize output to match input image dimensions
        upsampled_logits = nn.functional.interpolate(logits,
                        size=(image_array.shape[0], image_array.shape[1]), # H x W
                        mode='bilinear',
                        align_corners=False)

        # get label masks
        labels = upsampled_logits.argmax(dim=1)[0]

        # move to CPU to visualize in matplotlib
        labels_viz = labels.cpu().numpy()

        mask = np.zeros_like(labels_viz, dtype=np.uint8)
        # for pi in range(1, 14):
        for pi in range(1, 19):
            index = np.where(labels_viz == pi)
            mask[index[0], index[1]] = 1  # 蓝色 面部
        mask = mask.astype(np.uint8)

        image_array = image_array * mask[..., None]  # 应用掩码
        res_image = torch.from_numpy(image_array).permute(2, 0, 1)  # 转换回 CHW 格式

        return res_image

    def build_video(self, video_path, output_path, seg_head=False, crop_head=False):
        def smooth_bbox(all_bbox, alpha=0.7):
            smoothed_bbox = [all_bbox[0]]  # Initialize the smoothed data with the first value of the input data
            for i in range(1, len(all_bbox)):
                smoothed_value = alpha * all_bbox[i] + (1 - alpha) * smoothed_bbox[i-1]
                smoothed_bbox.append(smoothed_value)
            return smoothed_bbox
            
        video_name = os.path.basename(video_path).split('.')[0]
        video_name = '{:06d}'.format(int(video_name)) # Ensure video_name is a string of 6 digits
        print(f'Processing video: {video_name}')
        if os.path.exists(output_path):
            print(f'Output path {output_path} exists, replace it.')
            shutil.rmtree(output_path) 
        os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, 'img_lmdb')):
            frames_data, _, meta_data = torchvision.io.read_video(video_path, output_format='TCHW')
            assert frames_data.shape[0] > 0, 'No frames in the video, reading video failed.'
            print(f'Processing video {video_path} with {frames_data.shape[0]} frames.')
            if crop_head:
                all_frames_boxes, all_frames_idx = [], []
                for fidx, frame in tqdm(enumerate(frames_data), total=frames_data.shape[0]):
                    if meta_data['video_fps'] > 50:
                        if fidx % 2 == 0:
                            continue
                    _, bbox, _ = self.vgghead_encoder(frame, fidx, only_vgghead=True)
                    if bbox is not None:
                        all_frames_idx.append(fidx)
                        all_frames_boxes.append(bbox.cpu())
                if not len(all_frames_boxes):
                    print('No face detected in the video: {}, tracking failed.'.format(video_path))
                    return None
                # frames_data = frames_data[all_frames_idx]
                all_frames_boxes = smooth_bbox(all_frames_boxes, alpha=0.03)
                lmdb_engine = LMDBEngine(os.path.join(output_path, 'img_lmdb'), write=True)
                for fidx, ori_fidx in tqdm(enumerate(all_frames_idx), total=len(all_frames_idx)):
                    frame = frames_data[ori_fidx]
                    frame_bbox = all_frames_boxes[fidx]
                    frame_bbox = expand_bbox(frame_bbox, scale=1.60).long()
                    crop_frame = torchvision.transforms.functional.crop(
                        frame, top=frame_bbox[1], left=frame_bbox[0], height=frame_bbox[3]-frame_bbox[1], width=frame_bbox[2]-frame_bbox[0]
                    )
                    crop_frame = torchvision.transforms.functional.resize(crop_frame, (512, 512), antialias=True)
                    # frame = torchvision.transforms.functional.center_crop(frame, 512)
                    if seg_head:
                        crop_frame = self.seg_face(crop_frame)
                    lmdb_engine.dump(f'{video_name}_{fidx}', payload=crop_frame, type='image')
                lmdb_engine.random_visualize(os.path.join(output_path, 'img_lmdb', 'visualize.jpg'))
                lmdb_engine.close()
            else:
                lmdb_engine = LMDBEngine(os.path.join(output_path, 'img_lmdb'), write=True)
                for fidx, frame in tqdm(enumerate(frames_data), total=frames_data.shape[0]):
                    if meta_data['video_fps'] > 50:
                        if fidx % 2 == 0:
                            continue
                    frame = torchvision.transforms.functional.resize(frame, 512, antialias=True) 
                    frame = torchvision.transforms.functional.center_crop(frame, 512)
                    if seg_head:
                        frame = self.seg_face(frame)
                    lmdb_engine.dump(f'{video_name}_{fidx}', payload=frame, type='image')
                lmdb_engine.random_visualize(os.path.join(output_path, 'img_lmdb', 'visualize.jpg'))
                lmdb_engine.close()
            return meta_data['video_fps']
        else:
            video_reader = torchvision.io.VideoReader(src=video_path)
            meta_data = video_reader.get_metadata()['video']
            return meta_data['fps'][0]

    def track_base(self, lmdb_engine, output_path):
        if output_path is not None and os.path.exists(os.path.join(output_path, 'base.pkl')):
            # 已经存在 base.pkl , 那么直接读取
            with open(os.path.join(output_path, 'base.pkl'), 'rb') as f:
                base_results = pickle.load(f)
            return base_results
        else:
            images_dataset = ImagesData(lmdb_engine)
            num_workers = 0 if len(images_dataset) < 5 else 2
            images_loader = torch.utils.data.DataLoader(
                images_dataset, batch_size=1, num_workers=num_workers, shuffle=False
            ) # batch_size=1
            images_loader = iter(images_loader) # 将 DataLoader 转换为迭代器
            base_results = {}
            for image_data in tqdm(images_loader):
                image_data = data_to_device(image_data, device=self._device)
                image, image_key = image_data['image'][0], image_data['image_key'][0]
                emica_inputs = self.emica_data_engine(image, image_key)
                if emica_inputs is None:
                    continue
                emica_inputs = torch.utils.data.default_collate([emica_inputs])
                emica_inputs = data_to_device(emica_inputs, device=self._device)
                emica_results = self.emica_encoder(emica_inputs)

                vgg_results, bbox, lmks_2d70 = self.vgghead_encoder(image, image_key)
                if vgg_results is None:
                    continue
                emica_results, vgg_results = self._process_emica_vgg(emica_results, vgg_results, lmks_2d70)
                base_results[image_key] = {
                    'emica_results': emica_results, 
                    'vgg_results': vgg_results, 
                    'bbox': bbox.cpu().numpy() / 512.0 # bbox.shape = [4,], 归一化到 [0, 1]
                }
            if output_path is not None:
                with open(os.path.join(output_path, 'base.pkl'), 'wb') as f:
                    pickle.dump(base_results, f)
            return base_results

    def track_optim(self, base_result, output_path, lmdb_engine=None, share_id=False):
        # 已经存在 optim.pkl , 那么直接读取
        if output_path is not None and os.path.exists(os.path.join(output_path, 'optim.pkl')):
            with open(os.path.join(output_path, 'optim.pkl'), 'rb') as f:
                optim_results = pickle.load(f)
            return optim_results
        else:
            # self.optim_engine.init_model(self.calibration_results, image_size=512)
            base_result = {k: v for k, v in base_result.items() if v is not None}
            mini_batchs = build_minibatch(list(base_result.keys()), share_id=share_id) # mini_batchs 是一个列表，每个元素是一个小批量的帧名称列表

            if lmdb_engine is not None:
                batch_frames = torch.stack([lmdb_engine[key] for key in mini_batchs[0][:20]]).to(self._device).float() # torch.stack()函数将检索到的多个独立张量在新的维度（默认是第0维）上堆叠起来，创建一个批量张量
            else:
                batch_frames = None
            optim_results = {}
            for mini_batch in tqdm(mini_batchs):
                mini_batch_emica = [base_result[key] for key in mini_batch] # mini_batch_emica 是一个列表，每个元素是一个字典，包含了每一帧的 emica 结果
                mini_batch_emica = torch.utils.data.default_collate(mini_batch_emica)
                mini_batch_emica = data_to_device(mini_batch_emica, device=self._device)
                optim_result, visualization = self.optim_engine.lightning_optimize(
                    mini_batch, mini_batch_emica, batch_frames=batch_frames, share_id=share_id
                )
                batch_frames = None
                if visualization is not None:
                    torchvision.utils.save_image(visualization, os.path.join(output_path, 'optim.jpg'))
                optim_results.update(optim_result)
            if output_path is not None:
                with open(os.path.join(output_path, 'optim.pkl'), 'wb') as f:
                    pickle.dump(optim_results, f)
            return optim_results

    def track_image(self, inp_images, inp_keys, seg_head=True):
        assert type(inp_images) == list, f'Image must be a list, but got {type(inp_images)}.'
        assert inp_images[0].dim() == 3, f'Image dim must be 3, but got {inp_images[0].dim()}.'
        assert inp_images[0].max() > 1.0, f'Image in [0, 255.0], but got {inp_images[0].max()}.'
        assert len(inp_images) == len(inp_keys), f'Image and key length must be equal, but got {inp_images.shape[0]} and {len(inp_keys)}.'

        croped_images, croped_keys = [], []
        for inp_key, inp_image in tqdm(zip(inp_keys, inp_images), total=len(inp_images)):
            croped_image = inp_image
            if inp_image is not None:
                # 分割人脸和背景，背景移除  background_rgb=0.0 设置了背景颜色为黑色（RGB 值均为 0）
                if seg_head:
                    croped_image = self.seg_face(croped_image)
                croped_images.append(croped_image)
                croped_keys.append(inp_key)
        
        if not len(croped_images):
            print('No face detected in all the images, tracking failed.')
            return None
        images_engine = {key: image.cpu() for key, image in zip(croped_keys, croped_images)} # 将图像和键值对存储在字典中
        base_results = self.track_base(images_engine, None)
        if not len(base_results.keys()):
            print('No face detected in all the images, tracking failed.')
            return None
        optim_results = self.track_optim(base_results, None, None, share_id=False)
        for key in optim_results:
            # NOTE Input type (double) and bias type (float) should be the same
            optim_results[key]['image'] = images_engine[key].float().cpu().numpy() / 255.0
        # do visualization
        flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=self.calibration_results['verts_scale'])
        mesh_render = RenderMesh(512, faces=flame_model.get_faces().cpu().numpy(), device=self._device)
        for key in optim_results:
            pred_vertices, _ = flame_model(
                shape_params=torch.tensor(optim_results[key]['shapecode'])[None],
                expression_params=torch.tensor(optim_results[key]['expcode'])[None],
                pose_params=torch.tensor(optim_results[key]['posecode'])[None], 
                eye_pose_params=torch.tensor(optim_results[key]['eyecode'])[None],
            )
            # alpha_image 由 mesh_render 返回，通常表示每个像素点的透明度（取值范围一般为0~1）。它的作用是指示渲染结果中哪些区域属于前景（如人脸、三维模型），哪些属于背景
            rendered_image, alpha_image = mesh_render(
                pred_vertices.to(self._device), focal_length=self.calibration_results['focal_length'],
                transform_matrix=torch.tensor(optim_results[key]['transform_matrix'])[None].to(self._device),
            )
            rendered_image = rendered_image[0].cpu().numpy() / 255.0
            alpha_image = alpha_image[0].expand(3, -1, -1).cpu().numpy()
            vis_image = optim_results[key]['image'].copy()
            vis_image[alpha_image>0.5] *= 0.5
            vis_image[alpha_image>0.5] += (rendered_image[alpha_image>0.5] * 0.5)
            optim_results[key]['vis_image'] = vis_image
        """
        optim_results[key]['image'] = images_engine[key].cpu().numpy() / 255.0 原始图像
        optim_results[key]['vis_image'] = vis_image 融合 渲染结果和原始图像
        """
        return optim_results

    @staticmethod
    def _process_emica_vgg(emica_results, vgg_results, lmks_2d70):
        """
        静态方法不依赖类实例，可以通过类名或实例直接调用，不需要也不能访问 self 或类的属性。它只是类名空间下的一个普通函数
        """
        processed_emica_results = {
            'shapecode': emica_results['shapecode'][0].cpu().numpy(),
            'expcode': emica_results['expcode'][0].cpu().numpy(),
            'globalpose': emica_results['globalpose'][0].cpu().numpy(),
            'jawpose': emica_results['jawpose'][0].cpu().numpy(),
        }
        processed_vgg_results = {
            'shapecode': vgg_results['shapecode'].cpu().numpy(),
            'expcode': vgg_results['expcode'].cpu().numpy(),
            'posecode': vgg_results['posecode'].cpu().numpy(),
            'transform': {
                'rotation_6d': vgg_results['rotation_6d'].cpu().numpy(),
                'translation': vgg_results['translation'].cpu().numpy(),
                'scale': vgg_results['scale'].cpu().numpy(),
            },
            'normalize': vgg_results['normalize'],
            'lmks_2d70': lmks_2d70.cpu().numpy(),
        }
        return processed_emica_results, processed_vgg_results


class ImagesData(torch.utils.data.Dataset):
    """
    相当于 dataset, 用于加载图片数据
    """
    def __init__(self, lmdb_engine):
        super().__init__()
        self._lmdb_engine = lmdb_engine
        self._image_keys = list(lmdb_engine.keys())

    def __getitem__(self, index):
        """
        获取图片
        """
        image_key = self._image_keys[index]
        image = self._lmdb_engine[image_key]
        return {'image': image, 'image_key': image_key}

    def __len__(self, ):
        return len(self._image_keys)


def data_to_device(data_dict, device='cuda'):
    assert isinstance(data_dict, dict), 'Data must be a dictionary.'
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], np.ndarray):
            data_dict[key] = torch.tensor(data_dict[key], device=device)
        elif isinstance(data_dict[key], dict):
            data_dict[key] = data_to_device(data_dict[key], device=device)
        else:
            continue
    return data_dict


def build_minibatch(all_frames, batch_size=1024, share_id=False):
    """
    all_frames：所有需要处理的帧名称列表
    batch_size：每个小批量包含的最大帧数（默认1024）
    share_id：是否按视频ID分组处理（默认False,不按video_id进行mini_batch处理）
    """
    if share_id: # 按视频ID分组处理
        all_frames = sorted(all_frames)
        video_names = list(set(['_'.join(frame_name.split('_')[:-1]) for frame_name in all_frames]))
        # 对应id的帧名称列表
        video_frames = {video_name: [] for video_name in video_names}
        for frame in all_frames:
            video_name = '_'.join(frame.split('_')[:-1])
            video_frames[video_name].append(frame)
        # 每个mini_batch都是一个video_id的帧
        all_mini_batch = []
        for video_name in video_names:
            mini_batch = []
            for frame_name in video_frames[video_name]:
                mini_batch.append(frame_name)
                if len(mini_batch) % batch_size == 0:
                    all_mini_batch.append(mini_batch)
                    mini_batch = []
            if len(mini_batch): # 剩下的帧
                all_mini_batch.append(mini_batch)
    else: # 不按视频ID分组处理,随机生成mini_batch
        try:
            all_frames = sorted(all_frames, key=lambda x: int(x.split('_')[-1]))
        except:
            all_frames = sorted(all_frames)
        all_mini_batch, mini_batch = [], []
        for frame_name in all_frames:
            mini_batch.append(frame_name)
            if len(mini_batch) % batch_size == 0:
                all_mini_batch.append(mini_batch)
                mini_batch = []
        if len(mini_batch):
            all_mini_batch.append(mini_batch)
    return all_mini_batch


def expand_bbox(bbox, scale=1.1):
    xmin, ymin, xmax, ymax = bbox.unbind(dim=-1)
    cenx, ceny = (xmin + xmax) / 2, (ymin + ymax) / 2
    # ceny = ceny - (ymax - ymin) * 0.05
    extend_size = torch.sqrt((ymax - ymin) * (xmax - xmin)) * scale
    xmine, xmaxe = cenx - extend_size / 2, cenx + extend_size / 2
    ymine, ymaxe = ceny - extend_size / 2, ceny + extend_size / 2
    expanded_bbox = torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
    return torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
