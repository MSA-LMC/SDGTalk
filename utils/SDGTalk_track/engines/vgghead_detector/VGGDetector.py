#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Modified based on code from Orest Kupyn (University of Oxford).

import os
import torch
import numpy as np
import torchvision

from .utils_vgghead import nms
from .utils_lmks_detector import LmksDetector

class VGGHeadDetector(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.image_size = 640
        self._device = device

    def _init_models(self, ):
        self.lmks_detector = LmksDetector(self._device)
        # vgg_heads_l
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        _model_path = os.path.join(_abs_path, '../../assets/vgghead/vgg_heads_l.trcd')
        self.model = torch.jit.load(_model_path, map_location='cpu')
        self.model.to(self._device).eval()

    @torch.no_grad()
    def forward(self, image_tensor, image_key, conf_threshold=0.5, only_vgghead=False):
        if not hasattr(self, 'model'):
            self._init_models()
        image_tensor = image_tensor.to(self._device).float()
        image, padding, scale = self._preprocess(image_tensor)
        bbox, scores, flame_params = self.model(image)
        bbox, vgg_results = self._postprocess(bbox, scores, flame_params, conf_threshold)
        if bbox is None:
            print('VGGHeadDetector: No face detected: {}!'.format(image_key))
            return None, None, None
        vgg_results['normalize'] = {'padding': padding, 'scale': scale}
        # bbox
        bbox = bbox.clip(0, self.image_size)
        bbox[[0, 2]] -= padding[0]; bbox[[1, 3]] -= padding[1]; bbox /= scale
        bbox = bbox.clip(0, self.image_size / scale)
        if only_vgghead:
            return vgg_results, bbox, None
        lmks_2d70 = self.lmks_detector(image_tensor, bbox)
        return vgg_results, bbox, lmks_2d70

    def _preprocess(self, image):
        _, h, w = image.shape
        if h > w:
            # 如果图像高度大于宽度（竖向图像），则保持新高度为目标尺寸self.image_size，而宽度按比例缩放。
            new_h, new_w = self.image_size, int(w * self.image_size / h)
        else:
            # 如果图像宽度大于或等于高度（横向或方形图像），则保持新宽度为目标尺寸，而高度按比例缩放。
            new_h, new_w = int(h * self.image_size / w), self.image_size
        scale = self.image_size / max(h, w)
        # image 先进行了缩放再计算pad
        image = torchvision.transforms.functional.resize(image, (new_h, new_w), antialias=True)
        pad_w = self.image_size - image.shape[2]
        pad_h = self.image_size - image.shape[1]
        image = torchvision.transforms.functional.pad(image, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill=127)
        image = image.unsqueeze(0).float() / 255.0
        return image, np.array([pad_w // 2, pad_h // 2]), scale

    def _postprocess(self, bbox, scores, flame_params, conf_threshold):
        # flame_params = {"shape": 300, "exp": 100, "rotation": 6, "jaw": 3, "translation": 3, "scale": 1}
        """
        首先,nms 函数会根据置信度阈值(confidence_threshold)和 IOU 阈值，对输入的 bbox、scores 和 flame_params 进行筛选和去重，返回经过 NMS 筛选后的候选框、分数和参数。如果经过 NMS 后没有剩余的候选框(即 bbox.shape[0] == 0),函数会直接返回 None,表示没有检测到有效目标。

        如果还有候选框，代码会计算每个框的面积（(bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])），然后用 argmax() 找到面积最大的框的索引 max_idx。最后,将 bbox 和 flame_params 都更新为面积最大的那一个，方便后续只处理最显著的检测结果。

        这种做法常用于只需要关注最主要目标（如最大人脸）的场景，忽略其他较小或不重要的检测结果。
        """
        bbox, scores, flame_params = nms(bbox, scores, flame_params, confidence_threshold=conf_threshold)
        if bbox.shape[0] == 0:
            return None, None
        max_idx = ((bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])).argmax().long()
        bbox, flame_params = bbox[max_idx], flame_params[max_idx]
        if bbox[0] < 5 and bbox[1] < 5 and bbox[2] > 635 and bbox[3] > 635:
            return None, None
        # flame
        posecode = torch.cat([flame_params.new_zeros(3), flame_params[400:403]])
        vgg_results = {
            'rotation_6d': flame_params[403:409], 'translation': flame_params[409:412], 'scale': flame_params[412:],
            'shapecode': flame_params[:300], 'expcode': flame_params[300:400], 'posecode': posecode, 
        }
        # 这里 bbox 是 [x1, y1, x2, y2] 格式
        return bbox, vgg_results
