import os
import glob
import json
import torch
from torch import nn
import numpy as np
import argparse
import torchvision
from tqdm.rich import tqdm
from PIL import Image
from engines.utils_lmdb import LMDBEngine
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class FaceDetector:
    def __init__(self, device='cuda'):
        self._device = device
        self.seg_image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.seg_model.to(device)
    
    def seg_face(self, crop_image):
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
        # https://huggingface.co/jonathandinu/face-parsing
        # for pi in range(1, 14):
        for pi in range(1, 19):
            index = np.where(labels_viz == pi)
            mask[index[0], index[1]] = 1
        mask = mask.astype(np.uint8)

        image_array = image_array * mask[..., None]
        res_image = torch.from_numpy(image_array).permute(2, 0, 1)

        return res_image

    def forward(self, image_path):
        inp_image = torchvision.io.read_image(image_path, mode=torchvision.io.image.ImageReadMode.RGB)
        inp_image = inp_image.to(self._device).float()
        crop_image = torchvision.transforms.functional.resize(inp_image, (512, 512), antialias=True)
        crop_image = self.seg_face(crop_image)  
        return crop_image


if __name__ == '__main__':
    # video_dirs is the list of video directories, each contains frames of the video, named as [frameid.jpg].
    PATH_TO_YOUR_STORAGE = "img_lmdb"
    BASE_DIR = "extract_imgs"
    VIDEO_DIRS = [str(os.path.join(BASE_DIR, f'{i}'))+"/" for i in  os.listdir(BASE_DIR) ]
    # print(VIDEO_DIRS)
    data_processor = FaceDetector()
    lmdb_engine = LMDBEngine(PATH_TO_YOUR_STORAGE, write=True)
    for vidx, video_dir_path in enumerate(VIDEO_DIRS):
        frames = glob.glob(f'{video_dir_path}/*.jpg') # image name shuold be [frameid.jpg]
        frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        video_id = int(os.path.basename(video_dir_path[:-1] if video_dir_path.endswith('/') else video_dir_path)) # video_id should be INT
        print(f'Processing video: {video_id}')
        for frame_path in tqdm(frames, desc=f'Processing video {vidx+1}/{len(VIDEO_DIRS)}:'):
            frame_tensor = data_processor.forward(frame_path)
            frame_id = int(os.path.splitext(os.path.basename(frame_path))[0])
            if frame_tensor is None:
                continue
            frame_tensor = frame_tensor.to(torch.uint8).cpu()
            # torchvision.io.write_jpeg(frame_tensor.to(torch.uint8).cpu(), 'debug.jpg', quality=90)
            dump_name = '{:06d}_{}'.format(video_id, frame_id)
            if lmdb_engine.exists(dump_name):
                print('Frame already exists: {}'.format(frame_path))
                continue
            # if image_tensor.max() < 5.0 
            if frame_tensor.max() < 5.0: 
                print('Frame empty: {}'.format(frame_path))
                continue
            # print(dump_name)
            lmdb_engine.dump(dump_name, payload=frame_tensor, type='image')
    lmdb_engine.random_visualize(os.path.join(PATH_TO_YOUR_STORAGE, 'visualize.jpg'))
    lmdb_engine.close()
