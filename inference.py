#!/usr/bin/env python
# Modified from GAGAvatar
import os
import torch
import argparse
import subprocess
import lightning
import numpy as np
import torchvision
from tqdm.rich import tqdm

from data import InferTestData
from models import build_model
from utils.utils import ConfigDict
from utils.SDGTalk_track.engines import CoreEngine as TrackEngine

def inference(image_path, driver_path, resume_path, force_retrack=False, save_video=True , device='cuda'):
    lightning.fabric.seed_everything(42)
    driver_path = driver_path[:-1] if driver_path.endswith('/') else driver_path
    driver_name = os.path.basename(driver_path).split('.')[0]
    # load model
    print(f'Loading model...')
    lightning_fabric = lightning.Fabric(accelerator=device, strategy='auto', devices=[0],)
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    # model ------------------------------------------------------
    model = build_model(model_cfg=meta_cfg.AUDIO_DRIVE_MODEL)
    model.load_state_dict(full_checkpoint['model'])
    model = lightning_fabric.setup(model)
    model.eval()
    
    track_engine = TrackEngine(focal_length=12.0, device=device)
    feature_name = os.path.basename(image_path).split('.')[0] 
    feature_data = get_tracked_results(image_path, track_engine, force_retrack=force_retrack)
    if feature_data is None:
        print(f'Finish inference, no face in input: {image_path}.')
        return
    
    driver_name = os.path.basename(driver_path[:-1] if driver_path.endswith('/') else driver_path)
    driver_dataset = InferTestData(driver_path, feature_data, 296)
    driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False) # batch_size=1 , shuffle=False

    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)

    images = []
    gt_images = []
    for idx, batch in enumerate(tqdm(driver_dataloader)):
        render_results = model.forward_infer_test(batch)
        gt_rgb = render_results['t_image'].clamp(0, 1)
        # pred_rgb = render_results['gen_image'].clamp(0, 1)
        pred_sr_rgb = render_results['sr_gen_image'].clamp(0, 1)
        gt_rgbs = torchvision.utils.make_grid([gt_rgb[0]], nrow=4, padding=0)
        visulize_rgbs = torchvision.utils.make_grid([pred_sr_rgb[0]], nrow=4, padding=0)
        # visulize_rgbs.shape = (C, H, W)
        gt_images.append(gt_rgbs.cpu())
        images.append(visulize_rgbs.cpu())

    dump_dir = os.path.join('render_results', "res_video")
    gt_dir = os.path.join('render_results', "gt_video")
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    if save_video:
        dump_path = os.path.join(dump_dir, f'{driver_name}.mp4')
        gt_dump_path = os.path.join(gt_dir, f'{driver_name}.mp4')
        merged_images = torch.stack(images)
        gt_merged_images = torch.stack(gt_images)
        merged_images = (merged_images * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
        gt_merged_images = (gt_merged_images * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(dump_path, merged_images, fps=25.0)
        torchvision.io.write_video(gt_dump_path, gt_merged_images, fps=25.0)
    else:
        dump_path = os.path.join(dump_dir, f'{driver_name}.jpg')
        merged_images = torchvision.utils.make_grid(images, nrow=5, padding=0)
        feature_images = torchvision.utils.make_grid([feature_data['image']]*(merged_images.shape[-2]//512), nrow=1, padding=0)
        merged_images = torch.cat([feature_images, merged_images], dim=-1)
        torchvision.utils.save_image(merged_images, dump_path)
    print(f'Finish inference: {dump_path}.')


def get_tracked_results(image_path, track_engine, force_retrack=False):
    if not is_image(image_path):
        print(f'Please input a image path, got {image_path}.')
        return None
    
    tracked_pt_path = 'render_results/tracked/tracked.pt'
    if not os.path.exists(tracked_pt_path):
        os.makedirs('render_results/tracked', exist_ok=True)
        torch.save({}, tracked_pt_path)

    tracked_data = torch.load(tracked_pt_path, weights_only=False)
    image_base = os.path.basename(image_path)

    if image_base in tracked_data and not force_retrack:
        print(f'Load tracking result from cache: {tracked_pt_path}.')
    else:
        print(f'Tracking {image_path}...')
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()
        feature_data = track_engine.track_image([image], [image_path],seg_head=False)
        if feature_data is not None:
            feature_data = feature_data[image_path]
            torchvision.utils.save_image(
                torch.tensor(feature_data['vis_image']), 'render_results/tracked/{}.jpg'.format(image_base.split('.')[0])
            )
        else:
            print(f'No face detected in {image_path}.')
            return None
        tracked_data[image_base] = feature_data
        other_names = [i for i in os.listdir(os.path.dirname(image_path)) if is_image(i)]
        other_paths = [os.path.join(os.path.dirname(image_path), i) for i in other_names]
        if len(other_paths) <= 150:
            print('Track on all images in this folder to save time.')
            other_images = [torchvision.io.read_image(imp, mode=torchvision.io.ImageReadMode.RGB).float() for imp in other_paths]
            try:
                other_feature_data = track_engine.track_image(other_images, other_names,seg_head=False)
                for key in other_feature_data:
                    torchvision.utils.save_image(
                        torch.tensor(other_feature_data[key]['vis_image']), 'render_results/tracked/{}.jpg'.format(key.split('.')[0])
                    )
                tracked_data.update(other_feature_data)
            except Exception as e:
                print(f'Error: {e}.')
        torch.save(tracked_data, tracked_pt_path)
    feature_data = tracked_data[image_base]
    for key in list(feature_data.keys()):
        if isinstance(feature_data[key], np.ndarray):
            feature_data[key] = torch.tensor(feature_data[key])
    return feature_data


def is_image(image_path):
    extension_name = image_path.split('.')[-1].lower()
    return extension_name in ['jpg', 'png', 'jpeg']


def inference_batch(image_dir, driver_dir, audio_dir, resume_path, force_retrack=False, save_video=True):

    torch.set_float32_matmul_precision('high')

    merge_dir = os.path.join('render_results', "merge_video")
    os.makedirs(merge_dir, exist_ok=True)

    video_names = ['50','133','145','161','162','171']
    for video_name in video_names:
        image_path = os.path.join(image_dir, f"{video_name}.jpg")
        driver_path = os.path.join(driver_dir, video_name)
        
        inference(image_path, driver_path, resume_path, force_retrack, save_video)

        # merge audio and video
        input_video_path = os.path.join('render_results', "res_video", f"{video_name}.mp4")
        input_audio_path = os.path.join(audio_dir, f"{video_name}.wav")
        output_video_path = os.path.join(merge_dir, f"{video_name}.mp4")
        print(f'[INFO] ===== merge audio from {input_video_path} to {output_video_path} =====')
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', input_video_path,'-i', input_audio_path,'-c:v', 'copy', '-c:a', 'aac','-shortest', output_video_path])
        print(f'[INFO] ===== merge audio =====')

if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    # build args
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-i', default='source_imgs', type=str)
    parser.add_argument('--pose_dir', '-d', default='track_res', type=str)
    parser.add_argument('--audio_dir', '-a', default='raw_audio', type=str)
    parser.add_argument('--resume_path', '-r', default='assets/SDGTalk.pt', type=str)
    args = parser.parse_args()
    # launch
    inference_batch(args.image_dir, args.pose_dir, args.audio_dir, args.resume_path)
