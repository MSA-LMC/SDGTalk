
# SDTalk: Structured Facial Priors and Dual-Branch Motion Fields for Generalizable Gaussian Talking Head Synthesis

## Demo

It is shown in demo/demo.mp4

## Installation


Tested on Ubuntu 20.04, CUDA 12.1, PyTorch 2.4.1

```bash
cd SDTalk
```
### Environment

```bash
conda env create --file environment.yml
conda activate sdgtalk
```

### Install the 3DGS renderer

```bash
git clone --recurse-submodules git@github.com:xg-chu/diff-gaussian-rasterization.git
pip install ./diff-gaussian-rasterization
rm -rf ./diff-gaussian-rasterization
```
### Preparation

```bash
bash prepare.sh
```
## Inference

```bash
python inference.py --image_dir demo/raw/video/source_imgs --pose_dir demo/track_res --audio_dir demo/raw/audio/raw_audio --resume_path assets/SDGTalk.pt
```

The result is in the `render_results`.

## Acknowledgement

Partial codes are from [GAGAvatar](https://github.com/xg-chu/GAGAvatar), [TalkingGaussian](https://github.com/Fictionarry/TalkingGaussian/tree/main). Face Parsing is from [face-parsing](https://huggingface.co/jonathandinu/face-parsing). Thanks for these great projects!