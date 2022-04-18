# RealBasicVSR (CVPR 2022)

\[[Paper](https://arxiv.org/pdf/2111.12704.pdf)\]

This is the official repository of "Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv". This repository contains *codes*, *colab*, *video demos* of our work.

**Authors**: [Kelvin C.K. Chan](https://ckkelvinchan.github.io/), [Shangchen Zhou](https://shangchenzhou.com/), [Xiangyu Xu](https://sites.google.com/view/xiangyuxu), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/), *Nanyang Technological University*

**Acknowedgement**: Our work is built upon [MMEditing](https://github.com/open-mmlab/mmediting). The code will also appear in MMEditing soon. Please follow and star this repository and MMEditing!

**Feel free to ask questions. I am currently working on some other stuff but will try my best to reply. If you are also interested in [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus), which is also accepted to CVPR 2022, please don't hesitate to star!** 




## News
- 11 Mar 2022: Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/RealBasicVSR)
- 3 Mar 2022: Our paper has been accepted to CVPR 2022
- 4 Jan 2022: Training code released
- 2 Dec 2021: Colab demo released <a href="https://colab.research.google.com/drive/1JzWRUR34hpKvtCHm84IGx6nv35LCv20J?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
- 29 Nov 2021: Test code released
- 25 Nov 2021: Initialize with video demos

## Table of Content
1. [Video Demos](#video-demos)
2. [Code](#code)
3. [VideoLQ Dataset](#videolq-dataset)
4. [Citations](#citations)

## Video Demos
The videos have been compressed. Therefore, the results are inferior to that of the actual outputs.

https://user-images.githubusercontent.com/7676947/143370499-9fe4069b-46cc-4f12-b6ff-5595e8e5e0b8.mp4

https://user-images.githubusercontent.com/7676947/143370350-91f751f3-0f33-4ee4-9b1a-b9279bf41c18.mp4

https://user-images.githubusercontent.com/7676947/143370556-9e7019d4-e718-46af-859f-54d5576cd370.mp4

https://user-images.githubusercontent.com/7676947/143370859-e0293b97-f962-476f-acf8-14fad27cea77.mp4

## Code
### Installation
1. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/locally/), e.g.,
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

2. Install mim and mmcv-full
```
pip install openmim
mim install mmcv-full
```

3. Install mmedit
```
pip install mmedit
```

### Inference
1. Download the pre-trained weights to `checkpoints/`. ([Dropbox](https://www.dropbox.com/s/eufigxmmkv5woop/RealBasicVSR.pth?dl=0) / [Google Drive](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view) / [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/chan0899_e_ntu_edu_sg/EfMvf8H6Y45JiY0xsK4Wy-EB0kiGmuUbqKf0qsdoFU3Y-A?e=9p8ITR))

2. Run the following command:
```
python inference_realbasicvsr.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${INPUT_DIR} ${OUTPUT_DIR} --max-seq-len=${MAX_SEQ_LEN} --is_save_as_png=${IS_SAVE_AS_PNG}  --fps=${FPS}
```

This script supports both images and videos as inputs and outputs. You can simply change ${INPUT_DIR} and ${OUTPUT_DIR} to the paths corresponding to the video files, if you want to use videos as inputs and outputs. But note that saving to videos may induce additional compression, which reduces output quality.

For example:
1. Images as inputs and outputs
```
python inference_realbasicvsr.py configs/realbasicvsr_x4.py checkpoints/RealBasicVSR_x4.pth data/demo_000 results/demo_000
```

2. Video as input and output
```
python inference_realbasicvsr.py configs/realbasicvsr_x4.py checkpoints/RealBasicVSR_x4.pth data/demo_001.mp4 results/demo_001.mp4 --fps=12.5
```

### Crop REDS dataset into sub-images
We crop the REDS dataset into sub-images for faster I/O. Please follow the instructions below:
1. Put the original REDS dataset in `./data`
2. Run the following command:
```
python crop_sub_images.py --data-root ./data/REDS  --scales 4
```
### Training
The training is divided into two stages:
1. Train a model without perceptual loss and adversarial loss using [realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds.py](realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds.py).
```
mim train mmedit configs/realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds.py --gpus 8 --launcher pytorch
```

2. Finetune the model with perceptual loss and adversarial loss using [realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py](realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py). (You may want to replace `load_from` in the configuration file with your checkpoints pre-trained at the first stage
```
mim train mmedit configs/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py --gpus 8 --launcher pytorch
```

**Note**: We use UDM10 with bicubic downsampling for validation. You can download it from [here](https://www.terabox.com/web/share/link?surl=LMuQCVntRegfZSxn7s3hXw&path=%2Fproject%2Fpfnl).

### Generating Video Demo
Assuming you have created two sets of images (e.g. input vs output), you can use `generate_video_demo.py` to generate a video demo. Note that the two sets of images must be of the same resolution. An example has been provided in the code.

## VideoLQ Dataset
You can download the dataset using [Dropbox](https://www.dropbox.com/sh/hc06f1livdhutbo/AAAMPy92EOqVjRN8waT0ie8ja?dl=0) / [Google Drive](https://drive.google.com/drive/folders/1-1iJRNdqdFZWOnoUU4xG1Z1QhwsGwMDy?usp=sharing) / [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/chan0899_e_ntu_edu_sg/ErSugvUBxoBMlvSAHhqT5BEB9-4ZaqxzJIcc9uvVa8JGHg?e=WpHJTc).

## Citations
```
@inproceedings{chan2022investigating,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {Investigating Tradeoffs in Real-World Video Super-Resolution},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2022}
}
```
