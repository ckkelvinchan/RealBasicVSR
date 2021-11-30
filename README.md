# RealBasicVSR

\[[Paper](https://arxiv.org/pdf/2111.12704.pdf)\]

This is the official repository of "Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv". This repository contains *codes*, *colab*, *video demos* of our work.

**Authors**: [Kelvin C.K. Chan](https://ckkelvinchan.github.io/), [Shangchen Zhou](https://shangchenzhou.com/), [Xiangyu Xu](https://sites.google.com/view/xiangyuxu), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/), *Nanyang Technological University*

**Acknowedgement**: Our work is built upon [MMEditing](https://github.com/open-mmlab/mmediting). The code will also appear in MMEditing soon. Please follow and star this repository and MMEditing!




## News

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
1. Download the pre-trained weights to `checkpoints/`. ([Dropbox](https://www.dropbox.com/s/eufigxmmkv5woop/RealBasicVSR.pth?dl=0) / [Google Drive](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view))

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

### Training
To be appeared.

## VideoLQ Dataset
You can download the dataset using [Dropbox](https://www.dropbox.com/sh/hc06f1livdhutbo/AAAMPy92EOqVjRN8waT0ie8ja?dl=0) or [Google Drive](https://drive.google.com/drive/folders/1-1iJRNdqdFZWOnoUU4xG1Z1QhwsGwMDy?usp=sharing). 

## Citations
```
@article{chan2021investigating,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {Investigating Tradeoffs in Real-World Video Super-Resolution},
  journal = {arXiv preprint arXiv:2111.12704},
  year = {2021}
}
```
