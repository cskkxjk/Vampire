[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.11837)
# (AAAI2024) Regulating Intermediate 3D Features for Vision-Centric Autonomous Driving
## Introduction
This is the official pytorch implementation of Regulating Intermediate 3D Features for Vision-Centric Autonomous Driving, In AAAI'24, Junkai Xu, Liang Peng, Haoran Cheng, Linxuan Xia, Qi Zhou, Dan Deng, Wei Qian, Wenxiao Wang and Deng Cai.

![Framework](./docs/framework.png)

[[Paper]](https://arxiv.org/abs/2312.11837)  
## News
- [2023-12-20] [Paper](https://arxiv.org/abs/2312.11837) is released on arxiv!
- [2023-12-19] Code is released.
- [2023-12-14] Demo release.
- [2023-12-9] Vampire is accepted at AAAI 2024!! Code is comming soon.

## Demo
![scene-0012-0018](./docs/scene-0012-0018.gif)

## Quick Start
### Installation
**Step 0.** Install [pytorch](https://pytorch.org/) (v1.9.0).
```
conda create --name vampire python=3.7 -y
conda activate vampire
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
**Step 1.** Install [mmdet](https://github.com/open-mmlab/mmdetection) (2.26.0), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) (0.29.1)[MMDetection3D](https://github.com/open-mmlab/mmdetection3d) (v1.0.0rc6).
```
pip install openmim==0.3.3
mim install mmcv-full==1.6.2
mim install mmdet==2.26.0
mim install mmsegmentation==0.29.1
git clone https://github.com/open-mmlab/mmdetection3d.git --branch v1.0.0rc6 --single-branch
cd mmdetection3d
pip install -e .
cd ..
```
**Step 2.** Install requirements.
```
pip install -r requirements.txt
python setup.py develop
```


### Data preparation
**Step 0.** Download nuScenes official dataset and [occupancy trainval subset](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction/tree/main) (including `gts.tar.gz` and `annotations.json`)

**Step 1.** Unzip all data in your disk and Symlink the dataset root to `./data/`.
```
ln -s [nuscenes root] ./data/
```
The directory will be as follows.
```
Vampire/
├── data/
│   ├── nuScenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
|   |   ├── lidarseg/
|   |   ├── panoptic/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── gts/
|   |   ├── annotations.json
```
**Step 2.** Prepare infos.
```
python scripts/gen_info.py
```

### Tutorials
**Train on 8 NVIDIA GPUs with a total batch size of 8.**
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
**Validation & Test (output submit file for nuscenes toolkit evaluation)**
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -v -b 8 --gpus 8
python [EXP_PATH] --ckpt_path [CKPT_PATH] -t -b 8 --gpus 8
```

### Benchmark
|Exp | Occ. | Seg. | Det.| weights |
| ------ | :---: | :---: | :---: |:---:|
|[Vampire](src/exps/nuscenes/ablation/vampire2_r50_256x704_24e_lss_inpaintor_depth_semantic.py)| 25.8 |62.6|0.318|[Google-drive](https://drive.google.com/file/d/1OKwvWTLeWXTg0syZx-byJ62r6MI23xKR/view?usp=sharing)|

## Acknowledgements
This project benefits from the following codebases. Thanks for their great works! 
* [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
* [TPVFormer](https://github.com/wzzheng/TPVFormer)
* [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
* [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)
