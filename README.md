# PSLT

Official Pytorch implementation of IEEE Transaction on Pattern Analysis and Machine Intelligence 2023, PSLT: A Light-weight Vision Transformer with
Ladder Self-Attention and Progressive Shift. Please refer to [paper](https://arxiv.org/abs/2304.03481) and [project page](https://isee-ai.cn/wugaojie/PSLT.html) for more details. The optimized models can be found in the [project page](https://isee-ai.cn/wugaojie/PSLT.html).

We currenently release the pytorch version code for:
- ImageNet-1k classification
- Object detection on COCO

Our repository is based on [DeiT](https://github.com/peternara/deit-Transformers), [timm](https://github.com/huggingface/pytorch-image-models), [mmdet](https://github.com/open-mmlab/mmdetection) and [PVT](https://github.com/whai362/PVT).

## Requirements:
1. Image classification:
    - pytorch 1.9.0, torchvision 0.10.0 and timm 0.4.12
2. Object Detection
    - mmcv-full 1.4.0 and mmdet 2.22.0
  
## Acknowledgement
This respository is built based on [DeiT](https://github.com/peternara/deit-Transformers), [timm](https://github.com/huggingface/pytorch-image-models), [mmdet](https://github.com/open-mmlab/mmdetection) and [PVT](https://github.com/whai362/PVT).


## Citation 
If you use this code for a paper, please cite:

```
@article{wu2023pslt,
title = {PSLT: A Light-weight Vision Transformer with Ladder Self-Attention and Progressive Shift},
author = {Gaojie Wu, Wei-Shi Zheng, Yutong Lu and Qi Tian},
journal = {IEEE Transaction on Pattern Analysis and Machine Intelligence},
year = {2023}
}
```
