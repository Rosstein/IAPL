<div align="center">
<!-- <h1>RCTrans</h1> -->
<h3>Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning</h3>
<h4>Yiheng Li, Zicahng Tan, Guoqing Xu, Zhen Lei, Xu Zhou and Yang Yang<h4>
<h5>MAIS&CASIA, UCAS, Sangfor<h5>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2508.01603)

## Introduction

This repository is an official implementation of IAPL, codes and weight will be released after paper accepted.

## News
- [2026/3/4] Codes and pre-trained weights are released.
- [2026/2/21] Our paper is accepted by CVPR 2026.

## Methods
![](IAPL_overview.png)

## Visualization
![](visualization.png)

## Environment Setting
```
pip install -r requirements.txt
```

## Data Preparation
Download [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) and [GenImage](https://github.com/GenImage-Dataset/GenImage) Datasets.

Organize the directory structure as follows:
```
Datasets
└── UniversalFakeDetect
    └── train
          ├── car
          ├── horse
          │      .
          │      .
    └── test					
          ├── progan	
          │── cyclegan   	
          │── biggan
          │      .
          │      .

└── GenImage
    └── train
          ├── SDv14
              ├── 0_real
              ├── 1_fake

    └── test     
          ├── ADM
              ├── 0_real
              ├── 1_fake
          │── BigGAN   	
          │── glide
          │      .
          │      .
```

## Experiments on 4-Class ProGAN
Training:
```
sh run_universalfake.sh
```
Testing on universalfakedetect:
```
sh tta_universalfake.sh
```
Testing on Chameleon:
```
sh tta_chameleon.sh
```


## Pre-trained Models

We release the pre-trained models on [ModelScope](https://modelscope.cn/models/yihengli/IAPL_pretrain) 

## Acknowledgement

We sincerely thank the following repos: [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect), [FatFormer](https://github.com/Michel-liu/FatFormer), [AIDE](https://github.com/shilinyan99/AIDE) and [TPT](https://github.com/azshue/TPT).


## 🚀 Modifications & Fixes (for Single-GPU / V100)

- **Distributed Training Bypass**: Added single-GPU fallback path to prevent NCCL/barrier crashes, removing uninitialized distributed APIs, `no_sync()`, and `.module` wrappers in `test_time.py`.
- **V100 Precision Compatibility**: Added automatic fallback from `bf16` to `fp16` in `test_time.py` (since V100 Volta architecture does not support bfloat16 natively).
- **Dynamic Weight Paths**: Fixed CLIP weight loading to strictly follow `args.clip_path` dynamically in `clip_models.py`.
- **Dataset Pathing**: Standardized GenImage dataset paths to `Datasets/GenImage` in `tta_genimage.sh`.

## 模型框架

# Training Process

基座模型是 CLIP 视觉编码器（ViT-L/14），在 models/clip_models.py 的 CLIPModel 构造里加载。核心流程：

- 在 main.py 中，训练时会调用 build_model(args)（models/\_\_init\_\_.py），返回 CLIPModel(args)。
- CLIPModel 内部调用 load_clip_to_cpu(...) 加载 CLIP 视觉编码器（clip_model.visual），作为 self.image_encoder。
- 输入图像经过 image_encoder 提取视觉特征，再送入后续适配层/判别头。
- 在 models/clip_models.py 中看到：
  - self.image_encoder = clip_model.visual（基座视觉模型）
  - self.fc_binary = nn.Linear(768, 1)（最终二分类头）

在 CLIPModel.\_\_init\_\_ 里明确冻结/解冻参数：

训练时更新的参数：运行时看到的可训练参数名单，会由 print(trained_clip) 输出

- CLIP 中 adapter / gamma 相关参数
- Prompt learner（可学习提示）
- 二分类头 fc_binary

prompt 结构：shared_ctx + bias(image)

- shared_ctx = ctx[n_ctx, vision_width]是每个图片共享的prompt
- bias(image)= conditional_ctx(image) 是一个“条件调制器”，它会根据当前图像生成 bias
- deep_compound_prompts_vision: 插入到 Transformer 的不同层里。


# Test-time

image_encoder 全冻结, prompt_learner 里只有名字包含 ctx 的参数保留可训练. 
fc_binary 冻结, conditional_ctx 冻结.  

- TTA 只更新ctx[n_ctx, vision_width], 随着image features更新。