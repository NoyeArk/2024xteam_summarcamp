# [MetaFormer Baselines for Vision](https://arxiv.org/abs/2210.13452) (TPAMI 2024)

<p align="left">
<a href="https://arxiv.org/abs/2210.13452" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2210.13452-b31b1b.svg?style=flat" /></a>
<a href="https://colab.research.google.com/drive/1raon_oZRnUBXb9ZYcMY3Au_r-3l4eP1I?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

This is a PyTorch implementation of several MetaFormer baslines including **IdentityFormer**, **RandFormer**, **ConvFormer** and **CAFormer** proposed by our paper "[MetaFormer Baselines for Vision](https://arxiv.org/abs/2210.13452)".

![Figure1](https://user-images.githubusercontent.com/49296856/197580831-fc937e24-9941-4794-b99d-822748fa0f11.png)
Figure 1: **Performance of MetaFormer baselines and other state-of-the-art models on ImageNet-1K at 224x224 resolution.** The architectures of our proposed models are shown in Figure 2. (a) IdentityFormer/RandFormer achieve over 80%/81% accuracy, indicating MetaFormer has solid lower bound of performance and works well on arbitrary token mixers. The accuracy of well-trained ResNet-50 is from "ResNet strikes back". (b) Without novel token mixers, pure CNN-based ConvFormer outperforms ConvNeXt, while CAFormer sets a new record of 85.5% accuracy on ImageNet-1K at 224x224 resolution under normal supervised training without external data or distillation.

![Overall](https://user-images.githubusercontent.com/49296856/212324452-ee5ccbcf-5577-42cb-9fa4-b9e6bdbb6d4a.png)
Figure 2: **(a-d)  Overall frameworks of IdentityFormer, RandFormer, ConvFormer and CAFormer.** Similar to ResNet, the models adopt hierarchical architecture of 4 stages, and stage $i$ has  $L_i$ blocks with feature dimension $D_i$. Each downsampling module is implemented by a layer of convolution. The first downsampling has kernel size of 7 and stride of 4, while the last three ones have kernel size of 3 and stride of 2. **(e-h) Architectures of IdentityFormer, RandFormer, ConvFormer and Transformer blocks**, which have token mixer of identity mapping, global random mixing, separable depthwise convolutions, or vanilla self-attention, respectively. 

![Comparision](https://user-images.githubusercontent.com/49296856/197601575-6a19ed8c-7bc2-433b-895b-e5363358ea77.png)


### News
Models of MetaFormer baselines are now integrated in [timm](https://github.com/huggingface/pytorch-image-models) by [Fredo Guan](https://github.com/fffffgggg54) and [Ross Wightman](https://github.com/rwightman). Many thanks!


## Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; [timm](https://github.com/rwightman/pytorch-image-models) (`pip install timm==0.6.11`)

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Validation

To evaluate our CAFormer-S18 models, run:

```bash
MODEL=caformer_s18
python3 validate.py /path/to/imagenet  --models $MODEL -b 128 \
  --checkpoint /path/to/checkpoint 
```

## Train
默认情况下，我们使用4096的批量大小，并展示如何使用8个GPU训练模型。对于多节点训练，请根据您的情况调整`--grad-accum-steps`。


```bash
DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/metaformer # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--models convformer_s18 --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.2 --head-dropout 0.0
```
Training (fine-tuning) scripts of other models are shown in [scripts](/scripts/).

## Acknowledgment
Weihao Yu would like to thank TRC program and GCP research credits for the support of partial computational resources. Our implementation is based on the wonderful [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) codebase.


## Bibtex
```
@article{yu2024metaformer,
  author={Yu, Weihao and Si, Chenyang and Zhou, Pan and Luo, Mi and Zhou, Yichen and Feng, Jiashi and Yan, Shuicheng and Wang, Xinchao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MetaFormer Baselines for Vision}, 
  year={2024},
  volume={46},
  number={2},
  pages={896-912},
  doi={10.1109/TPAMI.2023.3329173}}
}
```
