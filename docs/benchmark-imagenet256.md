# ImageNet 256x256 Benchmark



## Basic setup

- The network architecture for all the experiments follows the original [VQGAN](../models/autoencoder/vqgan_net.py).
- Results are evaluated on ImageNet validation set which contains 50000 images.



## Quantitative results

Using hyperparameters from ["Taming Transformers"](http://arxiv.org/abs/2012.09841) paper (see [config](../configs/imagenet256/vqgan-taming.yaml)):

|  Downsample ratio   | Codebook dim. | Codebook size | Codebook usage↑ |  PSNR↑  | SSIM↑  | LPIPS↓ | rFID↓  |
|:-------------------:|:-------------:|:-------------:|:---------------:|:-------:|:------:|:------:|:------:|
|         16          |      256      |     1024      |     37.50%      | 19.9142 | 0.5052 | 0.1778 | 5.8165 |

- ️🌱 The PSNR and SSIM are close to the results reported in the paper (19.4 & 0.50).
- ️🌱 The rFID is even better than the results reported in the paper (7.94).
- ⚠️ The model suffers from the low codebook usage problem.



Using hyperparameters from ["LlamaGen"](http://arxiv.org/abs/2406.06525) paper (see [config](../configs/imagenet256/vqgan-llamagen.yaml)):

|  Downsample ratio  | Codebook dim. | Codebook size | Codebook usage↑ |  PSNR↑  | SSIM↑  | LPIPS↓ | rFID↓  |
|:------------------:|:-------------:|:-------------:|:---------------:|:-------:|:------:|:------:|:------:|
|         16         |       8       |     16384     |      100%       | 20.7201 | 0.5509 | 0.1385 | 2.1073 |

- ️🌱 The PSNR is close to the results reported in the paper (20.79).
- ️🌱 The rFID is even slightly better than the results reported in the paper (2.19).
