# ImageNet 256x256 Benchmark



## Introduction

This benchmark aims to reproduce the results reported in the papers as closely as possible.



## VQGAN (Taming Transformers)

[[paper]](http://arxiv.org/abs/2012.09841) [[config]](../configs/imagenet256/vqgan-taming.yaml)

|  Downsample ratio   | Codebook dim. | Codebook size | Codebook usage↑ |  PSNR↑  | SSIM↑  | LPIPS↓ | rFID↓  |
|:-------------------:|:-------------:|:-------------:|:---------------:|:-------:|:------:|:------:|:------:|
|         16          |      256      |     1024      |     37.50%      | 19.9142 | 0.5052 | 0.1778 | 5.8165 |

- ️🌱 The PSNR and SSIM are close to the results reported in the paper (19.4 & 0.50).
- ️🌱 The rFID is even better than the results reported in the paper (7.94).
- 🎈 The model suffers from the low codebook usage problem.



## VQGAN (LlamaGen)

[[paper]](http://arxiv.org/abs/2406.06525) [[config]](../configs/imagenet256/vqgan-llamagen.yaml)

|  Downsample ratio  | Codebook dim. | Codebook size | Codebook usage↑ |  PSNR↑  | SSIM↑  | LPIPS↓ | rFID↓  |
|:------------------:|:-------------:|:-------------:|:---------------:|:-------:|:------:|:------:|:------:|
|         16         |       8       |     16384     |      100%       | 20.7201 | 0.5509 | 0.1385 | 2.1073 |

- ️🌱 The PSNR is close to the results reported in the paper (20.79).
- ️🌱 The rFID is even slightly better than the results reported in the paper (2.19).



## TiTok

[[paper]](https://arxiv.org/abs/2406.07550) [[project page]](https://yucornetto.github.io/projects/titok.html) [[config]](../configs/imagenet256/vqgan-titok.yaml)

| \# tokens | Codebook dim. | Codebook size | Codebook usage↑ |  PSNR↑  | SSIM↑  | LPIPS↓ | rFID↓  |
|:---------:|:-------------:|:-------------:|:---------------:|:-------:|:------:|:------:|:------:|
|    64     |      12       |     4096      |      100%       | 17.8995 | 0.4022 | 0.2681 | 4.6691 |

- ⚠️ The model is trained with a single-stage training strategy, which is different from the paper.
- ⚠️ The results are not good. Reconstructed images contain repeated patterns and artifacts. Need further investigation.
