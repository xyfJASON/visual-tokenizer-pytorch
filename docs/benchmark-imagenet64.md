# ImageNet 64x64 Benchmark



## Introduction

This benchmark does not aim to achieve the best performance, but to gain insights into the behavior of various vector quantization methods.
Therefore, we use the same basic setup for all the experiments:

- Dataset: ImageNet, resized to 64x64
- Network architecture: [SimpleCNN](../models/autoencoder/simple_cnn.py)
- Training hyperparameters:
  - Batch size: 256
  - Learning rate: 4e-4
  - Optimizer: Adam
  - Training steps: 500k



## VQVAE

| Codebook dim. | Codebook size | l2 norm | entropy reg | Codebook usage↑ |  PSNR↑  | SSIM↑  | LPIPS↓ |  rFID↓  |
|:-------------:|:-------------:|:-------:|:-----------:|:---------------:|:-------:|:------:|:------:|:-------:|
|      64       |      512      |  False  |     No      |     96.09%      | 26.8890 | 0.8646 | 0.1017 | 39.2825 |
|      64       |     2048      |  False  |     No      |     28.96%      | 27.0770 | 0.8704 | 0.0916 | 36.7708 |
|      64       |     16384     |  False  |     No      |      3.00%      | 26.8605 | 0.8670 | 0.0939 | 38.1272 |
|       8       |     2048      |  False  |     No      |     58.89%      | 27.6575 | 0.8833 | 0.0792 | 35.0210 |
|       8       |     16384     |  False  |     No      |     24.55%      | 28.5044 | 0.9017 | 0.0575 | 27.1896 |
|      64       |     2048      |  True   |     No      |     82.71%      | 27.3169 | 0.8864 | 0.0547 | 22.6812 |
|      64       |     16384     |  True   |     No      |     26.81%      | 27.8334 | 0.8987 | 0.0439 | 18.5483 |
|      64       |     2048      |  False  |     Yes     |     100.00%     | 26.3328 | 0.8617 | 0.0952 | 31.8191 |
|      64       |     16384     |  False  |     Yes     |     85.31%      | 26.7838 | 0.8695 | 0.0722 | 27.1384 |

Conclusions:

- As the size of the codebook increases, the codebook usage decreases, which is known as the codebook collapse problem.
- Decreasing the codebook dim. slightly alleviates the codebook collapse problem.
- Using l2 norm on the codes alleviates the codebook collapse problem, and significantly improves the reconstruction.
- Entropy regularization addresses the codebook collapse problem by encouraging the codebook to be used more evenly.



## FSQ-VAE

|   Levels    | Codebook size | Codebook usage↑ |  PSNR↑  | SSIM↑  | LPIPS↓ |  rFID↓  |
|:-----------:|:-------------:|:---------------:|:-------:|:------:|:------:|:-------:|
|   [8,8,8]   |      512      |     100.00%     | 26.0826 | 0.8456 | 0.1179 | 45.6580 |
|  [8,5,5,5]  |     1000      |     100.00%     | 26.1262 | 0.8526 | 0.1059 | 43.4865 |
| [8,8,8,6,5] |     15360     |     99.99%      | 27.7061 | 0.8907 | 0.0669 | 27.2670 |

Conclusions:

- FSQ addresses the codebook collapse problem without introducing any complicated codebook losses.
- FSQ lags behind VQ when the codebook size is small, but outperforms VQ when the codebook size grows large. This observation is consistent with the paper.



## SimVQ-VAE

| Codebook dim. | Codebook size | Codebook usage↑ |  PSNR↑  | SSIM↑  | LPIPS↓ |  rFID↓  |
|:-------------:|:-------------:|:---------------:|:-------:|:------:|:------:|:-------:|
|      64       |      512      |     100.00%     | 26.6992 | 0.8638 | 0.0960 | 36.4991 |
|      64       |     2048      |     100.00%     | 27.7483 | 0.8897 | 0.0654 | 27.4036 |

Conclusions:

- SimVQ addresses the codebook collapse problem by reparameterizing the codebook through a linear transformation layer.
