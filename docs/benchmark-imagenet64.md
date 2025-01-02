# ImageNet 64x64 Benchmark



## Introduction

This benchmark does not aim to achieve the best performance, but to gain insights into the behavior of different quantization methods.
Therefore, we use the same basic setup for all the experiments:

- Dataset: ImageNet, resized to 64x64
- Network architecture: [SimpleCNN](../models/autoencoder/simple_cnn.py)
- Training hyperparameters:
  - Batch size: 256
  - Learning rate: 4e-4
  - Optimizer: Adam
  - Training steps: 500k

We focus on the following aspects:
  
- Codebook dim
- Codebook size
- Use EMA k-means to update the codebook
- Use l2 norm on the codes
- Use entropy regularization



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

Conclusions:

- As the size of the codebook increases, the codebook usage decreases, which is known as the codebook collapse problem.
- Decreasing the codebook dim. slightly alleviates the codebook collapse problem.
- Using l2 norm on the codes alleviates the codebook collapse problem, and significantly improves the reconstruction.



## FSQ-VAE

|   Levels    | Codebook size | Codebook usage↑ | PSNR↑ | SSIM↑ | LPIPS↓ | rFID↓ |
|:-----------:|:-------------:|:---------------:|:-----:|:-----:|:------:|:-----:|
|   [8,8,8]   |      512      |        %        |       |       |        |       |
|  [8,5,5,5]  |     1000      |        %        |       |       |        |       |
| [8,8,8,6,5] |     15360     |        %        |       |       |        |       |



## SimVQ-VAE

| Codebook dim. | Codebook size | Codebook usage↑ |  PSNR↑  | SSIM↑  | LPIPS↓ |  rFID↓  |
|:-------------:|:-------------:|:---------------:|:-------:|:------:|:------:|:-------:|
|      64       |      512      |     100.00%     | 26.6992 | 0.8638 | 0.0960 | 36.4991 |
|      64       |     2048      |     100.00%     | 27.7483 | 0.8897 | 0.0654 | 27.4036 |

- SimVQ addresses the codebook collapse problem by reparameterizing the codebook through a linear transformation layer.
