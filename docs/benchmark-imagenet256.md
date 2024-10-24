# Benchmark on ImageNet (256x256)



## Basic setup

- The network architecture for all the experiments follows the original [VQGAN](../models/autoencoder/vqgan_net.py).
- Results are tested on ImageNet validation set which contains 50000 images.



## Quantitative results

Using hyperparameters from ["Taming Transformers"](http://arxiv.org/abs/2012.09841) paper (see [config](../configs/vqgan-imagenet.yaml)):

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Downsample ratio</th>
    <th style="text-align: center">Codebook dim.</th>
    <th style="text-align: center">Codebook size</th>
    <th style="text-align: center">l2-norm codes</th>
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">LPIPS↓</th>
    <th style="text-align: center">rFID↓</th>
</tr>
<tr>
    <td style="text-align: center">16</td>
    <td style="text-align: center">256</td>
    <td style="text-align: center">1024</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">19.4617</td>
    <td style="text-align: center">0.4644</td>
    <td style="text-align: center">0.2284</td>
    <td style="text-align: center">18.9091</td>
</tr>
</table>

⚠️ The rFID is much worse than the results reported in the paper (7.94).

Using hyperparameters from ["LlamaGen"](http://arxiv.org/abs/2406.06525) paper (see [config](../configs/vqgan-imagenet-llamagen.yaml)):

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Downsample ratio</th>
    <th style="text-align: center">Codebook dim.</th>
    <th style="text-align: center">Codebook size</th>
    <th style="text-align: center">l2-norm codes</th>
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">LPIPS↓</th>
    <th style="text-align: center">rFID↓</th>
</tr>
<tr>
    <td style="text-align: center">16</td>
    <td style="text-align: center">8</td>
    <td style="text-align: center">16384</td>
    <td style="text-align: center">Yes</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">20.0725</td>
    <td style="text-align: center">0.5201</td>
    <td style="text-align: center">0.1625</td>
    <td style="text-align: center">4.0617</td>
</tr>
</table>

⚠️ The SSIM and rFID are worse than the results reported in the paper (0.675 & 2.19).
