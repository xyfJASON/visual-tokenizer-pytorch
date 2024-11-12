# Benchmark on ImageNet (256x256)



## Basic setup

- The network architecture for all the experiments follows the original [VQGAN](../models/autoencoder/vqgan_net.py).
- Results are tested on ImageNet validation set which contains 50000 images.



## Quantitative results

Using hyperparameters from ["Taming Transformers"](http://arxiv.org/abs/2012.09841) paper (see [config](../configs/vqgan-imagenet.yaml)):

<table style="text-align: center;">
<tr>
    <th>Downsample ratio</th>
    <th>Codebook dim.</th>
    <th>Codebook size</th>
    <th>l2-norm codes</th>
    <th>Codebook usage↑</th>
    <th>PSNR↑</th>
    <th>SSIM↑</th>
    <th>LPIPS↓</th>
    <th>rFID↓</th>
</tr>
<tr>
    <td>16</td>
    <td>256</td>
    <td>1024</td>
    <td>No</td>
    <td>100.00%</td>
    <td>19.4617</td>
    <td>0.4614</td>
    <td>0.2284</td>
    <td>18.9091</td>
</tr>
</table>

- ️🌱 The PSNR and SSIM are close to the results reported in the paper (19.4 & 0.50).
- ⚠️ The rFID is much worse than the results reported in the paper (7.94).



Using hyperparameters from ["LlamaGen"](http://arxiv.org/abs/2406.06525) paper (see [config](../configs/vqgan-imagenet-llamagen.yaml)):

<table style="text-align: center;">
<tr>
    <th>Downsample ratio</th>
    <th>Codebook dim.</th>
    <th>Codebook size</th>
    <th>l2-norm codes</th>
    <th>Codebook usage↑</th>
    <th>PSNR↑</th>
    <th>SSIM↑</th>
    <th>LPIPS↓</th>
    <th>rFID↓</th>
</tr>
<tr>
    <td>16</td>
    <td>8</td>
    <td>16384</td>
    <td>Yes</td>
    <td>100.00%</td>
    <td>20.0723</td>
    <td>0.5231</td>
    <td>0.1625</td>
    <td>4.0617</td>
</tr>
</table>

- ️🌱 The PSNR is close to the results reported in the paper (20.79).
- 🌱 The SSIM is close to the expected value (0.675 reported in the paper, fixed to 0.56).
- ⚠️ The rFID is much worse than the results reported in the paper (2.19).
