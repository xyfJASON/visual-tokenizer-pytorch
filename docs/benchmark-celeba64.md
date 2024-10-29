# Benchmark on CelebA (64x64)



## Basic setup

- The network architecture for all the experiments is [SimpleCNN](../models/autoencoder/simple_cnn.py).
- Hyperparameters for all the experiments:
  - Batch size: 256
  - Learning rate: 4e-4
  - Optimizer: Adam
  - Training steps: 500k
- Results are tested on CelebA test split which contains 19962 images.



## VQVAE

**Effect of codebook dimension**:

<table style="text-align: center;">
<tr>
    <th>Codebook dim.</th>
    <th>Codebook size</th>
    <th>Codebook usage↑</th>
    <th>PSNR↑</th>
    <th>SSIM↑</th>
    <th>LPIPS↓</th>
    <th>rFID↓</th>
</tr>
<tr>
    <td>4</td>
    <td rowspan="5">512</td>
    <td>100.00%</td>
    <td>32.2119</td>
    <td>0.9456</td>
    <td>0.0239</td>
    <td>16.3249</td>
</tr>
<tr>
    <td>8</td>
    <td>100.00%</td>
    <td>32.2406</td>
    <td>0.9459</td>
    <td>0.0228</td>
    <td>16.6592</td>
</tr>
<tr>
    <td>16</td>
    <td>68.75%</td>
    <td>31.6908</td>
    <td>0.9412</td>
    <td>0.0263</td>
    <td>16.4272</td>
</tr>
<tr>
    <td>32</td>
    <td>66.41%</td>
    <td>31.7674</td>
    <td>0.9417</td>
    <td>0.0261</td>
    <td>16.3970</td>
</tr>
<tr>
    <td>64</td>
    <td>56.45%</td>
    <td>31.5486</td>
    <td>0.9389</td>
    <td>0.0275</td>
    <td>16.8227</td>
</tr>
</table>

- Smaller codebook dimension leads to higher codebook usage.

**Effect of codebook size**:

<table style="text-align: center;">
<tr>
    <th>Codebook dim.</th>
    <th>Codebook size</th>
    <th>Codebook usage↑</th>
    <th>PSNR↑</th>
    <th>SSIM↑</th>
    <th>LPIPS↓</th>
    <th>rFID↓</th>
</tr>
<tr>
    <td rowspan="3">64</td>
    <td>512</td>
    <td>56.45%</td>
    <td>31.5486</td>
    <td>0.9389</td>
    <td>0.0275</td>
    <td>16.8227</td>
</tr>
<tr>
    <td>1024</td>
    <td>30.18%</td>
    <td>31.3835</td>
    <td>0.9395</td>
    <td>0.0272</td>
    <td>16.4965</td>
</tr>
<tr>
    <td>2048</td>
    <td>16.06%</td>
    <td>31.6631</td>
    <td>0.9407</td>
    <td>0.0264</td>
    <td>16.5808</td>
</tr>
</table>

- With low codebook usage, increasing codebook size cannot improve the reconstruction quality.

**Effect of l2-norm codes**:

<table style="text-align: center;">
<tr>
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
    <td rowspan="2">4</td>
    <td rowspan="2">512</td>
    <td>No</td>
    <td>100.00%</td>
    <td>32.2119</td>
    <td>0.9456</td>
    <td>0.0239</td>
    <td>16.3249</td>
</tr>
<tr>
    <td>Yes</td>
    <td>100.00%</td>
    <td>32.2439</td>
    <td>0.9473</td>
    <td></td>
    <td>16.4495</td>
</tr>
<tr>
    <td rowspan="2">64</td>
    <td rowspan="2">512</td>
    <td>No</td>
    <td>56.45%</td>
    <td>31.5486</td>
    <td>0.9389</td>
    <td>0.0275</td>
    <td>16.8227</td>
</tr>
<tr>
    <td>Yes</td>
    <td>98.24%</td>
    <td>31.3334</td>
    <td>0.9442</td>
    <td>0.0209</td>
    <td>12.9127</td>
</tr>
</table>

- The l2-normalized codes can improve codebook usage even when codebook dimension is large.

**Effect of EMA update**:

<table style="text-align: center;">
<tr>
    <th>Codebook dim.</th>
    <th>Codebook size</th>
    <th>Codebook update</th>
    <th>Codebook usage↑</th>
    <th>PSNR↑</th>
    <th>SSIM↑</th>
    <th>LPIPS↓</th>
    <th>rFID↓</th>
</tr>
<tr>
    <td rowspan="2">4</td>
    <td rowspan="2">512</td>
    <td>VQ loss</td>
    <td>100.00%</td>
    <td>32.2119</td>
    <td>0.9456</td>
    <td>0.0239</td>
    <td>16.3249</td>
</tr>
<tr>
    <td>EMA</td>
    <td>100.00%</td>
    <td>32.3069</td>
    <td>0.9468</td>
    <td>0.0224</td>
    <td>16.3338</td>
</tr>
<tr>
    <td rowspan="2">64</td>
    <td rowspan="2">512</td>
    <td>VQ loss</td>
    <td>56.45%</td>
    <td>31.5486</td>
    <td>0.9389</td>
    <td>0.0275</td>
    <td>16.8227</td>
</tr>
<tr>
    <td>EMA</td>
    <td>100.00%</td>
    <td>32.0708</td>
    <td>0.9459</td>
    <td>0.0228</td>
    <td>15.5629</td>
</tr>
</table>

- Use EMA to update the codebook can improve codebook usage even when codebook dimension is large.

**Effect of entropy regularization**:

<table style="text-align: center;">
<tr>
    <th>Codebook dim.</th>
    <th>Codebook size</th>
    <th>Entropy reg. weight</th>
    <th>Codebook usage↑</th>
    <th>PSNR↑</th>
    <th>SSIM↑</th>
    <th>LPIPS↓</th>
    <th>rFID↓</th>
</tr>
<tr>
    <td rowspan="2">64</td>
    <td rowspan="2">512</td>
    <td>0.0</td>
    <td>56.45%</td>
    <td>31.5486</td>
    <td>0.9389</td>
    <td>0.0275</td>
    <td>16.8227</td>
</tr>
<tr>
    <td>0.1</td>
    <td>100.00%</td>
    <td>29.5755</td>
    <td>0.9218</td>
    <td>0.0422</td>
    <td>14.1500</td>
</tr>
</table>

- Entropy regularization can improve codebook usage, but it may hurt the reconstruction quality.



## FSQ-VAE

<table style="text-align: center;">
<tr>
    <th>Levels</th>
    <th>Codebook size</th> 
    <th>Codebook usage↑</th>
    <th>PSNR↑</th>
    <th>SSIM↑</th>
    <th>LPIPS↓</th>
    <th>rFID↓</th>
</tr>
<tr>
    <td style="text-align: center">[8,8,8]</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">30.8543</td>
    <td style="text-align: center">0.9434</td>
    <td style="text-align: center">0.0315</td>
    <td style="text-align: center">15.7079</td>
</tr>
<tr>
    <td style="text-align: center">[8,5,5,5]</td>
    <td style="text-align: center">1000</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">30.9024</td>
    <td style="text-align: center">0.9480</td>
    <td style="text-align: center">0.0266</td>
    <td style="text-align: center">15.8230</td>
</tr>
</table>

- FSQ-VAE does not suffer from the codebook collapse problem.
- FSQ-VAE can achieve comparable performance with VQVAE of the same codebook size.



## LFQ-VAE

<table style="text-align: center;">
<tr>
    <th>Dim.</th>
    <th>Codebook size</th> 
    <th>Codebook usage↑</th>
    <th>PSNR↑</th>
    <th>SSIM↑</th>
    <th>LPIPS↓</th>
    <th>rFID↓</th>
</tr>
<tr>
    <td>9</td>
    <td>512</td>
    <td>100.00%</td>
    <td>26.1391</td>
    <td>0.8630</td>
    <td>0.0700</td>
    <td>18.5518</td>
</table>

⚠️ The result is not as good as expected. Some details may be missing from the implementation.
