# Benchmark on CelebA (64x64)



## Basic setup

- The network architecture for all the experiments is [SimpleCNN](../models/autoencoder/simple_cnn.py).
- Results are tested on CelebA test set which contains 19962 images.



## VQVAE

**Effect of codebook dimension**:

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Codebook dim.</th>
    <th style="text-align: center">Codebook size</th>
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">LPIPS↓</th>
    <th style="text-align: center">rFID↓</th>
</tr>
<tr>
    <td style="text-align: center">4</td>
    <td style="text-align: center" rowspan="5">512</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.2119</td>
    <td style="text-align: center">0.9456</td>
    <td style="text-align: center">0.0239</td>
    <td style="text-align: center">16.3249</td>
</tr>
<tr>
    <td style="text-align: center">8</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.2406</td>
    <td style="text-align: center">0.9459</td>
    <td style="text-align: center">0.0228</td>
    <td style="text-align: center">16.6592</td>
</tr>
<tr>
    <td style="text-align: center">16</td>
    <td style="text-align: center">68.75%</td>
    <td style="text-align: center">31.6908</td>
    <td style="text-align: center">0.9412</td>
    <td style="text-align: center">0.0263</td>
    <td style="text-align: center">16.4272</td>
</tr>
<tr>
    <td style="text-align: center">32</td>
    <td style="text-align: center">66.41%</td>
    <td style="text-align: center">31.7674</td>
    <td style="text-align: center">0.9417</td>
    <td style="text-align: center">0.0261</td>
    <td style="text-align: center">16.3970</td>
</tr>
<tr>
    <td style="text-align: center">64</td>
    <td style="text-align: center">56.45%</td>
    <td style="text-align: center">31.5486</td>
    <td style="text-align: center">0.9389</td>
    <td style="text-align: center">0.0275</td>
    <td style="text-align: center">16.8227</td>
</tr>
</table>

- Smaller codebook dimension leads to higher codebook usage.

**Effect of codebook size**:

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Codebook dim.</th>
    <th style="text-align: center">Codebook size</th>
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">LPIPS↓</th>
    <th style="text-align: center">rFID↓</th>
</tr>
<tr>
    <td style="text-align: center" rowspan="3">64</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">56.45%</td>
    <td style="text-align: center">31.5486</td>
    <td style="text-align: center">0.9389</td>
    <td style="text-align: center">0.0275</td>
    <td style="text-align: center">16.8227</td>
</tr>
<tr>
    <td style="text-align: center">1024</td>
    <td style="text-align: center">30.18%</td>
    <td style="text-align: center">31.3835</td>
    <td style="text-align: center">0.9395</td>
    <td style="text-align: center">0.0272</td>
    <td style="text-align: center">16.4965</td>
</tr>
<tr>
    <td style="text-align: center">2048</td>
    <td style="text-align: center">16.06%</td>
    <td style="text-align: center">31.6631</td>
    <td style="text-align: center">0.9407</td>
    <td style="text-align: center">0.0264</td>
    <td style="text-align: center">16.5808</td>
</tr>
</table>

- With low codebook usage, increasing codebook size cannot improve the reconstruction quality.

**Effect of l2-norm codes**:

<table style="text-align: center;">
<tr>
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
    <td style="text-align: center" rowspan="2">4</td>
    <td style="text-align: center" rowspan="2">512</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.2119</td>
    <td style="text-align: center">0.9456</td>
    <td style="text-align: center">0.0239</td>
    <td style="text-align: center">16.3249</td>
</tr>
<tr>
    <td style="text-align: center">Yes</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.2439</td>
    <td style="text-align: center">0.9473</td>
    <td style="text-align: center"></td>
    <td style="text-align: center">16.4495</td>
</tr>
<tr>
    <td style="text-align: center" rowspan="2">64</td>
    <td style="text-align: center" rowspan="2">512</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">56.45%</td>
    <td style="text-align: center">31.5486</td>
    <td style="text-align: center">0.9389</td>
    <td style="text-align: center">0.0275</td>
    <td style="text-align: center">16.8227</td>
</tr>
<tr>
    <td style="text-align: center">Yes</td>
    <td style="text-align: center">98.24%</td>
    <td style="text-align: center">31.3334</td>
    <td style="text-align: center">0.9442</td>
    <td style="text-align: center">0.0209</td>
    <td style="text-align: center">12.9127</td>
</tr>
</table>

- The l2-normalized codes can improve codebook usage even when codebook dimension is large.

**Effect of EMA update**:

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Codebook dim.</th>
    <th style="text-align: center">Codebook size</th>
    <th style="text-align: center">Codebook update</th>
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">LPIPS↓</th>
    <th style="text-align: center">rFID↓</th>
</tr>
<tr>
    <td style="text-align: center" rowspan="2">4</td>
    <td style="text-align: center" rowspan="2">512</td>
    <td style="text-align: center">VQ loss</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.2119</td>
    <td style="text-align: center">0.9456</td>
    <td style="text-align: center">0.0239</td>
    <td style="text-align: center">16.3249</td>
</tr>
<tr>
    <td style="text-align: center">EMA</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.3069</td>
    <td style="text-align: center">0.9468</td>
    <td style="text-align: center">0.0224</td>
    <td style="text-align: center">16.3338</td>
</tr>
<tr>
    <td style="text-align: center" rowspan="2">64</td>
    <td style="text-align: center" rowspan="2">512</td>
    <td style="text-align: center">VQ loss</td>
    <td style="text-align: center">56.45%</td>
    <td style="text-align: center">31.5486</td>
    <td style="text-align: center">0.9389</td>
    <td style="text-align: center">0.0275</td>
    <td style="text-align: center">16.8227</td>
</tr>
<tr>
    <td style="text-align: center">EMA</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.0708</td>
    <td style="text-align: center">0.9459</td>
    <td style="text-align: center">0.0228</td>
    <td style="text-align: center">15.5629</td>
</tr>
</table>

- Use EMA to update the codebook can improve codebook usage.

**Effect of entropy regularization**:

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Codebook dim.</th>
    <th style="text-align: center">Codebook size</th>
    <th style="text-align: center">Entropy reg. weight</th>
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">LPIPS↓</th>
    <th style="text-align: center">rFID↓</th>
</tr>
<tr>
    <td style="text-align: center" rowspan="2">64</td>
    <td style="text-align: center" rowspan="2">512</td>
    <td style="text-align: center">0.0</td>
    <td style="text-align: center">56.45%</td>
    <td style="text-align: center">31.5486</td>
    <td style="text-align: center">0.9389</td>
    <td style="text-align: center">0.0275</td>
    <td style="text-align: center">16.8227</td>
</tr>
<tr>
    <td style="text-align: center">0.1</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">29.5755</td>
    <td style="text-align: center">0.9218</td>
    <td style="text-align: center">0.0422</td>
    <td style="text-align: center">14.1500</td>
</tr>
</table>

- Entropy regularization can improve codebook usage.



## FSQ-VAE

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Levels</th>
    <th style="text-align: center">Codebook size</th> 
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">LPIPS↓</th>
    <th style="text-align: center">rFID↓</th>
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
