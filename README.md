# visual-tokenizer-pytorch

Implement visual tokenizers with PyTorch.

<br/>



## Results

### VQVAE on CelebA(64x64)

**Implementation details**:

- The network architecture for all the experiments is [SimpleCNN](./models/autoencoder/simple_cnn.py).
- Results are tested on CelebA test set which contains 19962 images.
- Codebook usage is calculated as the percentage of used codes in a queue of size 65536 over the whole codebook size.

**Quantitative results**:

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Quantizer type</th>
    <th style="text-align: center">Codebook dim.</th>
    <th style="text-align: center">Codebook size</th>
    <th style="text-align: center">l2-norm codes</th> 
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">rFID↓</th>
</tr>
<tr>
    <td style="text-align: center" rowspan="9">VQ loss</td>
    <td style="text-align: center">4</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.2119</td>
    <td style="text-align: center">0.9456</td>
    <td style="text-align: center">16.3249</td>
</tr>
<tr>
    <td style="text-align: center">8</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.2406</td>
    <td style="text-align: center">0.9459</td>
    <td style="text-align: center">16.6592</td>
</tr>
<tr>
    <td style="text-align: center">16</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">68.55%</td>
    <td style="text-align: center">31.6908</td>
    <td style="text-align: center">0.9412</td>
    <td style="text-align: center">16.4272</td>
</tr>
<tr>
    <td style="text-align: center">32</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">66.21%</td>
    <td style="text-align: center">31.7674</td>
    <td style="text-align: center">0.9417</td>
    <td style="text-align: center">16.3970</td>
</tr>
<tr>
    <td style="text-align: center">64</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">56.05%</td>
    <td style="text-align: center">31.5486</td>
    <td style="text-align: center">0.9389</td>
    <td style="text-align: center">16.8227</td>
</tr>
<tr>
    <td style="text-align: center">64</td>
    <td style="text-align: center">1024</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">30.08%</td>
    <td style="text-align: center">31.3835</td>
    <td style="text-align: center">0.9395</td>
    <td style="text-align: center">16.4965</td>
</tr>
<tr>
    <td style="text-align: center">64</td>
    <td style="text-align: center">2048</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">16.02%</td>
    <td style="text-align: center">31.6631</td>
    <td style="text-align: center">0.9407</td>
    <td style="text-align: center">16.5808</td>
</tr>
<tr>
    <td style="text-align: center">4</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">Yes</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.2439</td>
    <td style="text-align: center">0.9473</td>
    <td style="text-align: center">16.4495</td>
</tr>
<tr>
    <td style="text-align: center">64</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">Yes</td>
    <td style="text-align: center">98.24%</td>
    <td style="text-align: center">31.3334</td>
    <td style="text-align: center">0.9442</td>
    <td style="text-align: center">12.9127</td>
</tr>
<tr>
    <td style="text-align: center" rowspan="2">EMA</td>
    <td style="text-align: center">4</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.3069</td>
    <td style="text-align: center">0.9468</td>
    <td style="text-align: center">16.3338</td>
</tr>
<tr>
    <td style="text-align: center">64</td>
    <td style="text-align: center">512</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">100.00%</td>
    <td style="text-align: center">32.0708</td>
    <td style="text-align: center">0.9459</td>
    <td style="text-align: center">15.5629</td>
</tr>
</table>

**Conclusions**:

- Smaller codebook dimension leads to higher codebook usage and better reconstruction quality.
- The l2-normalized codes can improve the codebook usage and reconstruction quality.
- EMA codebook achieves better codebook usage and reconstruction quality than the VQ loss codebook.

<br/>



### VQGAN on ImageNet(256x256)

**Implementation details**:

- The network architecture for all the experiments follows the original [VQGAN](./models/autoencoder/vqgan_net.py).
- Results are tested on ImageNet validation set which contains 50000 images.
- Codebook usage is calculated as the percentage of used codes in a queue of size 65536 over the whole codebook size.

**Quantitative results**:

Using hyperparameters from ["Taming Transformers"](http://arxiv.org/abs/2012.09841) paper (see [config](./configs/vqgan-imagenet.yaml)):

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Downsample ratio</th>
    <th style="text-align: center">Codebook dim.</th>
    <th style="text-align: center">Codebook size</th>
    <th style="text-align: center">l2-norm codes</th>
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">rFID↓</th>
</tr>
<tr>
    <td style="text-align: center">16</td>
    <td style="text-align: center">256</td>
    <td style="text-align: center">1024</td>
    <td style="text-align: center">No</td>
    <td style="text-align: center">99.90%</td>
    <td style="text-align: center">19.4618</td>
    <td style="text-align: center">0.4644</td>
    <td style="text-align: center">18.90909</td>
</tr>
</table>

⚠️ The rFID is much worse than the results reported in the paper (7.94).

Using hyperparameters from ["LlamaGen"](http://arxiv.org/abs/2406.06525) paper (see [config](./configs/vqgan-imagenet-llamagen.yaml)):

<table style="text-align: center;">
<tr>
    <th style="text-align: center">Downsample ratio</th>
    <th style="text-align: center">Codebook dim.</th>
    <th style="text-align: center">Codebook size</th>
    <th style="text-align: center">l2-norm codes</th>
    <th style="text-align: center">Codebook usage↑</th>
    <th style="text-align: center">PSNR↑</th>
    <th style="text-align: center">SSIM↑</th>
    <th style="text-align: center">rFID↓</th>
</tr>
<tr>
    <td style="text-align: center">16</td>
    <td style="text-align: center">8</td>
    <td style="text-align: center">16384</td>
    <td style="text-align: center">Yes</td>
    <td style="text-align: center">94.10%</td>
    <td style="text-align: center">20.0723</td>
    <td style="text-align: center">0.5201</td>
    <td style="text-align: center">4.061665</td>
</tr>
</table>

⚠️ The SSIM and rFID are worse than the results reported in the paper (0.675 & 2.19).

<br/>



## Installation

Clone this repo:

```shell
git clone https://github.com/xyfJASON/visual-tokenizer-pytorch.git
cd visual-tokenizer-pytorch
```

Create and activate a conda environment:

```shell
conda create -n vistok python=3.11
conda activate vistok
```

Install dependencies:

```shell
pip install -r requirements.txt
```

<br/>



## References

VQVAE:

```
@article{van2017neural,
  title={Neural discrete representation learning},
  author={Van Den Oord, Aaron and Vinyals, Oriol and others},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

VQGAN (Taming Transformers):

```
@inproceedings{esser2021taming,
  title={Taming transformers for high-resolution image synthesis},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bjorn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12873--12883},
  year={2021}
}
```

ViT-VQGAN:

```
@inproceedings{yuvector,
  title={Vector-quantized Image Modeling with Improved VQGAN},
  author={Yu, Jiahui and Li, Xin and Koh, Jing Yu and Zhang, Han and Pang, Ruoming and Qin, James and Ku, Alexander and Xu, Yuanzhong and Baldridge, Jason and Wu, Yonghui},
  booktitle={International Conference on Learning Representations}
}
```

VQGAN (LlamaGen):

```
@article{sun2024autoregressive,
  title={Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation},
  author={Sun, Peize and Jiang, Yi and Chen, Shoufa and Zhang, Shilong and Peng, Bingyue and Luo, Ping and Yuan, Zehuan},
  journal={arXiv preprint arXiv:2406.06525},
  year={2024}
}
```
