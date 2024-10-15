# visual-tokenizer-pytorch

Implement visual tokenizers with PyTorch.

<br/>



## Progress

- [x] VQVAE
- [x] VQGAN (Taming-Transformers)
- [x] VQGAN (LlamaGen)
- [ ] ViT-VQGAN
- [ ] RQVAE
- [x] FSQ
- [ ] LFQ
- [ ] BSQ

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



## Benchmarks

See [benchmarks](./docs) for more details.

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
@inproceedings{yu2022vectorquantized,
  title={Vector-quantized Image Modeling with Improved {VQGAN}},
  author={Jiahui Yu and Xin Li and Jing Yu Koh and Han Zhang and Ruoming Pang and James Qin and Alexander Ku and Yuanzhong Xu and Jason Baldridge and Yonghui Wu},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=pfNyExj7z2}
}
```

MaskGIT:

```
@inproceedings{chang2022maskgit,
  title={Maskgit: Masked generative image transformer},
  author={Chang, Huiwen and Zhang, Han and Jiang, Lu and Liu, Ce and Freeman, William T},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11315--11325},
  year={2022}
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

FSQ:

```
@inproceedings{mentzer2024finite,
  title={Finite Scalar Quantization: {VQ}-{VAE} Made Simple},
  author={Fabian Mentzer and David Minnen and Eirikur Agustsson and Michael Tschannen},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=8ishA3LxN8}
}
```
