import argparse
import os
import tqdm
from omegaconf import OmegaConf

import accelerate
import torch
import torch_fidelity
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from metrics import PSNR, SSIM, LPIPS
from models.vqmodel import VQModel
from utils.data import load_data
from utils.logger import get_logger
from utils.misc import instantiate_from_config, discard_label, image_norm_to_float


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--weights', type=str, required=True, help='Path to pretrained vqmodel weights')
    parser.add_argument('--bspp', type=int, default=256, help='Batch size on each process')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to directory saving samples (for rFID)')
    parser.add_argument('--seed', type=int, default=8888, help='Set random seed')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(use_tqdm_handler=True, is_main_process=accelerator.is_main_process)

    # SET SEED
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')
    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    split = 'test'
    if conf.data.name.lower() == 'imagenet':
        split = 'valid'
    dataset = load_data(conf.data, split=split)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.bspp,
        shuffle=False, drop_last=False, **conf.dataloader,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of dataset: {len(dataset)}')
    logger.info(f'Batch size per process: {args.bspp}')
    logger.info(f'Total batch size: {args.bspp * accelerator.num_processes}')

    # BUILD MODEL
    encoder = instantiate_from_config(conf.encoder)
    decoder = instantiate_from_config(conf.decoder)
    quantizer = instantiate_from_config(conf.quantizer)
    vqmodel = VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer).eval().to(device)

    # LOAD WEIGHTS
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')
    ckpt = torch.load(args.weights, map_location='cpu')
    vqmodel.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load vqvae from {args.weights}')
    logger.info('=' * 50)

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    vqmodel, dataloader = accelerator.prepare(vqmodel, dataloader)  # type: ignore
    unwrapped_vqmodel = accelerator.unwrap_model(vqmodel)
    accelerator.wait_for_everyone()

    # START EVALUATION
    logger.info('Start evaluating...')
    idx = 0
    if args.save_dir is not None:
        os.makedirs(os.path.join(args.save_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'reconstructed'), exist_ok=True)

    psnr_fn = PSNR(reduction='none')
    ssim_fn = SSIM(reduction='none')
    lpips_fn = LPIPS(reduction='none').to(device)
    psnr_list, ssim_list, lpips_list = [], [], []
    codebook_size = unwrapped_vqmodel.codebook_size
    indices_count = torch.zeros((codebook_size, ), device=device)

    with torch.no_grad():
        for x in tqdm.tqdm(dataloader, desc='Evaluating', disable=not accelerator.is_main_process):
            x = discard_label(x)
            out = vqmodel(x)
            decx, indices = out['decx'], out['indices']
            decx = decx.clamp(-1, 1)
            indices = indices.reshape(-1, out['quantized_z'].shape[2], out['quantized_z'].shape[3])

            x = image_norm_to_float(x)
            decx = image_norm_to_float(decx)
            psnr = psnr_fn(decx, x)
            ssim = ssim_fn(decx, x)
            lpips = lpips_fn(decx, x)

            psnr = accelerator.gather_for_metrics(psnr)
            ssim = accelerator.gather_for_metrics(ssim)
            lpips = accelerator.gather_for_metrics(lpips)
            indices = accelerator.gather_for_metrics(indices).flatten()
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips)
            indices_count += F.one_hot(indices, num_classes=codebook_size).sum(dim=0).float()

            if args.save_dir is not None:
                x = accelerator.gather_for_metrics(x)
                decx = accelerator.gather_for_metrics(decx)
                if accelerator.is_main_process:
                    for ori, dec in zip(x, decx):
                        save_image(ori, os.path.join(args.save_dir, 'original', f'{idx}.png'))
                        save_image(dec, os.path.join(args.save_dir, 'reconstructed', f'{idx}.png'))
                        idx += 1

    psnr = torch.cat(psnr_list, dim=0).mean().item()
    ssim = torch.cat(ssim_list, dim=0).mean().item()
    lpips = torch.cat(lpips_list, dim=0).mean().item()
    codebook_usage = torch.sum(indices_count > 0).item() / codebook_size
    probs = indices_count / indices_count.sum()
    perplexity = torch.exp(-torch.sum(probs * torch.log(torch.clamp(probs, 1e-10))))

    logger.info(f'PSNR: {psnr:.4f}')
    logger.info(f'SSIM: {ssim:.4f}')
    logger.info(f'LPIPS: {lpips:.4f}')
    logger.info(f'Codebook usage: {codebook_usage * 100:.2f}%')
    logger.info(f'Perplexity: {perplexity:.4f}')

    if accelerator.is_main_process and args.save_dir is not None:
        fid_score = torch_fidelity.calculate_metrics(
            input1=os.path.join(args.save_dir, 'original'),
            input2=os.path.join(args.save_dir, 'reconstructed'),
            fid=True, verbose=False,
        )['frechet_inception_distance']
        logger.info(f'rFID: {fid_score:.4f}')

    accelerator.end_training()


if __name__ == '__main__':
    main()
