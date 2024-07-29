import argparse
import os
from contextlib import nullcontext
from omegaconf import OmegaConf

import accelerate
import torch
import torch.nn.functional as F
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from models.vqmodel import VQModel
from losses.adversarial import HingeLoss
from losses.lpips import LPIPSLoss
from utils.data import load_data
from utils.logger import get_logger
from utils.misc import create_exp_dir, find_resume_checkpoint, instantiate_from_config
from utils.misc import get_time_str, check_freq, get_data_generator, discard_label
from utils.tracker import StatusTracker


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to configuration file',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/exp-{current time}/',
    )
    parser.add_argument(
        '-r', '--resume', type=str,
        help='Resume from a checkpoint. Could be a path or `best` or `latest`',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir, conf_yaml=OmegaConf.to_yaml(conf), time_str=args.time_str,
            exist_ok=args.resume is not None, no_interaction=args.no_interaction,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True, is_main_process=accelerator.is_main_process,
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger, exp_dir=exp_dir, print_freq=conf.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(conf.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    if conf.train.batch_size % accelerator.num_processes != 0:
        raise ValueError(
            f'Batch size should be divisible by number of processes, '
            f'get {conf.train.batch_size} % {accelerator.num_processes} != 0'
        )
    bspp = conf.train.batch_size // accelerator.num_processes
    micro_bs = conf.train.micro_batch_size or bspp
    train_set = load_data(conf.data, split='train')
    valid_set = load_data(conf.data, split='valid')
    valid_set = Subset(valid_set, torch.randperm(len(valid_set))[:16])
    train_loader = DataLoader(
        dataset=train_set, batch_size=bspp,
        shuffle=True, drop_last=True, **conf.dataloader,
    )
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=bspp,
        shuffle=False, drop_last=False, **conf.dataloader,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Micro batch size: {micro_bs}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # BUILD MODEL AND OPTIMIZERS
    encoder = instantiate_from_config(conf.encoder)
    decoder = instantiate_from_config(conf.decoder)
    quantizer = instantiate_from_config(conf.quantizer)
    disc = instantiate_from_config(conf.disc)
    model = VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer)
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    optimizer_d = instantiate_from_config(conf.train.optim_d, params=disc.parameters())
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Number of parameters of vq model: {sum(p.numel() for p in model.parameters()):,}')
    logger.info(f'Number of parameters of discriminator: {sum(p.numel() for p in disc.parameters()):,}')
    logger.info('=' * 50)

    # RESUME TRAINING
    step = 0
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        # load model
        ckpt_model = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt_model['model'])
        disc.load_state_dict(ckpt_model['disc'])
        logger.info(f'Successfully load model from {resume_path}')
        # load optimizer
        ckpt_optimizer = torch.load(os.path.join(resume_path, 'optimizer.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        optimizer_d.load_state_dict(ckpt_optimizer['optimizer_d'])
        logger.info(f'Successfully load optimizer from {resume_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(resume_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1
        logger.info(f'Restart training at step {step}')

    # DEFINE LOSSES
    hinge_loss = HingeLoss(discriminator=disc).to(device)
    lpips = LPIPSLoss().to(device)

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model, disc, optimizer, optimizer_d, train_loader = accelerator.prepare(
        model, disc, optimizer, optimizer_d, train_loader,  # type: ignore
    )
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_disc = accelerator.unwrap_model(disc)

    accelerator.wait_for_everyone()

    # TRAINING FUNCTIONS
    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model
        accelerator.save(dict(
            model=unwrapped_model.state_dict(),
            disc=unwrapped_disc.state_dict(),
        ), os.path.join(save_path, 'model.pt'))
        # save optimizer
        accelerator.save(dict(
            optimizer=optimizer.state_dict(),
            optimizer_d=optimizer_d.state_dict(),
        ), os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        accelerator.save(dict(step=step), os.path.join(save_path, 'meta.pt'))

    def calc_adaptive_weight(loss_nll, loss_adv, last_layer):
        nll_grads = torch.autograd.grad(loss_nll, last_layer, retain_graph=True)[0]
        adv_grads = torch.autograd.grad(loss_adv, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(adv_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def run_step(batch):
        optimizer.zero_grad()
        x = discard_label(batch).float()
        bs = x.shape[0]
        for i in range(0, bs, micro_bs):
            micro_x = x[i:i+micro_bs]
            loss_scale = micro_x.shape[0] / bs
            model_no_sync = accelerator.no_sync(model) if i + micro_bs < bs else nullcontext()
            disc_no_sync = accelerator.no_sync(disc) if i + micro_bs < bs else nullcontext()
            with model_no_sync, disc_no_sync:
                out = model(micro_x)
                # reconstruction loss
                if conf.train.type_rec == 'l2':
                    loss_rec = F.mse_loss(out['decx'], x)
                elif conf.train.type_rec == 'l1':
                    loss_rec = F.l1_loss(out['decx'], x)
                else:
                    raise ValueError(f'Unknown reconstruction loss type: {conf.train.type_rec}')
                # perceptual loss
                loss_lpips = lpips(out['decx'], x).mean()
                # commitment loss
                loss_commit = out['loss_commit']
                # vq loss
                loss_vq = out['loss_vq']
                # adversarial loss
                loss_adv = torch.tensor(0.0, device=out['decx'].device, requires_grad=True)
                if step >= conf.train.start_adv:
                    loss_adv = hinge_loss('G', out['decx'])
                    if conf.train.adaptive_adv_weight:
                        loss_nll = conf.train.coef_rec * loss_rec + conf.train.coef_lpips * loss_lpips
                        adaptive_weight = calc_adaptive_weight(loss_nll, loss_adv, unwrapped_model.last_layer)
                        loss_adv = adaptive_weight * loss_adv
                # total loss
                loss = (
                    conf.train.coef_rec * loss_rec +
                    conf.train.coef_lpips * loss_lpips +
                    conf.train.coef_commit * loss_commit +
                    conf.train.coef_vq * loss_vq +
                    conf.train.coef_adv * loss_adv
                )
                accelerator.backward(loss * loss_scale)
        optimizer.step()
        return dict(
            loss_rec=loss_rec.item(),
            loss_lpips=loss_lpips.item(),
            loss_commit=loss_commit.item(),
            loss_vq=loss_vq.item(),
            loss_adv_g=loss_adv.item(),
            perplexity=out['perplexity'].item(),
            lr=optimizer.param_groups[0]['lr'],
        )

    def run_step_d(batch):
        optimizer_d.zero_grad()
        x = discard_label(batch).float()
        bs = x.shape[0]
        for i in range(0, bs, micro_bs):
            micro_x = x[i:i+micro_bs]
            loss_scale = micro_x.shape[0] / bs
            model_no_sync = accelerator.no_sync(model) if i + micro_bs < bs else nullcontext()
            disc_no_sync = accelerator.no_sync(disc) if i + micro_bs < bs else nullcontext()
            with model_no_sync, disc_no_sync:
                loss_adv_d = torch.tensor(0.0, device=device, requires_grad=True)
                if step >= conf.train.start_adv:
                    out = model(micro_x)
                    loss_adv_d = hinge_loss('D', out['decx'], micro_x)
                accelerator.backward(loss_adv_d * loss_scale)
        optimizer_d.step()
        return dict(
            loss_adv_d=loss_adv_d.item(),
            lr_d=optimizer_d.param_groups[0]['lr'],
        )

    @accelerator.on_main_process
    @torch.no_grad()
    def sample(savepath):
        shows = []
        for x in valid_loader:
            x = discard_label(x).float().to(device)
            recx = unwrapped_model(x)['decx']
            C, H, W = recx.shape[1:]
            show = torch.stack((x, recx), dim=1).reshape(-1, C, H, W)
            shows.append(show)
        shows = torch.cat(shows, dim=0)
        save_image(shows, savepath, nrow=8, normalize=True, value_range=(-1, 1))

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        tqdm_kwargs=dict(desc='Epoch', leave=False, disable=not accelerator.is_main_process),
    )
    while step < conf.train.n_steps:
        # get a batch of data
        _batch = next(train_data_generator)
        # run a step
        model.train()
        train_status = run_step(_batch)
        status_tracker.track_status('Train', train_status, step)
        train_status = run_step_d(_batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()
        # validate
        model.eval()
        # save checkpoint
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>7d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(conf.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>7d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>7d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info('End of training')


if __name__ == '__main__':
    main()
