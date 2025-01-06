import os
import math
import tqdm
import argparse
from omegaconf import OmegaConf
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from models.ema import EMA
from models.vqmodel import VQModel
from losses.lpips import LPIPS as LPIPSLoss
from losses.perceptual_loss import PerceptualLoss
from losses.adversarial import AdversarialLoss
from utils.data import load_data
from utils.logger import get_logger
from utils.tracker import StatusTracker
from utils.misc import get_time_str, check_freq, set_seed
from utils.experiment import create_exp_dir, find_resume_checkpoint, instantiate_from_config
from utils.experiment import discard_label, toggle_on_gradients, toggle_off_gradients
from utils.distributed import init_distributed_mode, is_main_process, on_main_process, is_dist_avail_and_initialized
from utils.distributed import wait_for_everyone, cleanup, get_rank, get_world_size, get_local_rank


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('-e', '--exp_dir', type=str, help='Path to the experiment directory. Default to be ./runs/exp-{current time}/')
    parser.add_argument('-r', '--resume', type=str, help='Resume from a checkpoint. Could be a path or `best` or `latest`')
    parser.add_argument('-mp', '--mixed_precision', type=str, default=None, choices=['fp16', 'bf16'], help='Mixed precision training')
    parser.add_argument('-cd', '--cover_dir', action='store_true', default=False, help='Cover the experiment directory if it exists')
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

    # INITIALIZE DISTRIBUTED MODE
    device = init_distributed_mode()
    print(f'Process {get_rank()} using device: {device}', flush=True)
    wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if is_main_process():
        create_exp_dir(
            exp_dir=exp_dir, conf_yaml=OmegaConf.to_yaml(conf), subdirs=['ckpt', 'samples'],
            time_str=args.time_str, exist_ok=args.resume is not None, cover_dir=args.cover_dir,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True, is_main_process=is_main_process(),
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger, print_freq=conf.train.print_freq,
        tensorboard_dir=os.path.join(exp_dir, 'tensorboard'),
        is_main_process=is_main_process(),
    )

    # SET MIXED PRECISION
    if args.mixed_precision == 'fp16':
        mp_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        mp_dtype = torch.bfloat16
    else:
        mp_dtype = torch.float32

    # SET SEED
    set_seed(conf.seed + get_rank())
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {get_world_size()}')
    logger.info(f'Distributed mode: {is_dist_avail_and_initialized()}')
    logger.info(f'Mixed precision: {args.mixed_precision}')
    wait_for_everyone()

    # BUILD DATASET AND DATALOADER
    assert conf.train.batch_size % get_world_size() == 0
    bspp = conf.train.batch_size // get_world_size()  # batch size per process
    micro_batch_size = conf.train.micro_batch_size or bspp  # actual batch size in each iteration
    train_set = load_data(conf.data, split='train')
    valid_set = load_data(conf.data, split='valid')
    valid_set = Subset(valid_set, torch.randperm(len(valid_set))[:16])
    train_sampler = DistributedSampler(train_set, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    train_loader = DataLoader(train_set, batch_size=bspp, sampler=train_sampler, drop_last=True, **conf.dataloader)
    valid_loader = DataLoader(valid_set, batch_size=bspp, shuffle=False, drop_last=False, **conf.dataloader)
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Micro batch size: {micro_batch_size}')
    logger.info(f'Gradient accumulation steps: {math.ceil(bspp / micro_batch_size)}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # BUILD MODEL
    encoder = instantiate_from_config(conf.encoder)
    decoder = instantiate_from_config(conf.decoder)
    quantizer = instantiate_from_config(conf.quantizer)
    disc = instantiate_from_config(conf.disc).to(device)
    model = VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer).to(device)

    ema = None
    if conf.train.get('ema', None) is not None:
        ema = EMA(model.parameters(), **getattr(conf.train, 'ema', dict())).to(device)

    # BUILD OPTIMIZERS AND SCHEUDLERS
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    optimizer_d = instantiate_from_config(conf.train.optim_d, params=disc.parameters())
    scheduler, scheduler_d = None, None
    if conf.train.get('sched', None):
        scheduler = instantiate_from_config(conf.train.sched, optimizer=optimizer)
        scheduler_d = instantiate_from_config(conf.train.sched, optimizer=optimizer_d)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == 'fp16')
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Number of parameters of vq model: {sum(p.numel() for p in model.parameters()):,}')
    logger.info(f'Number of parameters of discriminator: {sum(p.numel() for p in disc.parameters()):,}')
    logger.info('=' * 50)

    # DEFINE LOSSES
    assert conf.train.type_rec in ['l2', 'l1']
    loss_rec_fn = nn.MSELoss() if conf.train.type_rec == 'l2' else nn.L1Loss()

    loss_lpips_fn = LPIPSLoss().eval().to(device)

    loss_perc_fn = None
    if conf.train.get('type_perc', None):
        loss_perc_fn = PerceptualLoss(conf.train.type_perc).eval().to(device)

    loss_adv_fn = AdversarialLoss(
        discriminator=disc,
        loss_type=conf.train.get('adv_loss_type', 'hinge'),
        coef_lecam_reg=conf.train.get('coef_lecam_reg', 0.0),
    ).to(device)

    # RESUME TRAINING
    step, epoch = 0, 0
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        # load model
        ckpt = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt['model'])
        logger.info(f'Successfully load model from {resume_path}')
        # load training states (loss_adv_fn, optimizers, schedulers, scaler, ema, step, epoch)
        ckpt = torch.load(os.path.join(resume_path, 'training_states.pt'), map_location='cpu')
        loss_adv_fn.load_state_dict(ckpt['loss_adv_fn'])
        optimizer.load_state_dict(ckpt['optimizer'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        if conf.train.get('sched', None):
            scheduler.load_state_dict(ckpt['scheduler'])
            scheduler_d.load_state_dict(ckpt['scheduler_d'])
        if ckpt.get('scaler', None):
            scaler.load_state_dict(ckpt['scaler'])
        if ema is not None:
            ema.load_state_dict(ckpt['ema'])
        step = ckpt['step'] + 1
        epoch = ckpt['epoch']
        logger.info(f'Successfully load training states from {resume_path}')
        logger.info(f'Restart training at step {step}')
        del ckpt

    # PREPARE FOR DISTRIBUTED TRAINING
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank())
        loss_adv_fn = DDP(loss_adv_fn, device_ids=[get_local_rank()], output_device=get_local_rank())
    model_wo_ddp = model.module if is_dist_avail_and_initialized() else model
    loss_adv_fn_wo_ddp = loss_adv_fn.module if is_dist_avail_and_initialized() else loss_adv_fn
    wait_for_everyone()

    # TRAINING FUNCTIONS
    @on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model
        torch.save(dict(model=model_wo_ddp.state_dict()), os.path.join(save_path, 'model.pt'))
        if ema is not None:
            with ema.scope(model.parameters()):
                torch.save(dict(model=model_wo_ddp.state_dict()), os.path.join(save_path, 'model_ema.pt'))
        # save training states (loss_adv_fn, optimizers, schedulers, scaler, ema, step, epoch)
        training_states = dict(
            loss_adv_fn=loss_adv_fn_wo_ddp.state_dict(), step=step, epoch=epoch,
            optimizer=optimizer.state_dict(), optimizer_d=optimizer_d.state_dict(),
        )
        if conf.train.get('sched', None):
            training_states.update(scheduler=scheduler.state_dict(), scheduler_d=scheduler_d.state_dict())
        if args.mixed_precision == 'fp16':
            training_states.update(scaler=scaler.state_dict())
        if ema is not None:
            training_states.update(ema=ema.state_dict())
        torch.save(training_states, os.path.join(save_path, 'training_states.pt'))

    def calc_adaptive_weight(loss_nll, loss_adv, last_layer):
        nll_grads = torch.autograd.grad(loss_nll, last_layer, retain_graph=True)[0]
        adv_grads = torch.autograd.grad(loss_adv, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(adv_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def train_micro_batch(x, loss_scale, no_sync):
        model_no_sync = model.no_sync() if no_sync else nullcontext()
        loss_adv_fn_no_sync = loss_adv_fn.no_sync() if no_sync else nullcontext()
        with model_no_sync, loss_adv_fn_no_sync:
            # ===================================================
            # Train vq model
            # ===================================================
            with torch.autocast(device_type='cuda', dtype=mp_dtype):
                toggle_off_gradients(disc)
                # forward
                out = model(x)
                # reconstruction loss
                loss_rec = loss_rec_fn(out['decx'], x)
                # lpips loss
                loss_lpips = loss_lpips_fn(out['decx'], x).mean()
                # perceptual loss
                loss_perc = torch.tensor(0.0, device=out['decx'].device, requires_grad=True)
                if loss_perc_fn is not None:
                    loss_perc = loss_perc_fn((out['decx'] + 1) / 2, (x + 1) / 2)
                # commitment loss
                loss_commit = out['loss_commit']
                # vq loss
                loss_vq = out['loss_vq']
                # adversarial loss
                loss_adv = torch.tensor(0.0, device=out['decx'].device, requires_grad=True)
                if step >= conf.train.start_adv:
                    loss_adv = loss_adv_fn('G', fake_data=out['decx'])
                    if conf.train.get('adaptive_adv_weight', False):
                        loss_nll = conf.train.coef_rec * loss_rec + conf.train.coef_lpips * loss_lpips
                        adaptive_weight = calc_adaptive_weight(loss_nll, loss_adv, model_wo_ddp.last_layer)
                        loss_adv = adaptive_weight * loss_adv
                # total loss
                loss = (
                    conf.train.coef_rec * loss_rec +
                    conf.train.coef_lpips * loss_lpips +
                    conf.train.get('coef_perc', 0.) * loss_perc +
                    conf.train.coef_commit * loss_commit +
                    conf.train.coef_vq * loss_vq +
                    conf.train.coef_adv * loss_adv
                )
                loss = loss * loss_scale
            # backward
            scaler.scale(loss).backward()

            # ===================================================
            # Train discriminator
            # ===================================================
            with torch.autocast(device_type='cuda', dtype=mp_dtype):
                toggle_on_gradients(disc)
                # adversarial loss
                loss_adv_d = loss_adv_fn('D', fake_data=out['decx'].detach(), real_data=x)
                loss_adv_d = loss_adv_d * loss_scale
            # backward
            scaler.scale(loss_adv_d).backward()

        return dict(
            loss_rec=loss_rec.item(),
            loss_lpips=loss_lpips.item(), loss_perc=loss_perc.item(),
            loss_commit=loss_commit.item(), loss_vq=loss_vq.item(),
            loss_adv=loss_adv.item(), loss_adv_d=loss_adv_d.item(),
            perplexity=out['perplexity'].item(),
        )

    def train_step(batch):
        status = dict()
        x = discard_label(batch).float().to(device)
        # zero the gradients
        optimizer.zero_grad()
        optimizer_d.zero_grad()
        # gradient accumulation
        for i in range(0, bspp, micro_batch_size):
            micro_x = x[i:i+micro_batch_size]
            loss_scale = micro_x.shape[0] / bspp
            no_sync = i + micro_batch_size < bspp and is_dist_avail_and_initialized()
            status = train_micro_batch(micro_x, loss_scale, no_sync)
        # optimize
        if conf.train.get('clip_grad_norm', None):
            scaler.unscale_(optimizer)
            scaler.unscale_(optimizer_d)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf.train.clip_grad_norm)
            nn.utils.clip_grad_norm_(disc.parameters(), max_norm=conf.train.clip_grad_norm)
        scaler.step(optimizer)
        scaler.step(optimizer_d)
        scaler.update()
        if ema is not None:
            ema.update(model.parameters())
        if conf.train.get('sched', None):
            scheduler.step()
            scheduler_d.step()
        status.update(lr=optimizer.param_groups[0]['lr'], lr_d=optimizer_d.param_groups[0]['lr'])
        return status

    @on_main_process
    @torch.no_grad()
    def sample(savepath):
        shows = []
        for x in valid_loader:
            x = discard_label(x).float().to(device)
            with ema.scope(model.parameters()) if ema is not None else nullcontext():
                recx = model_wo_ddp(x)['decx']
            C, H, W = recx.shape[1:]
            show = torch.stack((x, recx), dim=1).reshape(-1, C, H, W)
            shows.append(show)
        shows = torch.cat(shows, dim=0)
        save_image(shows, savepath, nrow=8, normalize=True, value_range=(-1, 1))

    # START TRAINING
    logger.info('Start training...')
    while step < conf.train.n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        for _batch in tqdm.tqdm(train_loader, desc='Epoch', leave=False, disable=not is_main_process()):
            # train a step
            model.train()
            train_status = train_step(_batch)
            status_tracker.track_status('Train', train_status, step)
            wait_for_everyone()
            # validate
            model.eval()
            # save checkpoint
            if check_freq(conf.train.save_freq, step):
                save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>7d}'))
                wait_for_everyone()
            # sample from current model
            if check_freq(conf.train.sample_freq, step):
                sample(os.path.join(exp_dir, 'samples', f'step{step:0>7d}.png'))
                wait_for_everyone()
            step += 1
            if step >= conf.train.n_steps:
                break
        epoch += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>7d}'))
    wait_for_everyone()

    # END OF TRAINING
    status_tracker.close()
    cleanup()
    logger.info('End of training')


if __name__ == '__main__':
    main()
