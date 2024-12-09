import os
import tqdm
import argparse
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from models.vqmodel import VQModel
from utils.data import load_data
from utils.logger import get_logger
from utils.tracker import StatusTracker
from utils.misc import get_time_str, check_freq, set_seed
from utils.experiment import create_exp_dir, find_resume_checkpoint, instantiate_from_config, discard_label
from utils.distributed import init_distributed_mode, is_main_process, on_main_process, is_dist_avail_and_initialized
from utils.distributed import wait_for_everyone, cleanup, get_rank, get_world_size, get_local_rank, reduce_tensor


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('-e', '--exp_dir', type=str, help='Path to the experiment directory. Default to be ./runs/exp-{current time}/')
    parser.add_argument('-r', '--resume', type=str, help='Resume from a checkpoint. Could be a path or `best` or `latest`')
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

    # SET SEED
    set_seed(conf.seed + get_rank())
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {get_world_size()}')
    logger.info(f'Distributed mode: {is_dist_avail_and_initialized()}')
    wait_for_everyone()

    # BUILD DATASET AND DATALOADER
    assert conf.train.batch_size % get_world_size() == 0
    bspp = conf.train.batch_size // get_world_size()  # batch size per process
    train_set = load_data(conf.data, split='train')
    valid_set = load_data(conf.data, split='valid')
    valid_set = Subset(valid_set, range(32))
    train_sampler = DistributedSampler(train_set, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    train_loader = DataLoader(train_set, batch_size=bspp, sampler=train_sampler, drop_last=True, **conf.dataloader)
    valid_loader = DataLoader(valid_set, batch_size=bspp, shuffle=False, drop_last=False, **conf.dataloader)
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # BUILD MODEL AND OPTIMIZERS
    encoder = instantiate_from_config(conf.encoder)
    decoder = instantiate_from_config(conf.decoder)
    quantizer = instantiate_from_config(conf.quantizer)
    use_ema_update = getattr(quantizer, 'use_ema_update', False)
    use_entropy_reg = getattr(quantizer, 'use_entropy_reg', False)
    model = VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer).to(device)
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Number of parameters of model: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # RESUME TRAINING
    step, epoch = 0, 0
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        # load model
        ckpt = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt['model'])
        logger.info(f'Successfully load model from {resume_path}')
        # load training states (optimizer, step, epoch)
        ckpt = torch.load(os.path.join(resume_path, 'training_states.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step'] + 1
        epoch = ckpt['epoch']
        logger.info(f'Successfully load training states from {resume_path}')
        logger.info(f'Restart training at step {step}')
        del ckpt

    # PREPARE FOR DISTRIBUTED TRAINING
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank())
    model_wo_ddp = model.module if is_dist_avail_and_initialized() else model
    wait_for_everyone()

    # TRAINING FUNCTIONS
    @on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model
        torch.save(dict(
            model=model_wo_ddp.state_dict(),
        ), os.path.join(save_path, 'model.pt'))
        # save training states (loss_adv_fn, optimizers, step, epoch)
        torch.save(dict(
            optimizer=optimizer.state_dict(),
            step=step,
            epoch=epoch,
        ), os.path.join(save_path, 'training_states.pt'))

    def train_step(batch):
        x = discard_label(batch).float().to(device)
        # zero the gradients
        optimizer.zero_grad()
        # forward
        out = model(x)
        # reconstruction loss
        loss_rec = F.mse_loss(out['decx'], x)
        # commitment loss
        loss_commit = out['loss_commit']
        # vq loss
        loss_vq = out['loss_vq'] if not use_ema_update else None
        # entropy regularization
        loss_entropy = out['loss_entropy'] if use_entropy_reg else None
        # total loss
        loss = loss_rec + conf.train.coef_commit * loss_commit
        if not use_ema_update:
            loss = loss + loss_vq
        if use_entropy_reg:
            loss = loss + conf.train.coef_entropy * loss_entropy
        # backward
        loss.backward()
        # optimize
        optimizer.step()
        # use EMA update for codebook
        if use_ema_update:
            # count used codebook entries
            codebook_num = quantizer.codebook_num
            codebook_dim = quantizer.codebook_dim
            flat_z = out['z'].detach().permute(0, 2, 3, 1).reshape(-1, codebook_dim)
            indices_count = torch.bincount(out['indices'], minlength=codebook_num)
            new_sumz = torch.zeros((codebook_num, codebook_dim), device=device)
            new_sumz.scatter_add_(dim=0, index=out['indices'][:, None].repeat(1, codebook_dim), src=flat_z)
            # reduce sumz and sumn across all processes
            new_sumz = reduce_tensor(new_sumz) * get_world_size()
            new_sumn = reduce_tensor(indices_count) * get_world_size()
            # update codebook
            quantizer.update_codebook(new_sumz, new_sumn)
        # return
        status = dict(loss_rec=loss_rec.item(), loss_commit=loss_commit.item())
        status.update(dict(loss_vq=loss_vq.item())) if not use_ema_update else None
        status.update(dict(loss_entropy=loss_entropy.item())) if use_entropy_reg else None
        status.update(dict(perplexity=out['perplexity'].item(), lr=optimizer.param_groups[0]['lr']))
        return status

    @on_main_process
    @torch.no_grad()
    def sample(savepath):
        shows = []
        for x in valid_loader:
            x = discard_label(x).float().to(device)
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
