import logging
import multiprocessing
import os
import pathlib
import time

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Embedding
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, nsf_HiFigan_dataset
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss, RSSLoss
from modules.mel_processing import mel_spectrogram_torch
from modules.models import MultiPeriodDiscriminator, TrainModel

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich import print
from muon import Muon

from utils.config_utils import read_full_config, print_config
from utils import get_hparams

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

progress = Progress(
    TextColumn("Running: "),
    BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn(),
    "•",
    TextColumn("[progress.description]{task.description}"),
    transient=True
    )



@click.command(help='')
@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
def main(config):
    assert torch.cuda.is_available(), "CPU training is not allowed."
    config = pathlib.Path(config)
    config = get_hparams(config)
    print_config(config)

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.port

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, config,))

def run(rank, n_gpus, config):
    global global_step
    if rank == 0:
        print("Hyperparameters:", config)
        writer = SummaryWriter(log_dir=config.model_dir)
     
    dist.init_process_group(
        backend = 'gloo' if os.name == 'nt' else 'nccl', 
        init_method='env://', world_size=n_gpus, 
        rank=rank
    )
    print("Backend:", dist.get_backend())
    torch.manual_seed(config.seed)
    torch.cuda.set_device(rank)
       
    print(f"Use GPU: {rank} for training")
    


    #all_in_mem = config.all_in_mem
    train_dataset = nsf_HiFigan_dataset(
        config=config,
        data_dir=pathlib.Path(config["DataIndexPath"]) / "train"
    )
    num_workers = multiprocessing.cpu_count()
    #if all_in_mem:
    #    num_workers = 0
    train_loader = DataLoader(
        train_dataset, 
        num_workers=num_workers, 
        shuffle=False, 
        pin_memory=True, 
        batch_size=config.batch_size, 
        collate_fn=TextAudioCollate(config=config), 
        persistent_workers=True
    )
    if rank == 0:
        eval_dataset = nsf_HiFigan_dataset(
            config=config,
            data_dir=pathlib.Path(config["DataIndexPath"]) / "valid", 
            infer=True
        )
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False, batch_size=1, pin_memory=False, drop_last=False, collate_fn=TextAudioCollate(config=config,infer=True))

    net_g = TrainModel(config).cuda(rank)
    net_d = MultiPeriodDiscriminator(config.use_spectral_norm).cuda(rank)

    net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank])
    rss_loss = RSSLoss(256, 2048, 8).cuda(rank)
    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(config.model_dir, "G_*.pth"), net_g, optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(config.model_dir, "D_*.pth"), net_d, optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        name=utils.latest_checkpoint_path(config.model_dir, "D_*.pth")
        global_step=int(name[name.rfind("_") + 1:name.rfind(".")]) + 1
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    warmup_epoch = config.warmup_epochs
    
    # set Muon optimizer
    muon_params = []
    adamw_params = []
    
    for module in net_g.modules():
        for name, param in module.named_parameters(recurse=False):
            if not isinstance(module, (Embedding)) and param.ndim >= 2:
                # 所有不是emb的层使用muon优化器
                muon_params.append(param)
            else:
                adamw_params.append(param)
    optim_g_muon = Muon(muon_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    optim_g_adamw = torch.optim.AdamW(
        adamw_params,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    muon_params_d = []
    adamw_params_d = []

    for module in net_d.modules():
        for name, param in module.named_parameters(recurse=False):
            if not isinstance(module, (Embedding)) and param.ndim >= 2:
                muon_params_d.append(param)
            else:
                adamw_params_d.append(param)

    optim_d_muon = Muon(muon_params_d, lr=config.learning_rate)
    optim_d_adamw = torch.optim.AdamW(
        adamw_params_d,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps
    )

    optim_g = [optim_g_muon, optim_g_adamw]
    optim_d = [optim_d_muon, optim_d_adamw]

    schedulers_g = []
    schedulers_d = []

    for optimizer in optim_g: 
        schedulers_g.append(
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.lr_decay,
                last_epoch=epoch_str - 2
            )
        )

    for optimizer in optim_d: 
        schedulers_d.append(
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.lr_decay,
                last_epoch=epoch_str - 2
            )
        )

    scaler = GradScaler(enabled=config.fp16_run)

    print("======= Start training =======")
    global train_task
    with progress:
        train_task = progress.add_task("Train", total=len(train_loader) - 1)
        for epoch in range(epoch_str, config.epochs + 1):
            if epoch <= warmup_epoch:
                for param_group in optim_g.param_groups:
                    param_group['lr'] = config.learning_rate / warmup_epoch * epoch
                for param_group in optim_d.param_groups:
                    param_group['lr'] = config.learning_rate / warmup_epoch * epoch
            if rank == 0:
                train_and_evaluate(rank, rss_loss, epoch, config, [net_g, net_d], [optim_g, optim_d], [schedulers_g, schedulers_d], scaler, [train_loader, eval_loader], writer)
            else:
                train_and_evaluate(rank, rss_loss, epoch, config, [net_g, net_d], [optim_g, optim_d], [schedulers_g, schedulers_d], scaler, [train_loader, None], None)

            for scheduler_g in schedulers_g: 
                scheduler_g.step()

            for scheduler_d in schedulers_d: 
                scheduler_d.step()

def train_and_evaluate(rank, rss_loss, epoch, config, nets, optims, schedulers, scaler, loaders, writer):
    net_g, net_d = nets
    optim_g, optim_d = optims
    schedulers_g, schedulers_d = schedulers
    train_loader, eval_loader = loaders
    
    half_type = torch.bfloat16 if config.half_type=="bf16" else torch.float16

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, sample in enumerate(train_loader):
        start_time = time.time()
        wav, spec, f0 = sample['audio'], sample['spec'], sample['f0']

        wav = wav.cuda(rank, non_blocking=True)
        spec = spec.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        sample = {
            'audio': wav,
            'spec': spec,
            'f0': f0
        }
        print("wav.shape:", wav.shape)
        print("spec.shape:", spec.shape)
        print("f0.shape:", f0.shape)
        print("wav.dtype:", wav.dtype)
        print("wav.device", wav.device)
        mel = mel_spectrogram_torch(
            wav.squeeze(1),
            config.filter_length,
            config.n_mel_channels,
            config.sampling_rate,
            config.hop_size,
            config.win_size,
            config.mel_fmin,
            config.mel_fmax)
        
        with autocast(enabled=config.fp16_run, dtype=half_type):
            pc_aug_num = int(np.ceil(sample['audio'].shape[0] * config.pc_aug_rate))
            pc_aug = config.pc_aug and pc_aug_num > 0
            if pc_aug:
                if not config.mini_nsf:
                    raise ValueError("PC augmentation is only available for generator with MiniNSF module.")
                z, Goutput, (m, logs), commit_loss = net_g(sample,pc_aug_num=pc_aug_num)
                audio_fake = torch.cat((
                    Goutput['audio'],
                    Goutput['audio_shift_c'],
                    Goutput['audio_shift_a'],
                    Goutput['audio_shift_ab']
                ), dim=0)
            else:
                z, Goutput, (m, logs), commit_loss = net_g(sample, pc_aug_num=0)
                audio_fake = Goutput['audio']
            print("audio_fake.shape:", audio_fake.shape)
            print("Goutput['audio'].shape:", Goutput['audio'].shape)

            y_hat_mel = mel_spectrogram_torch(
                Goutput['audio'].squeeze(1),
                config.filter_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.hop_size,
                config.win_size,
                config.mel_fmin,
                config.mel_fmax
            )
            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wav, audio_fake.detach())


            with autocast(enabled=False, dtype=half_type):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc * config.c_disc
        
        if random.random() > 0.7 or global_step < 2000:
            loss_disc_all *= 0

        optim_d_muon, optim_d_adamw = optim_d

        optim_d_muon.zero_grad()
        optim_d_adamw.zero_grad()

        scaler.scale(loss_disc_all).backward()

        scaler.unscale_(optim_d_muon)
        scaler.unscale_(optim_d_adamw)

        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        
        scaler.step(optim_d_adamw)
        scaler.step(optim_d_muon)


        with autocast(enabled=config.fp16_run, dtype=half_type):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wav, audio_fake)
            with autocast(enabled=False, dtype=half_type):
                if pc_aug:
                    pc_wav_loss = F.l1_loss(Goutput['audio_shift_ab'], Goutput['audio_shift_c']) * 30
                    loss_mel_ab = rss_loss(Goutput['audio_shift_ab'], Goutput['audio_shift_c']) * config.c_mel
                else:
                    pc_wav_loss = 0
                    loss_mel_ab = 0
                loss_mel = rss_loss(sample['audio'], Goutput['audio']) * config.c_mel
                loss_kl = kl_loss(logs, m) * config.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                if global_step < 2000:
                    loss_gen *= 0
                    loss_fm *= 0
                commit_loss = commit_loss * config.c_vq
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_mel_ab + loss_kl + pc_wav_loss + commit_loss
        
        optim_g_muon, optim_g_adamw = optim_g
        optim_g_muon.zero_grad()
        optim_g_adamw.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g_muon)
        scaler.unscale_(optim_g_adamw)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g_muon)
        scaler.step(optim_g_adamw)
        scaler.update()

        if rank == 0:
            if global_step % config.log_interval == 0:
                lr = optim_g_adamw.param_groups[0]['lr']

                scalar_dict = {"loss/g": loss_gen, "loss/d": loss_disc_all, "lr": lr, "grad_norm/g": grad_norm_g, "grad_norm/d": grad_norm_d}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel,"loss/g/mel_ab": loss_mel_ab, "loss/g/kl": loss_kl, "loss/pc_wav_loss":pc_wav_loss, "loss/vq_loss": commit_loss})

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                }

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % config.eval_interval == 0:
                evaluate(config, net_g, eval_loader, writer)
                utils.save_checkpoint(net_g, optim_g, config.learning_rate, epoch,
                                      os.path.join(config.model_dir, f"G_{global_step}.pth"))
                utils.save_checkpoint(net_d, optim_d, config.learning_rate, epoch,
                                      os.path.join(config.model_dir, f"D_{global_step}.pth"))
                keep_ckpts = getattr(config, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=config.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                    print(f"Save checkpoint: G_{global_step}.pth D_{global_step}.pth | epoch={epoch}, step={global_step}, lr={optim_g.param_groups[0]['lr']:.5f}, loss_g={loss_gen.item():.2f}, loss_fm={loss_fm.item():.2f}, loss_mel={loss_mel.item():.2f}, loss_kl={loss_kl.item():.4f}, loss_pc_wav={pc_wav_loss.item():.2f}, vq_loss={commit_loss.item():.2f}")
            end_time = time.time()
            progress.update(train_task, advance=1, description=f"speed={1 / (end_time - start_time):.2f}it/s, epoch={epoch}, step={global_step}, lr={optim_g.param_groups[0]['lr']:.5f}, loss_g={loss_gen.item():.2f}, loss_fm={loss_fm.item():.2f}, loss_mel={loss_mel.item():.2f}, loss_kl={loss_kl.item():.2f}, loss_pc_wav={pc_wav_loss.item():.2f}, vq_loss={commit_loss.item():.4f}, grad_norm={grad_norm_g:.2f}")
        global_step += 1
    progress.reset(train_task)

def evaluate(config, generator, eval_loader, writer):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        evaluate_task = progress.add_task("Evaluate", total=len(eval_loader) - 1)
        for batch_idx, sample in enumerate(eval_loader):
            progress.update(evaluate_task, description=f"audio=_{batch_idx}")
            wav = sample['audio']
            wav = wav.cuda(0)

            mel = mel_spectrogram_torch(
                wav,
                config.filter_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.hop_size,
                config.win_size,
                config.mel_fmin,
                config.mel_fmax)

            z, y_hat, (m, logs), commit_loss = generator(sample, pc_aug_num=0)["audio"]

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                config.filter_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.hop_size,
                config.win_size,
                config.mel_fmin,
                config.mel_fmax
            )

            audio_dict.update({f"gen/audio_{batch_idx}": y_hat[0], f"gt/audio_{batch_idx}": wav[0]})
            progress.update(evaluate_task, advance=1)
        image_dict.update({"mel/gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()), "mel/gt": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
    progress.update(evaluate_task, description=f"Writing Summarize")
    utils.summarize(
        writer=writer,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=config.sampling_rate
    )
    progress.remove_task(evaluate_task)
    generator.train()


if __name__ == "__main__":
    main()