import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from models import DFFNet
from losses import TotalLoss
from data import build_dataloader
from utils import set_seed, AverageMeter, save_checkpoint, load_checkpoint, compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser(description='DFFNet Training')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def build_model(config):
    cfg = config['model']
    return DFFNet(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        base_channels=cfg['base_channels'],
        num_levels=cfg['num_levels'],
        mamba_blocks_per_level=cfg['mamba_blocks_per_level'],
        gabor_blocks_per_level=cfg['gabor_blocks_per_level'],
        d_state=cfg['d_state'],
        d_conv=cfg['d_conv'],
        expand=cfg['expand'],
        num_heads=cfg['num_heads'],
        gabor_n_orientations=cfg['gabor_n_orientations'],
        gabor_n_scales=cfg['gabor_n_scales'],
        use_residual_learning=cfg['use_residual_learning'],
    )


def build_optimizer(model, config):
    cfg = config['training']
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        betas=tuple(cfg['betas']),
    )


def build_scheduler(optimizer, config):
    cfg = config['training']
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['T_max'], eta_min=cfg['eta_min'],
    )


def warmup_lr(epoch, warmup_epochs, warmup_lr_val, base_lr):
    if epoch < warmup_epochs:
        return warmup_lr_val + (base_lr - warmup_lr_val) * epoch / warmup_epochs
    return base_lr


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch in val_loader:
        hazy = batch['hazy'].to(device)
        gt = batch['gt'].to(device)
        pred = model(hazy)

        pred = pred[:, :, :gt.shape[2], :gt.shape[3]]

        psnr_val = compute_psnr(pred, gt)
        ssim_val = compute_ssim(pred, gt)
        psnr_meter.update(psnr_val.item(), hazy.shape[0])
        ssim_meter.update(ssim_val.item(), hazy.shape[0])

    model.train()
    return psnr_meter.avg, ssim_meter.avg


def train(config, resume=None):
    hw_cfg = config['hardware']
    train_cfg = config['training']
    log_cfg = config['logging']

    set_seed(hw_cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(config).to(device)
    if len(hw_cfg['gpu_ids']) > 1:
        model = nn.DataParallel(model, device_ids=hw_cfg['gpu_ids'])

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    criterion = TotalLoss(**config['loss']).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(enabled=False)

    train_loader = build_dataloader(config, mode='train')
    val_loader = build_dataloader(config, mode='val')

    log_dir = log_cfg['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

    start_epoch = 0
    best_psnr = 0.0
    if resume:
        start_epoch, best_psnr = load_checkpoint(resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    global_step = start_epoch * len(train_loader)
    collapse_count = 0
    prev_psnr = 0.0
    max_collapse = 5

    for epoch in range(start_epoch, train_cfg['total_epochs']):
        model.train()
        loss_meter = AverageMeter()

        if epoch < train_cfg['warmup_epochs']:
            lr = warmup_lr(epoch, train_cfg['warmup_epochs'],
                          train_cfg['warmup_lr'], train_cfg['lr'])
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        for i, batch in enumerate(train_loader):
            hazy = batch['hazy'].to(device)
            gt = batch['gt'].to(device)

            with autocast('cuda', dtype=torch.bfloat16, enabled=hw_cfg['mixed_precision']):
                pred = model(hazy)
                loss, loss_dict = criterion(pred, gt, hazy)

            if not torch.isfinite(loss):
                print(f"\n[ALERT] Loss became NaN/Inf at epoch {epoch+1}, step {i+1}. Stopping training.")
                raw_model = model.module if hasattr(model, 'module') else model
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                }, log_dir, 'crash_checkpoint.pth')
                writer.close()
                return

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            if torch.isfinite(total_norm):
                scaler.step(optimizer)
            else:
                print(f"  [WARN] Non-finite grad norm at epoch {epoch+1}, step {i+1}, skipped.")
                optimizer.zero_grad()
            scaler.update()

            loss_meter.update(loss.item(), hazy.shape[0])
            global_step += 1

            if global_step % log_cfg['log_interval'] == 0:
                lr_current = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{train_cfg['total_epochs']}] "
                      f"Step [{i+1}/{len(train_loader)}] "
                      f"Loss: {loss_meter.avg:.4f} LR: {lr_current:.6f}")
                for k, v in loss_dict.items():
                    writer.add_scalar(f'train/{k}', v, global_step)
                writer.add_scalar('train/lr', lr_current, global_step)

        if epoch >= train_cfg['warmup_epochs']:
            scheduler.step()

        if (epoch + 1) % log_cfg['val_interval'] == 0:
            psnr, ssim = validate(model, val_loader, device)
            print(f"Epoch [{epoch+1}] Val PSNR: {psnr:.2f} SSIM: {ssim:.4f}")
            writer.add_scalar('val/psnr', psnr, epoch + 1)
            writer.add_scalar('val/ssim', ssim, epoch + 1)

            if epoch >= train_cfg['warmup_epochs']:
                if prev_psnr > 0 and psnr < prev_psnr * 0.7:
                    collapse_count += 1
                    print(f"  [WARNING] PSNR dropped >30% ({prev_psnr:.2f} -> {psnr:.2f}), collapse count: {collapse_count}/{max_collapse}")
                else:
                    collapse_count = max(0, collapse_count - 1)
                prev_psnr = max(prev_psnr, psnr)

                if collapse_count >= max_collapse:
                    print(f"\n[ALERT] Training collapsed! PSNR dropped significantly for {max_collapse} consecutive epochs. Stopping.")
                    raw_model = model.module if hasattr(model, 'module') else model
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_psnr': best_psnr,
                    }, log_dir, 'crash_checkpoint.pth')
                    writer.close()
                    print(f"Best model is saved at {log_dir}/best_model.pth (PSNR: {best_psnr:.2f})")
                    return

            if psnr > best_psnr:
                best_psnr = psnr
                raw_model = model.module if hasattr(model, 'module') else model
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                }, log_dir, 'best_model.pth')
                print(f"  -> New best PSNR: {best_psnr:.2f}")

        if (epoch + 1) % log_cfg['save_interval'] == 0:
            raw_model = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, log_dir, f'epoch_{epoch+1}.pth')

    writer.close()
    print(f"Training complete. Best PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train(config, resume=args.resume)
