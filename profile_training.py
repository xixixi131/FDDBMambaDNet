"""Profile VRAM usage and training speed on the local GPU."""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml

from models import DFFNet
from losses import TotalLoss

def get_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {name}")
    print(f"Total VRAM: {total:.1f} GB")
    return total

def profile_batch_size(config, batch_size, crop_size=256, warmup=3, measure=10):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device('cuda')
    cfg = config['model']

    model = DFFNet(
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
    ).to(device)

    criterion = TotalLoss(**config['loss']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.train()
    param_count = sum(p.numel() for p in model.parameters())

    try:
        for step in range(warmup + measure):
            hazy = torch.rand(batch_size, 3, crop_size, crop_size, device=device)
            gt = torch.rand(batch_size, 3, crop_size, crop_size, device=device)

            if step == warmup:
                torch.cuda.synchronize()
                t_start = time.time()

            with torch.cuda.amp.autocast(enabled=True):
                pred = model(hazy)
                loss, loss_dict = criterion(pred, gt, hazy)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        t_end = time.time()

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        avg_time = (t_end - t_start) / measure

        return True, peak_mem, avg_time, param_count

    except torch.cuda.OutOfMemoryError:
        return False, 0, 0, param_count
    finally:
        del model, criterion, optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    total_vram = get_gpu_info()
    print()

    with open(os.path.join(os.path.dirname(__file__), 'config', 'default.yaml')) as f:
        config = yaml.safe_load(f)

    num_train = 6000
    total_epochs = config['training']['total_epochs']

    print(f"Model params config: base_ch={config['model']['base_channels']}, "
          f"levels={config['model']['num_levels']}")
    print(f"Dataset: ~{num_train} pairs, {total_epochs} epochs")
    print(f"Crop size: {config['data']['crop_size']}")
    print()

    results = []
    for bs in [1, 2, 4, 8, 12, 16, 20, 24]:
        print(f"--- Testing batch_size={bs} ---")
        ok, peak, time_per_iter, params = profile_batch_size(config, bs)
        if ok:
            iters_per_epoch = num_train // bs
            total_iters = iters_per_epoch * total_epochs
            total_hours = (time_per_iter * total_iters) / 3600
            total_days = total_hours / 24

            print(f"  Peak VRAM:       {peak:.2f} GB / {total_vram:.1f} GB ({peak/total_vram*100:.0f}%)")
            print(f"  Time/iter:       {time_per_iter:.3f} s")
            print(f"  Iters/epoch:     {iters_per_epoch}")
            print(f"  Total iters:     {total_iters:,}")
            print(f"  Est. total time: {total_hours:.1f} h ({total_days:.1f} days)")
            print(f"  Throughput:      {bs/time_per_iter:.1f} img/s")
            results.append((bs, peak, time_per_iter, total_hours, total_days))
        else:
            print(f"  OOM! batch_size={bs} exceeds VRAM")
            break
        print()

    if results:
        print("=" * 65)
        print(f"{'BS':>3} | {'VRAM':>8} | {'Time/it':>8} | {'img/s':>6} | {'Hours':>7} | {'Days':>5}")
        print("-" * 65)
        for bs, peak, t, hours, days in results:
            print(f"{bs:>3} | {peak:>6.2f}GB | {t:>7.3f}s | {bs/t:>5.1f} | {hours:>6.1f}h | {days:>4.1f}d")
        print("=" * 65)
        print(f"\nModel parameters: {params:,}")

        best = max(results, key=lambda r: r[0] / r[2])
        print(f"\nRecommendation: batch_size={best[0]}, "
              f"VRAM {best[1]:.1f}GB/{total_vram:.0f}GB, "
              f"~{best[4]:.1f} days to complete {total_epochs} epochs")
