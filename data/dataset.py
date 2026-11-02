import os
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from .transforms import PairedTransform


class DehazeDataset(Dataset):

    def __init__(self, hazy_dir, gt_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.transform = transform

        valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.filenames = sorted([
            f for f in os.listdir(hazy_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hazy = Image.open(os.path.join(self.hazy_dir, fname)).convert('RGB')

        gt_path = os.path.join(self.gt_dir, fname)
        if not os.path.exists(gt_path):
            base = os.path.splitext(fname)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                candidate = os.path.join(self.gt_dir, base + ext)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break

        gt = Image.open(gt_path).convert('RGB')

        if self.transform is not None:
            hazy, gt = self.transform(hazy, gt)

        return {'hazy': hazy, 'gt': gt, 'filename': fname}


def build_dataloader(config, mode='train'):
    crop_size = config['data'].get('crop_size', 256)
    transform = PairedTransform(crop_size=crop_size, mode=mode)

    key = 'train_datasets' if mode == 'train' else 'val_datasets'
    dataset_configs = config['data'].get(key, [])

    datasets = []
    for dc in dataset_configs:
        ds = DehazeDataset(
            hazy_dir=dc['hazy_dir'],
            gt_dir=dc['gt_dir'],
            transform=transform,
        )
        datasets.append(ds)

    if len(datasets) == 0:
        raise ValueError(f"No datasets configured for mode={mode}")
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    batch_size = config['data'].get('batch_size', 8) if mode == 'train' else 1
    num_workers = config['data'].get('num_workers', 4) if mode == 'train' else 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
    )
