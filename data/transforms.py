import random
import torch
import torchvision.transforms.functional as TF


class PairedTransform:
    """Apply identical spatial transforms to hazy-GT image pairs."""

    def __init__(self, crop_size=256, mode='train'):
        self.crop_size = crop_size
        self.mode = mode

    def __call__(self, hazy, gt):
        hazy = TF.to_tensor(hazy)
        gt = TF.to_tensor(gt)

        if self.mode == 'train':
            _, h, w = hazy.shape
            if h >= self.crop_size and w >= self.crop_size:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
                hazy = TF.crop(hazy, top, left, self.crop_size, self.crop_size)
                gt = TF.crop(gt, top, left, self.crop_size, self.crop_size)
            else:
                hazy = TF.resize(hazy, [self.crop_size, self.crop_size])
                gt = TF.resize(gt, [self.crop_size, self.crop_size])

            if random.random() > 0.5:
                hazy = TF.hflip(hazy)
                gt = TF.hflip(gt)

            if random.random() > 0.5:
                hazy = TF.vflip(hazy)
                gt = TF.vflip(gt)

            angle = random.choice([0, 90, 180, 270])
            if angle > 0:
                hazy = TF.rotate(hazy, angle)
                gt = TF.rotate(gt, angle)

        else:
            _, h, w = hazy.shape
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16
            if new_h != h or new_w != w:
                hazy = torch.nn.functional.pad(hazy, (0, new_w - w, 0, new_h - h), mode='reflect')
                gt = torch.nn.functional.pad(gt, (0, new_w - w, 0, new_h - h), mode='reflect')

        return hazy, gt
