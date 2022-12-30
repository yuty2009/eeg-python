
import numpy as np
from scipy import signal
from copy import deepcopy
import torch
import torch.nn as nn


""" We assume a (timepoints x channels) or (T, C) EEG epoch as input here """

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, epoch, target=None):
        for t in self.transforms:
            if target is not None:
                epoch, target = t(epoch, target)
            else:
                epoch = t(epoch, None)
        if target is not None:
            return epoch, target
        return epoch


class ToTensor(nn.Module):
    """ 
    Turn a (timepoints x channels) or (T, C) epoch into 
    a (depth x timepoints x channels) or (D, T, C) image for torch.nn.Convnd
    """
    def forward(self, epoch, target=None):
        if isinstance(epoch, np.ndarray):
            epoch = torch.tensor(epoch[None, :, :], dtype=torch.float32)
        elif isinstance(epoch, torch.Tensor):
            epoch = epoch[None, :, :]
        if target is not None:
            return epoch, torch.tensor(target, dtype=torch.long)
        return epoch


class TemporalGaussNoise(nn.Module):
    """Add temporal Gaussian noise to the EEG epoch
    """
    def __init__(self, snr = 0.2):
        super().__init__()
        self.snr = snr

    def forward(self, epoch, target=None):
        signal_std = torch.std(epoch, unbiased=True)
        noise_std = signal_std * self.snr
        noise = torch.randn(epoch.shape) * noise_std
        epoch = epoch + noise
        if target is not None:
            return epoch, target
        return epoch


class TemporalFiltering(nn.Module):
    """Apply temporal filtering on the EEG epoch
    For digital filters, `fpass` are in the same units as `fs`.  By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. 
    """
    def __init__(self, fpass = 0.2, order = 3):
        super().__init__()
        self.filter = signal.butter(order, fpass, btype='low')

    def forward(self, epoch, target=None):
        epoch = signal.lfilter(self.filter[0], self.filter[1], epoch, axis=-2)
        if target is not None:
            return epoch, target
        return epoch


class RandomTemporalShift(nn.Module):
    """Shift the EEG epoch along with the temporal dimention
    """
    def __init__(self, ratio = 0.25):
        super().__init__()
        self.ratio = ratio

    def forward(self, epoch, target=None):
        len_t = epoch.size(-2)
        len_shift = int(len_t * self.ratio)
        shift = torch.randint(-len_shift, len_shift+1, (1,))[0]
        # print(shift)
        epoch_new = deepcopy(epoch)
        if shift != 0:
            epoch_new[..., shift:, :] = epoch[..., :-shift, :]
            epoch_new[..., :shift, :] = epoch[..., -shift:, :]
        if target is not None:
            return epoch, target
        return epoch


class RandomTemporalMask(nn.Module):
    """Mask part of the EEG epoch along with the temporal dimention
    """
    def __init__(self, ratio = 0.1, maskv = 0.0):
        super().__init__()
        self.ratio = ratio
        self.maskv = maskv

    def forward(self, epoch, target=None):
        len_t = epoch.size(-2)
        len_mask = int(len_t * self.ratio)
        mask_start = torch.randint(0, len_t-len_mask+1, (1,))[0]
        # print(mask_start)
        epoch[..., mask_start:mask_start+len_mask, :] = self.maskv
        if target is not None:
            return epoch, target
        return epoch


class RandomTemporalFlip(nn.Module):
    """Flip the temporal infomation of the given epoch randomly 
    with a given probability. If the image is torch Tensor, it is expected
    to have [..., T, C] shape, where ... means an arbitrary number of leading
    dimensions
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, epoch, target=None):
        if torch.rand(1) < self.p:
            epoch[..., :, :] = epoch[..., -1::-1, :]
        if target is not None:
            return epoch, target
        return epoch


if __name__ == '__main__':

    aa = torch.arange(1, 101, 1).view(5, 20).transpose(1, 0)
    print(aa)
    # tf1 = RandomTemporalShift()
    tf1 = RandomTemporalMask()
    bb, _ = tf1(aa)
    print(bb)
