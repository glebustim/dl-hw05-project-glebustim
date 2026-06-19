import math

import torch
import torch.nn as nn

from src.model.drunet import DRUNet


def _next_power_of_2(n):
    return 1 << (max(1, int(n)) - 1).bit_length()


class LeADMM(nn.Module):
    def __init__(self, num_iters=20, trainable=True, mu_1_init=1e-4, mu_2_init=1e-4, mu_3_init=1e-4, tau_init=2e-4, pad_factor=2.0, eps=1e-12, normalize_output=True):
        super().__init__()
        self.num_iters = num_iters
        self.pad_factor = pad_factor
        self.eps = eps
        self.normalize_output = normalize_output

        log_mu_1 = torch.log(torch.as_tensor(mu_1_init, dtype=torch.float32).repeat(num_iters))
        log_mu_2 = torch.log(torch.as_tensor(mu_2_init, dtype=torch.float32).repeat(num_iters))
        log_mu_3 = torch.log(torch.as_tensor(mu_3_init, dtype=torch.float32).repeat(num_iters))
        log_tau = torch.log(torch.as_tensor(tau_init, dtype=torch.float32).repeat(num_iters))

        if trainable:
            self.log_mu_1 = nn.Parameter(log_mu_1)
            self.log_mu_2 = nn.Parameter(log_mu_2)
            self.log_mu_3 = nn.Parameter(log_mu_3)
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_mu_1", log_mu_1)
            self.register_buffer("log_mu_2", log_mu_2)
            self.register_buffer("log_mu_3", log_mu_3)
            self.register_buffer("log_tau", log_tau)

    def forward(self, y, psf):
        batch_size, channels, height, width = y.shape
        padded_height = _next_power_of_2(math.ceil(self.pad_factor * height))
        padded_width = _next_power_of_2(math.ceil(self.pad_factor * width))
        psf = self._prepare_psf(psf, batch_size, channels, y.device, y.dtype)
        psf_fft = self._psf_fft(psf, padded_height, padded_width)
        psf_fft_conj = torch.conj(psf_fft)
        psf_fft_abs2 = psf_fft.real.square() + psf_fft.imag.square()
        diff_power = self._diff_power_spectrum(padded_height, padded_width, y.device, y.dtype)

        x = torch.zeros(batch_size, channels, height, width, device=y.device, dtype=y.dtype)
        alpha_1 = torch.zeros_like(x)
        alpha_2 = torch.zeros(batch_size, channels, 2, height, width, device=y.device, dtype=y.dtype)
        alpha_3 = torch.zeros_like(x)

        for i in range(self.num_iters):
            mu_1 = torch.exp(self.log_mu_1[i]).to(device=y.device, dtype=y.dtype)
            mu_2 = torch.exp(self.log_mu_2[i]).to(device=y.device, dtype=y.dtype)
            mu_3 = torch.exp(self.log_mu_3[i]).to(device=y.device, dtype=y.dtype)
            tau = torch.exp(self.log_tau[i]).to(device=y.device, dtype=y.dtype)

            u = self._soft_threshold(self._center_crop(self._psi(self._center_pad(x, padded_height, padded_width)), height, width) + alpha_2 / mu_2, tau)

            hx = self._apply_h(x, psf_fft, padded_height, padded_width)
            v = (alpha_1 + mu_1 * hx + y) / (1.0 + mu_1)

            w = torch.relu(alpha_3 / mu_3 + x)

            r = (self._center_pad(mu_3 * w - alpha_3, padded_height, padded_width) + self._psi_t(self._center_pad(mu_2 * u - alpha_2, padded_height, padded_width)) + self._center_pad(self._apply_ht(mu_1 * v - alpha_1, psf_fft_conj, padded_height, padded_width), padded_height, padded_width))
            denom = mu_1 * psf_fft_abs2 + mu_2 * diff_power + mu_3 + self.eps
            x_padded = torch.fft.irfft2(torch.fft.rfft2(r) / denom, s=(padded_height, padded_width))
            x = self._center_crop(x_padded, height, width)

            hx = self._apply_h(x, psf_fft, padded_height, padded_width)
            alpha_1 = alpha_1 + mu_1 * (hx - v)
            alpha_2 = alpha_2 + mu_2 * (self._center_crop(self._psi(self._center_pad(x, padded_height, padded_width)), height, width) - u)
            alpha_3 = alpha_3 + mu_3 * (x - w)

        if self.normalize_output:
            x = self._normalize_01(x, self.eps)
        return x

    @staticmethod
    def _soft_threshold(x, threshold):
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

    @classmethod
    def _psi(cls, x):
        return torch.stack((cls._diff_h(x), cls._diff_w(x)), dim=2)

    @classmethod
    def _psi_t(cls, x):
        return cls._diff_h_t(x[:, :, 0]) + cls._diff_w_t(x[:, :, 1])

    @staticmethod
    def _diff_h(x):
        return x - torch.roll(x, shifts=1, dims=-2)

    @staticmethod
    def _diff_w(x):
        return x - torch.roll(x, shifts=1, dims=-1)

    @staticmethod
    def _diff_h_t(x):
        return x - torch.roll(x, shifts=-1, dims=-2)

    @staticmethod
    def _diff_w_t(x):
        return x - torch.roll(x, shifts=-1, dims=-1)

    @staticmethod
    def _apply_h(x, psf_fft, padded_height, padded_width):
        height, width = x.shape[-2:]
        x_padded = LeADMM._center_pad(x, padded_height, padded_width)
        hx = torch.fft.irfft2(torch.fft.rfft2(x_padded) * psf_fft, s=(padded_height, padded_width))
        return LeADMM._center_crop(hx, height, width)

    @staticmethod
    def _apply_ht(x, psf_fft_conj, padded_height, padded_width):
        height, width = x.shape[-2:]
        x_padded = LeADMM._center_pad(x, padded_height, padded_width)
        htx = torch.fft.irfft2(torch.fft.rfft2(x_padded) * psf_fft_conj, s=(padded_height, padded_width))
        return LeADMM._center_crop(htx, height, width)

    @staticmethod
    def _center_pad(x, padded_height, padded_width):
        height, width = x.shape[-2:]
        pad_top = (padded_height - height) // 2
        pad_left = (padded_width - width) // 2
        x_padded = x.new_zeros(*x.shape[:-2], padded_height, padded_width)
        x_padded[..., pad_top:pad_top + height, pad_left:pad_left + width] = x
        return x_padded

    @staticmethod
    def _center_crop(x, height, width):
        pad_top = (x.shape[-2] - height) // 2
        pad_left = (x.shape[-1] - width) // 2
        return x[..., pad_top:pad_top + height, pad_left:pad_left + width]

    @staticmethod
    def _normalize_01(x, eps=1e-6):
        x_min = x.amin(dim=(-3, -2, -1), keepdim=True)
        x_max = x.amax(dim=(-3, -2, -1), keepdim=True)
        x = (x - x_min) / (x_max - x_min).clamp_min(eps)
        return x.clamp(eps, 1.0 - eps)

    @staticmethod
    def _prepare_psf(psf, batch_size, channels, device, dtype):
        psf = psf.to(device=device, dtype=dtype)
        if psf.dim() == 2:
            psf = psf.unsqueeze(0).unsqueeze(0)
        elif psf.dim() == 3:
            psf = psf.unsqueeze(0)
        if psf.shape[1] == 1 and channels != 1:
            psf = psf.expand(psf.shape[0], channels, psf.shape[-2], psf.shape[-1])
        if psf.shape[0] == 1 and batch_size != 1:
            psf = psf.expand(batch_size, channels, psf.shape[-2], psf.shape[-1])
        normalizer = psf.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        return psf / normalizer

    @staticmethod
    def _psf_fft(psf, padded_height, padded_width):
        psf_height, psf_width = psf.shape[-2:]
        psf_padded = psf.new_zeros(psf.shape[0], psf.shape[1], padded_height, padded_width)
        start_h = (padded_height - psf_height) // 2
        start_w = (padded_width - psf_width) // 2
        psf_padded[..., start_h:start_h + psf_height, start_w:start_w + psf_width] = psf
        psf_padded = torch.fft.ifftshift(psf_padded, dim=(-2, -1))
        return torch.fft.rfft2(psf_padded)

    @staticmethod
    def _diff_power_spectrum(padded_height, padded_width, device, dtype):
        freq_h = (2 * torch.pi * torch.arange(padded_height, device=device, dtype=dtype) / padded_height)
        freq_w = (2 * torch.pi * torch.arange(padded_width // 2 + 1, device=device, dtype=dtype) / padded_width)
        power_h = 2.0 - 2.0 * torch.cos(freq_h)
        power_w = 2.0 - 2.0 * torch.cos(freq_w)
        return (power_h[:, None] + power_w[None, :]).unsqueeze(0).unsqueeze(0)

class ModularLeADMM(nn.Module):
    def __init__(self, num_iters=5, pre_channels=None, post_channels=None, trainable=True, mu_1_init=1e-4, mu_2_init=1e-4, mu_3_init=1e-4, tau_init=2e-4, pad_factor=2.0, image_channels=3, normalize_output=True):
        super().__init__()

        self.pre_processor = self._make_processor(pre_channels, image_channels)
        self.leadmm = LeADMM(num_iters=num_iters, trainable=trainable, mu_1_init=mu_1_init, mu_2_init=mu_2_init, mu_3_init=mu_3_init, tau_init=tau_init, pad_factor=pad_factor, normalize_output=normalize_output)
        self.post_processor = self._make_processor(post_channels, image_channels)
        self.normalize_output = normalize_output

    def forward(self, y, psf):
        if self.pre_processor is not None:
            y = self.pre_processor(y)

        x = self.leadmm(y, psf)

        if self.post_processor is not None:
            x = self.post_processor(x)
            if self.normalize_output:
                x = LeADMM._normalize_01(x)

        return x

    @staticmethod
    def _make_processor(channels, image_channels):
        if channels is None:
            return None
        return DRUNet(in_channels=image_channels, out_channels=image_channels, channels=list(channels))
