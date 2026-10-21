"""Inference for the original two-stem Spleeter-style U-Net checkpoint.

The model only estimates the lowest 1024 frequency bins. The unmodelled
high-frequency residual is retained in the final (accompaniment) stem.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .unet import UNet


def load_ckpt(model, ckpt):
    """Load converted weights without silently retaining random parameters."""
    state = model.state_dict()
    unknown = sorted(set(ckpt) - set(state))
    missing = sorted(
        key for key in state if key not in ckpt and not key.endswith("num_batches_tracked")
    )
    if unknown or missing:
        raise ValueError(f"Checkpoint keys do not match the model: missing={missing}, unknown={unknown}")
    for key, value in ckpt.items():
        tensor = torch.as_tensor(value)
        if tensor.shape != state[key].shape:
            raise ValueError(
                f"Checkpoint shape mismatch for {key}: expected {tuple(state[key].shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        state[key] = tensor
    model.load_state_dict(state, strict=True)
    return model


def pad_and_partition(tensor, T):
    """Pad B x C x F x L into segment-major (B * ceil(L/T)) x C x F x T."""
    if tensor.ndim != 4 or tensor.size(-1) == 0:
        raise ValueError("Expected a nonempty B x C x F x L tensor.")
    if not isinstance(T, int) or isinstance(T, bool) or T <= 0:
        raise ValueError("Partition length T must be a positive integer.")
    length = tensor.size(-1)
    padded = F.pad(tensor, (0, math.ceil(length / T) * T - length))
    return torch.cat(torch.split(padded, T, dim=-1), dim=0)


class Estimator(nn.Module):
    """Load a TensorFlow checkpoint and separate 44.1 kHz stereo waveforms.

    ``separate`` accepts a floating-point tensor of shape ``(2, samples)``.
    Use ``.to(device)`` to select the inference device. Waveform resampling and
    file I/O belong to the public wrapper, not this model.
    """

    def __init__(self, num_instrumments, checkpoint_path):
        super().__init__()
        if num_instrumments != 2:
            raise ValueError("This checkpoint interface supports exactly two stems.")
        self.F = 1024
        self.T = 512
        self.win_length = 4096
        self.hop_length = 1024
        self.register_buffer("win", torch.hann_window(self.win_length))

        # TensorFlow is needed for checkpoint conversion, not for importing the
        # inference helpers or testing the DSP path with deterministic networks.
        from .util import tf2pytorch

        ckpts = tf2pytorch(checkpoint_path, num_instrumments)
        self.instruments = nn.ModuleList(load_ckpt(UNet(2), ckpt) for ckpt in ckpts)
        self.eval()

    def compute_stft(self, wav):
        """Return the full complex spectrum and model-band magnitudes."""
        window = self.win.to(device=wav.device, dtype=wav.dtype)
        # Reflection padding cannot represent clips shorter than half a frame.
        # Zero padding makes short, nonempty clips well-defined instead.
        padding = "reflect" if wav.size(-1) > self.win_length // 2 else "constant"
        spectrum = torch.stft(
            wav,
            self.win_length,
            hop_length=self.hop_length,
            window=window,
            center=True,
            pad_mode=padding,
            return_complex=True,
        )
        return spectrum, spectrum[:, : self.F].abs()

    def inverse_stft(self, spectrum, length):
        """Invert a complex spectrum to exactly ``length`` samples."""
        if not torch.is_complex(spectrum):
            raise ValueError("inverse_stft expects a complex spectrum.")
        bins = self.win_length // 2 + 1
        if spectrum.size(-2) > bins:
            raise ValueError("Spectrum has more bins than the configured FFT.")
        if spectrum.size(-2) < bins:
            spectrum = F.pad(spectrum, (0, 0, 0, bins - spectrum.size(-2)))
        return torch.istft(
            spectrum,
            self.win_length,
            hop_length=self.hop_length,
            window=self.win.to(device=spectrum.device, dtype=spectrum.real.dtype),
            center=True,
            length=length,
        )

    @torch.inference_mode()
    def separate(self, wav):
        """Return vocals and accompaniment, with original length and channels.

        U-Nets see one time partition at a time, which bounds activation memory
        for long recordings. The full waveform and STFT still reside in memory.
        The accompaniment is the mixture residual, preserving high frequencies
        and ensuring that both returned stems reconstruct the input mixture.
        """
        if not isinstance(wav, torch.Tensor) or wav.ndim != 2 or wav.size(0) != 2:
            raise ValueError("Expected a stereo tensor with shape (2, samples).")
        if wav.size(1) == 0 or not torch.is_floating_point(wav):
            raise ValueError("Expected a nonempty floating-point waveform.")
        if not torch.isfinite(wav).all():
            raise ValueError("Waveform contains NaN or infinite samples.")
        if len(self.instruments) != 2:
            raise ValueError("Exactly two instrument networks are required.")

        self.eval()
        wav = wav.to(device=self.win.device, dtype=self.win.dtype)
        spectrum, magnitude = self.compute_stft(wav)
        frame_count = magnitude.size(-1)
        vocals = torch.zeros_like(spectrum[:, : self.F])

        for start in range(0, frame_count, self.T):
            stop = min(start + self.T, frame_count)
            chunk = F.pad(magnitude[..., start:stop], (0, self.T - (stop - start)))
            chunk = chunk.unsqueeze(0).transpose(2, 3)  # 1 x 2 x T x F
            estimates = [net(chunk) for net in self.instruments]
            for estimate in estimates:
                if estimate.shape != chunk.shape or not torch.isfinite(estimate).all():
                    raise ValueError("Instrument network returned invalid spectrogram estimates.")
            # U-Net outputs are magnitude estimates. Squaring implements the
            # original generalized Wiener mask; epsilon handles silence.
            powers = [estimate.square() for estimate in estimates]
            denominator = powers[0] + powers[1] + 1e-10
            mask = ((powers[0] + 5e-11) / denominator)[0].transpose(1, 2)
            vocals[..., start:stop] = (
                spectrum[:, : self.F, start:stop] * mask[..., : stop - start]
            )

        vocal_wave = self.inverse_stft(vocals, length=wav.size(-1))
        return [vocal_wave, wav - vocal_wave]
