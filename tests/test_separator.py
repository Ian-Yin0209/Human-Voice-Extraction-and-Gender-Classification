"""Deterministic DSP tests; no pretrained model, TensorFlow, or network needed."""

import math
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

if torch is not None:
    from spleeter_style.estimator import Estimator, load_ckpt, pad_and_partition

    class FixedEstimate(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self.calls = []

        def forward(self, value):
            self.calls.append((tuple(value.shape), self.training, torch.is_grad_enabled()))
            return torch.full_like(value, self.scale)

    def make_estimator(scales=(1.0, 1.0)):
        # Bypass checkpoint loading to isolate inference/DSP from model quality.
        estimator = Estimator.__new__(Estimator)
        nn.Module.__init__(estimator)
        estimator.F = 8
        estimator.T = 4
        estimator.win_length = 16
        estimator.hop_length = 4
        estimator.register_buffer("win", torch.hann_window(16))
        estimator.instruments = nn.ModuleList(FixedEstimate(scale) for scale in scales)
        return estimator


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed.")
class SeparatorTests(unittest.TestCase):
    def test_partition_preserves_samples_and_pads_only_the_tail(self):
        value = torch.arange(2 * 2 * 3 * 5).reshape(2, 2, 3, 5)
        actual = pad_and_partition(value, 4)
        self.assertEqual(tuple(actual.shape), (4, 2, 3, 4))
        torch.testing.assert_close(actual[:2], value[..., :4])
        torch.testing.assert_close(actual[2:, ..., 0], value[..., 4])
        self.assertEqual(torch.count_nonzero(actual[2:, ..., 1:]).item(), 0)

    def test_invalid_partition_arguments_fail_clearly(self):
        for value, length in ((torch.empty(1, 2, 3, 0), 4), (torch.ones(2, 3), 4),
                              (torch.ones(1, 2, 3, 4), 0)):
            with self.subTest(shape=tuple(value.shape), length=length):
                with self.assertRaises(ValueError):
                    pad_and_partition(value, length)

    def test_complex_stft_roundtrip_preserves_short_and_odd_lengths(self):
        estimator = make_estimator()
        generator = torch.Generator().manual_seed(7)
        for length in (1, 7, 17, 103):
            with self.subTest(length=length):
                waveform = torch.randn(2, length, generator=generator)
                spectrum, magnitude = estimator.compute_stft(waveform)
                self.assertTrue(torch.is_complex(spectrum))
                self.assertEqual(spectrum.shape[1], 9)
                torch.testing.assert_close(magnitude, spectrum[:, :8].abs())
                result = estimator.inverse_stft(spectrum, length)
                self.assertEqual(tuple(result.shape), (2, length))
                torch.testing.assert_close(result, waveform, atol=1e-6, rtol=1e-5)

    def test_stems_preserve_channels_length_and_mixture(self):
        estimator = make_estimator()
        waveform = torch.stack((torch.linspace(-0.5, 0.5, 103), torch.linspace(0.2, -0.1, 103)))
        vocals, accompaniment = estimator.separate(waveform)
        self.assertEqual(tuple(vocals.shape), (2, 103))
        self.assertEqual(tuple(accompaniment.shape), (2, 103))
        torch.testing.assert_close(vocals + accompaniment, waveform)
        self.assertFalse(torch.equal(vocals[0], vocals[1]))

    def test_mask_uses_both_models_and_keeps_high_frequency_residual(self):
        estimator = make_estimator(scales=(1.0, 2.0))
        waveform = torch.randn(2, 103, generator=torch.Generator().manual_seed(19))
        spectrum, _ = estimator.compute_stft(waveform)
        # Squared estimates 1 and 4 assign one fifth of model-band energy to vocals.
        expected = estimator.inverse_stft(spectrum[:, :8] * 0.2, waveform.shape[-1])
        vocals, accompaniment = estimator.separate(waveform)
        torch.testing.assert_close(vocals, expected)
        torch.testing.assert_close(accompaniment, waveform - expected)

    def test_inference_uses_one_partition_without_gradients(self):
        estimator = make_estimator()
        waveform = torch.ones(2, 103, requires_grad=True)
        stems = estimator.separate(waveform)
        expected_calls = math.ceil((103 // estimator.hop_length + 1) / estimator.T)
        for network in estimator.instruments:
            self.assertEqual(len(network.calls), expected_calls)
            for shape, training, grad_enabled in network.calls:
                self.assertEqual(shape, (1, 2, 4, 8))
                self.assertFalse(training)
                self.assertFalse(grad_enabled)
        self.assertTrue(all(not stem.requires_grad for stem in stems))

    def test_silence_is_finite(self):
        estimator = make_estimator(scales=(0.0, 0.0))
        for result in estimator.separate(torch.zeros(2, 3)):
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(torch.count_nonzero(result).item(), 0)

    def test_invalid_waveforms_are_rejected(self):
        estimator = make_estimator()
        values = (torch.ones(1, 10), torch.empty(2, 0), torch.ones(2, 3, dtype=torch.int16),
                  torch.full((2, 3), float("nan")), torch.full((2, 3), float("inf")))
        for value in values:
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype):
                with self.assertRaises(ValueError):
                    estimator.separate(value)

    def test_window_is_registered_and_moves_with_dtype(self):
        estimator = make_estimator().to(dtype=torch.float64)
        self.assertIn("win", dict(estimator.named_buffers()))
        waveform = torch.ones(2, 17, dtype=torch.float64)
        for stem in estimator.separate(waveform):
            self.assertEqual(stem.dtype, torch.float64)

    def test_bad_network_output_is_rejected(self):
        class BadEstimate(nn.Module):
            def forward(self, value):
                return value[..., :1]

        estimator = make_estimator()
        estimator.instruments[0] = BadEstimate()
        with self.assertRaisesRegex(ValueError, "invalid spectrogram"):
            estimator.separate(torch.ones(2, 17))

    def test_checkpoint_shape_or_missing_parameters_fail(self):
        model = nn.Linear(2, 1)
        valid = {key: value.clone() for key, value in model.state_dict().items()}
        load_ckpt(model, valid)
        with self.assertRaisesRegex(ValueError, "missing"):
            load_ckpt(model, {"weight": valid["weight"]})
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            load_ckpt(model, {"weight": torch.ones(1, 3), "bias": valid["bias"]})


if __name__ == "__main__":
    unittest.main()
