"""File-based, zero-mean SI-SDR without alignment or signal modification.

This is a scale-invariant comparison against a supplied reference, not the
filter-based SDR used in some music-separation benchmarks. Each channel is
evaluated independently. The overall result is the arithmetic mean of channel
scores in dB only when every channel has a finite score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import load_audio


def _channel_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> tuple[float | None, str | None]:
    """Project the centered estimate onto the centered reference in float64.

    Exact zero energies are reported explicitly instead of adding an epsilon
    that would assign arbitrary finite scores to silence or perfect estimates.
    Subtracting a channel mean is part of the metric; no alignment is performed.
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    reference_energy = float(np.dot(reference, reference))
    estimate_energy = float(np.dot(estimate, estimate))
    if reference_energy == 0.0:
        return None, "reference_zero_variance"
    if estimate_energy == 0.0:
        return None, "estimate_zero_variance"

    scale = float(np.dot(estimate, reference)) / reference_energy
    target = scale * reference
    error = estimate - target
    target_energy = float(np.dot(target, target))
    error_energy = float(np.dot(error, error))

    if target_energy == 0.0:
        return None, "orthogonal_estimate"
    if error_energy == 0.0:
        return None, "perfect_reconstruction"

    # Taking the difference of logarithms avoids overflowing an energy ratio.
    score = 10.0 * (float(np.log10(target_energy)) - float(np.log10(error_energy)))
    return score, None


def evaluate_files(reference: str | Path, estimate: str | Path) -> dict:
    """Compare equally sampled mono/stereo files and return JSON-safe SI-SDR.

    Rates, channels and frame counts must already match. In particular, this
    function does not resample, trim, normalize, shift or otherwise align files.
    A zero-variance reference (silence or constant DC), zero-variance estimate,
    exact orthogonality, or exact scale-perfect reconstruction has no finite dB
    score; its value is None with a machine-readable reason. If any channel is
    undefined, the overall score is also None rather than discarding it.
    """
    reference_audio, reference_rate = load_audio(reference)
    estimate_audio, estimate_rate = load_audio(estimate)
    if reference_rate != estimate_rate:
        raise ValueError(
            "Evaluation requires the same sample rate; "
            f"reference is {reference_rate} Hz and estimate is {estimate_rate} Hz. "
            "No automatic resampling is performed."
        )
    if reference_audio.shape != estimate_audio.shape:
        raise ValueError(
            "Evaluation requires the same frame count and channel count; "
            f"reference shape is {reference_audio.shape} and estimate shape is {estimate_audio.shape}. "
            "No automatic alignment or trimming is performed."
        )

    channel_results = []
    for index in range(reference_audio.shape[1]):
        score, reason = _channel_si_sdr(reference_audio[:, index], estimate_audio[:, index])
        channel_results.append({"channel": index + 1, "si_sdr_db": score, "reason": reason})

    all_finite = all(result["si_sdr_db"] is not None for result in channel_results)
    overall_score = (
        float(np.mean([result["si_sdr_db"] for result in channel_results]))
        if all_finite else None
    )
    return {
        "metric": "SI-SDR",
        "reference": str(Path(reference)),
        "estimate": str(Path(estimate)),
        "sample_rate": reference_rate,
        "frames": int(reference_audio.shape[0]),
        "channels": int(reference_audio.shape[1]),
        "zero_mean": True,
        "alignment": "none",
        "aggregation": "arithmetic_mean_of_channel_db",
        "undefined_policy": "overall_null_if_any_channel_undefined",
        "si_sdr_db": overall_score,
        "reason": None if all_finite else "undefined_channel",
        "per_channel": channel_results,
    }
