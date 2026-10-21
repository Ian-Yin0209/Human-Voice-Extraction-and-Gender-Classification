"""Compatibility inference for the upstream 128-feature binary voice model.

Architecture and feature convention: x4nth055/gender-recognition-by-voice (MIT).
See THIRD_PARTY_NOTICES.md. Scores refer to the original dataset labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import load_audio, select_segment


def create_legacy_model():
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Input
    except ImportError as exc:
        raise ValueError("Classification requires model dependencies: pip install '.[models]'.") from exc
    return Sequential([
        Input(shape=(128,)),
        Dense(256), Dropout(0.3),
        Dense(256, activation="relu"), Dropout(0.3),
        Dense(128, activation="relu"), Dropout(0.3),
        Dense(128, activation="relu"), Dropout(0.3),
        Dense(64, activation="relu"), Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])


def extract_mel_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise ValueError("Classification requires model dependencies: pip install '.[models]'.") from exc
    mono = samples.mean(axis=1)
    mono = librosa.resample(y=mono, orig_sr=sample_rate, target_sr=22050, res_type="soxr_hq")
    spectrum = librosa.feature.melspectrogram(y=mono, sr=22050, n_mels=128)
    return np.mean(spectrum, axis=1, dtype=np.float64).astype(np.float32).reshape(1, 128)


def scores_from_probability(probability: float) -> dict:
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Classifier returned an invalid score.")
    return {
        "label": "male" if probability > 0.5 else "female",
        "scores": {"male": probability, "female": 1.0 - probability},
        "interpretation": "Legacy binary training-label scores; not a person's gender identity.",
    }


def classify_file(
    input_path: str | Path,
    *,
    weights: str | Path,
    start: float = 0.0,
    duration: float | None = None,
) -> dict:
    weight_path = Path(weights)
    if not weight_path.is_file():
        raise ValueError(f"Classifier weights do not exist: {weight_path}")
    if weight_path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("Legacy classifier weights must be a local .h5 or .hdf5 file.")
    samples, sample_rate = load_audio(input_path)
    samples = select_segment(samples, sample_rate, start, duration)
    features = extract_mel_features(samples, sample_rate)
    if features.shape != (1, 128) or not np.isfinite(features).all():
        raise ValueError("Audio did not produce 128 finite Mel features.")
    model = create_legacy_model()
    try:
        model.load_weights(str(weight_path))
        prediction = np.asarray(model(features, training=False))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load or run the legacy classifier: {exc}") from exc
    if prediction.shape != (1, 1):
        raise ValueError("Classifier must return one binary score.")
    return {"path": str(Path(input_path)), **scores_from_probability(float(prediction[0, 0]))}
