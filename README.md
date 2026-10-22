# Voice Extraction Toolkit · MIMO Stem Lab

[![CI](https://github.com/Ian-Yin0209/Human-Voice-Extraction-and-Gender-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Ian-Yin0209/Human-Voice-Extraction-and-Gender-Classification/actions/workflows/ci.yml)
![Version 0.3.0](https://img.shields.io/badge/version-0.3.0-176B56)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB)
[![Code MIT](https://img.shields.io/badge/code-MIT-176B56)](LICENSE)

**One song. Four stems. A local workflow you can inspect.**

Turn a music file into **vocals, bass, drums and other**, plus an accompaniment mix and a JSON processing report. Version 0.3.0 adds a local backend for **Sony Research's September 2026 MIMO-SCNet** model, alongside audio inspection, preparation and reference-based evaluation.

[Download source ZIP](https://github.com/Ian-Yin0209/Human-Voice-Extraction-and-Gender-Classification/archive/refs/heads/master.zip) · [Research & validation](docs/RESEARCH.md) · [Contributing](CONTRIBUTING.md)

## Start in three commands

Download and unzip the source, then open a terminal in the folder containing `pyproject.toml`. Use Python 3.10–3.12 and put your input at `song.wav`.

```sh
python -m pip install '.[modern]'
voice-extract fetch-model mimo-scnet --models-dir models
voice-extract separate song.wav stems --backend mimo --weights models/mimo-scnet-small-v1.pth
```

This installs the local checkout; the project is **not published on PyPI**. The modern backend uses PyTorch and the base audio libraries. It does not require TensorFlow, torchaudio or FFmpeg.

The model download is a **39.4 MB ZIP**; its extracted weights are **42.5 MB**. PyTorch and other dependencies are additional downloads. These numbers are not the complete installation size.

The `stems` directory must be new. MIMO separation refuses existing output directories and has no `--force` override.

## What you get

```text
stems/
├── vocals.wav
├── bass.wav
├── drums.wav
├── other.wav
├── accompaniment.wav
└── report.json
```

Outputs are **44.1 kHz stereo floating-point WAV**. Mono input is duplicated to stereo. The accompaniment is the sum of bass, drums and other; `report.json` records the model identity, processing settings and mixture-reconstruction error. Add `--json` for a machine-readable console result too.

CPU is the default. `--device cuda` is available with a compatible CUDA-enabled PyTorch installation; CUDA has not been validated in this release.

| MIMO option | Default | Accepted values |
| --- | --- | --- |
| `--chunk-seconds` | `6` | `0.1` to `30` seconds |
| `--overlap` | `0.25` | Fraction `0` to `0.75` |
| `--iterations` | `2` | `1` or `2` refinement steps |

Chunking limits model activation sizes. It does **not** make the whole workflow stream from disk: the input and accumulated output arrays still reside in RAM.

## Explicit downloads, offline inference

Inspect the pinned catalog without networking or importing an ML runtime:

```sh
voice-extract models --json
```

`fetch-model` verifies both the official ZIP and extracted weights against pinned sizes and SHA-256 hashes. To install from an already downloaded [official ZIP](https://github.com/SonyResearch/mimo-audio-separation/releases/download/v1.0.0/mimo-scnet_small.zip), use:

```sh
voice-extract fetch-model mimo-scnet --models-dir models --archive mimo-scnet_small.zip
```

That archive-based installation is offline. **`separate` never downloads models or makes a network request.** Weights are obtained separately and are not bundled with this repository.

## Inspect, prepare and compare

These commands need only the base installation (`python -m pip install .`), with no ML runtime:

```sh
voice-extract inspect song.wav --json
voice-extract prepare song.wav clip.wav --start 10 --duration 15 --sample-rate 44100 --mono
voice-extract evaluate reference-vocals.wav stems/vocals.wav --json
```

`prepare` writes a floating-point WAV and resamples to 44.1 kHz by default. Use `--normalize` to normalize a non-silent peak to 0.99. Existing files require its explicit `--force` option; input and output must be different files.

`evaluate` compares a supplied reference with an estimate using zero-mean, per-channel **SI-SDR**. Sample rates, frame counts and channel counts must already match: it does not align, crop or resample them. Undefined or unbounded cases produce JSON `null` with a reason; if any channel is undefined, the overall score is also `null`. **SI-SDR is not the museval SDR used in the upstream MUSDB18-HQ results.**

## What has actually run

The published [Windows CPU validation record](docs/validation/2026-10-22-mimo-cpu.json) contains two real-weight checks: a two-second, 16 kHz **stereo** tone became 88,200 stereo frames at 44.1 kHz through five short chunks; a one-frame, 8 kHz **mono** input became six stereo frames.

For the two-second case, model loading, inference and overlap-add took **1.7684 seconds**, excluding audio I/O and weight verification. Maximum stem-sum reconstruction error was **6.71 × 10⁻⁸**.

This is a generated-signal execution check, not a music-quality benchmark or a speed promise. It does not establish current state-of-the-art separation quality. See [Research & validation](docs/RESEARCH.md) for settings, upstream comparisons and remaining limits.

## Research and licensing

The implementation builds on [Sony's MIMO research](https://arxiv.org/abs/2609.07226), with a small inference adapter and preserved third-party notices. [Code is MIT licensed](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md). **That code license does not establish the model-weight license.** No separate explicit weight license was found in the inspected upstream repository, release or checkpoint archive; review the upstream terms before redistribution or commercial use.

## Legacy adapters

<details>
<summary>Spleeter and the historical voice-label classifier</summary>

The earlier adapters remain available with `python -m pip install '.[models]'` and separately obtained weights:

```sh
voice-extract separate song.wav legacy-stems --backend spleeter --checkpoint models/2stems/model
voice-extract classify song.wav --weights models/model.h5 --json
```

Sources: [Deezer Spleeter](https://github.com/deezer/spleeter), [Tuan Nguyen's PyTorch port](https://github.com/tuan3w/spleeter-pytorch), and [x4nth055's classifier](https://github.com/x4nth055/gender-recognition-by-voice). The classifier reproduces upstream binary dataset labels, not a person's gender or identity. Historical origins and notices are retained in [Provenance](docs/PROVENANCE.md) and the [model card](docs/MODEL_CARD.md).

</details>

## Development

```sh
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

Both `voice-extract` and `python -m voice_extraction` are supported. [Contributing](CONTRIBUTING.md) · [Roadmap](docs/ROADMAP.md) · [Research](docs/RESEARCH.md)

## 中文简述

一首歌，本地拆成四条音轨：人声、贝斯、鼓与其他乐器，另外输出伴奏和 JSON 报告。0.3.0 接入 Sony 2026 年 MIMO-SCNet 研究模型，CPU 默认可用；模型下载显式执行并校验哈希，分离过程不联网。现代路径不需要 TensorFlow、torchaudio 或 FFmpeg，基础音频工具也不依赖机器学习运行时。

39.4 MB 只是模型 ZIP，另需安装 PyTorch。当前实测证明真实权重能在 Windows CPU 上完成短音频处理，尚不能证明音乐分离质量、GPU 性能或全局领先。模型权重许可与代码 MIT 许可分别说明。
