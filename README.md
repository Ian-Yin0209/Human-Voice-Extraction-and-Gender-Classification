# Voice Extraction Toolkit

A local command-line toolkit for inspecting and preparing audio, with optional Spleeter-style vocal separation and a legacy voice-label classifier.

The audio utilities work without model weights or a microphone. Model commands use explicitly supplied local weights and do not download models automatically. See [Provenance](docs/PROVENANCE.md) for the project's history and upstream components.

## Install

Use Python 3.10–3.12:

```sh
python -m pip install .
voice-extract --help
```

These commands install the local checkout; this project is not published on PyPI. The base installation uses NumPy, SciPy and SoundFile. For optional model commands:

```sh
python -m pip install '.[models]'
```

This adds PyTorch, TensorFlow, librosa and h5py. Available package builds depend on your Python version, operating system and accelerator. You can also run `python -m voice_extraction`.

## Inspect and prepare audio

Use a mono or stereo audio file supported by SoundFile, such as WAV or FLAC:

```sh
voice-extract inspect input.wav
voice-extract inspect input.wav --json
voice-extract prepare input.wav prepared.wav --start 2 --duration 10 --sample-rate 44100 --mono --normalize
```

`prepare` writes a floating-point WAV file. It resamples to 44.1 kHz by default; choose another rate with `--sample-rate`. Without crop options it uses the whole input. Channel mixing and peak normalization to 0.99 are opt-in. Existing outputs are rejected unless `--force` is supplied, and the input file cannot be used as the output.

For a reproducible demonstration without downloading audio, generate an original two-second stereo tone:

```sh
python examples/generate_demo.py demo.wav
voice-extract inspect demo.wav --json
voice-extract prepare demo.wav prepared.wav --start 0.25 --duration 1 --sample-rate 22050
```

This signal checks file processing, not speech recognition or separation quality.

## Optional: separate vocals

Obtain the original **2-stem** checkpoint from [Deezer's Spleeter v1.4.0 release](https://github.com/deezer/spleeter/releases/tag/v1.4.0), then extract `2stems.tar.gz` into a local model directory. Point `--checkpoint` at the shared checkpoint **prefix**, not at the `.index` or `.data-*` file:

```sh
voice-extract separate input.wav stems --checkpoint models/2stems/model --device cpu
```

The command resamples to 44.1 kHz, duplicates mono input to stereo when needed, and writes `vocals.wav` and `accompaniment.wav`. Use `--device cuda` with a working CUDA-enabled PyTorch installation. Existing outputs require `--force` to replace.

The separator is based on an older PyTorch port of Spleeter. Pretrained-weight loading and inference have passed a Windows CPU smoke check on generated audio; this does not establish separation quality. See [validation](docs/VALIDATION.md) and the [roadmap](docs/ROADMAP.md).

## Optional: run the legacy classifier

The original weights are available from upstream [gender-recognition-by-voice](https://github.com/x4nth055/gender-recognition-by-voice/blob/master/results/model.h5). Obtain them separately and supply a local file:

```sh
voice-extract classify input.wav --weights models/model.h5 --start 0 --duration 5 --json
```

This reproduces an upstream model's binary `male` / `female` dataset labels. They are not a determination of a person's gender or identity. The model uses average Mel-spectrum features and a fully connected neural network. It is retained for research compatibility; this repository does not claim to have trained the model or independently measured its accuracy. See the [model card](docs/MODEL_CARD.md).

## Development

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

Optional model smoke check, using separately obtained local weights:

```sh
python examples/model_smoke.py --checkpoint models/2stems/model --weights models/model.h5 --output-dir smoke-results
```

Use a new output directory. The script generates its own stereo tone, checks both model adapters, and records dependency versions and weight SHA-256 checksums. It does not measure model accuracy or source-separation quality.

Supported entry points are `voice-extract` and `python -m voice_extraction`. The historical training and recording scripts are not included in the maintained tree; `legacy/README.md` links to their original history and upstream sources.

- [Contributing](CONTRIBUTING.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Provenance](docs/PROVENANCE.md)
- [Model card](docs/MODEL_CARD.md)
- [Roadmap](docs/ROADMAP.md)

## 中文概览

这是一个本地音频处理命令行工具：基础安装可查看音频信息、裁剪、重采样、转单声道及归一化；安装可选模型依赖后，可使用自己提供的本地权重进行人声分离和历史分类模型实验。工具不会自动下载权重，也不默认调用麦克风。已有输出默认不会被覆盖。

历史来源与第三方署名见独立文档。分类结果只表示上游训练集的二元标签，不能用来确定个人身份或性别。预训练模型的实际效果仍需独立验证。
