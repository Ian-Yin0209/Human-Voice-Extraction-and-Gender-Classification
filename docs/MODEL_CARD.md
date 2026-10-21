# Model card

Optional adapters use inherited pretrained models. Base audio utilities do not use a learned model. Weights must be supplied as explicit local paths; they are not bundled or downloaded automatically.

## Vocal separation

| Item | Description |
| --- | --- |
| Source | [Deezer Spleeter](https://github.com/deezer/spleeter), through the [tuan3w PyTorch port](https://github.com/tuan3w/spleeter-pytorch). |
| Architecture | Two-stem Spleeter-style U-Net masks over a short-time Fourier representation. |
| Checkpoint | TensorFlow checkpoint prefix for the original 2-stem model. |
| Audio handling | 44.1 kHz stereo; mono input is duplicated when needed. |
| Output | `vocals.wav` and `accompaniment.wav`. |
| Intended use | Local vocal/accompaniment separation experiments. |
| Evaluation | Pretrained-weight loading and synthetic-audio inference passed on Windows CPU; no independent quality benchmark. See [validation](VALIDATION.md). |

The model was designed for music source separation. A vocal stem does not guarantee that all speech is retained, all background sound is removed, or individual speakers are separated. The toolkit makes no denoising, transcription or speaker-identification performance claim.

Get `2stems.tar.gz` from [Deezer's v1.4.0 release](https://github.com/deezer/spleeter/releases/tag/v1.4.0). Extract it and pass the path ending in `model`, alongside the `.index` and `.data-*` files. The release contains upstream checksum information. Review source terms separately; this repository's license does not grant rights to external checkpoints or input audio.

## Legacy binary voice-label classifier

| Item | Description |
| --- | --- |
| Source | [x4nth055/gender-recognition-by-voice](https://github.com/x4nth055/gender-recognition-by-voice). |
| Weights | Upstream [`results/model.h5`](https://github.com/x4nth055/gender-recognition-by-voice/blob/master/results/model.h5); the formerly bundled copy has the same Git blob ID. |
| Features | Mean Mel-spectrum values: a 128-value vector. |
| Architecture | Five hidden fully connected Dense layers with Dropout and one sigmoid output. |
| Labels | Upstream binary training labels `male` and `female`. |
| Intended use | Reproducing a historical classifier workflow and studying its limitations. |
| Evaluation | No independent accuracy, calibration, demographic-coverage or fairness evaluation has been completed for this maintained interface. |

The upstream tutorial describes preparation from Common Voice data. This repository has not independently audited the exact training split, speaker coverage, label collection or training procedure. Copied pretrained weights must not be described as a newly trained project model.

The output is a model score for an upstream dataset label. It does not establish a person's gender, identity, age or health. The binary label scheme does not represent all people. An uncalibrated score is not a reliable probability about an individual.

The old recording/training scripts are not included in this version; `legacy/README.md` links to their history and upstream sources. There is no supported training pipeline. The classifier command reads a supplied audio file and does not activate a microphone.

## Evaluation still needed

The recorded Windows smoke check covers package versions, checkpoint checksums, generated audio and output shape. It confirms model integration on that environment; it does not validate model quality. Repeat it on other supported environments and add a licensed speech/music evaluation set before making quality claims. See [VALIDATION.md](VALIDATION.md) and [ROADMAP.md](ROADMAP.md).
