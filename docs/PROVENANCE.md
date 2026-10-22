# Project provenance

## 2026 research integration

Version 0.3 adds an inference adapter for Sony Research's MIMO-SCNet small configuration from [Iterative Audio Separation with Mixture Consistency via MIMO Model Extension](https://arxiv.org/abs/2609.07226). Source revision, modifications, notices, checkpoint hashes and limits are recorded in [Research](RESEARCH.md) and `voice_extraction/_vendor/sony_scnet/SOURCE.md`. This project did not train that model. External weights are obtained separately and are not included in the MIT source distribution.

## Educational origin

This repository began as **Human Voice Extraction and Gender Classification**, an educational prototype described in its original README as coursework for **ESC7013P**. Its original source was published in April 2022. It combined a Spleeter-style separator with a voice-label classifier and local audio recording scripts.

The 2026 maintenance work introduces the **Voice Extraction Toolkit** CLI, separates optional model dependencies from basic audio utilities, and documents the inherited implementation. This does not imply years of continuous maintenance, independent model training or external adoption.

## Source lineage

### Spleeter-style separator

Files under `spleeter_style/` derive from [tuan3w/spleeter-pytorch](https://github.com/tuan3w/spleeter-pytorch). Against commit [`853d4bb6048fae879543342a8278c298854637f3`](https://github.com/tuan3w/spleeter-pytorch/tree/853d4bb6048fae879543342a8278c298854637f3), dated 2020-07-12:

- The original `spleeter_style/unet.py` and `spleeter_style/util.py` match upstream `spleeter/unet.py` and `spleeter/util.py` after whitespace normalization.
- The original `spleeter_style/estimator.py` differs in the `istft` import: `torch` here rather than `torchaudio.functional` upstream.
- The original root `estimator.py` adapts the loading, normalization, separation and writing pattern from upstream `test_estimator.py`.

The port credits [Deezer Spleeter](https://github.com/deezer/spleeter), the source of the separation architecture and pretrained models. The early README's claim of reimplementation from Spleeter was incomplete: the PyTorch port is also a direct source.

### Classifier and historical scripts

The classifier utilities and recording helpers derive from [x4nth055/gender-recognition-by-voice](https://github.com/x4nth055/gender-recognition-by-voice). Against commit [`e0f0f3887ea57a0d3c2309c7238242fa7e6badf5`](https://github.com/x4nth055/gender-recognition-by-voice/tree/e0f0f3887ea57a0d3c2309c7238242fa7e6badf5), dated 2020-07-30, the original `utils.py` matches after removing docstrings and normalizing whitespace. The original `test.py` recording helpers and feature extraction also closely correspond to upstream.

The historical training and recording scripts are not included in the maintained tree. `legacy/README.md` points to the original history and upstream references. The old recording helpers also closely match the earlier answer by cryo (2011-07-19) to [Detect & Record Audio in Python](https://stackoverflow.com/questions/892199/detect-record-audio-in-python). Their exact source revision and complete permission chain remain unresolved; the upstream repository's MIT notice alone does not establish that chain. These helpers are no longer distributed or used by the supported CLI.

The early README called the classifier a CNN; the actual `create_model()` defines a fully connected network with Dense and Dropout layers.

## Previously bundled artifacts

The maintenance tree removes bundled model files and recorded/music audio examples. This changes the current checkout, not historical commits.

| Historical artifact | Established provenance |
| --- | --- |
| `model.h5` | Identical to upstream `x4nth055/gender-recognition-by-voice/results/model.h5`: Git blob SHA `a1c0d0cc622b6858ebb54b269407f4d971781d5a`, 1,936,312 bytes. It was not independently trained by this work. |
| `checkpoints/2stems/checkpoint` | Matches `tuan3w/spleeter-pytorch`; blob SHA `a6e034f303e6fd8795171c44ad5816e660476552`. |
| `checkpoints/2stems/model.data-00000-of-00001` | Matches that upstream; blob SHA `2f7f72a4b56abdd2fdc75faa26afa98e009d1f7a`. |
| `checkpoints/2stems/model.index` | Matches that upstream; blob SHA `19b2f3a4ccef42c47eeb9b4a5e8e10cdb6143d37`. |
| `checkpoints/2stems/model.meta` | Matches that upstream; blob SHA `0ff10b4890443661ff750f70e7a38312c0f05e97`. |
| Audio examples and generated outputs | Complete source and redistribution terms have not been established. They are not distributed as maintained examples. |

Git blob IDs establish repository object identity; they are not raw-file SHA-256 checksums.

## Licensing and remaining work

The verified source projects publish MIT licenses. Their notices are preserved in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The new repository license does not relicense third-party materials or assert a license for externally obtained weights, training data or media.

Independent evaluation of model quality, training-data coverage and fairness remains on the [roadmap](ROADMAP.md).
