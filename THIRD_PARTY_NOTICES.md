# Third-party notices

This project contains source derived from the projects below. Original copyrights and licenses remain in effect for those portions and must be preserved when distributing substantial copies. The new maintenance code and documentation are licensed as described in [LICENSE](LICENSE).

## Sony MIMO-SCNet integration (0.3.0)

Source: [SonyResearch/mimo-audio-separation](https://github.com/SonyResearch/mimo-audio-separation/tree/e172761a0798bef8a549d440a276218529b53edb), fixed revision `e172761a0798bef8a549d440a276218529b53edb`.

`voice_extraction/_vendor/sony_scnet/scnet.py` and `separation.py` retain the upstream backbone with local helper imports. Their complete MIT notices ship in [Sony LICENSE](voice_extraction/_vendor/sony_scnet/LICENSE) (Copyright 2025 Yukara Ikemiya) and [inherited SCNet LICENSE](voice_extraction/_vendor/sony_scnet/LICENSE.SCNet). Changes are listed in [SOURCE.md](voice_extraction/_vendor/sony_scnet/SOURCE.md). The new inference-only adapter follows the upstream MIMO normalization and direct-prediction consistency projection; it does not redistribute the training pipeline.

The checkpoint is an external user-obtained artifact, not part of this source distribution. No separate explicit weight license was found in the inspected release archive. A source-code MIT badge does not license external recordings, datasets or checkpoints; see [Research](docs/RESEARCH.md).

## 1. spleeter-pytorch — Tuan Nguyen

Source: [tuan3w/spleeter-pytorch](https://github.com/tuan3w/spleeter-pytorch/tree/853d4bb6048fae879543342a8278c298854637f3).

Affected material includes `spleeter_style/unet.py`, `spleeter_style/util.py`, `spleeter_style/estimator.py` and adaptations of the original `test_estimator.py` workflow. The baseline source comparison is recorded in [Provenance](docs/PROVENANCE.md). Later modifications do not remove this upstream attribution.

Upstream license, reproduced in full:

```text
MIT License

Copyright (c) 2019 Tuan Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. gender-recognition-by-voice — x4nth055

Source: [x4nth055/gender-recognition-by-voice](https://github.com/x4nth055/gender-recognition-by-voice/tree/e0f0f3887ea57a0d3c2309c7238242fa7e6badf5).

Affected material includes adaptations of the classifier's Dense-network architecture and audio-feature workflow. The old utilities, training/preparation and recording scripts are not included in this version; `legacy/README.md` links to their history and upstream references. The historically bundled `model.h5` was identical to the upstream model; it is no longer distributed in the maintained tree. The older recording-snippet lineage described below is a separate unresolved attribution item; the presence of this upstream MIT notice does not resolve it.

Upstream license, reproduced in full:

```text
MIT License

Copyright (c) 2020 x4nth055

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. Spleeter — Deezer SA

Source: [deezer/spleeter](https://github.com/deezer/spleeter). The PyTorch port above builds on Spleeter's architecture and refers to its pretrained checkpoints. Deezer's [README](https://github.com/deezer/spleeter#license) identifies the source code as MIT licensed.

The code-license text is retained below for this lineage. It does not establish separate licensing terms for externally obtained checkpoints, training data or audio examples.

Upstream source-code license, reproduced in full:

```text
MIT License

Copyright (c) 2019-present, Deezer SA.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Earlier recording-snippet lineage

The original microphone-recording helpers closely match the answer by **cryo**, posted on 2011-07-19, to [Detect & Record Audio in Python on Stack Overflow](https://stackoverflow.com/questions/892199/detect-record-audio-in-python). The answer also credits Nick Fortescue's earlier response. Historical adaptations include the sample rate, silence threshold handling and Python 3 syntax.

The exact source revision and complete permission chain for these snippets have not been established. Their implementation is not distributed in this maintained version, is not imported by the supported CLI, and is not relicensed by this repository's new MIT license. Consult the original answer and [Stack Overflow's licensing information](https://stackoverflow.com/help/licensing) before reintroducing these recording helpers.

## External models, data and audio

No pretrained weights, datasets or original music/recorded examples are included as maintained distributable assets. Optional commands accept user-supplied local weights. Model provenance and remaining validation gaps are described in [MODEL_CARD.md](docs/MODEL_CARD.md).

- The original two-stem checkpoint can be obtained from [Deezer's v1.4.0 release](https://github.com/deezer/spleeter/releases/tag/v1.4.0). Exact weight-license scope has not been independently established here.
- The classifier's historical pretrained weights are available at [upstream results/model.h5](https://github.com/x4nth055/gender-recognition-by-voice/blob/master/results/model.h5). The upstream repository carries the MIT notice above; this maintenance work has not independently audited training-data rights or model-specific terms.
- The source and redistribution terms of every original audio sample have not been established. Removing files from the maintained tree does not alter historical Git commits.

Normal package dependencies are installed separately and retain their respective licenses. This notice does not replace their distribution notices.
