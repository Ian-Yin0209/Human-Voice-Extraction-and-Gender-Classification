# Research & validation

Selection snapshot: **22 October 2026**. This page separates upstream research results from this toolkit's local execution checks.

## Selected: Sony MIMO-SCNet small

Sony Research's [MIMO paper](https://arxiv.org/abs/2609.07226) appeared in September 2026. It extends one-pass separators into iterative models and projects each step back to mixture consistency. We selected its **10.6M-parameter, four-stem, two-step SCNet small** checkpoint for a compact local integration.

The toolkit pins upstream source revision [`e172761a0798bef8a549d440a276218529b53edb`](https://github.com/SonyResearch/mimo-audio-separation/tree/e172761a0798bef8a549d440a276218529b53edb). The [training configuration](https://github.com/SonyResearch/mimo-audio-separation/blob/e172761a0798bef8a549d440a276218529b53edb/configs/data/musdb18_src-4_sec-11.yaml) fixes output order as **vocals, bass, drums, other**. The adapter uses the selected backbone with an inference-only refinement loop, rather than the upstream training/discriminator stack.

### Author-reported results, not toolkit measurements

The [upstream README](https://github.com/SonyResearch/mimo-audio-separation/blob/e172761a0798bef8a549d440a276218529b53edb/README.md) reports these **museval SDR** results on MUSDB18-HQ test, using median-of-medians. Both small models were trained on MUSDB18-HQ train. The authors explicitly distinguish this protocol from numbers in the paper.

| Upstream model | Parameters | Vocals | Bass | Drums | Other |
| --- | --- | ---: | ---: | ---: | ---: |
| SCNet small | 10.6M | 9.71 | 9.65 | 10.61 | 7.26 |
| MIMO SCNet small | 10.6M | 9.82 | 10.46 | 10.93 | 7.69 |

Values are in dB. This is an attributed comparison of two upstream checkpoints, not an independent reproduction, a score for this toolkit, or a universal current SOTA ranking.

## Fixed artifacts and license boundary

Weights come from the authors' [v1.0.0 release](https://github.com/SonyResearch/mimo-audio-separation/releases/tag/v1.0.0), not a mirror. They are not included in the source tree or package.

| Artifact | Identity |
| --- | --- |
| Official ZIP | [`mimo-scnet_small.zip`](https://github.com/SonyResearch/mimo-audio-separation/releases/download/v1.0.0/mimo-scnet_small.zip), 39,354,515 bytes — approximately 39.4 MB |
| ZIP SHA-256 | `f0eaaab7e899ffc548b29d51d76b89727253fb0a529b787b258eabfcf47db6fb` |
| Installed file | `mimo-scnet-small-v1.pth`, 42,491,051 bytes — approximately 42.5 MB |
| Weight SHA-256 | `1d19305440e8439faeb4163d809b2dcaaffacf67f6f926425be581290d6536ee` |

PyTorch and base dependencies are additional to these sizes. `models --json` reads an offline catalog; `fetch-model` is the explicit download/install command. A supplied `--archive` uses only the local official ZIP. Both archive and weights are checked, and separation accepts only the pinned local weights without networking.

The upstream [code LICENSE](https://github.com/SonyResearch/mimo-audio-separation/blob/e172761a0798bef8a549d440a276218529b53edb/LICENSE) is MIT; the reused SCNet code has its own retained [MIT notice](https://github.com/SonyResearch/mimo-audio-separation/blob/e172761a0798bef8a549d440a276218529b53edb/_LICENSES/SCNet.LICENSE). No separate explicit license for the released weights was found in the inspected repository, release or checkpoint archive. The root code license is not a confirmation of weight redistribution or commercial-use rights. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Local runtime evidence

A published [2026-10-22 Windows CPU validation record](validation/2026-10-22-mimo-cpu.json) contains two checks with the pinned real weights. The recorded environment is Python 3.12.14 and PyTorch 2.14.1+cpu on Windows 10. Inputs are generated signals, not commercial music or private recordings.

| Case | Input | Output per stem | Chunks | Maximum reconstruction error |
| --- | --- | --- | ---: | ---: |
| `stereo_tone_2s` | Two seconds, 16 kHz, already stereo | 88,200 stereo frames at 44.1 kHz | 5 | 6.7055225 × 10⁻⁸ |
| `one_frame_mono` | One mono frame at 8 kHz | Six stereo frames at 44.1 kHz | 1 | 7.4505806 × 10⁻⁹ |

Both use two refinement steps, 0.5-second chunks and 25% overlap, producing four stems, their accompaniment sum and `report.json`. The second case exercises very short input and mono-to-stereo handling; the first starts with stereo input.

For `stereo_tone_2s`, **model loading, inference and overlap-add** took **1.7683967 seconds**. This timer excludes audio I/O and weight verification; it is not total workflow wall time. The JSON record contains the exact per-case measurements and runtime versions.

This checks that real weights load and the complete file workflow runs. It does not measure how well singing is isolated from music. The reconstruction check measures whether the four outputs sum back to the resampled mixture, not whether each source is correct. No GPU validation, MUSDB18-HQ evaluation or listening study is claimed. Timing depends on the machine, signal, chunk settings and runtime; it is not a speed guarantee.

Normal CLI settings use 6-second chunks, 25% overlap and two iterations; the short-chunk smoke above is not a default-settings performance benchmark. Chunking limits activation size, while the input and accumulated output arrays remain in RAM. Outputs are 44.1 kHz stereo float WAV; a new output directory is required.

## How to compare your own estimates

```sh
voice-extract evaluate reference-vocals.wav stems/vocals.wav --json
```

This implements zero-mean SI-SDR independently per channel. Inputs must already have identical sample rate, frame count and channel count. No time alignment, trimming or resampling is performed. The overall result is the arithmetic mean of channel scores in dB only when every channel has a finite score.

Zero-variance references or estimates, exact orthogonality and exact scale-perfect reconstruction have no finite score in this implementation. JSON represents those scores as `null` and includes a reason; any undefined channel also makes the overall score `null`. A `null` is not zero dB. This SI-SDR is **not** the filter-based museval SDR in the upstream comparison table; do not compare the two as interchangeable metrics.

## Other research considered

- **Windowed RoFormer:** the [Smule paper](https://arxiv.org/abs/2510.25745) and [official repository](https://github.com/smulelabs/windowed-roformer) offer a Mel-Band RoFormer derivative with MIT code and checkpoints. Its provided checkpoint is approximately 914 MB, and the default implementation enables compiled FlexAttention. We did not integrate or validate that Windows compilation path; the smaller Sony checkpoint was a more bounded first integration. The paper's FLOP reduction is not a runtime measurement for this toolkit.
- **SAM Audio:** [Meta's official implementation](https://github.com/facebookresearch/sam-audio) supports text, visual and temporal prompts for general sound extraction rather than this fixed four-stem workflow. It requires Python 3.11+, recommends CUDA, and asks users to request/authenticate checkpoint access. It uses a separate [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE). It is not integrated or benchmarked here.

BS-RoFormer's SDX23 result and early Mel-Band RoFormer/SCNet SOTA claims describe their original experiments, not a standing 2026 leaderboard. Newer music-source restoration challenges also use different targets and metrics. This project claims a transparent, tested integration of a recent model, not invention of that model or globally best separation quality.
