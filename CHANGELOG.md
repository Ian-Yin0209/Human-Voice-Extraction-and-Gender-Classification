# Changelog

## 0.2.0 (unreleased)

- Add an installable package and file-based `inspect`, `prepare`, `separate`, and `classify` commands.
- Replace hardcoded audio paths and fixed sample offsets with explicit arguments in seconds.
- Validate mono/stereo audio, preserve channel layout, and resample separator input to 44.1 kHz.
- Keep microphone dependencies out of file processing and load model runtimes only on demand.
- Keep successful CLI processing successful when redirected output uses an encoding that cannot represent the input filename.
- Modernize the inherited PyTorch separator and retain the original U-Net architecture. Accompaniment now includes the mixture residual, retaining unmodelled high frequencies; outputs are not numerically identical to the original implementation.
- Add synthetic-audio regression tests, packaging checks, and continuous integration.
- Document upstream code/model provenance and link to historical training scripts from `legacy/` without redistributing the retired recording/training implementation.
- Stop distributing bundled song recordings, generated audio, IDE settings, and pretrained weights in the current source tree. Git history is retained.

Pretrained-weight compatibility passed on generated audio in a documented Windows CPU environment. Model quality remains unevaluated; see `docs/VALIDATION.md`.
