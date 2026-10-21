# Roadmap

These are planned tasks, not delivery dates or performance promises. Mark an item complete only with a reviewable change and relevant checks.

## Packaging and audio utilities

- [ ] Validate installation and CLI behavior across Python 3.10–3.12 on Linux and Windows.
- [x] Check crop boundaries, resampling, mono conversion, silence normalization and overwrite protection using generated audio on Windows Python 3.12.
- [ ] Document supported input/output formats from verified examples.

## Optional models

- [x] Run the inherited Deezer 2-stem checkpoint from the fixed upstream PyTorch port; record Windows CPU environment, checksums and synthetic-audio results.
- [x] Check classifier loading and feature extraction with upstream pretrained weights in a documented Windows CPU environment.
- [ ] Add redistributable evaluation fixtures and report limitations before making quality claims.
- [ ] Audit exact external model licensing and training-data documentation separately from source-code licenses.
- [ ] Review whether retaining the binary-label classifier remains useful after evaluating its limitations.

## Maintenance

- [ ] Review notices whenever reused source or model integration changes.
- [ ] Audit provenance and permissions before reintroducing any historical training or recording implementation.
- [ ] Turn reproducible user reports into regression tests.
- [ ] Check installation, documentation examples and declared compatibility before publishing a release.
