# Historical training references

This directory links to the original experiment and upstream training workflow. The retired scripts remain in Git history and are not redistributed in the current source tree. The maintained entry points live in `voice_extraction/`.

- Original `utils.py`, `test.py`, `train.py`, and `preparation.py`: [prototype at eb261be](https://github.com/Ian-Yin0209/Human-Voice-Extraction-and-Gender-Classification/tree/eb261be21bda886d838b60e1be59778b28d505ef).
- Upstream training reference: [x4nth055/gender-recognition-by-voice](https://github.com/x4nth055/gender-recognition-by-voice). See `../THIRD_PARTY_NOTICES.md` for attribution and the earlier recording-code provenance note.
- Original training/preparation scripts expect local datasets/CSV metadata that are not distributed here. They are not a reproducible training recipe.
- The maintained file-based commands do not record audio or import PyAudio.
- Historical pretrained weights and sample recordings are not part of the current distribution. Their provenance is documented in `../docs/PROVENANCE.md`.

No new training result or accuracy claim is made by retaining these files.
