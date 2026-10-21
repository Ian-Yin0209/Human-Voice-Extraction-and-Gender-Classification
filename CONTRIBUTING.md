# Contributing

Contributions that make local audio processing more predictable, easier to reproduce and better documented are welcome.

## Set up

Use Python 3.10–3.12 and install from the repository root:

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m voice_extraction --help
```

The base tools should remain usable without PyTorch, TensorFlow, model downloads or microphone access. Install `'.[models]'` only when working on model integration. Such checks require separately obtained local weights; state clearly when a change was checked without them.

## Propose a change

1. Explain the problem in an issue or pull request.
2. Keep changes focused and preserve refusal to overwrite outputs unless `--force` is passed.
3. Add a regression test for changed behavior. Prefer small, generated audio fixtures over recorded speech or music.
4. Run relevant checks and include their commands and results.

For audio bugs, include the command, complete error, Python and package versions, operating system, and non-sensitive file metadata. A small synthetic reproduction is preferable to uploading private recordings. Do not commit model weights, large audio assets, credentials or material you cannot redistribute.

## Code, models and attribution

- New repository-authored code and documentation are contributed under [LICENSE](LICENSE).
- Preserve upstream copyright and license notices. Record reused components in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Include the source and terms for a new model integration. Do not claim training, evaluation results or dataset coverage without reproducible evidence.
- `legacy/README.md` links to historical sources; the old training and recording scripts are not distributed in this version. Supported behavior belongs in `voice_extraction` and its tests.

There is no promised response or release schedule. Current work and validation gaps are listed in the [roadmap](docs/ROADMAP.md).
