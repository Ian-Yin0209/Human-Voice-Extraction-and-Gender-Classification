# Source and modifications

Source: https://github.com/SonyResearch/mimo-audio-separation

Fixed revision: `e172761a0798bef8a549d440a276218529b53edb` (2026-10-07).

`scnet.py` and `separation.py` come from `src/model/backbone/scnet/`. Only their imports of the `exists` and gradient-checkpoint helper were redirected to local equivalents in `common.py`. Backbone layers and state-dict keys are preserved. This is inference integration, not the upstream training pipeline.

Sony's MIT notice and the inherited SCNet MIT notice are retained in LICENSE and LICENSE.SCNet. Upstream checkpoint files are not bundled. The repository's code license is not a claim about the separate terms of external weights, training data or user recordings.
