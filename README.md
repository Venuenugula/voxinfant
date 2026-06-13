# VoxInfant

Infant cry analysis pipeline: DSP front-end → segment-level acoustic + GFCC
features → XGBoost/MLP soft-vote ensemble. Classifies a cry into
**`pain` / `burping` / `distress`**.

> ⚠️ **Research preview, not a medical device.** GroupKFold macro-F1 ≈ 0.56 over
> 3 coarse classes. Do not use for clinical decisions.

The deployable model uses **reproducible features only** (62-d acoustic + GFCC) —
no wav2vec/torch. This was deliberate: Colab-computed features (wav2vec, GFCC) do
not reproduce at inference and caused predictions to collapse. Training now
re-extracts from **raw audio** with the exact inference code, so train == inference.
See [docs/experiments.md](docs/experiments.md).

## Layout
```
voxinfant_production/
├── config.yaml              # all pipeline constants (single source of truth)
├── src/voxinfant/           # the package
│   ├── config.py            # loads config.yaml
│   ├── dsp.py               # normalize → pre-emphasis → bandpass → noise reduce
│   ├── segmentation.py      # RMS VAD + 1.25s/50%-overlap windows
│   ├── features.py          # acoustic(64) + gfcc(26) + wav2vec(256) = 346
│   ├── ensemble.py          # WeightedSoftVoteEnsemble (picklable)
│   ├── inference.py         # CryAnalyzer: wav → prediction dict
│   └── train.py             # retrain → regenerate model artifacts
├── api/app.py               # FastAPI: /health, /predict
├── scripts/prepare_data.py  # raw audio → features/*.npy
├── models/                  # trained artifacts live here (git-ignored)
├── docs/                    # architecture, feature_contract, experiments
└── tests/                   # feature-contract smoke tests
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs deps from requirements.txt
```

## Workflow (artifacts must be regenerated — none are committed)
```bash
# 1. Put raw audio under data/raw/<raw_label>/*.wav  (see config.yaml: class_map)
# 2. Build segment features  → features/raw_features.npy, labels.npy, groups.npy
python scripts/prepare_data.py
# 3. Train + evaluate (GroupKFold) + save → models/voxinfant_ensemble.pkl, scaler.pkl, label_classes.json
python -m voxinfant.train
# 4. Serve
uvicorn api.app:app --reload --port 8000
```

## Predict (once artifacts exist)
```bash
curl -F "file=@sample.wav" http://localhost:8000/predict
```
```python
from voxinfant.inference import CryAnalyzer
print(CryAnalyzer().load().predict("sample.wav").to_dict())
```

## Tests
```bash
pytest          # contract smoke tests; heavy-dep tests auto-skip if deps absent
```

See [docs/feature_contract.md](docs/feature_contract.md) for the exact 346-d
vector layout — the one thing you must not break.
