# Architecture

```
 audio (.wav/.mp3)
        │
        ▼
 DSP  ──────────────  dsp.process_audio
   normalize → pre-emphasis(0.97) → bandpass(200–4000 Hz) → noise reduce
        │
        ▼
 VAD (RMS) ─────────  segmentation.apply_vad
        │
        ▼
 segment (1.25 s, 50% overlap)  ── segmentation.segment_audio
        │
        ▼
 per-segment features  ── features.build_segment_vector
   [ acoustic(36) | gfcc(26) ] = 62      (reproducible; no torch)
        │
        ▼
 StandardScaler  ──  models/scaler.pkl
        │
        ▼
 Ensemble (0.65·XGB + 0.35·MLP)  ──  models/voxinfant_ensemble.pkl
        │
        ▼
 mean probability over segments → label + alternatives  ── inference.CryAnalyzer
```

> wav2vec/SSL was dropped from the deployable model (not reproducible across
> environments, no accuracy gain). See docs/feature_contract.md.

## Modules (`src/voxinfant/`)
| Module | Responsibility |
|--------|----------------|
| `config.py` | Load `config.yaml`; single source of truth for all constants |
| `dsp.py` | DSP front-end (load, normalize, pre-emphasis, bandpass, noise reduce) |
| `segmentation.py` | RMS VAD + fixed-window segmentation |
| `features.py` | Acoustic + GFCC + wav2vec blocks; segment vector assembly |
| `ensemble.py` | `WeightedSoftVoteEnsemble` (picklable; imported at inference) |
| `inference.py` | `CryAnalyzer`: wav → prediction dict |
| `train.py` | Retrain from saved features; regenerate artifacts |

## Service (`api/app.py`)
FastAPI: `GET /health`, `POST /predict`. Model loaded lazily; 503 if artifacts absent.

## Two inference profiles (planned)
- **Full (current):** acoustic + GFCC + wav2vec → the trained 346-d model.
- **Fast (future):** acoustic + GFCC only (no wav2vec). **Requires a separately
  trained model** — you cannot drop the 256 wav2vec dims from the current model.
  Tracked in `docs/experiments.md`.
