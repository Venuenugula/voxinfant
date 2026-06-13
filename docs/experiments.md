# Experiment Log

One block per experiment; the deployable config is at the top.

---

## EXP_PROD — Hierarchical acoustic+GFCC model (CURRENT, deployed)
- **Features:** acoustic(36) + GFCC(26) = **62**, segment-level. No wav2vec.
- **Extraction:** raw audio → local DSP → VAD → 1.25 s / 50 % segments → features,
  via `scripts/prepare_data.py` (the SAME code path as inference).
- **Model:** **two-stage hierarchical** (`HierarchicalCryClassifier`):
  stage 1 = distress-vs-rest, stage 2 = pain-vs-burping. Each stage is an
  XGB(600·d3·lr0.05) with `class_weight='balanced'` sample weights.
- **Scaling:** global `StandardScaler` (fit on all data; also used at inference).
- **Decision:** **argmax** of the mean per-segment probability over a file.
- **Eval:** GroupKFold(5) by `file_id`. Report **file-level** (the metric users
  experience — inference averages a file's segments):
  **file macro-F1 = 0.593, file acc = 0.692** (segment-level 0.508 / 0.547).
- **Status:** inference-ready. `python -m voxinfant.train` regenerates artifacts.

### Why hierarchical (the chosen tradeoff)
The flat soft-vote ensemble had the *same* file macro-F1 (~0.59) but was "lazy":
it defaulted to `distress` and missed most minorities (burping recall 0.42, pain
0.43). The hierarchical model distinguishes the classes — **burping recall
0.42→0.53, pain 0.43→0.48** — at the cost of distress recall (0.89→0.79). For a
product meant to tell parents *which* cry it is, the balanced behaviour wins.

### Why this feature/extraction design (hard-won)
- **wav2vec dropped.** Colab embeddings don't reproduce locally and gave *worse*
  F1 (0.47 vs 0.52). Pulled torch + a 360 MB model for no benefit.
- **Re-extract from RAW, not Colab artifacts.** Colab `processed_audio`/features
  carry library fingerprints (spafe GFCC, noisereduce DSP); training on them while
  inferring locally collapsed every prediction to `distress`. Extract locally with
  the inference code so train == inference exactly.

### Model-improvement sweep (Phases 0–3, 2026-06-13) — what did NOT work
GroupKFold, file-level macro-F1. **We are at a data-diversity ceiling (~0.60):**
only ~120 source files each for pain/burping, so nothing generalises much.
- **Richer features** (F0/pyin, MFCC Δ/ΔΔ, spectral contrast/flatness, onset;
  62→134 d): **flat** (0.5509→0.5504 seg; +0.01 file). Not worth the pyin latency.
  Code kept behind `features.build_segment_vector(rich=True)` + `prepare_data --rich`.
- **Audio augmentation** (pitch/time/noise on pain+burping, group-safe, test on
  originals only): **slightly hurt** (file-F1 0.605→0.602). Same ~120 files → no
  new diversity. Code: `scripts/prepare_aug.py`, `scripts/cv_aug.py`.
- **Higher-capacity / RandomForest:** no better than the ensemble.
- **Balanced-prior decision:** badly hurt (0.605→0.573). Over-corrects.
- **Hierarchical:** the one genuine win (chosen). Probe/sweep: `scripts/probe.py`,
  `scripts/cv_eval.py`, `scripts/cv_aug.py`.

---

## EXP_PAPER — Full research system (reference only, NOT deployed)
- Features: acoustic + GFCC + **wav2vec(256)** = 318. Macro-F1 ≈ 0.54 in Colab.
- Not reproducible outside Colab (see above). Kept in the paper, not the product.

## Backlog
- [ ] More distress subtypes are confusable; consider hierarchical (distress-vs-rest
      then subtype) modelling.
- [ ] Calibrate probabilities before surfacing confidence in the UI.
- [ ] Hold out whole files from a *different* corpus for an honest out-of-sample number.
