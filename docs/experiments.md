# Experiment Log

One block per experiment; the deployable config is at the top.

---

## EXP_PROD — Reproducible acoustic+GFCC model (CURRENT, deployed)
- **Features:** acoustic(36) + GFCC(26) = **62**, segment-level. No wav2vec.
- **Extraction:** raw audio → local DSP → VAD → 1.25 s / 50 % segments → features,
  via `scripts/prepare_data.py` (the SAME code path as inference).
- **Balancing:** downsample `distress` to the combined minority count. No oversampling.
- **Scaling:** global `StandardScaler`.
- **Model:** XGB(400·d3·lr0.05) + MLP(128,64), soft-vote 0.65/0.35 (`ensemble.py`).
- **Class weights:** `{burping:1.5, distress:1, pain:1.5}` (gentle — see below).
- **Decision:** **argmax** of mean per-segment probability (what `CryAnalyzer` uses).
- **Eval:** GroupKFold(5) by `file_id`. **macro-F1 = 0.558**, acc = 0.599.
- **Status:** inference-ready. `python -m voxinfant.train` regenerates artifacts.

### Why this design (hard-won)
- **wav2vec dropped.** Colab-computed embeddings don't reproduce locally
  (transformers drift) and gave *worse* F1 (0.47 vs 0.52). Pulled torch + a
  360 MB model + an OpenMP/CUDA crash for no benefit.
- **Re-extract from RAW, not from Colab artifacts.** Features AND `processed_audio`
  made in Colab carry library fingerprints (spafe GFCC, noisereduce DSP). Using
  them for training while inferring locally made every prediction collapse to
  `distress`. Fix: extract everything locally from raw audio with the inference
  code, so train == inference exactly.
- **Gentle class weights + argmax.** The notebook's `{b4,d1,p5}` + pain
  oversampling + soft-threshold rule over-triggered `pain` at inference. Argmax
  with mild weights scores best (0.558 vs 0.535) and balances the classes.

### Live sanity (raw donateacry, partly in-sample)
`belly_pain→pain .83 · burping→burping .62 · hungry→distress .92 ·
tired→distress 1.0 · discomfort→distress 1.0` — 50/56 = 0.89. No collapse.

---

## EXP_PAPER — Full research system (reference only, NOT deployed)
- Features: acoustic + GFCC + **wav2vec(256)** = 318. Macro-F1 ≈ 0.54 in Colab.
- Not reproducible outside Colab (see above). Kept in the paper, not the product.

## Backlog
- [ ] More distress subtypes are confusable; consider hierarchical (distress-vs-rest
      then subtype) modelling.
- [ ] Calibrate probabilities before surfacing confidence in the UI.
- [ ] Hold out whole files from a *different* corpus for an honest out-of-sample number.
