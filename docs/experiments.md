# Experiment Log

Convert the paper's 14 iterations into trackable entries here. One block per
experiment. Keep the *final* deployable config at the top.

---

## EXP_FINAL — Publishable ensemble (current production target)
- **Features:** acoustic(64) + GFCC(26) + wav2vec(256) = 346, segment-level.
- **Balancing:** downsample `distress` → `len(others)`; oversample `pain` +500.
- **Scaling:** global `StandardScaler` (fit on balanced set).
- **Model:** XGB(600·d3·lr0.05, λ3/α2/γ2) + MLP(128,64), soft-vote 0.65/0.35.
- **Class weights:** pain=5, burping=4, distress=1.
- **Eval:** GroupKFold(5) by `file_id`. **Macro-F1 ≈ 0.54.**
- **Decision rule (notebook):** pain if p>0.20 → elif burping p>0.20 → else distress.
- **Status:** inference-ready once artifacts regenerated (`python -m voxinfant.train`).

---

## Backlog / open questions
- [ ] **Fast inference profile** — train an acoustic+GFCC-only model (no wav2vec)
      for low-latency live prediction. Needs its own `models/voxinfant_fast.pkl`.
- [ ] Calibration: earlier cells tried `CalibratedClassifierCV`; decide if the
      deployed probabilities need calibration before exposing confidence in the UI.
- [ ] Replace `emb[:256]` truncation of wav2vec with PCA-256 (truncation discards
      potentially useful later dims).
- [ ] Per-fold vs global scaling — global is used now; document leakage analysis.

## Template
```
## EXP_00X — <name>
- Features: ...
- Change vs previous: ...
- Result (macro-F1 / acc): ...
- Keep? why: ...
```
