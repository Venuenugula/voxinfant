# Feature Contract

The trained model expects a **346-dimensional** segment feature vector, formed
by concatenating three blocks **in this exact order**. Any drift here silently
breaks inference (wrong dimension → scaler/model error or garbage predictions).

| Block | Source | Dims | Notes |
|-------|--------|-----:|-------|
| **Acoustic / prosodic** | `extract_acoustic_features` | **64** | per **segment** |
| **GFCC** | `extract_gfcc_features` (spafe) | **26** | per **segment** |
| **wav2vec2** | `Wav2VecEmbedder` | **256** | per **file**, broadcast to every segment |
| | | **346** | |

## Acoustic block (64) — breakdown
| Feature | Dims |
|---------|-----:|
| MFCC mean (n_mfcc=13) | 13 |
| MFCC std | 13 |
| MFCC delta mean | 13 |
| MFCC delta² mean | 13 |
| spectral centroid mean/std | 2 |
| spectral bandwidth mean/std | 2 |
| spectral rolloff mean/std | 2 |
| zero-crossing-rate mean/std | 2 |
| RMS mean/std | 2 |
| pitch (pyin f0) mean/std | 2 |
| **total** | **64** |

## GFCC block (26)
`spafe.features.gfcc.gfcc(y, fs=sr)` with default `num_ceps=13` → `mean(13) ⧺ std(13)`.

## wav2vec2 block (256)
`facebook/wav2vec2-base` → `last_hidden_state.mean(dim=1)` (768-d) → **first 256 dims**.
Computed **once per file** on the DSP-processed signal; the same 256-d vector is
attached to every segment of that file. This is why grouping by `file_id` in
GroupKFold is essential — segments of one file share a feature block and would
leak across a naive split.

## Critical invariants
1. **Order** of the three blocks is fixed.
2. **wav2vec is per-file, not per-segment.**
3. The same DSP chain (`process_audio`) must run before both wav2vec and
   acoustic/GFCC extraction.
4. `config.yaml: features.expected_dim` must equal 346. The smoke test asserts it.
