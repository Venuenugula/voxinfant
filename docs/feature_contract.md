# Feature Contract

The deployable model expects a **62-dimensional** segment feature vector, formed
by concatenating two **reproducible** blocks in this exact order. Both are
computed identically at training and inference time.

| Slice | Block | Source | Dims |
|-------|-------|--------|-----:|
| `[0:36]`  | **Acoustic** | `extract_acoustic_features` (librosa) | **36** |
| `[36:62]` | **GFCC** | `extract_gfcc_features` (spafe) | **26** |
| | | | **62** |

## Acoustic block (36)
| Feature | Dims |
|---------|-----:|
| MFCC mean (n_mfcc=13) | 13 |
| MFCC std | 13 |
| spectral centroid mean/std | 2 |
| spectral bandwidth mean/std | 2 |
| spectral rolloff mean/std | 2 |
| zero-crossing-rate mean/std | 2 |
| RMS mean/std | 2 |
| **total** | **36** |

## GFCC block (26)
`spafe.features.gfcc.gfcc(y, fs=sr)` with default `num_ceps=13` → `mean(13) ⧺ std(13)`.

## Why wav2vec was dropped
The research pipeline added a 256-d `facebook/wav2vec2-base` embedding per file
(total 318 dims). It was removed from the **product** for three reasons:

1. **Not reproducible** outside the original Colab run. Live embeddings differ
   from the cached training embeddings (library/version + processed-audio
   round-trip), which made every live prediction collapse to `distress`.
2. **No accuracy gain.** In a clean GroupKFold comparison, acoustic+gfcc scored
   **macro-F1 0.52** vs **0.47** for the full 318-d set — wav2vec hurt slightly.
3. **Deployment cost.** It pulled in torch + transformers (~360 MB model, an
   OpenMP/CUDA crash class) for no benefit.

The legacy 318-d `raw_features.npy` is still usable: `train.py` slices it to the
first 62 columns. wav2vec remains part of the *paper*, not the *product*.

## Critical invariants
1. Order is fixed: `[acoustic | gfcc]`.
2. The same DSP chain (`process_audio`) runs before feature extraction.
3. `config.yaml: features.expected_dim` must equal **62**. The smoke test asserts it.
4. No per-file features remain, so segment vectors are fully independent given
   the file — GroupKFold by `file_id` is still used to avoid same-file leakage.
