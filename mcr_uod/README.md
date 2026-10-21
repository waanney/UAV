# MCR-UOD — Causal Reasoning Package

Scaffold for the **causal half** of the research project *Uncertainty-guided
causal reasoning for multimodal object detection* (HCMUT, 2026).

This package is separated from the vendored `ultralytics/` tree so the research
code stays clean and both sub-teams can work in parallel without file conflicts.

## Repository context

The repository is a locally patched copy of **Ultralytics 8.3.82** plus the
upstream MCR-UOD code (NeurIPS 2025, `lnxwow/MCR-UOD`). The upstream code
implements a *heuristic* version of the confounder idea directly inside
`ultralytics/nn/modules/block.py` (`C2fAttn`, `FeatureInteraction`,
`TextFeatureEnhancer`). This package re-implements the idea cleanly and
**mathematically correctly** (do-operator + prior `P(Z)` + NWGM).

## Research pillars

| Pillar | Owner | Where |
|---|---|---|
| Causal zoom (distant objects) | causal team | `mcr_uod/causal/zoom/` |
| Causal prompt (similar objects) | causal team | `mcr_uod/causal/prompt_intervene.py` |
| Conformal prediction + rerank | uncertainty team | `mcr_uod/uncertainty/` (external) |

## Work split (2 + 2)

**Causal team (2 people)**

- Owner A — *core causal math*: `confounder.py`, `nwgm.py`, `bacr.py`, `lgre.py`
- Owner B — *applications*: `zoom/`, `prompt_intervene.py`, `adapter.py`

**Uncertainty team (2 people)**

- Owns `mcr_uod/uncertainty/` (not created here). Creates `conformal.py`,
  `rerank.py`, `pipeline.py` with its own interface.

`causal/adapter.py` is the single join point, added by the causal team once the
uncertainty interface is frozen.

## Frozen interfaces

These signatures are agreed; do not change them without syncing both teams.

```python
# confounder.py
ConfounderDictionary(num_classes, d2, descriptions_path, K=4, M=9, learnable_prior=True)
    .build_text_bank(clip_text_encoder) -> Tensor[S, D2]
    .prior()                            -> Tensor[S]          # P(Z), sums to 1
    .update(F_p, z_idx)                 -> None              # Eq.11, in place
    .nearest(F)                         -> Tensor[N]

# lgre.py
LGRE(d2, num_heads).forward(visual_feat[B,C,H,W], txt_feats[1,nc,D2]) -> s[B,1,H,W]
topk_select(score[B,1,H,W], tau) -> (F[B,N,D2], I[B,N])

# bacr.py
BACR(d2, num_heads, fuse_mode, use_nwgm).forward(F[B,N,D2], Z[S,D2], prior[S]) -> F'[B,N,D2]
scatter_back(z[B,C,H,W], F_prime[B,N,C], indices[B,N]) -> [B,C,H,W]

# nwgm.py
nwgm_intervene(F, Z, prior, f) -> Tensor                       # Eq.7
```

## Paper symbol map

| Symbol | Code | Meaning |
|---|---|---|
| `tau` | `causal.tau` | top-k pixel ratio, `N = tau*H*W` (Eq.5) |
| `D2` | `causal.d2` | unified text/feature dim (512) |
| `alpha` | `causal.alpha` | dictionary momentum (Eq.11) |
| `K, M` | `causal.K`, `causal.M` | weather x (occlusion x view) strata |
| `P(Z)` | `prior()` | confounder prior (Eq.1,6,7) |
| NWGM | `nwgm_intervene` | Eq.7 approximation |

## Config

All hyper-parameters live in `configs/mcr_causal.yaml`. No absolute paths are
hard-coded; assets are resolved via `mcr_uod/paths.py`.

## How to run (once implemented)

```bash
python train.py     # YOLOv8x-Worldv2 + causal modules, VisDrone, imgsz=1536
python val.py       # mAP evaluation
python detect.py    # folder inference
```

## Known issues (inherited from upstream)

1. **`ultralytics/engine/results.py` is missing** in the committed tree, so
   `from ultralytics import YOLO` fails out of the box. Restore it from the
   stock `ultralytics==8.3.82` wheel before running anything.
2. Upstream hard-codes absolute paths (`/home/linianxin/aaai2026/...`); use
   `mcr_uod.paths` instead (`confunder.txt` is now repo-relative).
3. Upstream imports `torch_geometric` unconditionally for a disabled GCN path.
   Keep it optional in this package.

## Status

Scaffold only: every function raises `NotImplementedError`. Logic is added by
each team. See `docs/ARCHITECTURE.md` for diagrams and data flow.
