# Architecture — MCR-UOD causal package

Diagrams for the causal half of *Uncertainty-guided causal reasoning for
multimodal object detection*. See `../README.md` for the interface contract.

## 1. Overall pipeline

```mermaid
flowchart LR
    IMG[UAV image] --> BACKBONE[YOLOv8x-Worldv2 backbone]
    BACKBONE --> F1[Low-level feat C1]
    BACKBONE --> F2[Mid-level feat C2]
    BACKBONE --> F3[High-level feat C3]

    F1 --> LGRE[LGRE: existence map s]
    LGRE --> TOPK[topk_select tau]
    TOPK --> BACR[BACR: do-operator over Z]
    DICT[(Confounder dict Z + prior P Z)] --> BACR
    BACR --> SCAT[scatter_back to C1'']
    SCAT --> HEAD[WorldDetect head]
    F2 --> HEAD
    F3 --> HEAD

    HEAD --> RAW[Raw boxes + scores]
    RAW --> CONF[Conformal calibration]
    CONF --> RERANK[Rerank]
    RERANK --> OUT[mAP / final detections]

    subgraph causal [causal team]
        LGRE
        TOPK
        BACR
        DICT
        SCAT
    end
    subgraph unc [uncertainty team]
        CONF
        RERANK
    end
```

## 2. Causal module data flow

```mermaid
flowchart TD
    X[feature map x: B,C,H,W] --> C2F[C2f trunk]
    TXT[class text feats: 1,nc,D2] --> LGRE
    C2F --> LGRE
    LGRE --> S[existence map s: B,1,H,W]
    S --> TOPK[topk_select tau]
    C2F --> TOPK
    TOPK --> F[selected feats F: B,N,D2]
    TOPK --> I[indices I: B,N]

    F --> QK[Q = Wq F]
    Z[[dict Z: S,D2]] --> KKV[K = Wk Z, V = Wv Z]
    PRIOR[[prior P Z: S]] --> ATT
    QK --> ATT[A = softmax Q K^T / sqrt D2]
    KKV --> ATT
    ATT --> FUSE["F' = f_F Cat(F, A V^T)"]
    F --> FUSE
    FUSE --> NWGM{use_nwgm?}
    NWGM -- yes --> FPRIME[F' after NWGM Eq.7]
    NWGM -- no --> FPRIME
    FPRIME --> SCAT[scatter_back at I]
    I --> SCAT
    SCAT --> ZNEW[C''1: B,C,H,W]

    FPRIME --> UPD[dict update Eq.11]
    UPD --> Z
```

## 3. Zoom strategies

```mermaid
flowchart TD
    S[existence map s] --> ROI[derive ROI boxes]
    ROI --> MODE{zoom.mode}

    MODE -- region --> RZ[RegionZoom.crop_and_rescale]
    RZ --> DET1[detect on zoomed patch]
    DET1 --> MB[RegionZoom.map_back to original frame]

    MODE -- feature --> FZ[FeatureZoom: causal scale de-bias]
    FZ --> FEAT[multiplied feature map]

    MODE -- both --> RZ
    MODE -- both --> FZ

    MB --> NMS[global NMS / merge]
    FEAT --> HEAD[detection head]
    HEAD --> NMS
```

## 4. Prompt intervention (MT2)

```mermaid
flowchart LR
    CLASS[class names] --> BANK[PromptBank: attribute-enriched prompts]
    ATTR[attributes: colour, shape, view] --> BANK
    BANK --> CLIP[CLIP text encoder]
    CLIP --> LOGITS[per-context logits: B,S,nc]
    PRIOR[[prior P Z: S]] --> PI[PromptIntervener]
    LOGITS --> PI
    PI --> OUT[P Y | do T logits: B,nc]
```

## 5. Ownership boundaries

```mermaid
flowchart LR
    subgraph causal_pkg [mcr_uod/causal]
        CConf[confounder.py]
        CLgre[lgre.py]
        CBacr[bacr.py]
        CNwgm[nwgm.py]
        CZoom[zoom/]
        CPrompt[prompt_intervene.py]
        CAdapter[adapter.py]
    end
    subgraph unc_pkg [mcr_uod/uncertainty]
        UConf[conformal.py]
        URerank[rerank.py]
        UPipe[pipeline.py]
    end
    CAdapter -->|join point| UPipe
    CBacr --> CAdapter
    CZoom --> CAdapter
    CPrompt --> CAdapter
```

## 6. Where this hooks into YOLO

`registry.py::register_causal_modules` monkey-patches the causal block into
`ultralytics.nn.modules` so `parse_model` can resolve it from a model YAML.
This replaces the upstream hook at
`ultralytics/nn/tasks.py::WorldModel.forward` (currently `C2fAttn`).
