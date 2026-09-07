# Closing the web→field gap on DeepWeeds (label-free)

**Setup.** Train on GBIF (global, non-Australian) using **source labels only**; test on DeepWeeds
(Australian field). No DeepWeeds label is ever used for training — only for scoring.
Self-training is transductive (adapts on the unlabelled test images it then scores).

---

## HEADLINE — full 8-species task (chance 12.5%, 1,600 test images)

| Method (all label-free) | DeepWeeds acc | macro-F1 |
|---|---|---|
| CLIP zero-shot | 11.7% | 0.07 |
| CLIP + LoRA | 35.2% | 0.35 |
| DINOv2 @336 | 57.8% | — |
| DINOv2 @448 | 63.8% | — |
| + naive self-training | 68.8% | 0.691 |
| + Sinkhorn distribution alignment (DA) only | 73.2% | — |
| + class-balanced self-training (CBST) only | 71.8% | — |
| **+ DA + CBST (3 rounds)** | **80.6%** | **0.806** |

GBIF in-domain (same model) = **95.2%** → remaining domain gap ≈ **14.6 pts** (was 31.4).

### Per-class F1 before vs after the class-balance fix
| Species | naive self-training | DA + CBST |
|---|---|---|
| rubber_vine | 0.895 | 0.902 |
| chinee_apple | 0.852 | 0.880 |
| parkinsonia | 0.830 | 0.861 |
| lantana | 0.638 | **0.841** |
| parthenium | 0.607 | **0.809** |
| snake_weed | 0.626 | **0.730** |
| siam_weed | 0.648 | 0.719 |
| prickly_acacia | 0.433 | **0.708** |

---

## What actually mattered

1. **Backbone dominates.** ResNet-50 → DINOv2 (self-supervised ViT) was the single biggest lever;
   on 4 species it was worth +22 pts. Bigger than every adaptation technique combined.
2. **Resolution is a major lever.** 224→336→448 kept paying (4-species: 78.2→88.2→90.8;
   8-species: —→57.8→63.8).
3. **Class-bias correction is essential at 8 classes.** Without it, `parthenium` and `snake_weed`
   became *sink* classes (precision ~0.45–0.49, recall ~0.87–0.96) that starved
   `prickly_acacia` (recall 0.32), `lantana` (0.49) and `siam_weed` (0.53).
   Distribution alignment + class-balanced selection recovered **+16.8 pts** over the base model
   and **+11.8** over naive self-training.
4. **Mechanism:** naive self-training *amplified* the bias — pseudo-label precision decayed
   81.3% → 77.2% → 74.9% across rounds. With DA it **held**: 89.6% → 89.0% → 87.6%.
   Alignment stops self-training from eating itself.
5. **Didn't work:** multi-resolution fusion (below best single resolution); naive equal-weight
   ensembling of a strong model with weak ones (DINOv2 88.2% → 85.2% when averaged with
   ResNet/CLIP at ~67%). Ensembling only helps between *comparable* experts.

---

## 4-species result (kept for reference — and as a caution)

On the easier 4-species subset (chance 25%) the same pipeline reached **94.2%**
(DINOv2 @448 + self-training), vs 95.7% for the *fully-supervised in-domain* DeepWeeds benchmark.
**That number did not survive the move to 8 species** (→ 68.8% naive, 80.6% balanced).
Small-subset results on this task are strongly optimistic — always report the full class set.

| Stage (4 species) | DeepWeeds |
|---|---|
| ResNet-50 | 66.6% |
| ResNet+CLIP ensemble | 73.3% |
| CLIP-guided self-training | 84.1% |
| few-shot (10 labels/class) | 89.2% |
| DINOv2 @448 | 90.8% |
| DINOv2 @448 + self-training | 94.2% |

---

## Honest caveats
- Single seed, single split; no CV folds or significance tests yet.
- Self-training is **transductive** (needs the unlabelled target batch in hand).
- DA assumes a roughly **balanced** target class distribution — true here by construction
  (200/class); in a real deployment the prior would have to be estimated.
- 448px inference is heavier — matters for on-robot deployment.

## Next steps
1. Multiple seeds + DeepWeeds official CV folds + significance testing.
2. DINOv2 ViT-L and native 518px.
3. Estimate the target prior rather than assuming uniform (removes the DA assumption).
4. Compare head-to-head against SHOT / CBST / ReCLIP as published baselines.

## Files
`accuracy_push.py` (4-species), `accuracy_push2.py` (resolution + self-training),
`accuracy_push8.py` (8-species), `accuracy_push8_balanced.py` (class-balance fix).
Results in `results_novel/accuracy_push*.json`; DINOv2 features cached in `results_novel/feat_cache*/`.
