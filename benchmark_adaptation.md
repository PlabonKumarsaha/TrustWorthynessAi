# Closing the GBIF→DeepWeeds gap to ≥80% (4 species)

**Goal.** Raise DeepWeeds (Australian target) accuracy from the ~66–67% source-trained baseline to
**≥80%**, using literature-backed domain-adaptation methods. Models trained on GBIF stay frozen
unless a method explicitly adapts them. Target = 1,200 images (300/class), chance = 25%.
In-domain GBIF ceiling ≈ 94.6% (ResNet).

Few-shot uses a **held-out DeepWeeds pool** (400 imgs, disjoint from the test set) → no leakage.

## Results (sorted by DeepWeeds accuracy)

| Method | DeepWeeds acc | Setting | Target labels used |
|---|---|---|---|
| Ensemble ResNet+CLIP+Tip (10-shot) | **89.2%** | few-shot | 40 (10/class) |
| Ensemble (20-shot) | 88.4% | few-shot | 80 |
| ResNet linear-probe (10-shot) | 87.8% | few-shot | 40 |
| **Self-trained ResNet + CLIP** | **84.1%** | **label-free** | **0** |
| CLIP linear-probe (20-shot) | 83.1% | few-shot | 80 |
| **Self-trained ResNet** | **82.8%** | **label-free** | **0** |
| Ensemble (5-shot) | 81.3% | few-shot | 20 |
| CLIP linear-probe (10-shot) | 81.3% | few-shot | 40 |
| ResNet linear-probe (5-shot) | 80.1% | few-shot | 20 |
| ResNet + CLIP ensemble | 73.3% | label-free | 0 |
| TENT-ResNet + CLIP | 73.3% | label-free | 0 |
| AdaBN-ResNet + CLIP | 71.0% | label-free | 0 |
| TENT-ResNet | 67.8% | label-free | 0 |
| CLIP only (baseline) | 67.2% | label-free | 0 |
| ResNet only (baseline) | 66.6% | label-free | 0 |
| AdaBN-ResNet | 63.5% | label-free | 0 |

## The two ways we cleared 80%

1. **Fully label-free — CLIP-guided self-training → 84.1%.** Use the ResNet+CLIP ensemble as a
   teacher, keep only high-confidence pseudo-labels (301 then 498 of 1,200; **98% correct**), and
   fine-tune the ResNet on them. No ground-truth DeepWeeds labels at all. **+17.5 points** over
   the ResNet baseline. This is the headline result.
2. **Few-shot — 5 labels/class → 80–81%, 10 labels/class → 89%.** A linear probe on ResNet
   features plus CLIP Tip-Adapter, ensembled. Just 20 labelled images total crosses 80%.

## What worked, what didn't
- **Ensembling ResNet+CLIP is the biggest free lever** (+6–7 pts) — the two models make
  different errors.
- **Self-training is the star label-free method** — it works *because* the ensemble teacher is
  highly accurate on its confident subset (98%), so the pseudo-labels are clean.
- **AdaBN alone hurt** this ResNet (63.5%) — target BN stats were noisier than the source stats;
  a documented failure mode. **TENT** (which also learns BN affine) recovered to +1.2.
- **Tip-Adapter stayed near zero-shot** here — the frozen CLIP features aren't discriminative
  enough on these species for a cache model to help much; linear-probing the features worked better.

## Takeaway for the write-up
- The domain gap (≈28 pts) is **almost fully closable**: label-free self-training reaches 84%
  (within ~11 pts of the in-domain ceiling), and 10 labels/class reaches 89% (within ~6 pts).
- Progression: baseline 67% → ensemble 73% → **self-training 84% (0 labels)** → few-shot 89%.

## Files (one technique per file)
`adapt_common.py`, `adapt_adabn.py`, `adapt_tent.py`, `adapt_ensemble.py`,
`adapt_pseudolabel.py`, `adapt_fewshot.py`, `adapt_benchmark.py` — results in `results_adapt/`.
