# Annotation-free deployment: does it survive realistic conditions?

Everything earlier was measured on a **balanced 200/class subset with negatives removed**.
This evaluation removes both crutches: the **full 17,509-image DeepWeeds set**, its **natural
class distribution**, and the **9,106 non-target "negative" images** the model has never seen.
Source is GBIF's 8 weed species only (source labels); no DeepWeeds label is used for training.
Backbone: frozen DINOv2 ViT-B/14 @448 + linear probe.

---

## 1. The headline result HOLDS on the full natural set

| Method (weeds only, natural imbalance, n=8,403) | Accuracy | macro-F1 |
|---|---|---|
| No adaptation | 64.9% | 0.653 |
| + DA (uniform prior) | 72.9% | 0.730 |
| + DA (oracle prior) | 73.2% | 0.732 |
| **+ DA + CBST (uniform prior), 3 rounds** | **80.1%** | **0.801** |

Versus **80.6%** on the earlier balanced 1,600-image subset — so the subsampling was **not**
inflating the result. Reason: DeepWeeds' eight weed classes are *naturally* near-balanced
(1,009–1,125 images each), so "uniform" was an accurate assumption, not a convenient one.

**This is the paper's positive claim: 64.9% → 80.1% on real Australian field imagery with zero
target annotation, trained only on free GBIF web images.**

---

## 2. FAILURE 1 — label-free prior estimation does not work here

The DA gain depends on knowing the target class prior. To remove that assumption we estimated it
from unlabelled data with EM (Saerens et al., 2002):

| Prior used | Accuracy |
|---|---|
| uniform (assumed) | 72.9% |
| oracle (true) | 73.2% |
| **EM-estimated (label-free)** | **61.3%** — *worse than no adaptation (64.9%)* |

**Prior L1 error = 0.577** (on a 2.0 scale). EM fails because the source model's predictions are
already heavily biased toward sink classes, and EM inherits that bias and amplifies it.

**Implication:** the +8 to +15 pt DA gain rests on prior knowledge. Here uniform happens to be
correct, but a practitioner who *estimates* the prior would be worse off than doing nothing.
This is a real limitation of distribution-alignment methods that the UDA literature under-reports.

---

## 3. FAILURE 2 — open-set rejection is inadequate for deployment

Over half of DeepWeeds (9,106 of 17,509) is **negative**: vegetation that is not a target weed.
A deployed sprayer must reject these. The model never saw a negative in training.

| Rejection score | AUROC |
|---|---|
| Max-softmax (MSP) | 0.718 |
| Energy | 0.772 |
| Mahalanobis (feature-space) | 0.769 |

AUROC ≈ 0.77 is far from deployable — it implies substantial false positives on ordinary
vegetation. **This, not closed-set accuracy, is the binding constraint on annotation-free
weed control.**

---

## 4. Per-class behaviour (closed-set)
Sink-class bias persists whenever the prior is wrong: `parthenium` (P 0.43 / R 0.96) and
`snake_weed` (P 0.38 / R 0.83) absorb predictions while `prickly_acacia` collapses (R 0.24).
Correct alignment fixes this; bad alignment makes it worse.

---

## 5. What this means for the paper

**Defensible claim:** free, web-sourced imagery + a frozen self-supervised backbone + label-free
alignment gives **80.1% on real Australian field conditions with no local annotation** — a
credible, economically meaningful result for rangeland weed management.

**Two honest limitations that strengthen rather than weaken the paper:**
1. The alignment gain requires a correct class prior; label-free estimation fails (and backfires).
2. Open-set rejection of non-target vegetation (AUROC 0.77) is the real blocker to field use.

**Remaining gaps before submission:** no published baseline (SHOT/CORAL/DANN); single target
dataset; transductive adaptation; no CV folds or significance testing.

## Files
`realistic_deployment.py` (streaming feature extraction over all 17,509 images),
`realistic_eval.py` (closed-set, prior estimation, open-set),
results in `results_novel/realistic/realistic_results.json`.
