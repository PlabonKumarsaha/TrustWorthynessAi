# Cross-Domain Weed Species Classification: Full Project Report

**Research question.** Can classifiers trained on *freely available, globally-sourced web imagery*
(GBIF) be deployed on *Australian field imagery* (DeepWeeds) without any local annotation — and
how far can the resulting domain gap be closed?

**Hardware.** All experiments on a single Apple M4 (24 GB, MPS backend). No GPU cluster.

---

# 1. Datasets

## 1.1 Source domain — GBIF "global" weed imagery

Built by querying the **Global Biodiversity Information Facility** for occurrence photographs of
eight weed species, deliberately **excluding Australia** so the source and target domains differ.

| Property | Value |
|---|---|
| Origin | GBIF citizen-science / herbarium occurrence photos (iNaturalist, Pl@ntNet, etc.) |
| Geographic filter | `EXCLUDE_COUNTRY = "AU"` — non-Australian only |
| Licence filter | CC0 and CC-BY only |
| Cap | 400 images per class (3,200 total) |
| Splits | 70/15/15 → 280 train / 60 val / 60 test per class |
| Format | 224×224 RGB JPEG, 5–23 KB (median ~13 KB) |
| Character | Curated/deliberate plant, flower and specimen shots |

**Species (matched by scientific name, not common name):**

| idx | Folder | Scientific name | Match quality |
|---|---|---|---|
| 0 | chinee_apple | *Ziziphus mauritiana* | exact species |
| 1 | lantana | *Lantana camara* | exact species |
| 2 | parkinsonia | *Parkinsonia aculeata* | exact species |
| 3 | parthenium | *Parthenium hysterophorus* | exact species |
| 4 | prickly_acacia | *Vachellia nilotica* (syn. *Acacia nilotica*) | exact, updated nomenclature |
| 5 | rubber_vine | *Cryptostegia grandiflora* | exact species |
| 6 | siam_weed | *Chromolaena odorata* | exact species |
| 7 | snake_weed | ***Stachytarpheta*** | **genus level only** ⚠ |

⚠ `snake_weed` is the one class matched only at genus level. It behaves measurably worse
throughout (it becomes a "sink" class), which is consistent with a heterogeneous source class.

## 1.2 Target domain — DeepWeeds (Australian)

| Property | Value |
|---|---|
| Origin | In-situ photographs, northern Australian rangelands (Queensland) |
| Size | **17,509 images**, native 256×256 |
| Classes | 8 weed species (8,403 images) + **"negative"** (9,106 images) |
| Natural weed distribution | 1,009–1,125 per class — **near-balanced** |
| Access | TensorFlow Datasets `deep_weeds`; repo `AlexOlsen/DeepWeeds` |
| Character | Cluttered field scenes: natural light, soil, occlusion, varied scale |

**Note:** over half the dataset (9,106 / 17,509) is *negative* — vegetation that is not a target
weed. Early experiments dropped this class; the final evaluation includes it.

## 1.3 Evaluation subsets used

| Subset | Composition | Used in |
|---|---|---|
| 4-species | 1,200 imgs (300/class), negatives dropped | Phases 2–6 |
| 8-species balanced | 1,600 imgs (200/class), negatives dropped | Phases 1, 7–8 |
| **Full realistic** | **17,509 imgs, natural distribution, negatives included** | Phases 9–10 |

---

# 2. Related work (state of the art)

## 2.1 The benchmark itself
- **DeepWeeds** (Olsen et al., *Scientific Reports* 2019): ResNet-50 **95.7%**, Inception-v3 95.1%
  — both *in-domain, fully supervised*. This is the ceiling reference.
- Recent in-domain work reaches **98.5%** using latent-diffusion data augmentation.

## 2.2 Domain adaptation in agriculture
- **UDA with self-training for weed segmentation** (Computers and Electronics in Agriculture, 2024)
  — teacher–student with EMA, pseudo-labelled target.
- **UDA for weed segmentation via greedy pseudo-labelling** (CVPR-W 2024).
- **From Web Data to Real Fields** (2025) — internet imagery → unlabelled robot field data via an
  attention-based adversarial discriminator (MAAD); +7.5% detection, +5.1% keypoint.
- **Domain Adaptation for Big Data in Agricultural Image Analysis** (2025 review) — taxonomy:
  discrepancy (CORAL/MMD), adversarial (DANN), style transfer, pseudo-labelling, Fourier, TTA, VLM.
- Cross-dataset plant disease transfer typically collapses (PlantVillage→PlantDoc ≈ **68%**).

## 2.3 Source-free / test-time adaptation
- **TENT** (ICLR 2021) — entropy minimisation over BatchNorm affine parameters.
- **AdaBN** — recompute BN statistics on the target.
- **SHOT** — frozen source classifier + information maximisation + pseudo-labels.
- **CBST** — class-balanced self-training.
- **Source-Free UDA survey** (Neurocomputing 2024); **TTA survey** (arXiv 2303.15361).

## 2.4 Vision-language models
- **Tip-Adapter** (ECCV 2022) — training-free cache adapter; **CoOp**, **CLIP-Adapter**.
- **ReCLIP** (2023) — cross-modal agreement pseudo-labels for source-free CLIP adaptation.
- **Co-learn++**, **PADCLIP**, **CLIP-powered dual-branch SFUDA** (2024) — CLIP as pseudo-label teacher.
- **RCL** (2024/25) — MLLM-guided reliability curriculum, **SOTA, +9.4% DomainNet**.
- **AgriCLIP**, **CropVLM**, **WeedCLR**, **AgroBench**, *"Are VLMs ready to zero-shot replace
  supervised classification in agriculture?"* (2025) — VLMs still weak on fine-grained agriculture.
- **iNatAg** — 4.7M images, 2,959 crop/weed species from citizen-science sources.

## 2.5 Foundation models vs. domain adaptation ⚠ directly relevant
- **Rethinking the Backbone in Class-Imbalanced Federated Source-Free DA** (2025) — argues for
  *"rethinking the focus from enhancing DA methods to improving feature extractors"*.
- **Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection** (2025).
- **Do Foundation Models Truly Outperform Domain-Specific Models?** (digital pathology).

**Implication:** the claim "a strong backbone matters more than the DA method" is **already
published** for other settings. Our contribution must be a *nuance* on it, not a restatement.

---

# 3. Approaches attempted, with benchmarks

## Phase 1 — VLM baseline (CLIP), 8 species, balanced subset

| Model | GBIF test | DeepWeeds | Drop |
|---|---|---|---|
| CLIP ViT-B/32 zero-shot (single prompt) | 35.4% | 12.3% | −23.1 |
| CLIP ViT-B/32 zero-shot (prompt ensemble) | 38.8% | 11.7% | −27.1 |
| CLIP + LoRA (1.16% of params trainable) | 88.5% | 35.2% | −53.3 |

Chance = 12.5%. Zero-shot CLIP is **at chance** on the target.

## Phase 1b — Test-time adaptation on CLIP (label-free), 8 species

| Backbone | base | +DA | +TTA | +TTA+DA | TENT |
|---|---|---|---|---|---|
| ViT-B/32 | 11.7% | 14.9% | 11.5% | 15.4% | 12.5% |
| ViT-L/14 (laion2b) | 14.8% | 26.2% | 15.2% | **27.5%** | — |

TENT **failed** (macro-F1 collapsed to 0.03) — entropy minimisation reinforces confident-but-wrong
predictions when the base model is near chance.

## Phase 2 — Three model families, 4 species (chance 25%)

| Model | GBIF (in-domain) | DeepWeeds | Gap |
|---|---|---|---|
| DL — ResNet-50 fine-tuned | 94.6% | 65.3% | −29.3 |
| VLM — CLIP + LoRA | 98.8% | 58.4% | −40.3 |
| VLM — CLIP zero-shot | 80.8% | 28.1% | −52.8 |
| ML — Random Forest (HSV + GLCM) | 68.3% | 27.7% | −40.7 |

**Finding:** highest in-domain ≠ best out-of-domain. CLIP+LoRA wins in-domain but overfits the
source domain harder than ResNet-50.

## Phase 3 — Preprocessing techniques (label-free), 4 species

| Technique | ML | DL | VLM |
|---|---|---|---|
| Baseline | 27.7% | 66.6% | 67.2% |
| **CLAHE** | **30.8% (+3.2)** | **68.3% (+1.8)** | 67.2% (+0.1) |
| Histogram matching → GBIF | 29.9% | 65.4% (−1.2) | 68.5% (+1.3) |
| Fourier Domain Adaptation (β=0.01) | 27.0% | 65.3% (−1.2) | 68.5% (+1.3) |
| Reinhard colour transfer | 26.8% | **46.0% (−20.6)** | **37.2% (−30.0)** |

**Findings:** CLAHE is the only universally safe gain. **Reinhard colour transfer is catastrophic** —
forcing source colour statistics onto the target destroys discriminative colour. The best
preprocessing is **model-dependent** (CNN prefers contrast normalisation; CLIP prefers spectral).

## Phase 4 — Domain adaptation, 4 species

| Method | DeepWeeds | Target labels |
|---|---|---|
| Baseline (best single model) | ~67% | 0 |
| AdaBN-ResNet | 63.5% (**worse**) | 0 |
| TENT-ResNet | 67.8% | 0 |
| AdaBN-ResNet + CLIP | 71.0% | 0 |
| ResNet + CLIP ensemble | 73.3% | 0 |
| TENT-ResNet + CLIP | 73.3% | 0 |
| CLIP-guided self-training (ResNet) | 82.8% | 0 |
| **Self-training + CLIP ensemble** | **84.1%** | **0** |
| Few-shot 5/class | 80.1–81.3% | 20 |
| Few-shot 10/class (ensemble) | **89.2%** | 40 |
| Few-shot 20/class | 88.4% | 80 |

Self-training worked because the ensemble teacher was **98% precise** on its confident subset.
**Tip-Adapter barely moved** (67.8–68.8%) — frozen CLIP features aren't discriminative enough here.

## Phase 5 — Conformal cross-architecture selection (our own method attempt)

Controlled comparison — identical self-training pipeline, only the pseudo-label selection rule differs:

| Selection rule | Coverage | Precision | Final ResNet | Final ResNet+CLIP |
|---|---|---|---|---|
| Confidence ≥ 0.9 | 35% | 97% | 73.1% | 75.6% |
| **Conformal + cross-arch agreement** | **51%** | **98%** | **77.5%** | **80.3%** |

**+4.7 points** from the selection rule alone. Supporting evidence: **72.8% of all errors fall in
the cross-architecture disagreement set**; agreed subset precision 87.6% vs 53.3% disagreed.

## Phase 6 — DINOv2 and resolution scaling, 4 species

| Model | GBIF | DeepWeeds |
|---|---|---|
| DINOv2 @224 (linear probe) | 99.2% | 78.2% |
| DINOv2 @336 | 99.6% | 88.2% |
| DINOv2 @448 | — | 90.8% |
| + label-free self-training (3 rounds) | — | **94.2%** |

Also tested and **failed**: multi-resolution concat (88.8%) and prob-average (90.1%), both below
best single resolution; and naive ensembling of DINOv2 with weaker models (88.2% → **85.2%**).

## Phase 7 — Full 8-species validation (chance 12.5%)

| Method | DeepWeeds |
|---|---|
| DINOv2 @336 | 57.8% |
| DINOv2 @448 | 63.8% |
| + naive self-training (3 rounds) | 68.8% |
| + Sinkhorn DA only | 73.2% |
| + CBST only | 71.8% |
| **+ DA + CBST** | **80.6%** (macro-F1 0.806) |

⚠ **The 4-species 94.2% did NOT survive** → 68.8%. Subset evaluation inflated the result by ~25 pts.

**Mechanism found:** naive self-training *amplifies* class bias (pseudo-label precision decays
81.3% → 77.2% → 74.9%); with distribution alignment it *holds* (89.6% → 89.0% → 87.6%).

## Phase 8 — Factorial study (backbone × resolution × method)

| Backbone @ res | in-domain | none | DA | ST | CBST | DA+CBST |
|---|---|---|---|---|---|---|
| ResNet-50 @224 | 79.4 | 29.8 | 33.2 | 29.1 | 29.8 | 36.0 |
| ResNet-50 @336 | 81.9 | 33.2 | 38.0 | 32.6 | 33.9 | 41.2 |
| ResNet-50 @448 | 84.0 | 37.6 | 41.8 | 37.5 | 37.8 | 45.6 |
| CLIP ViT-B/32 @224 | 85.8 | 34.2 | 35.4 | 32.8 | 35.6 | 38.6 |
| DINOv2 @224 | 93.1 | 54.6 | 64.2 | 56.0 | 55.6 | 72.9 |
| DINOv2 @336 | 94.2 | 57.8 | 69.1 | 60.8 | 62.8 | 79.8 |
| DINOv2 @448 | 95.2 | 63.8 | 73.2 | 68.8 | 71.8 | 80.6 |

**Spread attributable to each factor:** backbone **30.8 pp** > adaptation method **11.9 pp** >
resolution **9.8 pp**.

**Hypothesis tested and REFUTED:** method rankings do *not* collapse across backbones —
Spearman ρ = **0.80–0.90**; ordering (`DA+CBST > DA/CBST > none/ST`) is stable.

**Better-supported finding:** adaptation gains **grow with representation quality** —
DA+CBST adds **+6.2 pp** on ResNet-50@224 but **+16.8 pp** on DINOv2@448. Backbone and
adaptation are **complementary, not substitutes** — a nuance on the 2025 "foundation models
replace DA" literature.

## Phase 9 — Realistic deployment (FULL 17,509 images, natural distribution, negatives included)

| Method (weeds only, n = 8,403, natural imbalance) | Accuracy | macro-F1 |
|---|---|---|
| No adaptation | 64.9% | 0.653 |
| + DA (uniform prior) | 72.9% | 0.730 |
| + DA (oracle prior) | 73.2% | 0.732 |
| + DA (**EM-estimated** prior, label-free) | **61.3%** ← *worse than nothing* | 0.613 |
| **+ DA + CBST (uniform), 3 rounds** | **80.1%** | **0.801** |

**The headline result held** (80.1% vs 80.6% on the balanced subset) because DeepWeeds' weed
classes are *naturally* near-balanced — so the earlier subsampling was not distorting.

**FAILURE:** label-free prior estimation via EM (Saerens et al.) has **L1 error 0.577** and makes
adaptation *worse than no adaptation*. The DA gain depends on prior knowledge.

## Phase 10 — Open-set rejection (9,106 unseen negatives)

| Score | AUROC | AUPR | **FPR@95TPR** |
|---|---|---|---|
| **kNN (feature-space, k=10)** | **0.808** | 0.806 | **63.8%** |
| Energy | 0.772 | 0.774 | 76.4% |
| MaxLogit | 0.770 | 0.773 | 75.6% |
| Mahalanobis | 0.769 | 0.768 | 76.3% |
| fuse kNN+CLIPtext | 0.752 | 0.746 | 73.1% |
| Entropy | 0.733 | 0.740 | 78.8% |
| fuse kNN+CLIPtext+XArch | 0.728 | 0.733 | 82.2% |
| MSP | 0.718 | 0.733 | 83.7% |
| Relative Mahalanobis | 0.677 | 0.706 | 91.9% |
| CLIP text-defined negatives | 0.634 | 0.599 | 85.3% |
| Cross-architecture disagreement | 0.545 | 0.545 | 94.4% |

**Only kNN improved** on the baseline. Two of our own proposals failed: **language-defined
negatives** (0.634) and **cross-architecture disagreement as an OOD signal** (0.545 ≈ chance).
Score **fusion degraded** the best score.

---

# 4. Current findings and status

## 4.1 Headline result
> Training **only on free GBIF web imagery**, with **zero Australian labels**, the system reaches
> **80.1% accuracy (macro-F1 0.801)** on the full, naturally-distributed DeepWeeds weed set —
> up from 64.9% unadapted, and from 11.7% for zero-shot CLIP.
> Fully-supervised in-domain reference: 95.7%.

## 4.2 Confirmed findings
1. **Backbone dominates** — DINOv2 over ResNet-50 was worth more than every adaptation method combined.
2. **Resolution is a first-class lever** — 224→448 consistently pays (up to +12 pts).
3. **Adaptation gains scale *with* backbone quality** (+6.2 → +16.8 pp) — complementary, not redundant.
4. **Class-bias correction is essential at 8 classes** — invisible at 4 classes, worth +17 pts at 8.
5. **Distribution alignment stops self-training from degrading itself** (precision holds ~88% vs decaying to 75%).
6. **Subset evaluation inflates cross-domain results** by ~25 pts (4 species 94.2% → 8 species 68.8%).

## 4.3 Confirmed failures (well-evidenced negative results)
1. **Label-free prior estimation fails** (EM L1 0.577) and backfires (61.3% vs 64.9%).
2. **Open-set rejection is not deployable** — best FPR@95TPR = 63.8%.
3. **Reinhard colour transfer** is catastrophic (−20/−30 pts).
4. **AdaBN alone hurts**; **TENT fails** on a near-chance base model.
5. **Tip-Adapter** ≈ zero-shot on this fine-grained task.
6. **Language-defined negatives** and **cross-architecture disagreement** fail for OOD detection.
7. **Combining unequal-quality signals consistently hurts** — observed 3× independently
   (backbone ensembling, pseudo-label fusion, OOD score fusion).

## 4.4 Limitations (honest)
- **No published baseline implemented** (SHOT / CORAL / DANN) — the largest credibility gap.
- **Seeds did not produce variance**: the factorial's "3 seeds" gave std = 0.0 because the
  linear-probe + selection pipeline is deterministic. **We have no true variance estimate.**
- **Frozen linear-probe protocol** handicaps ResNet-50 (29.8% frozen vs 66.6% fine-tuned), so the
  30.8 pp backbone spread is partly an artefact of the evaluation design.
- **Transductive** adaptation — needs the unlabelled target batch in hand.
- **Single target dataset**; no CV folds; no significance testing.
- `snake_weed` matched at genus level only.

## 4.5 Recommended next steps
1. Implement **SHOT and CORAL** baselines (cheap on cached features) — closes the biggest gap.
2. Real variance: resample data splits / bootstrap, then significance tests.
3. A **second target dataset** (e.g. CWFID) to show the finding is not dataset-specific.
4. Treat **open-set rejection** as the primary open problem — it is the binding constraint.

## 4.6 Publication assessment
This is a strong **application / benchmark** contribution, not a novel-method one. The
"backbone > DA method" claim is already published elsewhere; our defensible nuance is that
**adaptation and representation quality are complementary and super-additive**, plus a
well-characterised set of failure modes. Realistic targets: *Computers and Electronics in
Agriculture* or *Frontiers in Plant Science*, framed as **annotation-free deployment with a
characterised open problem**.

---

# 5. Code map

| File | Purpose |
|---|---|
| `bench_common.py`, `bench_ml.py`, `bench_dl.py`, `bench_vlm.py` | 4-species tri-model benchmark |
| `prep_common.py`, `prep_{clahe,fda,histmatch,reinhard}.py` | preprocessing techniques |
| `adapt_{adabn,tent,ensemble,pseudolabel,fewshot}.py` | domain adaptation suite |
| `novel_reliability.py`, `novel_method.py` | conformal cross-architecture selection |
| `accuracy_push*.py` | DINOv2 + resolution scaling, 8-species, class-balance fix |
| `qi_features.py`, `qi_factorial.py` | factorial study |
| `realistic_deployment.py`, `realistic_eval.py`, `openset.py` | full realistic + open-set evaluation |

Results: `results_bench/`, `results_prep/`, `results_adapt/`, `results_novel/`.
Datasets, model weights and feature caches are git-ignored.
