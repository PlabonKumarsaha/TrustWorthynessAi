# Cross-domain weed classification: progress summary

**Research question.** Can models trained on *globally-sourced* (web/GBIF) weed imagery generalise
to *Australian field* conditions (DeepWeeds), and how far can we close the domain gap — ideally
without annotating Australian data?

**Compute.** All experiments on a single Apple M4 (24 GB, MPS). Code + per-experiment result JSON
saved; models/datasets git-ignored.

---

## 1. Datasets

| Dataset | Role | Origin | Content | Size used |
|---|---|---|---|---|
| **GBIF "global"** | Source (train + in-domain test) | GBIF citizen-science, **Australia excluded**, CC0/CC-BY only | Curated/specimen photos of the weed species | 400/class (280 train / 60 val / 60 test) |
| **DeepWeeds** | Target (out-of-domain test) | In-situ Australian rangeland (Queensland) | Field photos, same species | Full = 17,509 imgs; balanced subsets used |

Same 8 species exist in both (chinee apple, lantana, parkinsonia, parthenium, prickly acacia,
rubber vine, siam weed, snake weed). **Species matched by scientific name** (snake weed matched at
genus level only). The core benchmark uses **4 clean, distinct species** — lantana, parkinsonia,
parthenium, rubber vine (chance = 25%); DeepWeeds test = 300/class (1,200 imgs).

---

## 2. Techniques used

**Models (each trained on GBIF, tested on both domains):**
- **ML** — Random Forest on hand-crafted colour (HSV) + texture (GLCM) features
- **DL** — ResNet-50 (ImageNet-pretrained) fine-tuned
- **VLM** — CLIP ViT-B/32: zero-shot (text prompts) and LoRA fine-tuned

**Gap-closing techniques tried:**
- *Preprocessing* (label-free): CLAHE, histogram matching, Fourier Domain Adaptation (FDA), Reinhard colour transfer
- *Test-time / source-free* (label-free): AdaBN, TENT, TTA + distribution alignment, model ensembling, **CLIP-guided self-training**
- *Few-shot* (a few target labels): CLIP Tip-Adapter, linear probes, ensemble

---

## 3. Results BEFORE modification — the domain gap (4 species)

Trained on GBIF, tested in-domain (GBIF) vs out-of-domain (DeepWeeds):

| Model | GBIF (in-domain) | DeepWeeds (target) | Gap |
|---|---|---|---|
| DL — ResNet-50 | 94.6% | 65–67% | ≈ −28 pts |
| VLM — CLIP LoRA | 98.8% | 58–67% | ≈ −33 pts |
| VLM — CLIP zero-shot | 80.8% | 28.1% | −52.8 pts |
| ML — Random Forest | 68.3% | 27.7% | −40.6 pts |

*(Reference: DeepWeeds published in-domain SOTA ≈ 95.7%. Chance = 25%.)*

**Finding:** every model drops sharply on Australian data. Best source-trained model on the target
≈ **67%**. Highest in-domain accuracy (CLIP LoRA) did **not** give the best target accuracy — it
over-fits the source domain.

---

## 4. Results AFTER modification — closing the gap (4 species, DeepWeeds target)

Baseline (source-trained, no adaptation): ResNet 66.6%, CLIP 67.2%.

### 4a. Preprocessing (label-free) — small gains
| Technique | ML | DL | VLM |
|---|---|---|---|
| CLAHE | 30.8% (+3.2) | **68.3% (+1.8)** | 67.2% |
| Histogram matching | 29.9% | 65.4% | 68.5% (+1.3) |
| FDA | 27.0% | 65.3% | 68.5% (+1.3) |
| Reinhard colour transfer | 26.8% | 46.0% (−20.6) | 37.2% (−30.0) ⚠ harmful |

### 4b. Domain adaptation — the real gains
| Method | DeepWeeds acc | Target labels |
|---|---|---|
| Baseline (best single model) | ~67% | 0 |
| Ensemble ResNet + CLIP | 73.3% | 0 |
| TENT-ResNet + CLIP | 73.3% | 0 |
| AdaBN-ResNet + CLIP | 71.0% | 0 |
| **CLIP-guided self-training (ResNet)** | **82.8%** | **0** |
| **Self-training + CLIP ensemble** | **84.1%** | **0** |
| Few-shot, 5 labels/class | 80.1–81.3% | 20 |
| Few-shot, 10 labels/class | 87.8–**89.2%** | 40 |
| Few-shot, 20 labels/class | 88.4% | 80 |

---

## 5. Before → After summary

| | DeepWeeds accuracy |
|---|---|
| Source-trained baseline (best) | ~67% |
| + label-free adaptation (self-training) | **84.1%** (+17 pts, **0 target labels**) |
| + few-shot (10 labels/class) | **89.2%** (+22 pts) |
| In-domain ceiling (no shift) | ~94.6% |

**The ~28-point domain gap was closed to ~11 pts with no Australian labels, and ~6 pts with 40
labelled images.** The label-free result works because the ResNet+CLIP ensemble teacher produced
pseudo-labels that were 98% accurate on its confident subset.

---

## 6. Key findings
1. Web-sourced (GBIF) training transfers to Australian field imagery, but with a large gap (~28 pts).
2. **Ensembling ResNet + CLIP** is the biggest free lever (+6–7 pts).
3. **Self-training is the standout label-free method** — reaches 84% with zero target labels.
4. **Negative results:** Reinhard colour transfer, and AdaBN-alone, *hurt*; Tip-Adapter barely helped.
5. Highest in-domain ≠ best on target — don't select deployment models on in-domain accuracy.

## 7. Honest limitations (for planning)
- Only **4 of 8 species**; **single seed** — CLIP LoRA showed ~9-pt run-to-run variance.
- No comparison yet to standard UDA baselines (DANN, CORAL, SHOT).
- All methods are off-the-shelf (no methodological novelty yet).
- These are current gaps between the work and a top-tier (Q1) publication.

## 8. Proposed next steps
1. Scale to **all 8 species** with **DeepWeeds official CV folds**, multiple seeds + significance.
2. Add **DANN / CORAL / SHOT** baselines for a fair comparison.
3. Ablations (confidence threshold, teacher choice, backbone) + feature-space analysis.
4. Decide publication target: solid **Q2** now, or invest in genuine novelty for a **Q1** attempt.
