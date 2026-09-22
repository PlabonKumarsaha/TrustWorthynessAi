# HD(LM)²D in reverse: train on DeepWeeds → shift-test on GBIF

Mirror of `hd_findings.md`. Same protocol (6 ImageNet backbones fine-tuned end to end, 12
classical heads refit on penultimate features, 78 combinations), with the domains swapped.

- **Train / val / in-domain test:** DeepWeeds, 5,039 / 1,677 / 1,687 (stratified 60/20/20 of 8,403)
- **Shift test:** GBIF, all 3,200 images
- 8 classes, chance 12.5%. Results in `results_hd_rev/`.

## 1. End-to-end fine-tuned models

| Backbone | DeepWeeds test | GBIF (shift) | Gap |
|---|---|---|---|
| regnet_y_32gf | **98.4%** | **34.4%** | 64.0 |
| resnet50 | 97.2% | 28.9% | 68.2 |
| densenet201 | 97.1% | 29.8% | 67.3 |
| vgg19 | 96.2% | 28.4% | 67.8 |
| mobilenet_v2 | 95.4% | 22.2% | 73.2 |
| vgg16 | 95.3% | 30.8% | 64.5 |

In-domain accuracy reproduces the published DeepWeeds benchmark (~95.7%) and exceeds it with
RegNetY (98.4%), so the source-domain training is sound. GBIF accuracy nonetheless collapses to
22–34% against a 12.5% chance level.

## 2. Mean gain of each head over the end-to-end model

| Head | In-domain gain | Improves | Shift gain | Improves |
|---|---|---|---|---|
| RF | **+0.20** | 4/6 | **+0.03** | 4/6 |
| LDA | +0.08 | 3/6 | −2.27 | 1/6 |
| LR | +0.01 | 3/6 | −0.25 | 3/6 |
| kNN | −0.00 | 2/6 | −0.41 | 3/6 |
| HGB | −0.13 | 2/6 | −1.84 | 2/6 |
| XGBoost | −0.24 | 0/6 | −1.96 | 1/6 |
| SVM | −0.32 | 1/6 | −0.84 | 3/6 |
| GradBoost | −0.44 | 1/6 | −2.69 | 0/6 |
| NB | −0.55 | 2/6 | −1.23 | 1/6 |
| Bagging | −0.81 | 0/6 | −1.63 | 1/6 |
| DT | −2.30 | 0/6 | −4.39 | 0/6 |
| AdaBoost | −31.24 | 0/6 | −7.69 | 0/6 |

## Findings

**1. The paper's gain shrinks further, to +0.20.**
Against their claimed **+16.56**, the best head here adds **+0.20** (random forest), and their
proposed SVM is **−0.32**. Eight of twelve heads are negative even in domain.

**2. Baseline strength predicts the gain — across all three studies.**
This is the clearest evidence that their effect is a broken-baseline artefact:

| Study | End-to-end baseline | Best head gain |
|---|---|---|
| Shaukat et al. | weak/broken (losses 30–54; one backbone at 52.8%) | **+16.56** |
| Forward (GBIF → DeepWeeds) | 77.8–86.4% | **+1.12** |
| Reverse (DeepWeeds → GBIF) | 95.3–98.4% | **+0.20** |

The stronger the end-to-end model, the less a refitted head recovers. At 95–98% there is
essentially nothing left to recover.

**3. Under shift the head still buys nothing.** Random forest is nominally positive (+0.03) but
that is noise; every other head is negative, worst being AdaBoost (−7.69) and DT (−4.39).

**4. In-domain selection is far less reliable here.** The best in-domain combination
(regnet_y_32gf + RF) does happen to be best under shift too, so selection cost 0 points — but the
rank correlation across all 78 combinations is **ρ = 0.477**, against **0.858** in the forward
direction. In-domain ranking carries much less information about shift performance when the
source domain is narrow.

**5. AdaBoost fails catastrophically on high-dimensional features** — 25.4% and 25.1% in domain
on VGG16/VGG19 (barely above chance, from a 95% backbone), and 63.3% on ResNet-50. This drags its
mean to −31.24. The paper reports AdaBoost at +10.08, so this is a sharp divergence, likely a
consequence of boosting stumps over 4,096 standardised dimensions.

## Forward vs reverse: the asymmetry

| | Forward (GBIF → DeepWeeds) | Reverse (DeepWeeds → GBIF) |
|---|---|---|
| Train size | 1,920 | 5,039 |
| In-domain accuracy | 77.8–86.4% | 95.3–98.4% |
| Shift accuracy | 25.2–40.6% | 22.2–34.4% |
| **Mean gap (E2E)** | **48.2** | **67.5** |
| Best head gain (in-domain) | +1.12 | +0.20 |
| Spearman(in-domain, shift) | 0.858 | 0.477 |

Every backbone shows a larger gap in reverse, by 12–25 points. Combined with the earlier
ResNet-50/Inception-v3 reverse study (66–68 points), the asymmetry now holds across **eight
independently trained models**.

The reading is unchanged: transfer is governed by the **breadth** of the training distribution,
not its difficulty. DeepWeeds is narrow — one instrument, fixed 1 m height, 8 sites, one season —
so models trained on it reach 95–98% in domain by leaning on cues absent from GBIF's worldwide,
many-photographer imagery.

## Caveats
- **Training-set sizes differ** (5,039 vs 1,920), so the forward/reverse gap comparison is
  confounded by data quantity as well as domain. A matched-size run would separate the two.
- In-domain accuracy here is near ceiling (95–98%), which compresses the differences between
  heads and partly explains the smaller gains.
- PyTorch/MPS rather than Keras; 150-epoch cap with early stopping (patience 6); heads use a
  StandardScaler with library defaults; single split, single seed.
