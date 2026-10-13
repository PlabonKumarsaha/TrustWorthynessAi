# Why more GBIF data transferred worse: feature-space check

Training on GBIF-649 (649/class) instead of GBIF-400 raised in-domain accuracy (83–86% → 91–95%)
but lowered DeepWeeds accuracy on 5 of 6 backbones (mean over 78 combinations 32.1% → 25.8%).
This check measures the image sets with frozen networks never fine-tuned on either domain
(ImageNet ResNet-50, DINOv2 ViT-B/14). Script: `shift_check.py`; raw output:
`results_shift/shift_check.json`.

## 1. The new set sits farther from DeepWeeds

Class-balanced, equal-size samples (400/class) of each GBIF set, compared with all 8,403
DeepWeeds images:

| Extractor | Metric | GBIF-400 | GBIF-649 |
|---|---|---|---|
| ResNet-50 | Fréchet distance | 94.2 | **189.0** (2.0×) |
| ResNet-50 | MMD² | 0.086 | **0.225** (2.6×) |
| DINOv2 | Fréchet distance | 1,277 | **1,497** (+17%) |
| DINOv2 | MMD² | 0.094 | **0.141** (1.5×) |

Both extractors agree. (A domain classifier separates either set from DeepWeeds with >99%
accuracy, so it saturates and does not discriminate between them.)

## 2. The difference is herbarium specimens

| | GBIF-400 | GBIF-649 |
|---|---|---|
| Herbarium share, GBIF `basisOfRecord` (ground truth) | no manifest | **61.3%** (3,185 PRESERVED_SPECIMEN) |
| Herbarium share, CLIP zero-shot detector | **4.1%** | 63.2% |

The CLIP detector was validated against the ground truth on GBIF-649: **97.7% accuracy, 99.6%
recall, 96.7% precision**. Its estimate for the new set (63.2%) matches the true value (61.3%),
so its 4.1% estimate for the original set is credible.

**The original set was ~4% herbarium; the new one is ~61%.**

Inside GBIF-649, herbarium images are much farther from DeepWeeds than living-plant photos:

| Extractor | MMD² to DeepWeeds: herbarium | MMD² to DeepWeeds: living plants |
|---|---|---|
| ResNet-50 | **0.439** | 0.100 |
| DINOv2 | **0.226** | 0.110 |

The living-plant photos in the new set sit about as close to DeepWeeds as the original set did
(0.100 vs 0.086 ResNet-50; 0.110 vs 0.094 DINOv2). **The extra distance comes entirely from the
herbarium sheets.**

## 3. Leakage: the higher in-domain accuracy is not inflated, but the original set's was

Share of test images with a near-duplicate in their own training split (DINOv2 cosine):

| Set | > 0.90 | > 0.95 | > 0.98 |
|---|---|---|---|
| GBIF-400 (original) | 13.0% | **10.8%** | **10.0%** |
| GBIF-649 (one photo per occurrence) | 1.1% | 0.6% | 0.6% |
| DeepWeeds | 3.1% | 0.2% | 0.1% |

- GBIF-649's higher in-domain accuracy is **not** from leakage — it has the least. Herbarium sheets
  (flat specimen on a uniform background) are simply easier to classify, on top of the extra data.
- **The original GBIF-400 set has ~10% near-identical train/test pairs**, most likely several
  photos of the same occurrence. Its in-domain figures (77.8–88.1%) are therefore somewhat
  inflated. Its DeepWeeds figures are unaffected, since DeepWeeds is a separate test set.

## 4. Other observations
- The two GBIF sets are largely distinct: only 4.2% of new images have a near-duplicate (>0.95)
  in the old set, and a classifier tells the two sets apart with ~96% accuracy.
- DINOv2 rates the new set as *less* diverse than the old (0.795 vs 0.891 mean pairwise cosine
  distance) — consistent with herbarium sheets sharing paper, labels and layout.
- Both distance measures ranked the two training sets correctly by how well they transfer: the
  set closer to DeepWeeds transferred better. That is only two data points, but it is the
  thesis question — feature-space shift predicting degradation — answered in the right direction.

## Conclusion
More GBIF data did not hurt; **more herbarium data did.** The extra GBIF images were 61%
herbarium, those images sit 2–4× farther from DeepWeeds than living-plant photos, and the
living-plant photos alone are as close as the original set. Filtering GBIF by `basisOfRecord`
is the obvious remedy, and a controlled photos-vs-herbarium experiment is the test that would
establish it.
