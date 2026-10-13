# Classifier search on fixed features (GBIF-400 → DeepWeeds)

## A. Source-only classifiers (tuned by 5-fold CV on GBIF train)

| Classifier | clip_vitl14 GBIF | clip_vitl14 **DW** | regnet_y GBIF | regnet_y **DW** | dinov2 GBIF | dinov2 **DW** |
|---|---|---|---|---|---|---|
| LR | 91.6 | **44.6** | 88.1 | **42.9** | 93.6 | **50.1** |
| SVM-linear | 91.4 | **43.0** | 87.5 | **43.1** | 94.1 | **51.0** |
| SVM-RBF | 91.7 | **44.4** | 88.0 | **43.9** | 94.8 | **51.1** |
| kNN-cosine | 89.8 | **40.3** | 87.0 | **40.1** | 93.1 | **49.6** |
| MLP | 91.4 | **45.1** | 87.0 | **38.8** | 91.7 | **46.9** |
| Prototype | 77.3 | **34.9** | 85.9 | **40.0** | 85.8 | **52.1** |
| Ensemble | 91.6 | **45.6** | 87.0 | **41.0** | 93.6 | **50.1** |

## B. Label-free shift-aware variants (DeepWeeds accuracy)

| Variant | clip_vitl14 | regnet_y | dinov2 |
|---|---|---|---|
| LR (tuned, source-only) | 44.6 | 42.9 | 50.1 |
| + target re-standardisation | 50.6 | 47.8 | 61.1 |
| + prior alignment (Sinkhorn) | 51.4 | 48.8 | 62.5 |
| + class-balanced self-training | 52.0 | 48.1 | 61.0 |
| + all three | 63.0 | 50.4 | 77.6 |

Reference, untuned defaults: CLIP L/14 + SVM 43.6%; RegNetY + SVM 45.7%; Sinkhorn alignment assumes a roughly uniform DeepWeeds class prior.
