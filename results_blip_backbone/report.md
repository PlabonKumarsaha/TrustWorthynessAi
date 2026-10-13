# BLIP as backbone (frozen), GBIF-400 → DeepWeeds

| Backbone | Dim | Zero-shot GBIF | Zero-shot DeepWeeds | Best head | Head GBIF | **Head DeepWeeds** | Mean of 12 heads (DW) |
|---|---|---|---|---|---|---|---|
| blip_base_coco | 768 | 28.9% | 13.5% | LR | 87.2% | **33.1%** | 26.9% |
| blip_large_coco | 1024 | 25.6% | 13.4% | LR | 86.2% | **38.2%** | 27.6% |

Reference (fine-tuned CNN, same split): RegNetY-32GF + SVM = 88.1% GBIF / 45.7% DeepWeeds.
