# CLIP as backbone (frozen), GBIF-400 → DeepWeeds

| Backbone | Dim | Zero-shot GBIF | Zero-shot DeepWeeds | Best head | Head GBIF | **Head DeepWeeds** | Mean of 12 heads (DW) |
|---|---|---|---|---|---|---|---|
| clip_vitb32_openai | 512 | 35.6% | 11.0% | LDA | 83.0% | **33.0%** | 27.9% |
| clip_vitl14_laion2b | 768 | 51.7% | 14.7% | SVM | 92.2% | **43.6%** | 35.8% |

Reference (fine-tuned CNN, same split): RegNetY-32GF + SVM = 88.1% GBIF / 45.7% DeepWeeds.
