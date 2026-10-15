# Multi-backbone feature fusion — trained on GBIF-649, tested on GBIF-649 test and DeepWeeds

Train 3,112 → test 1,048; de-duplicated test = 1,042 test images with no near-copy in training. 5-fold CV: all 5,192 GBIF-649 images, near-duplicate groups kept within one fold, frozen backbones only. DeepWeeds: all 8,403 weed images, no adaptation.

| Model | Classifier | **Test** | **Test (de-dup)** | macro-F1 | 5-fold CV | DeepWeeds |
|---|---|---|---|---|---|---|
| dinov2_448+clip_l14 | LR | **98.5** | **98.5** | 0.985 | 98.2 ± 0.4 | 76.0 |
| dinov2_448 | SVM | **98.3** | **98.3** | 0.983 | 98.2 ± 0.3 | 77.7 |
| dinov2_448+clip_l14 | SVM | **98.2** | **98.2** | 0.982 | 98.4 ± 0.3 | 76.0 |
| dinov2_448 | LR | **98.1** | **98.1** | 0.981 | 98.0 ± 0.2 | 77.1 |
| dinov2_448+clip_l14 | Stack | **98.1** | **98.1** | 0.981 | 98.1 ± 0.4 | 70.1 |
| dinov2_224+clip_l14 | SVM | **97.7** | **97.7** | 0.977 | 98.0 ± 0.5 | 66.0 |
| dinov2_224 | LR | **97.6** | **97.6** | 0.976 | 97.4 ± 0.6 | 63.6 |
| dinov2_224 | SVM | **97.6** | **97.6** | 0.976 | 97.7 ± 0.5 | 66.1 |
| dinov2_224+clip_l14 | Stack | **97.5** | **97.5** | 0.975 | 97.9 ± 0.5 | 61.9 |
| dinov2_448+clip_l14+regnet_y | Stack | **97.0** | **97.0** | 0.970 | — | 47.8 |
| dinov2_224+clip_l14 | LR | **97.0** | **96.9** | 0.969 | 97.7 ± 0.5 | 66.5 |
| dinov2_448+clip_l14+regnet_y | SVM | **96.0** | **96.0** | 0.960 | — | 35.1 |
| dinov2_448+clip_l14+regnet_y | LR | **95.9** | **95.9** | 0.959 | — | 34.8 |
| regnet_y | LR | **95.5** | **95.5** | 0.955 | — | 31.6 |
| regnet_y | SVM | **95.2** | **95.2** | 0.952 | — | 31.9 |
| clip_l14 | SVM | **94.8** | **94.8** | 0.948 | 95.8 ± 0.4 | 52.2 |
| clip_l14 | LR | **93.6** | **93.6** | 0.936 | 94.8 ± 0.5 | 48.6 |
