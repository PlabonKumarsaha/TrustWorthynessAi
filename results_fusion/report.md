# Multi-backbone feature fusion — in-domain GBIF accuracy (Phase 1)

Fixed split: train 1,920 → test 640; de-duplicated test = 571 test images with no near-copy in training. 5-fold CV: all 3,200 GBIF images, near-duplicate groups kept within one fold, frozen backbones only. DeepWeeds is a secondary column (no adaptation).

| Model | Classifier | **Test** | **Test (de-dup)** | macro-F1 | 5-fold CV | DeepWeeds |
|---|---|---|---|---|---|---|
| dinov2_448+clip_l14 | SVM | **96.6** | **96.9** | 0.966 | 94.0 ± 3.9 | 64.8 |
| dinov2_448 | SVM | **95.6** | **95.8** | 0.957 | 94.0 ± 4.1 | 64.8 |
| dinov2_448+clip_l14 | LR | **95.6** | **95.8** | 0.957 | 93.2 ± 4.3 | 63.2 |
| dinov2_224+clip_l14 | SVM | **95.3** | **95.5** | 0.953 | 93.1 ± 3.8 | 54.0 |
| dinov2_448+clip_l14 | Stack | **95.2** | **95.3** | 0.951 | 95.1 ± 1.5 | 59.2 |
| dinov2_224 | SVM | **94.8** | **94.9** | 0.948 | 92.4 ± 3.8 | 51.1 |
| dinov2_224+clip_l14 | LR | **94.8** | **94.9** | 0.948 | 92.6 ± 4.1 | 53.5 |
| dinov2_224 | LR | **94.4** | **94.4** | 0.944 | 91.9 ± 4.0 | 49.2 |
| dinov2_448 | LR | **94.4** | **94.4** | 0.944 | 92.8 ± 4.0 | 63.9 |
| dinov2_224+clip_l14 | Stack | **94.2** | **94.2** | 0.942 | 92.8 ± 4.0 | 54.2 |
| dinov2_448+clip_l14+regnet_y | Stack | **92.0** | **91.8** | 0.920 | — | 51.2 |
| clip_l14 | LR | **91.7** | **91.4** | 0.917 | 90.2 ± 1.7 | 43.1 |
| clip_l14 | SVM | **91.7** | **91.4** | 0.917 | 91.0 ± 3.6 | 44.4 |
| dinov2_448+clip_l14+regnet_y | LR | **89.4** | **88.8** | 0.893 | — | 45.9 |
| dinov2_448+clip_l14+regnet_y | SVM | **89.1** | **88.4** | 0.890 | — | 46.8 |
| regnet_y | LR | **88.0** | **87.2** | 0.879 | — | 42.6 |
| regnet_y | SVM | **88.0** | **87.2** | 0.879 | — | 43.9 |
