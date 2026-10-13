# Step 3 — multi-view feature averaging (fixed GBIF-400 split)

| Model | Classifier | **Test** | **Test (de-dup)** | macro-F1 | DeepWeeds |
|---|---|---|---|---|---|
| dinov2_448+clip_l14 | SVM | **96.6** | **96.9** | 0.966 | 64.8 |
| dinov2_448_tta+clip_l14_tta | LR | **95.9** | **96.2** | 0.960 | — |
| dinov2_448_tta+clip_l14_tta | SVM | **95.8** | **96.0** | 0.958 | — |
| dinov2_448 | SVM | **95.6** | **95.8** | 0.957 | 64.8 |
| dinov2_448_tta | SVM | **95.3** | **95.5** | 0.954 | — |
| dinov2_448+clip_l14 | Stack | **95.2** | **95.3** | 0.951 | 59.2 |
| dinov2_448_tta+clip_l14_tta | Stack | **95.2** | **95.3** | 0.951 | — |
| dinov2_448_tta | LR | **94.7** | **94.8** | 0.948 | — |
| clip_l14_tta | SVM | **93.0** | **92.8** | 0.929 | — |
| clip_l14_tta | LR | **91.9** | **91.6** | 0.919 | — |
| clip_l14 | SVM | **91.7** | **91.4** | 0.917 | 44.4 |
