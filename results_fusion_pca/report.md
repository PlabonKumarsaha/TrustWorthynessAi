# Step 4 — PCA-256 dimension-balanced fusion (fixed GBIF-400 split)

| Model | Classifier | **Test** | **Test (de-dup)** | macro-F1 | DeepWeeds |
|---|---|---|---|---|---|
| dinov2_448+clip_l14 | SVM | **96.6** | **96.9** | 0.966 | 64.8 |
| dinov2_448+clip_l14 | LR | **95.6** | **95.8** | 0.957 | 63.2 |
| dinov2_448+clip_l14 | Stack | **95.2** | **95.3** | 0.951 | 59.2 |
| dinov2_448+clip_l14 | PCA-256 Stack | **94.7** | **94.8** | 0.947 | 59.7 |
| dinov2_448+clip_l14 | PCA-256 LR | **94.2** | **94.2** | 0.943 | 63.4 |
| dinov2_448+clip_l14 | PCA-256 SVM | **94.2** | **94.2** | 0.943 | 58.6 |
| dinov2_448+clip_l14+regnet_y | PCA-256 SVM | **93.9** | **93.9** | 0.939 | 50.2 |
| dinov2_448+clip_l14+regnet_y | PCA-256 LR | **93.6** | **93.5** | 0.936 | 61.4 |
| dinov2_448+regnet_y | PCA-256 SVM | **93.4** | **93.3** | 0.934 | 54.4 |
| dinov2_448+regnet_y | PCA-256 LR | **93.3** | **93.2** | 0.933 | 60.7 |
| dinov2_448+clip_l14+regnet_y | Stack | **92.0** | **91.8** | 0.920 | 51.2 |
| dinov2_448+clip_l14+regnet_y | PCA-256 Stack | **91.9** | **91.6** | 0.918 | 51.9 |
| dinov2_448+regnet_y | PCA-256 Stack | **90.8** | **90.4** | 0.907 | 49.7 |
| dinov2_448+regnet_y | Stack | **90.5** | **90.0** | 0.904 | 49.5 |
| dinov2_448+clip_l14+regnet_y | LR | **89.4** | **88.8** | 0.893 | 45.9 |
| dinov2_448+clip_l14+regnet_y | SVM | **89.1** | **88.4** | 0.890 | 46.8 |
| dinov2_448+regnet_y | LR | **88.8** | **88.1** | 0.887 | 45.5 |
| dinov2_448+regnet_y | SVM | **88.8** | **88.1** | 0.887 | 46.5 |
