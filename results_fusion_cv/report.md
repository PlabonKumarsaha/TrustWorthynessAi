# Repeated grouped CV with paired tests (in-domain GBIF accuracy)

5×5 = 25 folds over all 3,200 GBIF images, near-duplicate groups kept within one fold; every candidate on identical folds. Reference: **dinov2_448 | SVM** = 93.82 ± 3.90%. p-values: corrected resampled t-test (Nadeau & Bengio 2003), Holm-adjusted; Wilcoxon signed-rank as a check.

| Model | Classifier | Accuracy (mean ± std) | Δ vs reference | p (corrected t) | p (Holm) | p (Wilcoxon) | Significant at 0.05 (Holm)? |
|---|---|---|---|---|---|---|---|
| dinov2_448 | SVM | **93.82 ± 3.90** | reference | — | — | — | — |
| dinov2_448_tta+clip_l14_tta | Stack | 94.44 ± 3.40 | +0.62 | 0.454 | 1.000 | 0.072 | no |
| dinov2_448+clip336 | Stack | 94.16 ± 3.66 | +0.34 | 0.621 | 1.000 | 0.103 | no |
| dinov2_448+clip336 | SVM | 94.12 ± 3.82 | +0.30 | 0.459 | 1.000 | 0.054 | no |
| dinov2_448_tta | SVM | 94.11 ± 3.99 | +0.29 | 0.427 | 1.000 | 0.048 | no |
| dinov2_448_tta+clip_l14_tta | SVM | 94.08 ± 4.11 | +0.26 | 0.535 | 1.000 | 0.144 | no |
| dinov2_448+clip_l14 | Stack | 93.98 ± 3.62 | +0.16 | 0.839 | 1.000 | 0.605 | no |
| dinov2_448+clip_l14 | SVM | 93.89 ± 3.97 | +0.07 | 0.865 | 1.000 | 0.876 | no |
| dinov2_448+clip_l14 | LR | 93.77 ± 3.16 | -0.05 | 0.964 | 1.000 | 0.153 | no |
| dinov2_448+clip_l14 | PCA-256 Stack | 93.67 ± 3.70 | -0.15 | 0.704 | 1.000 | 0.444 | no |
| dinov2_448+clip_l14 | PCA-256 SVM | 93.04 ± 3.91 | -0.78 | 0.117 | 1.000 | 0.000 | no |
| dinov2_448 | LR | 93.03 ± 3.79 | -0.79 | 0.030 | 0.332 | 0.000 | no |
| dinov2_224 | SVM | 92.40 ± 4.04 | -1.42 | 0.013 | 0.164 | 0.000 | no |
| clip_l14_tta | SVM | 90.99 ± 3.92 | -2.83 | 0.000 | 0.000 | 0.000 | yes, worse |
| clip_l14 | SVM | 90.83 ± 3.65 | -2.98 | 0.000 | 0.001 | 0.000 | yes, worse |
| clip336 | SVM | 90.79 ± 2.83 | -3.03 | 0.022 | 0.268 | 0.000 | no |
