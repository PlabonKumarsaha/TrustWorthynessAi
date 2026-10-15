# Feature fusion — trained on GBIF-Living (4,008 train), tested on GBIF-Living test and DeepWeeds

| Model | Classifier | Test | Test (de-dup) | 5-fold CV | DeepWeeds |
|---|---|---|---|---|---|
| dinov2_448+clip_l14 | Stack | 98.7 | 98.7 | 97.7 ± 0.7 | 71.5 |
| dinov2_448+clip_l14 | SVM | 98.5 | 98.5 | 98.1 ± 0.4 | 77.2 |
| dinov2_448 | SVM | 98.4 | 98.4 | 97.7 ± 0.9 | 78.7 |
| dinov2_448+clip_l14 | LR | 98.4 | 98.4 | 97.6 ± 0.8 | 76.4 |
| dinov2_448 | LR | 97.7 | 97.7 | 97.4 ± 0.7 | 75.3 |
| clip_l14 | SVM | 96.0 | 96.0 | 95.7 ± 0.4 | 53.3 |
| clip_l14 | LR | 94.6 | 94.6 | 94.9 ± 0.7 | 52.8 |
