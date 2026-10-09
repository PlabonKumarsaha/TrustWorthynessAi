# HD(LM)²D replication: train on GBIF-900 → shift-test on DeepWeeds

Direction: `gbif900_2dw`. Backbones complete: 6/6 (mobilenet_v2, resnet50, densenet201, vgg16, vgg19, regnet_y_32gf). 8 classes, chance 12.5%.

## 1. End-to-end fine-tuned models (paper Tables 7–9)

| Backbone | GBIF-900 test | DeepWeeds | Gap |
|---|---|---|---|
| mobilenet_v2 | 92.1% | 28.2% | 63.9 |
| resnet50 | 91.0% | 26.7% | 64.3 |
| densenet201 | 92.9% | 26.4% | 66.6 |
| vgg16 | 92.6% | 29.2% | 63.3 |
| vgg19 | 91.2% | 19.2% | 72.1 |
| regnet_y_32gf | 94.8% | 31.7% | 63.2 |

## 2. Backbone × head accuracy — GBIF-900 test (in-domain) (paper Table 13)

| Backbone | E2E | SVM | LR | HGB | kNN | RF | XGBoost | LDA | Bagging | GradBoost | AdaBoost | DT | NB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | 92.1 | 92.1 | **92.2** | 91.4 | 91.9 | 92.1 | 91.2 | 91.6 | 90.5 | 91.2 | 87.5 | 84.4 | 92.1 |
| resnet50 | 91.0 | **92.7** | 92.1 | 91.2 | 91.7 | 92.0 | 91.1 | 91.4 | 90.9 | 91.2 | 88.8 | 86.3 | 91.9 |
| densenet201 | 92.9 | 92.4 | **93.5** | 92.2 | 93.1 | 92.9 | 92.2 | 92.8 | 92.1 | 92.2 | 89.7 | 86.6 | 89.3 |
| vgg16 | 92.6 | 91.6 | 92.6 | 92.6 | 92.0 | **92.9** | 92.3 | 75.7 | 92.4 | 92.3 | 90.9 | 90.7 | 90.9 |
| vgg19 | 91.2 | 88.6 | 89.8 | 91.1 | **91.4** | 91.2 | **91.4** | 77.4 | 90.8 | 90.6 | 90.8 | 89.4 | 89.2 |
| regnet_y_32gf | 94.8 | 95.5 | 95.0 | 94.8 | **95.7** | 95.6 | 93.9 | 91.5 | 95.1 | 93.0 | 85.0 | 91.0 | 95.2 |

## 2. Backbone × head accuracy — DeepWeeds (shift) (paper Table 13)

| Backbone | E2E | SVM | LR | HGB | kNN | RF | XGBoost | LDA | Bagging | GradBoost | AdaBoost | DT | NB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | 28.2 | 28.2 | 28.3 | 27.3 | **28.5** | 28.0 | 26.4 | 25.4 | 27.7 | 27.9 | 20.8 | 22.5 | 28.2 |
| resnet50 | **26.7** | 26.3 | 24.5 | 25.9 | 26.1 | 26.2 | 25.1 | 22.3 | 26.2 | 23.5 | 23.2 | 19.6 | 26.2 |
| densenet201 | 26.4 | 24.4 | 26.1 | 24.1 | 23.3 | **26.8** | 24.8 | 24.1 | 26.3 | 24.2 | 20.9 | 20.9 | 23.5 |
| vgg16 | 29.2 | 29.3 | 28.9 | 28.5 | 30.3 | 28.8 | 29.3 | 23.6 | 28.0 | 28.8 | **31.0** | 28.2 | 28.1 |
| vgg19 | 19.2 | 21.2 | 22.7 | 21.4 | 23.7 | 21.5 | 22.0 | 15.1 | **25.5** | 22.9 | 20.6 | 24.3 | 19.4 |
| regnet_y_32gf | 31.7 | **31.9** | 30.9 | 31.6 | 31.7 | 31.2 | 29.9 | 24.8 | 31.8 | 29.3 | 18.9 | 25.2 | 30.7 |

## 3. Mean gain of each head over the end-to-end model (paper Table 14)

| Head | In-domain gain | Improves (in-domain) | Shift gain | Improves (shift) |
|---|---|---|---|---|
| RF | +0.35 | 3/6 | +0.20 | 2/6 |
| kNN | +0.19 | 4/6 | +0.38 | 4/6 |
| LR | +0.08 | 4/6 | +0.03 | 2/6 |
| HGB | -0.22 | 1/6 | -0.40 | 1/6 |
| SVM | -0.30 | 2/6 | -0.02 | 3/6 |
| XGBoost | -0.43 | 2/6 | -0.61 | 2/6 |
| Bagging | -0.48 | 1/6 | +0.68 | 2/6 |
| GradBoost | -0.69 | 1/6 | -0.78 | 1/6 |
| NB | -1.00 | 2/6 | -0.84 | 2/6 |
| AdaBoost | -3.64 | 0/6 | -4.31 | 2/6 |
| DT | -4.37 | 0/6 | -3.41 | 1/6 |
| LDA | -5.72 | 1/6 | -4.33 | 0/6 |

## 4. Does in-domain model selection survive shift?

- Combinations: 78 (6 backbones × 13 classifiers incl. E2E)
- Best in-domain: **regnet_y_32gf + kNN** — 95.7% GBIF-900 → 31.7% DeepWeeds (rank 3/78 under shift)
- Best under shift: **regnet_y_32gf + SVM** — 31.9% DeepWeeds (95.5% GBIF-900)
- Cost of selecting on in-domain accuracy: 0.2 pts on DeepWeeds
- Spearman(in-domain, shift) over all combinations: ρ = 0.702 (p = 7.8e-13)
- Gap (GBIF-900 − DeepWeeds): mean 65.3 pts, range 52.1–72.1
