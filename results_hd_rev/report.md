# HD(LM)²D replication: train on DeepWeeds → shift-test on GBIF

Direction: `dw2gbif`. Backbones complete: 6/6 (mobilenet_v2, resnet50, densenet201, vgg16, vgg19, regnet_y_32gf). 8 classes, chance 12.5%.

## 1. End-to-end fine-tuned models (paper Tables 7–9)

| Backbone | DeepWeeds test | GBIF | Gap |
|---|---|---|---|
| mobilenet_v2 | 95.4% | 22.2% | 73.2 |
| resnet50 | 97.2% | 28.9% | 68.2 |
| densenet201 | 97.1% | 29.8% | 67.3 |
| vgg16 | 95.3% | 30.8% | 64.5 |
| vgg19 | 96.2% | 28.4% | 67.8 |
| regnet_y_32gf | 98.4% | 34.4% | 64.0 |

## 2. Backbone × head accuracy — DeepWeeds test (in-domain) (paper Table 13)

| Backbone | E2E | SVM | LR | HGB | kNN | RF | XGBoost | LDA | Bagging | GradBoost | AdaBoost | DT | NB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | 95.4 | 96.0 | 95.5 | 95.6 | 96.0 | 95.9 | 95.3 | **96.1** | 94.5 | 95.2 | 90.8 | 92.6 | 95.5 |
| resnet50 | 97.2 | 97.2 | 96.9 | 96.8 | 97.2 | 97.1 | 96.9 | **97.5** | 96.1 | 96.0 | 63.3 | 93.7 | 97.3 |
| densenet201 | 97.1 | 96.4 | 97.4 | 97.0 | 97.3 | **97.6** | 97.1 | 97.4 | 96.3 | 97.2 | 94.3 | 94.3 | 94.3 |
| vgg16 | 95.3 | 94.7 | 95.4 | **95.6** | 95.0 | **95.6** | 95.3 | 94.7 | 95.0 | 95.2 | 25.4 | 94.4 | 95.2 |
| vgg19 | **96.2** | 95.0 | 96.0 | 95.7 | 95.7 | 96.0 | 95.9 | 95.9 | 96.0 | 95.9 | 25.1 | 95.6 | 95.5 |
| regnet_y_32gf | 98.4 | 98.3 | 98.3 | 98.0 | 98.4 | **98.5** | 97.6 | 98.3 | 96.8 | 97.5 | 93.1 | 95.0 | 98.4 |

## 2. Backbone × head accuracy — GBIF (shift) (paper Table 13)

| Backbone | E2E | SVM | LR | HGB | kNN | RF | XGBoost | LDA | Bagging | GradBoost | AdaBoost | DT | NB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | 22.2 | **23.4** | 22.8 | 22.3 | 23.3 | 22.6 | 23.0 | 22.3 | 22.9 | 21.3 | 18.4 | 18.8 | 22.5 |
| resnet50 | **28.9** | 28.6 | 28.8 | 26.2 | 27.9 | 27.4 | 26.4 | 26.3 | 27.4 | 24.2 | 20.7 | 23.5 | 26.7 |
| densenet201 | **29.8** | 26.4 | 28.0 | 25.2 | 26.8 | 29.2 | 24.4 | 25.4 | 25.4 | 25.1 | 23.7 | 24.4 | 26.7 |
| vgg16 | 30.8 | 27.3 | 31.1 | 31.0 | 31.2 | **31.4** | 29.8 | 28.9 | 29.2 | 29.7 | 22.7 | 27.8 | 30.3 |
| vgg19 | 28.4 | **29.0** | 27.9 | 26.9 | 28.4 | 28.6 | 27.8 | 27.3 | 27.2 | 27.4 | 18.9 | 25.4 | 27.7 |
| regnet_y_32gf | 34.4 | 34.8 | 34.5 | 32.0 | 34.4 | **35.5** | 31.3 | 30.8 | 32.5 | 30.7 | 24.0 | 28.2 | 33.3 |

## 3. Mean gain of each head over the end-to-end model (paper Table 14)

| Head | In-domain gain | Improves (in-domain) | Shift gain | Improves (shift) |
|---|---|---|---|---|
| RF | +0.20 | 4/6 | +0.03 | 4/6 |
| LDA | +0.08 | 3/6 | -2.27 | 1/6 |
| LR | +0.01 | 3/6 | -0.25 | 3/6 |
| kNN | -0.00 | 2/6 | -0.41 | 3/6 |
| HGB | -0.13 | 2/6 | -1.84 | 2/6 |
| XGBoost | -0.24 | 0/6 | -1.96 | 1/6 |
| SVM | -0.32 | 1/6 | -0.84 | 3/6 |
| GradBoost | -0.44 | 1/6 | -2.69 | 0/6 |
| NB | -0.55 | 2/6 | -1.23 | 1/6 |
| Bagging | -0.81 | 0/6 | -1.63 | 1/6 |
| DT | -2.30 | 0/6 | -4.39 | 0/6 |
| AdaBoost | -31.24 | 0/6 | -7.69 | 0/6 |

## 4. Does in-domain model selection survive shift?

- Combinations: 78 (6 backbones × 13 classifiers incl. E2E)
- Best in-domain: **regnet_y_32gf + RF** — 98.5% DeepWeeds → 35.5% GBIF (rank 1/78 under shift)
- Best under shift: **regnet_y_32gf + RF** — 35.5% GBIF (98.5% DeepWeeds)
- Cost of selecting on in-domain accuracy: 0.0 pts on GBIF
- Spearman(in-domain, shift) over all combinations: ρ = 0.477 (p = 1e-05)
- Gap (DeepWeeds − GBIF): mean 66.7 pts, range 2.7–73.9
