# HD(LM)²D replication: train on GBIF → shift-test on DeepWeeds

Direction: `gbif2dw`. Backbones complete: 6/6 (mobilenet_v2, resnet50, densenet201, vgg16, vgg19, regnet_y_32gf). 8 classes, chance 12.5%.

## 1. End-to-end fine-tuned models (paper Tables 7–9)

| Backbone | GBIF test | DeepWeeds | Gap |
|---|---|---|---|
| mobilenet_v2 | 83.6% | 35.6% | 47.9 |
| resnet50 | 83.9% | 33.8% | 50.1 |
| densenet201 | 86.4% | 39.9% | 46.5 |
| vgg16 | 77.8% | 25.2% | 52.6 |
| vgg19 | 82.2% | 35.6% | 46.6 |
| regnet_y_32gf | 86.1% | 40.6% | 45.5 |

## 2. Backbone × head accuracy — GBIF test (in-domain) (paper Table 13)

| Backbone | E2E | SVM | LR | HGB | kNN | RF | XGBoost | LDA | Bagging | GradBoost | AdaBoost | DT | NB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | 83.6 | **84.4** | 84.2 | 84.2 | 84.1 | 84.1 | 83.9 | 81.2 | 82.5 | 83.0 | 80.2 | 78.9 | 84.2 |
| resnet50 | 83.9 | 85.3 | 85.0 | 84.4 | 85.0 | **85.9** | 83.8 | 80.3 | 85.5 | 82.8 | 46.2 | 79.5 | 85.0 |
| densenet201 | 86.4 | 87.7 | **88.0** | 86.4 | 85.9 | 86.7 | 85.5 | 74.4 | 83.4 | 85.0 | 80.2 | 77.3 | 83.8 |
| vgg16 | 77.8 | 78.4 | 80.8 | **81.6** | 76.6 | 80.2 | 79.8 | 70.9 | 78.9 | 80.9 | 77.2 | 77.7 | 77.3 |
| vgg19 | 82.2 | 79.2 | 80.9 | 82.0 | 78.6 | **82.3** | 82.0 | 76.1 | 82.2 | 81.9 | 79.7 | 79.1 | 79.7 |
| regnet_y_32gf | 86.1 | **88.1** | 87.8 | 86.6 | 86.6 | 87.3 | 86.2 | 84.5 | 85.8 | 85.2 | 80.5 | 82.2 | 87.2 |

## 2. Backbone × head accuracy — DeepWeeds (shift) (paper Table 13)

| Backbone | E2E | SVM | LR | HGB | kNN | RF | XGBoost | LDA | Bagging | GradBoost | AdaBoost | DT | NB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | **35.6** | 34.4 | 33.7 | 33.9 | 34.2 | 34.6 | 33.8 | 32.0 | 32.3 | 32.2 | 30.8 | 30.4 | 33.6 |
| resnet50 | 33.8 | 34.6 | **34.8** | 30.2 | 33.6 | 34.4 | 29.6 | 26.3 | 31.9 | 23.4 | 16.9 | 23.4 | 34.3 |
| densenet201 | **39.9** | 36.5 | 37.3 | 38.7 | 32.3 | 37.3 | 38.8 | 19.7 | 36.2 | 35.8 | 28.7 | 33.0 | 36.8 |
| vgg16 | 25.2 | 22.9 | 24.5 | 26.5 | 21.5 | 25.8 | 26.5 | 22.5 | 24.8 | 25.6 | 25.7 | 22.1 | **27.4** |
| vgg19 | **35.6** | 31.0 | 31.7 | 32.7 | 30.5 | 32.8 | 32.8 | 29.4 | 31.7 | 32.6 | 29.7 | 28.3 | 32.1 |
| regnet_y_32gf | 40.6 | **45.7** | 44.6 | 40.4 | 41.4 | 41.3 | 39.5 | 33.0 | 40.2 | 38.8 | 23.8 | 35.2 | 39.6 |

## 3. Mean gain of each head over the end-to-end model (paper Table 14)

| Head | In-domain gain | Improves (in-domain) | Shift gain | Improves (shift) |
|---|---|---|---|---|
| LR | +1.12 | 5/6 | -0.68 | 2/6 |
| RF | +1.09 | 6/6 | -0.74 | 3/6 |
| HGB | +0.86 | 4/6 | -1.38 | 1/6 |
| SVM | +0.52 | 5/6 | -0.94 | 2/6 |
| XGBoost | +0.21 | 3/6 | -1.62 | 1/6 |
| GradBoost | -0.21 | 1/6 | -3.70 | 1/6 |
| Bagging | -0.28 | 2/6 | -2.26 | 0/6 |
| NB | -0.47 | 3/6 | -1.15 | 2/6 |
| kNN | -0.55 | 3/6 | -2.87 | 1/6 |
| DT | -4.22 | 0/6 | -6.38 | 0/6 |
| LDA | -5.42 | 0/6 | -7.97 | 0/6 |
| AdaBoost | -9.35 | 0/6 | -9.18 | 1/6 |

## 4. Does in-domain model selection survive shift?

- Combinations: 78 (6 backbones × 13 classifiers incl. E2E)
- Best in-domain: **regnet_y_32gf + SVM** — 88.1% GBIF → 45.7% DeepWeeds (rank 1/78 under shift)
- Best under shift: **regnet_y_32gf + SVM** — 45.7% DeepWeeds (88.1% GBIF)
- Cost of selecting on in-domain accuracy: 0.0 pts on DeepWeeds
- Spearman(in-domain, shift) over all combinations: ρ = 0.858 (p = 1.2e-23)
- Gap (GBIF − DeepWeeds): mean 49.9 pts, range 29.4–59.4
