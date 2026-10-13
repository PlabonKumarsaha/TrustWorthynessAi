# Adaptation stage on frozen CLIP / BLIP backbones (GBIF-400 → DeepWeeds)

Same heads, techniques and data as hd_adapt.py (fine-tuned CNNs).


### clip_vitb32_openai

| Head | GBIF test | DW: paper pipeline | DW: + restd | DW: + align | DW: + self-train | **DW: + all three** | macro-F1 |
|---|---|---|---|---|---|---|---|
| SVM | 83.1 | 32.6 | 34.3 | 30.2 | 34.8 | **38.9** | 0.380 |
| LR | 82.5 | 30.5 | 33.4 | 31.9 | 33.2 | **35.6** | 0.356 |
| HGB | 80.2 | 32.6 | 34.1 | 34.0 | 33.3 | **37.9** | 0.377 |
| kNN | 76.6 | 26.5 | 31.2 | 27.4 | 31.0 | **40.9** | 0.409 |
| RF | 78.9 | 30.3 | 32.7 | 32.5 | 32.0 | **39.3** | 0.392 |

### clip_vitl14_laion2b

| Head | GBIF test | DW: paper pipeline | DW: + restd | DW: + align | DW: + self-train | **DW: + all three** | macro-F1 |
|---|---|---|---|---|---|---|---|
| SVM | 92.2 | 43.5 | 49.3 | 47.5 | 48.3 | **58.8** | 0.577 |
| LR | 90.8 | 42.6 | 47.0 | 47.3 | 46.8 | **54.7** | 0.548 |
| HGB | 90.0 | 39.6 | 45.8 | 42.6 | 43.1 | **55.5** | 0.555 |
| kNN | 87.7 | 41.6 | 45.9 | 47.1 | 45.7 | **68.8** | 0.686 |
| RF | 87.3 | 39.7 | 45.1 | 43.4 | 47.4 | **58.2** | 0.582 |

### blip_base_coco

| Head | GBIF test | DW: paper pipeline | DW: + restd | DW: + align | DW: + self-train | **DW: + all three** | macro-F1 |
|---|---|---|---|---|---|---|---|
| SVM | 87.0 | 31.9 | 38.8 | 37.3 | 31.6 | **44.3** | 0.431 |
| LR | 87.2 | 33.1 | 36.7 | 37.0 | 34.1 | **38.9** | 0.389 |
| HGB | 83.0 | 29.4 | 34.2 | 33.5 | 32.6 | **43.0** | 0.429 |
| kNN | 77.8 | 27.9 | 33.1 | 30.0 | 30.0 | **53.1** | 0.530 |
| RF | 78.8 | 28.4 | 34.6 | 34.9 | 25.5 | **43.4** | 0.430 |

### blip_large_coco

| Head | GBIF test | DW: paper pipeline | DW: + restd | DW: + align | DW: + self-train | **DW: + all three** | macro-F1 |
|---|---|---|---|---|---|---|---|
| SVM | 87.7 | 31.7 | 39.1 | 39.2 | 30.3 | **44.1** | 0.436 |
| LR | 86.2 | 38.2 | 41.5 | 39.8 | 39.0 | **42.4** | 0.423 |
| HGB | 85.5 | 30.1 | 35.5 | 35.2 | 26.4 | **41.2** | 0.411 |
| kNN | 78.4 | 29.7 | 33.0 | 31.7 | 34.2 | **48.4** | 0.483 |
| RF | 82.0 | 27.2 | 33.1 | 35.1 | 24.9 | **40.9** | 0.407 |

## Combined leaderboard: 6 fine-tuned CNNs + 4 frozen VLMs, top 15 by DeepWeeds (+ all three)

| # | Backbone | Type | Head | GBIF test | DW: paper pipeline | **DW: + all three** | macro-F1 |
|---|---|---|---|---|---|---|---|
| 1 | clip_vitl14_laion2b | frozen VLM | kNN | 87.7 | 41.6 | **68.8** | 0.686 |
| 2 | regnet_y_32gf | fine-tuned CNN | kNN | 86.6 | 41.4 | **61.9** | 0.620 |
| 3 | clip_vitl14_laion2b | frozen VLM | SVM | 92.2 | 43.5 | **58.8** | 0.577 |
| 4 | clip_vitl14_laion2b | frozen VLM | RF | 87.3 | 39.7 | **58.2** | 0.582 |
| 5 | clip_vitl14_laion2b | frozen VLM | HGB | 90.0 | 39.6 | **55.5** | 0.555 |
| 6 | regnet_y_32gf | fine-tuned CNN | RF | 87.3 | 41.3 | **54.9** | 0.550 |
| 7 | clip_vitl14_laion2b | frozen VLM | LR | 90.8 | 42.6 | **54.7** | 0.548 |
| 8 | densenet201 | fine-tuned CNN | kNN | 85.9 | 32.3 | **54.7** | 0.547 |
| 9 | blip_base_coco | frozen VLM | kNN | 77.8 | 27.9 | **53.1** | 0.530 |
| 10 | regnet_y_32gf | fine-tuned CNN | HGB | 86.6 | 40.4 | **52.0** | 0.520 |
| 11 | regnet_y_32gf | fine-tuned CNN | LR | 87.8 | 44.6 | **50.8** | 0.508 |
| 12 | regnet_y_32gf | fine-tuned CNN | SVM | 88.1 | 45.8 | **49.1** | 0.493 |
| 13 | blip_large_coco | frozen VLM | kNN | 78.4 | 29.7 | **48.4** | 0.483 |
| 14 | resnet50 | fine-tuned CNN | kNN | 85.0 | 33.6 | **47.3** | 0.474 |
| 15 | densenet201 | fine-tuned CNN | RF | 86.7 | 37.3 | **45.1** | 0.449 |
