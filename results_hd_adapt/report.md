# HD(LM)²D + label-free adaptation stage (GBIF-400 → DeepWeeds)

Same fine-tuned backbones and classical heads as the paper replication; adaptation uses unlabelled DeepWeeds features only.

## DeepWeeds accuracy: paper pipeline vs + all three techniques

| Backbone | SVM base → all | LR base → all | HGB base → all | kNN base → all | RF base → all |
|---|---|---|---|---|---|
| mobilenet_v2 | 34.3 → **39.0** | 33.7 → **35.7** | 33.9 → **36.7** | 34.2 → **43.8** | 34.6 → **38.9** |
| resnet50 | 34.6 → **39.6** | 34.8 → **40.1** | 30.2 → **38.3** | 33.6 → **47.3** | 34.4 → **39.9** |
| densenet201 | 36.6 → **40.8** | 37.3 → **43.5** | 38.7 → **41.8** | 32.3 → **54.7** | 37.3 → **45.1** |
| vgg16 | 22.8 → **26.6** | 24.5 → **24.5** | 26.5 → **26.6** | 21.5 → **25.8** | 25.8 → **27.7** |
| vgg19 | 31.0 → **32.6** | 31.7 → **30.3** | 32.7 → **31.5** | 30.5 → **33.1** | 32.8 → **32.7** |
| regnet_y_32gf | 45.8 → **49.1** | 44.6 → **50.8** | 40.4 → **52.0** | 41.4 → **61.9** | 41.3 → **54.9** |

## Mean gain over the paper pipeline, across 6 backbones (DeepWeeds, points)

| Head | + restd | + align | + st | **+ all** | all improves |
|---|---|---|---|---|---|
| SVM | +2.4 | +1.6 | -0.2 | **+3.8** | 6/6 |
| LR | +1.5 | +1.8 | +1.2 | **+3.1** | 5/6 |
| HGB | +2.4 | +0.5 | +0.8 | **+4.1** | 5/6 |
| kNN | +3.7 | +1.9 | +2.6 | **+12.2** | 6/6 |
| RF | +2.3 | +2.1 | +2.1 | **+5.5** | 5/6 |

Best: **regnet_y_32gf + kNN + all three = 61.9%** on DeepWeeds (paper pipeline best: RegNetY-32GF + SVM = 45.7%). Prior alignment assumes a roughly uniform DeepWeeds class prior.
