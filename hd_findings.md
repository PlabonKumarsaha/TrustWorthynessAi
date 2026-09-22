# Replicating HD(LM)²D on GBIF, and testing it under distribution shift

Method from Shaukat, Luo & Varadharajan (2023), *EAAI* 122:106030 — fine-tune an ImageNet CNN
end to end, take the penultimate ("last fully connected") features, refit classical classifiers
on them. Their claim is that the classical head adds **+16.56 accuracy points on average**, with
SVM best.

Applied here to weed imagery: source = GBIF (8 species, 60/20/20 → 1,920/640/640),
shift test = DeepWeeds (8,403 field images). 6 backbones × (1 end-to-end + 12 heads) = 78
combinations. Chance = 12.5%.

## 1. End-to-end fine-tuned models

| Backbone | GBIF test | DeepWeeds | Gap |
|---|---|---|---|
| regnet_y_32gf | 86.1% | **40.6%** | 45.5 |
| densenet201 | **86.4%** | 39.9% | 46.5 |
| resnet50 | 83.9% | 33.8% | 50.1 |
| mobilenet_v2 | 83.6% | 35.6% | 47.9 |
| vgg19 | 82.2% | 35.6% | 46.6 |
| vgg16 | 77.8% | 25.2% | 52.6 |

## 2. Mean gain of each head over the end-to-end model

| Head | In-domain gain | Improves | Shift gain | Improves |
|---|---|---|---|---|
| LR | **+1.12** | 5/6 | −0.68 | 2/6 |
| RF | +1.09 | **6/6** | −0.74 | 3/6 |
| HGB | +0.86 | 4/6 | −1.38 | 1/6 |
| SVM | +0.52 | 5/6 | −0.94 | 2/6 |
| XGBoost | +0.21 | 3/6 | −1.62 | 1/6 |
| GradBoost | −0.21 | 1/6 | −3.70 | 1/6 |
| Bagging | −0.28 | 2/6 | −2.26 | 0/6 |
| NB | −0.47 | 3/6 | −1.15 | 2/6 |
| kNN | −0.55 | 3/6 | −2.87 | 1/6 |
| DT | −4.22 | 0/6 | −6.38 | 0/6 |
| LDA | −5.42 | 0/6 | −7.97 | 0/6 |
| AdaBoost | −9.35 | 0/6 | −9.18 | 1/6 |

## Findings

**1. The headline gain does not replicate: +1.12, not +16.56.**
The best head (logistic regression) adds about **one point**, and SVM — the paper's proposed
method — adds **+0.52**. That is roughly **one-thirtieth** of their reported +16.56.

The likely reason is the one visible in their own tables: their gain is measured against
end-to-end baselines that look broken rather than weak (test losses of 30–54 for a 2-class
softmax, InceptionResNetV2 at 52.8%). Our end-to-end baselines are healthy (77.8–86.4%), so
there is little headroom for a refitted head to recover. The effect they report is plausibly
mostly a broken-baseline artefact.

**2. The SVM is not the mechanism.** LR (+1.12) beats SVM (+0.52), and RF is the only head that
improves every backbone (6/6). This matches the paper's own numbers, where LR came within 0.5
points of SVM — the benefit, such as it is, comes from refitting *any* reasonable linear or
ensemble head on deep features, not from the SVM specifically.

**3. Under distribution shift the advantage disappears and reverses.**
**Every one of the 12 heads has a negative mean gain on DeepWeeds.** The best is LR at −0.68;
the worst are AdaBoost (−9.18), LDA (−7.97) and DT (−6.38). Only RF beats the end-to-end model
on as many as 3 of 6 backbones. The paper never tested shift, and the practice it recommends
gives no benefit out of domain.

A reasonable reading: refitting a high-capacity head on frozen source features fits the source
feature distribution more tightly, which is worth ~1 point in domain and costs more than that
when the features move.

**4. In-domain selection survived shift here — unlike in our earlier experiments.**
The best in-domain combination, **regnet_y_32gf + SVM (88.1%)**, is also the best under shift
(**45.7%**, rank 1 of 78), so selecting on in-domain accuracy cost **0 points**. Rank correlation
across all 78 combinations is high (Spearman ρ = 0.858, p = 1.2e-23).

This contradicts our earlier four-species result, where CLIP+LoRA won in domain and lost by 7
points out of domain. The difference is that those were different *architecture families*
(a VLM vs a CNN), whereas here the backbones are all supervised ImageNet CNNs of similar vintage.
Selection appears to transfer within a family and break across families — worth stating as a
boundary condition rather than a general rule.

**5. The paper's backbone ranking does not carry over, but its best combination does.**
They found VGG16 best on every dataset; here VGG16 is **worst** (77.8%, and 25.2% under shift),
and VGG19 beats it. Their proposed RegNetY320 + SVM, however, is the best cell here too — and as
in their Table 13, SVM and LR are nearly tied (88.1 vs 87.8). Their backbone conclusion is likely
specific to byte-plot images; their best-combination conclusion is not.

**6. The shift itself is severe.** Mean gap 49.9 points (range 29.4–59.4). Best DeepWeeds
accuracy from any of the 78 combinations is 45.7%, against 40.6% with no head at all.

## Caveats
- PyTorch/MPS, not Keras/TensorFlow.
- 150-epoch cap with early stopping (patience 6); no backbone reached the cap.
- Heads use a StandardScaler and library defaults; the paper reports no preprocessing or tuning.
- GBIF is balanced, so their imbalance-augmentation stage does not apply.
- Single split, single seed, no cross-validation.
- The `seconds` field in `results_hd/e2e/*.json` includes time the machine spent asleep and is
  not a runtime measurement.

## Files
`hd_common.py`, `hd_train.py`, `hd_heads.py`, `hd_report.py`; results in `results_hd/`
(`e2e/`, `heads/`, `report.md`). Cached features are git-ignored.
