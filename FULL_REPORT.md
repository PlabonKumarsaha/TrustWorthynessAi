# Cross-domain weed classification: progress report

## Aim

The question I set out to answer is whether a weed classifier trained entirely on free,
web-sourced imagery can be deployed on Australian field photographs without any local
annotation. Labelling rangeland imagery is expensive, and there is a large amount of
already-labelled plant photography sitting in biodiversity databases, so if the transfer works
it would remove most of the annotation cost from a weed-mapping pipeline. The obvious risk is
domain shift: web photographs and field photographs of the same species do not look alike.

All experiments were run on a single M4 MacBook, which shaped some of the design decisions —
in particular the reliance on frozen features and linear probes rather than repeated
end-to-end fine-tuning.

## 1. Data

### Source: GBIF

I built the training set by querying GBIF for occurrence photographs of eight weed species,
resolving each species by scientific name rather than common name, and excluding any record
from Australia so that the source and target domains genuinely differ. Only CC0 and CC-BY
licensed media were kept. The download was capped at 400 images per class, giving 3,200 images
which I split 70/15/15 into 280 train, 60 validation and 60 test per class. Everything is
224×224 RGB JPEG.

The eight species and the names used to resolve them:

| Folder | Scientific name |
|---|---|
| chinee_apple | *Ziziphus mauritiana* |
| lantana | *Lantana camara* |
| parkinsonia | *Parkinsonia aculeata* |
| parthenium | *Parthenium hysterophorus* |
| prickly_acacia | *Vachellia nilotica* (syn. *Acacia nilotica*) |
| rubber_vine | *Cryptostegia grandiflora* |
| siam_weed | *Chromolaena odorata* |
| snake_weed | *Stachytarpheta* (genus only) |

Seven of the eight resolve to an exact species. Snake weed is the exception: I could only match
it at genus level, so that folder may contain several *Stachytarpheta* species. This turned out
to matter later, because snake weed consistently behaves as the worst-performing class.

In character these are mostly deliberate photographs — a plant, a flower, sometimes a herbarium
specimen — usually with the subject centred and reasonably well lit.

### Target: DeepWeeds

DeepWeeds (Olsen et al., 2019) contains 17,509 photographs taken in situ in northern Australian
rangelands, covering the same eight species plus a ninth "negative" class of vegetation that is
not a target weed. The weed classes are close to balanced, between 1,009 and 1,125 images each,
but the negative class is large: 9,106 images, slightly more than half the dataset. The images
are cluttered field scenes with natural lighting, soil and occlusion, which is exactly the
distribution shift I wanted to measure.

For most of the project I evaluated on subsets — first four species, then all eight, both with
the negative class removed and classes balanced by subsampling. Only in the final phase did I
evaluate on the complete 17,509 images with the natural distribution and the negatives included.
That decision turned out to be important and I return to it below.

## 2. Related work

The obvious reference point is the DeepWeeds paper itself, which reports 95.7% with a ResNet-50
and 95.1% with Inception-v3. Those numbers are in-domain and fully supervised — the model is
trained on DeepWeeds itself — so they represent a ceiling rather than a competitor. More recent
in-domain work using latent-diffusion augmentation reaches about 98.5%.

Domain adaptation for agricultural imagery is an active area. Computers and Electronics in
Agriculture published a self-training approach for weed segmentation in 2024, and a CVPR
workshop paper the same year used greedy pseudo-labelling for the same task. Closest to my own
setup is "From Web Data to Real Fields" (2025), which adapts detectors from internet-scraped
imagery to unlabelled robot-collected field data using an attention-based adversarial
discriminator, reporting a 7.5% detection improvement. A 2025 review of domain adaptation in
agricultural image analysis gives the standard taxonomy: discrepancy methods such as CORAL and
MMD, adversarial approaches such as DANN, style transfer, pseudo-labelling, Fourier methods and
test-time adaptation. For context on how badly cross-dataset transfer usually goes, plant disease
models trained on PlantVillage typically drop to around 68% on PlantDoc.

On the source-free and test-time side the standard methods are TENT, which minimises prediction
entropy over BatchNorm affine parameters; AdaBN, which simply recomputes BatchNorm statistics on
the target; SHOT, which freezes the source classifier and adapts the feature extractor using
information maximisation and pseudo-labels; and class-balanced self-training. There are recent
surveys of both source-free UDA and test-time adaptation.

Vision-language models have been applied to this problem too. Tip-Adapter provides a
training-free cache adapter for few-shot CLIP; CoOp and CLIP-Adapter learn prompts or small
adapters instead. ReCLIP uses agreement between CLIP's own image and text encoders to select
pseudo-labels, and Co-learn++ and related work use CLIP as a pseudo-label teacher for
source-free adaptation. The strongest recent result in this family is RCL, which uses multimodal
LLMs to guide a reliability curriculum and reports state-of-the-art results with a 9.4%
improvement on DomainNet. In agriculture specifically there is AgriCLIP, CropVLM, WeedCLR and
AgroBench, and a 2025 paper asking directly whether VLMs can zero-shot replace supervised
classifiers in agriculture — the answer being not yet for fine-grained species.

One line of work is directly relevant to the conclusion I initially wanted to draw. A 2025 paper
on class-imbalanced federated source-free adaptation argues explicitly for shifting attention
from improving adaptation methods to improving feature extractors, and another shows large
self-supervised models closing the gap in domain-adaptive object detection. There is a similar
reality-check paper in digital pathology. This means the claim "the backbone matters more than
the adaptation method" is already in the literature for other settings, and I cannot present it
as new. What I can contribute is a more specific finding about how the two interact, which I
come back to in section 4.

## 3. What I tried

I worked through this in roughly ten stages. The order matters because several later decisions
were reactions to earlier results.

### Starting with CLIP

I began with CLIP ViT-B/32 as a vision-language model, on all eight species. Zero-shot
classification using text prompts gave 38.8% on the GBIF test split with prompt ensembling, but
only 11.7% on DeepWeeds, which is essentially the 12.5% chance level for eight classes.
Fine-tuning with LoRA on about 1.2% of the parameters lifted GBIF to 88.5% and DeepWeeds to
35.2%. So adaptation helped the target substantially in relative terms, but the gap widened
rather than closed.

I then tried test-time methods on the frozen model: test-time augmentation, Sinkhorn
distribution alignment, and TENT. TTA alone did almost nothing. Distribution alignment helped
noticeably. Switching to a larger backbone, ViT-L/14 trained on LAION-2B, and combining it with
TTA and alignment reached 27.5% — still poor, but more than double the zero-shot baseline. TENT
failed outright here; its macro-F1 collapsed to 0.03. In hindsight this makes sense: entropy
minimisation sharpens whatever the model already believes, and when the model is at chance and
confidently wrong, sharpening makes things worse.

### Comparing three model families

To get a cleaner picture I dropped to four visually distinct species with exact taxonomic
matches (lantana, parkinsonia, parthenium, rubber vine) and compared three families trained on
GBIF: a Random Forest on HSV colour histograms and GLCM texture features, a fine-tuned
ResNet-50, and CLIP with LoRA.

| Model | GBIF test | DeepWeeds | Gap |
|---|---|---|---|
| ResNet-50, fine-tuned | 94.6% | 65.3% | 29.3 |
| CLIP ViT-B/32 + LoRA | 98.8% | 58.4% | 40.3 |
| CLIP ViT-B/32, zero-shot | 80.8% | 28.1% | 52.8 |
| Random Forest (colour + texture) | 68.3% | 27.7% | 40.7 |

The result I did not expect is that the model with the best in-domain accuracy was not the best
on the target. CLIP with LoRA wins on GBIF by four points but loses to ResNet-50 by seven points
on DeepWeeds. It fits the source domain harder and generalises worse. Selecting a deployment
model on in-domain accuracy would have picked the wrong one.

### Preprocessing

Since a good part of the shift is colour and lighting, I tried four preprocessing techniques
applied to the target images at inference, with the models frozen. CLAHE contrast normalisation
gave small but consistent gains (+1.8 on ResNet, +3.2 on the Random Forest) and was the only
technique that never hurt. Histogram matching and Fourier Domain Adaptation each gave about
+1.3 to CLIP but cost the CNN about a point. Reinhard colour transfer was a disaster: it cost
ResNet-50 twenty points and CLIP thirty. Forcing the source colour statistics onto the target
apparently destroys the colour information the classifier actually relies on.

The useful conclusion is that the best preprocessing depends on the model — the CNN prefers
contrast normalisation, CLIP prefers spectral or tonal alignment — and that aggressive global
colour remapping should be avoided.

### Adaptation methods

Working still on four species, I tried AdaBN, TENT, ensembling and self-training. AdaBN on its
own actually hurt, dropping ResNet from 66.6% to 63.5%. TENT gave about a point. Ensembling
ResNet with CLIP was the largest easy gain, taking either model from around 67% to 73.3%,
presumably because the two make different mistakes.

The best result in this phase came from self-training: using the ResNet-CLIP ensemble as a
teacher, keeping only predictions above 0.9 confidence, and fine-tuning ResNet on those
pseudo-labels. The confident subset turned out to be 98% correct, and two rounds took ResNet to
82.8% and the ensemble to 84.1%, without using a single DeepWeeds label. For comparison, giving
the model 10 real labels per class and training a linear probe reached 89.2%.

Tip-Adapter, which I had expected to do well, barely moved the needle — around 68% regardless of
the number of shots. The frozen CLIP features simply are not discriminative enough on these
species for a cache-based method to exploit.

### A selection rule of my own

Two observations from the above suggested something. First, single-model confidence is not
trustworthy under this shift — CLIP was confidently wrong. Second, when ResNet and CLIP agreed,
they were right 87.6% of the time, and when they disagreed, only 53.3%. In fact 72.8% of all
errors fell in the disagreement set.

So I built a pseudo-label selection rule that keeps a sample only if the two architectures agree
and it passes a conformal threshold calibrated on a small held-out pool, and compared it against
plain confidence thresholding in an otherwise identical self-training pipeline. The proposed rule
kept more data (51% vs 35% coverage) at equal or better precision, and improved the final result
from 75.6% to 80.3%. That is a 4.7 point gain attributable purely to the selection rule.

I later checked the literature more carefully and found that ReCLIP, Co-learn++ and CLIP-OT
already occupy much of this space, so the idea is less novel than I thought when I built it.

### Changing the backbone

The largest single improvement in the whole project came from replacing the backbone. A linear
probe on frozen DINOv2 ViT-B/14 features, trained only on GBIF labels, reached 78.2% on the
four-species DeepWeeds subset at 224 pixels, 88.2% at 336 and 90.8% at 448. Adding label-free
self-training took it to 94.2%. That is more than twenty points above the fine-tuned ResNet, with
no adaptation machinery at all beyond self-training.

Resolution mattered far more than I expected — ten points between 224 and 336 alone. Two things
I tried here did not work: fusing multiple resolutions was worse than simply using the best one,
and ensembling DINOv2 with the weaker ResNet and CLIP models dropped it from 88.2% to 85.2%.
Averaging models of unequal quality drags the good one down, which contradicts the conclusion I
had drawn earlier when the two models being ensembled were evenly matched.

### Scaling to eight species

At this point I had a 94.2% result and was fairly pleased with it, so the next step was to check
it held on all eight species. It did not. The same pipeline gave 57.8% at 336 pixels and 63.8% at
448, rising only to 68.8% with self-training. Moving from four classes to eight cost roughly
twenty-five points. The four-species result had been substantially an artefact of an easier task.

The per-class breakdown showed why, and pointed at the fix. Parthenium and snake weed had high
recall but low precision — 0.96 recall at 0.44 precision for parthenium — while prickly acacia
collapsed to 0.32 recall despite 0.69 precision. Two classes were acting as sinks, absorbing
predictions that belonged to others. Worse, self-training made this progressively worse:
pseudo-label precision fell from 81.3% to 77.2% to 74.9% across rounds as the sinks swallowed
more of the pseudo-labelled set.

Applying Sinkhorn distribution alignment and class-balanced selection fixed most of it:

| Method (8 species) | DeepWeeds |
|---|---|
| DINOv2 @448, no adaptation | 63.8% |
| plus naive self-training | 68.8% |
| plus distribution alignment only | 73.2% |
| plus class-balanced self-training only | 71.8% |
| plus both | 80.6% |

Per-class F1 evened out from a 0.43–0.90 spread to 0.71–0.90, and — the part I find most
interesting — pseudo-label precision stopped decaying, holding at 89.6%, 89.0%, 87.6% across the
three rounds instead of falling. Alignment does not just add accuracy; it prevents self-training
from degrading itself.

### A factorial comparison

To understand how much of the result came from each ingredient I ran a factorial over three
backbones (ResNet-50, CLIP ViT-B/32, DINOv2), three resolutions and five adaptation settings,
using an identical frozen-feature and linear-probe protocol throughout so the classifier was
held constant.

| Backbone at resolution | in-domain | none | DA | ST | CBST | DA+CBST |
|---|---|---|---|---|---|---|
| ResNet-50 @224 | 79.4 | 29.8 | 33.2 | 29.1 | 29.8 | 36.0 |
| ResNet-50 @336 | 81.9 | 33.2 | 38.0 | 32.6 | 33.9 | 41.2 |
| ResNet-50 @448 | 84.0 | 37.6 | 41.8 | 37.5 | 37.8 | 45.6 |
| CLIP ViT-B/32 @224 | 85.8 | 34.2 | 35.4 | 32.8 | 35.6 | 38.6 |
| DINOv2 @224 | 93.1 | 54.6 | 64.2 | 56.0 | 55.6 | 72.9 |
| DINOv2 @336 | 94.2 | 57.8 | 69.1 | 60.8 | 62.8 | 79.8 |
| DINOv2 @448 | 95.2 | 63.8 | 73.2 | 68.8 | 71.8 | 80.6 |

The spread attributable to the backbone is 30.8 points, against 11.9 for the adaptation method
and 9.8 for resolution. My hypothesis going in was that the ranking of adaptation methods would
break down once the backbone was strong enough — that conclusions drawn on ResNet-50 at 224
pixels would not transfer. That was wrong. Spearman correlation between the rankings from
different backbones is 0.80 to 0.90; the ordering is essentially stable.

What the data does support is the opposite of what I assumed. The benefit of adaptation grows
with the quality of the representation: distribution alignment plus class-balanced self-training
adds 6.2 points on ResNet-50 at 224 but 16.8 points on DINOv2 at 448. Backbone quality and
adaptation are complementary rather than substitutes, which is a more specific claim than the
2025 papers arguing that foundation models should replace adaptation methods.

### Comparing against published baselines

Late in the project I implemented CORAL, DANN and SHOT on the same frozen DINOv2 features and
scored them on the same full natural weed set, so the comparison is like for like. Since SHOT and
DANN need a trainable feature transform, I also trained a source-only model with the identical
MLP architecture, otherwise their gains would be confounded with the extra capacity.

| Method | Accuracy | macro-F1 |
|---|---|---|
| SHOT | 85.7% | 0.860 |
| DA + CBST (mine) | 79.9% | 0.799 |
| DA only (mine) | 72.9% | 0.730 |
| source-only, MLP (matched architecture) | 67.4% | 0.677 |
| source-only, linear probe | 64.9% | 0.653 |
| DANN | 63.9% | 0.633 |
| CORAL | 52.6% | 0.520 |

SHOT beats my approach by 5.8 points, and the matched-architecture control shows this is not a
capacity effect: SHOT gains 18.3 points over its own source-only baseline, against 15.0 for my
DA plus class-balanced self-training over the linear probe. Its adaptation is simply better. I
had expected it to be competitive and it was, which is the main reason for running the comparison
rather than assuming.

Two of the three baselines were worse than doing nothing. DANN came in below its own source-only
model, which I attribute to adversarial training being unstable with only 2,240 source samples.
CORAL was worst by a wide margin, and that is unsurprising in hindsight: whiten-and-recolour
requires estimating a 768 by 768 covariance matrix from 2,240 points, which is badly conditioned.

There is a secondary observation here that supports the factorial result. SHOT was developed and
evaluated on ResNet-era backbones, and applying it to DINOv2 features yields a larger absolute
gain than it reports on its original benchmarks. That is independent corroboration, from a method
I did not design, that adaptation and representation quality compound.

### Testing it under realistic conditions

Everything up to this point used balanced subsets with the negative class removed. Both of those
choices flatter the result, so the last stage removed them: the full 17,509 images, the natural
class distribution, and the 9,106 negatives included.

The closed-set result held up. On the natural, unbalanced weed set the unadapted probe gives
64.9%, alignment takes it to 72.9%, and alignment plus class-balanced self-training reaches
80.1% with a macro-F1 of 0.801 — against 80.6% on the balanced subset. The subsampling had not
been inflating anything, for the straightforward reason that DeepWeeds' weed classes are already
close to balanced.

Two things did not hold up.

The first is the class prior. Alignment needs to know the target class distribution, and I had
been assuming uniform, which happened to be roughly correct. To remove the assumption I estimated
the prior from unlabelled data using EM. The estimate was badly wrong — an L1 error of 0.577 —
and using it dropped accuracy to 61.3%, which is worse than not adapting at all. EM inherits the
model's existing bias toward the sink classes and then amplifies it. So the eight to fifteen
point gain from alignment rests on prior knowledge that a practitioner may not have.

The second is open-set rejection, and this is the more serious problem. A deployed system has to
reject the vegetation that is not a target weed, and over half of DeepWeeds is exactly that. I
compared eleven label-free rejection scores:

| Score | AUROC | FPR at 95% TPR |
|---|---|---|
| kNN in feature space | 0.808 | 63.8% |
| Energy | 0.772 | 76.4% |
| MaxLogit | 0.770 | 75.6% |
| Mahalanobis | 0.769 | 76.3% |
| Entropy | 0.733 | 78.8% |
| Max softmax | 0.718 | 83.7% |
| Relative Mahalanobis | 0.677 | 91.9% |
| CLIP text-defined negatives | 0.634 | 85.3% |
| Cross-architecture disagreement | 0.545 | 94.4% |

Only kNN improved on the earlier baseline, and only modestly. Two ideas of my own failed.
Describing non-target vegetation in language — grass, bare soil, gravel, native bushland — and
using CLIP to reject anything matching those prompts gave 0.634, well below simple energy
scoring. And cross-architecture disagreement, which had predicted classification errors so well
earlier, is essentially at chance (0.545) for detecting whether an image is out of distribution.
Those are different quantities and I had conflated them. Fusing scores also made things worse
rather than better, which is the third time in this project that combining signals of unequal
quality has hurt.

The number that matters operationally is the last column. To catch 95% of real weeds the best
method still accepts 64% of ordinary vegetation. An AUROC of 0.81 sounds respectable and is
useless for a sprayer.

## 4. Where things stand

Training only on free GBIF imagery, with no Australian labels at all, the best configuration
reaches 85.7% accuracy and 0.860 macro-F1 on the full DeepWeeds weed set. That configuration is
SHOT applied to frozen DINOv2 features, not my own method, which reaches 79.9%. Both are well
above the 64.9% unadapted probe and the 11.7% zero-shot CLIP result I started with. The fully
supervised in-domain reference is 95.7%, so roughly ten points of gap remain.

The findings I am reasonably confident in are these. Backbone choice dominates everything else,
and was worth more than all the adaptation methods combined. Input resolution is a first-class
variable and is under-reported in this literature. Adaptation and representation quality are
complementary, with adaptation gains growing rather than shrinking as the backbone improves.
Class-bias correction is unnecessary at four classes and essential at eight. Distribution
alignment stops self-training from degrading itself. And evaluating on a species subset inflated
my result by roughly twenty-five points, which makes me wary of subset results elsewhere.

The failures are worth recording too, since several cost me time and might save someone else's.
Label-free prior estimation does not work here and actively backfires. Open-set rejection is not
deployable at the accuracy I can currently reach. Reinhard colour transfer is harmful. AdaBN
alone hurts and TENT fails on a weak base model. Tip-Adapter is no better than zero-shot on this
task. Language-defined negatives and cross-architecture disagreement both fail as OOD signals.
And combining signals of unequal quality has hurt every time I have tried it.

## 5. Limitations

The published baselines are now implemented, and the outcome is that my own combination of
distribution alignment and class-balanced self-training is not competitive: SHOT beats it by 5.8
points on identical features. I therefore cannot present the method as a contribution, and the
value of the work lies in the empirical study and the characterised failure modes rather than in
a new algorithm.

I also need to be careful about the word "seeds". The factorial ran three seeds per cell but the
standard deviations came out as exactly zero, because a linear probe with fixed data and a
deterministic selection rule has nothing to vary. I therefore have no variance estimate at all,
and getting one requires resampling the splits rather than changing a random seed.

The frozen linear-probe protocol used in the factorial also handicaps ResNet-50, which scores
29.8% frozen against 66.6% when fine-tuned. Some of the 30.8-point backbone spread is a
consequence of that design choice rather than a property of the backbones.

Beyond that: the adaptation is transductive, so it assumes the unlabelled target batch is
available at adaptation time; there is only one target dataset; there are no cross-validation
folds or significance tests; and snake weed is matched only at genus level.

## 6. Next steps

In order of priority: obtain a real variance estimate by resampling the splits; add a second
target dataset such as CWFID so the findings are not tied to DeepWeeds; check whether SHOT and my
alignment approach are complementary, since SHOT does not use an explicit class-prior correction
and the sink-class problem may still be present in its output; and treat open-set rejection as the
main open problem, since it is what actually blocks deployment.

On publication, my assessment is that this is an application and benchmarking contribution
rather than a novel method, and the baseline comparison confirms that: SHOT outperforms what I
built. The backbone-versus-method argument is already published elsewhere. What remains defensible
is the interaction result, that adaptation and representation quality compound, which is now
supported by SHOT's behaviour as well as my own; the demonstration that annotation-free transfer
from web imagery reaches 85.7% on real field data; and a fairly thorough set of characterised
failure modes.
Computers and Electronics in Agriculture or Frontiers in Plant Science seem like the realistic
targets, framed around annotation-free deployment with an honestly characterised open problem.

## Code

The four-species comparison is in `bench_*.py`, preprocessing in `prep_*.py`, adaptation methods
in `adapt_*.py`, the conformal selection rule in `novel_*.py`, the DINOv2 and resolution work in
`accuracy_push*.py`, the factorial in `qi_*.py`, and the realistic and open-set evaluations in
`realistic_*.py` and `openset.py`. Results are in the corresponding `results_*` directories.
Datasets, model weights and feature caches are excluded from version control.
