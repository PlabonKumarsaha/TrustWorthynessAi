"""
VLM (CLIP) domain-generalisation experiment on 8 invasive weed species.

Story
-----
Source domain  : globally-sourced (GBIF, non-Australian) imagery  -> weeds/global_{train,val,test}
Target domain  : Australian field conditions (DeepWeeds)          -> weeds/_arrays/au_*.npy

1. Zero-shot CLIP (the fine-tuned-off-the-shelf VLM) -> initial accuracy on source.
2. Evaluate the SAME model on the Australian target -> accuracy falls (domain shift).
3. LoRA-adapt CLIP's visual tower on the source train set -> re-evaluate on both,
   testing whether the target accuracy recovers.

Everything reports top-1 accuracy and macro-F1 so the drop/recovery is unambiguous.
"""
import os, glob, json, time, random
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CLASS_NAMES = ["chinee_apple","lantana","parkinsonia","parthenium",
               "prickly_acacia","rubber_vine","siam_weed","snake_weed"]
# human-readable names for text prompts
READABLE = ["chinee apple","lantana","parkinsonia","parthenium weed",
            "prickly acacia","rubber vine","siam weed","snake weed"]
NUM = len(CLASS_NAMES)

MODEL_NAME, PRETRAINED = "ViT-B-32", "openai"

# ---------------------------------------------------------------- data
def load_folder(folder, preprocess):
    """Return (tensor[N,3,H,W], labels[N]) from an ImageFolder-style dir."""
    xs, ys = [], []
    for ci, c in enumerate(CLASS_NAMES):
        for fp in sorted(glob.glob(os.path.join(folder, c, "*"))):
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            xs.append(preprocess(img)); ys.append(ci)
    return torch.stack(xs), torch.tensor(ys)

def load_arrays(img_npy, lbl_npy, preprocess):
    imgs = np.load(img_npy); lbls = np.load(lbl_npy)
    xs = [preprocess(Image.fromarray(a)) for a in imgs]
    return torch.stack(xs), torch.tensor(lbls)

# ---------------------------------------------------------------- text classifier
PROMPT_TEMPLATES = [
    "a photo of {}.",
    "a photo of {}, a type of weed.",
    "a close-up photo of a {} plant.",
    "a field photograph of {}.",
    "an image of the invasive weed {}.",
]

@torch.no_grad()
def build_text_classifier(model, tokenizer, templates):
    weights = []
    for name in READABLE:
        toks = tokenizer([t.format(name) for t in templates]).to(DEVICE)
        tf_ = model.encode_text(toks)
        tf_ = F.normalize(tf_, dim=-1).mean(0)      # ensemble over templates
        tf_ = F.normalize(tf_, dim=-1)
        weights.append(tf_)
    return torch.stack(weights)                     # [NUM, D]

# ---------------------------------------------------------------- eval
@torch.no_grad()
def encode_images(model, x, bs=64):
    feats = []
    for i in range(0, len(x), bs):
        f = model.encode_image(x[i:i+bs].to(DEVICE))
        feats.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(feats)

def evaluate(model, text_w, x, y, scale=100.0):
    img_f = encode_images(model, x)
    logits = scale * img_f @ text_w.cpu().T
    pred = logits.argmax(1).numpy()
    yt = y.numpy()
    return dict(acc=float(accuracy_score(yt, pred)),
                macro_f1=float(f1_score(yt, pred, average="macro")))

# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    print(f"device={DEVICE}  model={MODEL_NAME}/{PRETRAINED}")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE).eval()

    print("loading data ...")
    Xtr, Ytr = load_folder("weeds/global_train", preprocess)
    Xva, Yva = load_folder("weeds/global_val",   preprocess)
    Xte, Yte = load_folder("weeds/global_test",  preprocess)
    Xau, Yau = load_arrays("weeds/_arrays/au_imgs.npy", "weeds/_arrays/au_labels.npy", preprocess)
    print(f"  source train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  test {tuple(Xte.shape)}  |  target(AU) {tuple(Xau.shape)}")

    results = {}

    # ---- 1 & 2: zero-shot baseline (single + ensembled prompts) ----
    for tag, templates in [("zeroshot_single", ["a photo of {}, a type of weed."]),
                           ("zeroshot_ensemble", PROMPT_TEMPLATES)]:
        text_w = build_text_classifier(model, tokenizer, templates)
        src = evaluate(model, text_w, Xte, Yte)
        tgt = evaluate(model, text_w, Xau, Yau)
        results[tag] = dict(source_test=src, australian=tgt)
        print(f"[{tag:18s}] source acc={src['acc']:.3f} f1={src['macro_f1']:.3f}  |  "
              f"AU acc={tgt['acc']:.3f} f1={tgt['macro_f1']:.3f}  |  drop={src['acc']-tgt['acc']:+.3f}")

    # ---- 3: LoRA-adapt the visual tower on the SOURCE train set ----
    print("\nattaching LoRA to CLIP visual tower ...")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=r".*visual\.transformer\.resblocks\.\d+\.(attn\.out_proj|mlp\.c_fc|mlp\.c_proj)",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # fixed text classifier (ensembled) from the frozen text tower
    model.eval()
    with torch.no_grad():
        text_w = build_text_classifier(model, tokenizer, PROMPT_TEMPLATES).to(DEVICE)

    logit_scale = model.logit_scale.exp().detach() if hasattr(model, "logit_scale") else torch.tensor(100.0, device=DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)

    EPOCHS, BS = 10, 32
    n = len(Xtr)
    best_val, best_state, best_epoch = -1.0, None, -1
    for ep in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(n)
        tot_loss = 0.0
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            xb = Xtr[idx].to(DEVICE); yb = Ytr[idx].to(DEVICE)
            f = F.normalize(model.encode_image(xb), dim=-1)
            logits = logit_scale * f @ text_w.T
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(idx)
        val = evaluate(model, text_w, Xva, Yva)
        print(f"  epoch {ep:2d}  loss={tot_loss/n:.4f}  val_acc={val['acc']:.3f}")
        if val["acc"] > best_val:
            best_val, best_epoch = val["acc"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k.lower()}

    # restore best LoRA weights
    if best_state is not None:
        sd = model.state_dict(); sd.update(best_state); model.load_state_dict(sd)
    print(f"  best val_acc={best_val:.3f} @ epoch {best_epoch}")

    model.eval()
    src = evaluate(model, text_w, Xte, Yte)
    tgt = evaluate(model, text_w, Xau, Yau)
    results["lora_source_adapted"] = dict(source_test=src, australian=tgt,
                                          best_val_acc=best_val, best_epoch=best_epoch,
                                          trainable_params=trainable)
    print(f"[lora_source_adapt ] source acc={src['acc']:.3f} f1={src['macro_f1']:.3f}  |  "
          f"AU acc={tgt['acc']:.3f} f1={tgt['macro_f1']:.3f}")

    results["_meta"] = dict(device=DEVICE, model=f"{MODEL_NAME}/{PRETRAINED}",
                            seconds=round(time.time()-t0,1),
                            n_source_test=len(Yte), n_australian=len(Yau))
    json.dump(results, open("clip_experiment_results.json","w"), indent=2)
    print(f"\nsaved -> clip_experiment_results.json   ({results['_meta']['seconds']}s)")

if __name__ == "__main__":
    main()
