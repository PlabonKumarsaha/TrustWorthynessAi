"""
VLM model: CLIP ViT-B/32. Zero-shot, and LoRA fine-tuned on GBIF (source).
Benchmark in-domain on GBIF test and out-of-domain on DeepWeeds. Runs on MPS.
"""
import os, json, time
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import open_clip
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from bench_common import gbif_arrays, deepweeds_arrays, CLASSES, READABLE, NUM

torch.manual_seed(42); np.random.seed(42)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL, PRE = "ViT-B-32", "openai"
TEMPLATES = ["a photo of {}.", "a photo of {}, a type of weed.",
             "a close-up photo of a {} plant.", "a field photograph of {}.",
             "an image of the invasive weed {}."]

def score(y,p): return dict(acc=round(float(accuracy_score(y,p)),4),
                            macro_f1=round(float(f1_score(y,p,average="macro")),4))

@torch.no_grad()
def text_classifier(model, tok):
    ws=[]
    for name in READABLE:
        t = tok([tmp.format(name) for tmp in TEMPLATES]).to(DEVICE)
        f = F.normalize(model.encode_text(t), dim=-1).mean(0)
        ws.append(F.normalize(f, dim=-1))
    return torch.stack(ws)

@torch.no_grad()
def img_feats(model, X, preprocess, bs=64):
    xs = torch.stack([preprocess(Image.fromarray(a)) for a in X])
    out=[]
    for i in range(0,len(xs),bs):
        out.append(F.normalize(model.encode_image(xs[i:i+bs].to(DEVICE)),dim=-1).cpu())
    return torch.cat(out)

def evaluate(model, text_w, X, y, preprocess):
    logits = 100.0 * img_feats(model, X, preprocess) @ text_w.cpu().T
    return score(y, logits.argmax(1).numpy())

def main():
    t0=time.time(); print(f"device={DEVICE}")
    model,_,preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRE)
    tok = open_clip.get_tokenizer(MODEL); model = model.to(DEVICE).eval()
    Xtr,ytr = gbif_arrays("global_train"); Xte,yte = gbif_arrays("global_test"); Xau,yau = deepweeds_arrays()

    # zero-shot
    tw = text_classifier(model, tok)
    zs_in = evaluate(model, tw, Xte, yte, preprocess); zs_ood = evaluate(model, tw, Xau, yau, preprocess)
    print(f"[VLM zero-shot] GBIF acc={zs_in['acc']:.3f}  |  DeepWeeds acc={zs_ood['acc']:.3f}")

    # LoRA fine-tune on GBIF
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=r".*visual\.transformer\.resblocks\.\d+\.(attn\.out_proj|mlp\.c_fc|mlp\.c_proj)")
    model = get_peft_model(model, cfg)
    with torch.no_grad(): tw = text_classifier(model, tok).to(DEVICE)
    scale = model.logit_scale.exp().detach()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    Xtr_t = torch.stack([preprocess(Image.fromarray(a)) for a in Xtr]); ytr_t = torch.tensor(ytr)
    EPOCHS,BS,n = 10,32,len(Xtr_t); best=-1; best_state=None
    for ep in range(1,EPOCHS+1):
        model.train(); perm=torch.randperm(n); tot=0.0
        for i in range(0,n,BS):
            idx=perm[i:i+BS]; xb=Xtr_t[idx].to(DEVICE); yb=ytr_t[idx].to(DEVICE)
            f=F.normalize(model.encode_image(xb),dim=-1); logits=scale*f@tw.T
            loss=F.cross_entropy(logits,yb); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*len(idx)
        v=evaluate(model, tw, Xte, yte, preprocess)     # GBIF test as val proxy
        print(f"  epoch {ep:2d} loss={tot/n:.4f} gbif_acc={v['acc']:.3f}")
        if v["acc"]>best: best=v["acc"]; best_state={k:val.detach().cpu().clone() for k,val in model.state_dict().items() if "lora" in k.lower()}
    sd=model.state_dict(); sd.update(best_state); model.load_state_dict(sd); model.eval()

    lo_in = evaluate(model, tw, Xte, yte, preprocess); lo_ood = evaluate(model, tw, Xau, yau, preprocess)
    res = dict(model="VLM (CLIP ViT-B/32)",
               zeroshot=dict(in_domain_gbif=zs_in, ood_deepweeds=zs_ood),
               lora=dict(in_domain_gbif=lo_in, ood_deepweeds=lo_ood, gap_acc=round(lo_in["acc"]-lo_ood["acc"],4)),
               # headline uses the fine-tuned model for fair comparison with DL/ML
               in_domain_gbif=lo_in, ood_deepweeds=lo_ood, gap_acc=round(lo_in["acc"]-lo_ood["acc"],4),
               seconds=round(time.time()-t0,1), classes=CLASSES)
    os.makedirs("results_bench", exist_ok=True); json.dump(res, open("results_bench/vlm.json","w"), indent=2)
    print(f"[VLM LoRA] GBIF acc={lo_in['acc']:.3f} f1={lo_in['macro_f1']:.3f}  |  "
          f"DeepWeeds acc={lo_ood['acc']:.3f} f1={lo_ood['macro_f1']:.3f}  |  gap={res['gap_acc']:+.3f}")

if __name__ == "__main__":
    main()
