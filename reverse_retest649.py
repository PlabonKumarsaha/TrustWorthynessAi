"""
Re-test the DeepWeeds-trained ResNet-50 and Inception-v3 (reverse_train.py) on GBIF-649
(or another GBIF set: --gbif weeds/gbif_living --out results_reverse_living).

No retraining: loads the saved best checkpoints (models_cache/reverse_*.pt), re-scores the
DeepWeeds test split as a check that the checkpoint matches results_reverse/, then scores all
images of the GBIF set as the shift test. Writes results_reverse_649/ by default.
"""
import os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from reverse_common import CLASSES, DEVICE, eval_tf, ImgDS, deepweeds_splits, list_dir, scores
from reverse_train import build, predict

def run(name, gbif_dir, out):
    t0 = time.time()
    m = build(name); m.load_state_dict(torch.load(f"models_cache/reverse_{name}.pt", map_location=DEVICE))
    dl = lambda f, y: DataLoader(ImgDS(f, y, eval_tf), batch_size=64, shuffle=False, num_workers=4)
    _, _, (fte, yte) = deepweeds_splits(); fg, yg = list_dir(gbif_dir)
    indom = scores(*predict(m, dl(fte, yte))); ood = scores(*predict(m, dl(fg, yg)))
    old = json.load(open(f"results_reverse/{name}.json"))["in_domain_deepweeds"]["acc"]
    assert abs(indom["acc"] - old) < 0.002, f"checkpoint mismatch: {indom['acc']} vs {old}"
    res = dict(model=name, in_domain_deepweeds=indom, ood_gbif=ood, gbif_dir=gbif_dir, n_gbif=int(len(yg)),
               gap_acc=round(indom["acc"] - ood["acc"], 4), seconds=round(time.time() - t0, 1), classes=CLASSES)
    json.dump(res, open(f"{out}/{name}.json", "w"), indent=2)
    print(f"[{name}] DeepWeeds test {indom['acc']:.4f}  |  {gbif_dir} {ood['acc']:.4f} (F1 {ood['macro_f1']:.4f})  "
          f"|  gap {res['gap_acc']:+.4f}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbif", default="weeds/global_900"); ap.add_argument("--out", default="results_reverse_649")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for n in ["resnet50", "inception_v3"]: run(n, a.gbif, a.out)
