"""
Shared data layer for the DL-vs-ML-vs-VLM domain-shift benchmark.

Source (train/in-domain) : GBIF global weed imagery   weeds/global_{train,val,test}/<class>/
Target (out-of-domain)   : DeepWeeds (Australian)      weeds/_arrays/au4_{imgs,labels}.npy

We start with 4 visually-distinct, exact-match, well-populated species.
"""
import os, glob
import numpy as np
from PIL import Image

# subset: (GBIF folder name, index in the canonical 8-class DeepWeeds order)
SUBSET = [
    ("lantana",     1),
    ("parkinsonia", 2),
    ("parthenium",  3),
    ("rubber_vine", 5),
]
CLASSES   = [c for c, _ in SUBSET]          # new label order 0..3
DW_INDEX  = {orig: new for new, (_, orig) in enumerate(SUBSET)}   # DeepWeeds orig idx -> new label
READABLE  = ["lantana", "parkinsonia", "parthenium weed", "rubber vine"]
NUM = len(CLASSES)

def gbif_files(split):
    """Return (list[filepath], np.array[label]) for a GBIF split folder."""
    fs, ys = [], []
    for ci, c in enumerate(CLASSES):
        for fp in sorted(glob.glob(os.path.join(f"weeds/{split}", c, "*"))):
            fs.append(fp); ys.append(ci)
    return fs, np.array(ys)

def gbif_arrays(split, size=224):
    """Return (uint8 [N,size,size,3], labels) for a GBIF split."""
    fs, ys = gbif_files(split)
    xs = np.stack([np.asarray(Image.open(f).convert("RGB").resize((size, size)), dtype=np.uint8) for f in fs])
    return xs, ys

def deepweeds_arrays():
    return np.load("weeds/_arrays/au4_imgs.npy"), np.load("weeds/_arrays/au4_labels.npy")
