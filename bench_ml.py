"""
ML baseline: Random Forest on hand-crafted colour (HSV histogram) + texture (GLCM) features.
Train on GBIF (source); benchmark in-domain on GBIF test and out-of-domain on DeepWeeds.
"""
import os, json, time
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from bench_common import gbif_arrays, deepweeds_arrays, CLASSES

def extract(img):                      # img: HxWx3 uint8 RGB
    feats = []
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [16], [0, 256]).flatten()
        feats.extend(h / (h.sum() + 1e-6))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    q = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(q, [1], [0], levels=8, symmetric=True, normed=True)
    for p in ["contrast", "homogeneity", "energy", "correlation"]:
        feats.append(float(graycoprops(glcm, p)[0, 0]))
    return np.array(feats, dtype=np.float32)

def feats(X): return np.stack([extract(im) for im in X])
def score(y, p): return dict(acc=round(float(accuracy_score(y, p)), 4),
                             macro_f1=round(float(f1_score(y, p, average="macro")), 4))

def main():
    t0 = time.time()
    Xtr, ytr = gbif_arrays("global_train", size=128)
    Xte, yte = gbif_arrays("global_test",  size=128)
    Xau, yau = deepweeds_arrays()
    Xau = np.stack([cv2.resize(a, (128, 128)) for a in Xau])
    print(f"features: train {Xtr.shape} test {Xte.shape} DeepWeeds {Xau.shape}")

    Ftr, Fte, Fau = feats(Xtr), feats(Xte), feats(Xau)
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    clf.fit(Ftr, ytr)

    indom = score(yte, clf.predict(Fte))
    ood   = score(yau, clf.predict(Fau))
    res = dict(model="ML (RandomForest, colour+texture)",
               in_domain_gbif=indom, ood_deepweeds=ood,
               gap_acc=round(indom["acc"]-ood["acc"], 4),
               seconds=round(time.time()-t0, 1), classes=CLASSES)
    os.makedirs("results_bench", exist_ok=True)
    json.dump(res, open("results_bench/ml.json", "w"), indent=2)
    print(f"[ML ] GBIF acc={indom['acc']:.3f} f1={indom['macro_f1']:.3f}  |  "
          f"DeepWeeds acc={ood['acc']:.3f} f1={ood['macro_f1']:.3f}  |  gap={res['gap_acc']:+.3f}")

if __name__ == "__main__":
    main()
