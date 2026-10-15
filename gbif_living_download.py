"""
Download GBIF-Living: a balanced GBIF set of living-plant photographs only.

GBIF-649 turned out to be 61% herbarium sheets (pressed, dried specimens), which look nothing
like field imagery. This set keeps the same eligibility rules as gbif_download.py (no Australian
records, one photo per occurrence, random result pages, manifest with attribution) and adds:
  - basisOfRecord = HUMAN_OBSERVATION only (photos of living plants; no herbarium sheets);
  - CC-BY-NC 4.0 admitted alongside CC0 and CC-BY 4.0, because most living-plant photos on GBIF
    come from iNaturalist under the non-commercial licence (fine for academic research; the
    manifest records the licence of every image).

Chinee apple has the fewest eligible records (~900), so it is downloaded first and its success
count sets the per-class target for every other class. Writes weeds/gbif_living/.
"""
import os, csv, time, random, argparse

from gbif_download import SPECIES, API, get, taxon_key, download_class

LICENSES = ["CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0"]
ORDER = ["chinee_apple", "rubber_vine", "prickly_acacia", "snake_weed", "siam_weed",
         "parkinsonia", "parthenium", "lantana"]                 # scarcest first

def candidates(key, want, rng):
    base = dict(taxonKey=key, mediaType="StillImage", license=LICENSES, basisOfRecord="HUMAN_OBSERVATION")
    total = get(f"{API}/occurrence/search", params={**base, "limit": 0}).json()["count"]
    pages = list(range(0, min(total, 100_000), 300)); rng.shuffle(pages)
    seen, out = set(), []
    for off in pages:
        for o in get(f"{API}/occurrence/search", params={**base, "limit": 300, "offset": off}).json()["results"]:
            if o.get("countryCode") == "AU" or o.get("basisOfRecord") != "HUMAN_OBSERVATION":
                continue
            media = [m for m in o.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")]
            if not media or media[0]["identifier"] in seen:
                continue
            seen.add(media[0]["identifier"])
            out.append(dict(occurrence=o["key"], country=o.get("countryCode", ""),
                            license=o.get("license", ""), url=media[0]["identifier"],
                            rights_holder=media[0].get("rightsHolder") or o.get("rightsHolder", ""),
                            dataset=o.get("datasetKey", "")))
        if len(out) >= want:
            break
        time.sleep(0.2)
    rng.shuffle(out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=900)
    ap.add_argument("--out", default="weeds/gbif_living")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    sci_of = {cls: sci for sci, cls in SPECIES}
    rng = random.Random(42); manifest = []; target = a.target
    for i, cls in enumerate(ORDER):
        t0 = time.time()
        cands = candidates(taxon_key(sci_of[cls]), int(target * 1.6), rng)
        kept = download_class(cls, cands, target, a.out, a.workers)
        if i == 0:
            target = min(a.target, len(kept))      # scarcest class caps everyone else
        for c in kept: c["class"] = cls
        manifest += kept
        print(f"{cls:16s} candidates={len(cands):5d} kept={len(kept):4d} target={target} ({time.time()-t0:.0f}s)", flush=True)
    n = min(sum(m["class"] == c for m in manifest) for c in ORDER)   # equalise if a later class fell short
    final = []
    for cls in ORDER:
        rows = [m for m in manifest if m["class"] == cls]
        for m in rows[n:]: os.remove(os.path.join(a.out, m["file"]))
        final += rows[:n]
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "class", "occurrence", "country", "license",
                                          "rights_holder", "dataset", "url"])
        w.writeheader(); w.writerows(final)
    print(f"done: {n} per class, {len(final)} images, manifest at {a.out}/manifest.csv")

if __name__ == "__main__":
    main()
