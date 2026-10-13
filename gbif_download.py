"""
Download a larger, balanced GBIF training set for the eight weed species.

Same eligibility rules as the original 3,200-image set: no Australian records, CC0 or CC-BY
licences only. Additionally:
  - one photo per occurrence, so near-duplicate shots of the same plant cannot straddle the
    train/test split;
  - records are drawn from random result pages rather than the first N, since GBIF's default
    ordering clusters by dataset;
  - a manifest records occurrence key, country, licence and rights holder for every image
    (CC-BY requires attribution).

Rubber vine has the fewest eligible records (~921), so it is downloaded first and its success
count sets the per-class target (capped at TARGET) for every other class.
"""
import os, io, csv, time, random, argparse
import concurrent.futures as cf
import requests
from PIL import Image

SPECIES = [("Cryptostegia grandiflora", "rubber_vine"),      # scarcest first: sets the target
           ("Ziziphus mauritiana", "chinee_apple"), ("Lantana camara", "lantana"),
           ("Parkinsonia aculeata", "parkinsonia"), ("Parthenium hysterophorus", "parthenium"),
           ("Vachellia nilotica", "prickly_acacia"), ("Chromolaena odorata", "siam_weed"),
           ("Stachytarpheta", "snake_weed")]
API = "https://api.gbif.org/v1"
LICENSES = ["CC0_1_0", "CC_BY_4_0"]
HEADERS = {"User-Agent": "MRes weed-domain-shift research (non-commercial)"}

def get(url, **kw):
    for attempt in range(5):
        try:
            r = requests.get(url, headers=HEADERS, timeout=kw.pop("timeout", 30), **kw)
            r.raise_for_status(); return r
        except requests.RequestException:
            if attempt == 4: raise
            time.sleep(2 * (attempt + 1))

def taxon_key(name):
    return get(f"{API}/species/match", params={"name": name}).json()["usageKey"]

def candidates(key, want, rng):
    base = dict(taxonKey=key, mediaType="StillImage", license=LICENSES)
    total = get(f"{API}/occurrence/search", params={**base, "limit": 0}).json()["count"]
    pages = list(range(0, min(total, 100_000), 300)); rng.shuffle(pages)
    seen, out = set(), []
    for off in pages:
        for o in get(f"{API}/occurrence/search", params={**base, "limit": 300, "offset": off}).json()["results"]:
            if o.get("countryCode") == "AU":
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

Image.MAX_IMAGE_PIXELS = 250_000_000      # herbarium scans exceed PIL's default; draft() shrinks them
MAX_BYTES = 25_000_000

def fetch(url):
    """One attempt per URL, short timeouts, size-capped; large JPEGs decoded at reduced scale."""
    urls = [url.replace("/original.", "/medium."), url] if "/original." in url else [url]
    for u in urls:
        try:
            r = requests.get(u, headers=HEADERS, timeout=(5, 15), stream=True)
            r.raise_for_status()
            data = r.raw.read(MAX_BYTES + 1, decode_content=True)
            if len(data) > MAX_BYTES:
                continue
            im = Image.open(io.BytesIO(data))
            im.draft("RGB", (448, 448))
            im = im.convert("RGB")
            if min(im.size) >= 64:
                return im.resize((224, 224))
        except Exception:
            continue
    return None

def download_class(cls, cands, target, out_dir, workers):
    """Save images as they complete; stop once the target is reached."""
    os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
    kept = []
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, c["url"]): c for c in cands}
        for fut in cf.as_completed(futs):
            im = fut.result()
            if im is None or len(kept) >= target:
                continue
            c = futs[fut]; c["file"] = f"{cls}/{cls}_{len(kept):04d}.jpg"
            im.save(os.path.join(out_dir, c["file"]), quality=95); kept.append(c)
            if len(kept) >= target:
                for f in futs: f.cancel()
                break
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=900)
    ap.add_argument("--out", default="weeds/global_900")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rng = random.Random(42); manifest = []; target = a.target
    for sci, cls in SPECIES:
        t0 = time.time()
        cands = candidates(taxon_key(sci), int(target * 1.6), rng)
        kept = download_class(cls, cands, target, a.out, a.workers)
        if cls == "rubber_vine":
            target = min(a.target, len(kept))      # scarcest class caps everyone else
        for c in kept: c["class"] = cls
        manifest += kept
        print(f"{cls:16s} candidates={len(cands):5d} kept={len(kept):4d} target={target} ({time.time()-t0:.0f}s)", flush=True)
    # trim rubber vine if a later class fell short, so every class is equal
    n = min(sum(m["class"] == c for m in manifest) for _, c in SPECIES)
    final = []
    for _, cls in SPECIES:
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
