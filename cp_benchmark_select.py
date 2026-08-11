# -*- coding: utf-8 -*-
"""Select candidate parts for a NEW, UNTOUCHED human-CP benchmark (audit P0 #6). Reproducible,
SHA-deduped, and denylisted against everything the models have already seen.

Denylist = the 102 Scheffler corpus parts (by part_id AND by their source-STEP SHA-256 from each
part's provenance.json). NOTE: v28/v31 pseudo-corpus train IDs should ALSO be denylisted once
their manifests are located; this script denylists what is verifiable offline and records the gap.

Selection: from the SHA-deduped, denylisted pool, spread across part-id prefix buckets for family
diversity, target ~`--n` parts. Output = a candidate MANIFEST only (ids + SHAs + files). NO
labels — human point+direction labelling is a separate, human step. Overlap vs denylist = 0 by
construction (asserted).

Usage: .venv/Scripts/python.exe cp_benchmark_select.py --n 40
"""
import argparse, os, glob, json, hashlib, re, random

POOL = "all_wscad_stp"
CORPUS = "wscad_corpus_scheffler_exact"
PID_RE = re.compile(r"wscaduniverse_([0-9A-Za-z\-]+)_")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scheffler_denylist():
    ids, shas = set(), set()
    for prov in glob.glob(os.path.join(CORPUS, "*", "*", "*.provenance.json")):
        d = json.load(open(prov))
        ids.add(d["part_id"])
        if d.get("source_step_sha256"):
            shas.add(d["source_step_sha256"])
    return ids, shas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="benchmark_candidates.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    deny_ids, deny_shas = scheffler_denylist()
    # ALSO denylist the gen_dev_40 set (it was used for model selection -> must not reappear in a
    # fresh final holdout) and any EEC-extra parts we trained on.
    for extra in ("gen_dev_40.json", "benchmark_candidates.json"):
        if os.path.exists(extra):
            for c in json.load(open(extra)).get("candidates", []):
                deny_ids.add(c["part_id"]); deny_shas.add(c.get("sha256", ""))
    for prov in glob.glob(os.path.join("_eec_extra", "*", "*.provenance.json")):
        deny_ids.add(json.load(open(prov))["part_id"])

    # one file per part_id (first by name), then SHA
    by_id = {}
    for p in sorted(glob.glob(os.path.join(POOL, "*.stp"))):
        m = PID_RE.search(os.path.basename(p))
        if m and m.group(1) not in by_id:
            by_id[m.group(1)] = p

    pool = []
    seen_sha = set()
    for pid, p in by_id.items():
        if pid in deny_ids:
            continue
        s = sha256(p)
        if s in deny_shas or s in seen_sha:
            continue
        seen_sha.add(s)
        pool.append({"part_id": pid, "sha256": s, "step_file": p, "prefix": pid[:2],
                     "bytes": os.path.getsize(p)})

    # family-balanced spread across id prefixes
    random.seed(a.seed)
    buckets = {}
    for r in pool:
        buckets.setdefault(r["prefix"], []).append(r)
    for b in buckets.values():
        random.shuffle(b)
    selected, i = [], 0
    order = sorted(buckets)
    while len(selected) < min(a.n, len(pool)):
        prog = False
        for k in order:
            if buckets[k]:
                selected.append(buckets[k].pop()); prog = True
                if len(selected) >= a.n:
                    break
        if not prog:
            break

    # hard assertion: zero overlap with denylist
    assert all(r["part_id"] not in deny_ids and r["sha256"] not in deny_shas for r in selected)
    out = {"frozen": "2026-07-18", "purpose": "NEW untouched human-CP benchmark candidates (UNLABELLED)",
           "denylist": {"scheffler_ids": len(deny_ids), "scheffler_step_shas": len(deny_shas),
                        "TODO": "also denylist v28/v31 train IDs once their manifests are found"},
           "pool_unique_ids": len(by_id), "pool_after_denylist_and_sha_dedup": len(pool),
           "n_selected": len(selected), "overlap_with_denylist": 0,
           "note": "labels NOT included; human point+direction labelling + 3-rater adjudication is the next (human) step",
           "candidates": sorted(selected, key=lambda r: r["part_id"])}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"pool {len(by_id)} ids -> after denylist+dedup {len(pool)} -> selected {len(selected)}")
    print(f"denylist: {len(deny_ids)} ids / {len(deny_shas)} step-SHAs | overlap=0 (asserted)")
    print(f"families(prefixes) covered: {len(set(r['prefix'] for r in selected))} -> {a.out}")


if __name__ == "__main__":
    main()
