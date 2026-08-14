# -*- coding: utf-8 -*-
"""Single MASTER manifest of the WSCAD STEP inventory (audit P1/data): every part in
all_wscad_stp with part_id, file, bytes, SHA-256, and flags (scheffler_denylist / benchmark_selected).
This is the one place to look up any WSCAD STEP part and its provenance; no more manifest-less files.

Output: wscad_master_manifest.json
"""
import glob, os, json, hashlib, re

POOL = "all_wscad_stp"; CORPUS = "wscad_corpus_scheffler_exact"
PID_RE = re.compile(r"wscaduniverse_([0-9A-Za-z\-]+)_")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    deny_ids = set()
    for prov in glob.glob(os.path.join(CORPUS, "*", "*", "*.provenance.json")):
        deny_ids.add(json.load(open(prov))["part_id"])
    selected = set()
    if os.path.exists("benchmark_candidates.json"):
        selected = {c["part_id"] for c in json.load(open("benchmark_candidates.json"))["candidates"]}

    rows = []; by_id = {}; seen_sha = {}
    for p in sorted(glob.glob(os.path.join(POOL, "*.stp"))):
        m = PID_RE.search(os.path.basename(p))
        if not m or m.group(1) in by_id:
            continue
        pid = m.group(1); s = sha256(p)
        by_id[pid] = p
        rows.append({"part_id": pid, "file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": s,
                     "duplicate_of": seen_sha.get(s), "scheffler_denylist": pid in deny_ids,
                     "benchmark_selected": pid in selected})
        seen_sha.setdefault(s, pid)
    dupes = sum(1 for r in rows if r["duplicate_of"])
    out = {"frozen": "2026-07-18", "pool": POOL, "n_unique_ids": len(rows),
           "n_sha_duplicates": dupes, "n_scheffler_denylist": sum(1 for r in rows if r["scheffler_denylist"]),
           "n_benchmark_selected": sum(1 for r in rows if r["benchmark_selected"]),
           "parts": rows}
    json.dump(out, open("wscad_master_manifest.json", "w"), indent=1)
    print(f"{len(rows)} unique ids | {dupes} sha-duplicates | denylist {out['n_scheffler_denylist']} | "
          f"benchmark {out['n_benchmark_selected']} -> wscad_master_manifest.json")


if __name__ == "__main__":
    main()
