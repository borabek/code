# -*- coding: utf-8 -*-
"""Download the human-labelled EEC parts we DON'T already have (Harvard Dataverse D3ODGT, 234 total
vs our 102 -> +132). Same source, same 5-class scheme, same 3-rater-checked labelling -> real clean
extra training data at zero labelling cost. Saves _eec_extra/<pid>/{<pid>.obj, <pid>.labels.txt,
<pid>.provenance.json}. Verifies vertex==label count. NONE overlap our locked-11 test (new ids).

Usage: .venv/Scripts/python.exe download_eec_extra.py
"""
import json, os, glob, subprocess

DV = "_dv.json"; OUT = "_eec_extra"; BASE = "https://dataverse.harvard.edu/api/access/datafile/"


def dl(fid, path):
    # urllib gets 403 here (proxy); curl works -> use curl (follow redirects)
    subprocess.run(["curl", "-sL", "--max-time", "60", "-o", path, BASE + str(fid)],
                   check=True, capture_output=True)
    return open(path, "rb").read()


def main():
    d = json.load(open(DV)); files = d["data"]["latestVersion"]["files"]
    ours = set(os.path.basename(os.path.dirname(p)) for p in glob.glob("wscad_corpus_scheffler_exact/*/*/"))
    # index by (pid, ext)
    idx = {}
    for f in files:
        fn = f["dataFile"]["filename"]
        if "." not in fn:
            continue
        pid, ext = fn.rsplit(".", 1)
        idx[(pid, ext)] = {"id": f["dataFile"]["id"], "dir": f.get("directoryLabel", ""),
                           "md5": f["dataFile"].get("md5")}
    new_pids = sorted(set(p for (p, e) in idx if e == "obj" and p not in ours))
    os.makedirs(OUT, exist_ok=True)
    ok = 0; bad = []
    for pid in new_pids:
        if (pid, "obj") not in idx or (pid, "txt") not in idx:
            bad.append((pid, "missing obj/txt")); continue
        pdir = os.path.join(OUT, pid); os.makedirs(pdir, exist_ok=True)
        try:
            ob = dl(idx[(pid, "obj")]["id"], os.path.join(pdir, f"{pid}.obj"))
            lb = dl(idx[(pid, "txt")]["id"], os.path.join(pdir, f"{pid}.labels.txt"))
        except Exception as e:
            bad.append((pid, str(e)[:40])); continue
        nv = sum(1 for ln in ob.decode("utf-8", "ignore").splitlines() if ln.startswith("v "))
        nl = sum(1 for ln in lb.decode("utf-8", "ignore").splitlines() if ln.strip() != "")
        if nv != nl or nv == 0:
            bad.append((pid, f"vert{nv}!=label{nl}")); continue
        json.dump({"part_id": pid, "source": "Harvard Dataverse D3ODGT (EEC)", "split_in_source": idx[(pid, "obj")]["dir"],
                   "n_verts": nv, "labels_kind": "human 5-class (same scheme as scheffler corpus)",
                   "role": "extra_pretrain_train (NOT test; locked-11 stays the only test)"},
                  open(os.path.join(pdir, f"{pid}.provenance.json"), "w"), indent=1)
        ok += 1
        if ok % 25 == 0:
            print(f"  ...{ok} downloaded", flush=True)
    print(f"\nDOWNLOADED {ok} new human-labelled EEC parts -> {OUT}/  (bad {len(bad)})")
    if bad:
        print("  bad:", bad[:10])
    print("These are EXTRA training data (same 5-class scheme). Next: pretrain ten 102+extra, keep locked-11 test.")


if __name__ == "__main__":
    main()
