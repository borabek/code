# -*- coding: utf-8 -*-
"""Batch 2 labelling package -- CATALOG-SOURCED rebuild (2026-07-20).

FIRST ATTEMPT WAS WRONG: selecting by "connection-gap" over the raw all_wscad_stp pool rewarded
ACCESSORIES (end-plates / partition plates), because those genuinely have no connections so the
model rightly finds none -> they scored highest gap. EPLAN confirmed 1050100000 is a WAP end-plate.
Geometry could not reliably filter them out (surface-complexity flagged real batch-1 terminals as
accessories -- remeshing seals the wire tunnels).

FIX: source from the CURATED terminal-block catalog (terminal_block_candidates_*.json, 1766 Phoenix/
Wago parts that a WSCAD query already classified as terminal blocks, each with typeName + English
description + subcategory). 1560 of them are in all_wscad_stp and not already used. Every candidate
is a REAL terminal with connection points -> the accessory trap is structurally impossible. Diverse
selection across subcategory + series; each part's TYPE is written into provenance and the README so
the annotator knows exactly what they are labelling.

Thesis-faithful: these ARE WSCAD terminal blocks (the product's target); nothing about the method
changes (DiffusionNet 5-class segmentation -> connection openings, human region labels).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe prepare_labeling_2.py --keep 40
"""
import os, sys, glob, json, argparse
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D
import connector3d, thesis_remesh
from infer_step_cp import step_to_mesh, load_any

OUT = "_label_targets_2"; OP = "results/step_infer/ops"
CKPT = "results/seg_extra/human77c_s0.pt"
CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
CLASSES = "0=Housing 1=Contact 2=SnapPoint 3=CableEntry 4=LabelSurface"
SUBCAT = {1: "feed-through / multi-level terminal", 2: "potential distributor",
          3: "knife-disconnect terminal", 4: "collective terminal"}


def save_obj(path, V, F):
    with open(path, "w") as f:
        for v in V: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F: f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def neutral_render(V, F, out_png):
    """Bias-safe reference: Lambert-shaded grey, NO class colours."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        light = np.array([0.4, 0.3, 1.0]); light = light / np.linalg.norm(light)
        tri = V[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        grey = 0.30 + 0.60 * np.abs(n @ light)
        fc = np.clip(np.stack([grey] * 3, 1), 0, 1)
        ext = V.max(0) - V.min(0); ctr = V.mean(0); rr = ext.max() * 0.55
        fig = plt.figure(figsize=(13, 4.2)); bg = (0.93, 0.94, 0.96)
        for i, (ev, az) in enumerate([(18, -60), (18, 30), (90, -90)]):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors=(0, 0, 0, 0.06), linewidths=0.1))
            ax.set_xlim(ctr[0]-rr, ctr[0]+rr); ax.set_ylim(ctr[1]-rr, ctr[1]+rr); ax.set_zlim(ctr[2]-rr, ctr[2]+rr)
            try: ax.set_box_aspect(ext)
            except Exception: pass
            ax.view_init(elev=ev, azim=az); ax.set_axis_off(); ax.set_facecolor(bg)
        fig.patch.set_facecolor(bg)
        plt.tight_layout(); plt.savefig(out_png, dpi=95, bbox_inches="tight", facecolor=bg); plt.close(fig)
    except Exception as e:
        print(f"    (render skipped: {str(e)[:40]})")


def load_catalog():
    parts = {}
    for f in glob.glob("terminal_block_candidates_*.json"):
        for p in json.load(open(f)).get("parts", []):
            parts[str(p["partNumber"])] = p
    return parts


def load_excludes(out_dir):
    excl = set()
    for f, key in [("gen_dev_40.json", "candidates"), ("final_holdout_40.json", "candidates"),
                   ("benchmark_candidates.json", "candidates")]:
        try: excl |= {c["part_id"] for c in json.load(open(f))[key]}
        except Exception: pass
    # exclude every existing label batch EXCEPT the output dir we are (re)building
    out_norm = os.path.normpath(out_dir)
    for d in glob.glob("_label_targets*/*/") + glob.glob("_pseudo_extra/*/"):
        part = os.path.normpath(d)                 # e.g. _label_targets_2/3281122
        batch = os.path.dirname(part)              # e.g. _label_targets_2
        if os.path.normpath(batch) == out_norm:
            continue
        excl |= {os.path.basename(part)}
    # explicit user-rejected out-of-scope parts (removed from a batch but must never re-appear)
    if os.path.exists("label_out_of_scope.txt"):
        for ln in open("label_out_of_scope.txt"):
            ln = ln.split("#")[0].strip()
            if ln: excl.add(ln)
    try:
        import scheffler_dataset as ds
        for sp in ("train", "val"):
            excl |= {s["part_id"] for s in ds.load_split("wscad_corpus_scheffler_exact", sp, verify_hashes=False)}
    except Exception as e:
        print("split exclude failed:", e)
    return excl


def series_of(parts, pn):
    return parts[pn].get("series") or (parts[pn].get("typeName") or "")[:4]


def prior_series():
    """Series (type-name first token) already labelled in earlier batches -> deprioritise them so a
    new batch covers NOVEL/rare families (better generalisation + more F1 headroom)."""
    seen = set()
    for pf in glob.glob("_label_targets*/*/*.provenance.json"):
        try:
            tn = json.load(open(pf)).get("type_name") or ""
            if tn: seen.add(tn.split()[0])
        except Exception:
            pass
    return seen


def diverse_select(free, parts, keep, per_series, novel_first=False):
    """Round-robin across subcategory, capping parts per series. novel_first pushes series NOT seen in
    earlier batches to the front so the batch maximises rare-family coverage."""
    prior = prior_series() if novel_first else set()
    by_sub = defaultdict(list)
    for pn in free:
        by_sub[parts[pn].get("subcategoryId")].append(pn)
    for sid in by_sub:   # novel series first, then by series/type for a clean spread
        by_sub[sid].sort(key=lambda pn: (series_of(parts, pn) in prior, series_of(parts, pn),
                                          parts[pn].get("typeName") or ""))
    order = sorted(by_sub, key=lambda s: -len(by_sub[s]))   # bigger subcategories first
    picked, series_count = [], defaultdict(int)
    idx = {s: 0 for s in order}
    while len(picked) < keep and any(idx[s] < len(by_sub[s]) for s in order):
        for s in order:
            while idx[s] < len(by_sub[s]):
                pn = by_sub[s][idx[s]]; idx[s] += 1
                if series_count[series_of(parts, pn)] < per_series:
                    series_count[series_of(parts, pn)] += 1; picked.append(pn); break
            if len(picked) >= keep: break
    n_novel = sum(1 for pn in picked if series_of(parts, pn) not in prior)
    if novel_first:
        print(f"  diversity: {n_novel}/{len(picked)} from series NOT in earlier batches", flush=True)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_label_targets_2", help="output batch dir (e.g. _label_targets_3)")
    ap.add_argument("--keep", type=int, default=40)
    ap.add_argument("--per-series", type=int, default=2)
    ap.add_argument("--exclude-type", nargs="*", default=[],
                    help="drop catalog parts whose typeName FIRST TOKEN is in this list "
                         "(e.g. BT BTO -- the screw-only-no-cable Phoenix series the user ruled out of CP scope)")
    ap.add_argument("--novel-first", action="store_true",
                    help="prefer series NOT in earlier batches (max rare-family coverage, more F1 headroom)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    global OUT; OUT = a.out

    parts = load_catalog()
    pool = {os.path.basename(s).split("_")[1]: s for s in glob.glob("all_wscad_stp/*.stp")}
    excl = load_excludes(OUT)
    xtype = set(a.exclude_type)
    def type_ok(pn):
        tn = (parts[pn].get("typeName") or "").split()
        return not (tn and tn[0] in xtype)
    free = [pn for pn in parts if pn in pool and pn not in excl and type_ok(pn)]
    print(f"catalog {len(parts)} | in-pool & unused {len(free)} | excluding {len(excl)} used"
          + (f" + typeName {sorted(xtype)}" if xtype else ""))
    picked = diverse_select(free, parts, a.keep, a.per_series, novel_first=a.novel_first)
    print(f"selected {len(picked)} diverse cataloged terminals")

    # rebuild OUT from scratch (old Weidmuller accessory-contaminated batch is replaced)
    if os.path.isdir(OUT):
        import shutil; shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    model, meta, _ = load_any(CKPT, dev=a.device)
    man = []
    for pn in picked:
        cat = parts[pn]; stp = pool[pn]
        try:
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            lab = np.asarray(D.predict(model, meta, V, F, device=a.device, op_cache_dir=OP))
        except Exception as e:
            print(f"  {pn}: ERR {str(e)[:50]}"); continue
        nconn = int(np.isin(lab, (CE, CT)).sum())
        d = os.path.join(OUT, pn); os.makedirs(d, exist_ok=True)
        save_obj(os.path.join(d, f"{pn}.obj"), V, F)
        np.savetxt(os.path.join(d, f"{pn}.labels.template.txt"), np.zeros(len(V), int), fmt="%d")
        neutral_render(V, F, os.path.join(d, f"{pn}.NEUTRAL.png"))
        prov = {"part_id": pn, "manufacturer": cat.get("manufacturer"), "type_name": cat.get("typeName"),
                "description": cat.get("searchEn"), "subcategory": SUBCAT.get(cat.get("subcategoryId")),
                "source_step": os.path.basename(stp), "n_verts": len(V), "model_n_conn": nconn,
                "classes": CLASSES, "frame": "STEP frame (remesh is identity)"}
        json.dump(prov, open(os.path.join(d, f"{pn}.provenance.json"), "w"), indent=1)
        man.append({"part_id": pn, "type_name": cat.get("typeName"), "subcategory": cat.get("subcategoryId"),
                    "manufacturer": cat.get("manufacturer"), "model_n_conn": nconn})
        print(f"  built {pn}  {cat.get('manufacturer')} {cat.get('typeName')}  ({len(V)}v, model_conn {nconn})", flush=True)

    bnum = OUT.rsplit("_", 1)[-1] if OUT.rsplit("_", 1)[-1].isdigit() else "2"
    json.dump({"batch": int(bnum), "source": "curated terminal_block_candidates catalog (real terminals, typed)",
               "excluded_typeName_first_token": sorted(xtype), "n": len(man), "parts": man},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    write_readme(len(man), bnum)
    print(f"\n{len(man)} REAL cataloged terminals -> {OUT}/  (+ README.md, manifest.json)")


def write_readme(n, bnum="2"):
    readme = f"""# Labelling package -- BATCH {bnum} ({n} parts)  [catalog-sourced, corrected brief]

These are {n} UNSEEN, CATALOGED terminal blocks (Phoenix Contact / Wago), each with its real type in
`<pid>.provenance.json` (e.g. "STTB 4 BU -- feed-through terminal"). Unlike the first attempt, every
part here is a genuine terminal with connection points -- no end-plates/accessories. None overlap
batch 1, train, val, or any held-out set.

## THE RULE (unchanged from the corrected brief)
> **Paint EVERY place a wire/conductor enters the housing as CableEntry (3).**
> Round hole, oval slot, square push-in clamp, screw clamp, cage clamp -- all class 3.
> Do NOT try to tell CableEntry from Contact; the model learns that itself, training supervises the
> union. Everything else stays Housing (0). Only classes 0 and 3 matter.

Mark the whole opening region (the recess/mouth), a patch a few rings wide -- not a single point.

## Task
{CLASSES}   (in practice only 0 and 3 matter)
Each part: <pid>/<pid>.obj + <pid>.labels.template.txt (all 0 -> edit) + <pid>.NEUTRAL.png (bias-free
grey render, no model guess) + <pid>.provenance.json (its catalog type -- read it to know the part).

## Workflow (`label_tool.html` in the repo root)
1. Double-click `label_tool.html` -> browser. "Load OBJ" -> pick `<pid>/<pid>.obj`.
2. Rotate to find every wire entry. Click the opening region + press **3**. Z = undo.
3. "Download labels" -> save as `<pid>/<pid>.labels.txt` (one int per line, len == #vertices).
4. Tell me when a few are done -- I run training + the arbiter automatically and report the honest
   number (target: beat the product CP F1 0.520 across seeds, not one lucky seed).

Every part is a real terminal, so every part has openings to mark -- if one truly looks featureless,
say so and I'll check the catalog entry.
"""
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)


if __name__ == "__main__":
    main()
