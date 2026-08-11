# -*- coding: utf-8 -*-
"""Apply the human adjudication of model-vs-human disagreements back into the training labels.

THE PROBLEM IT FIXES: train_seg_extra's masked BCE supervises the connection channel on EVERY vertex
-- positives (weight `partial_pos_weight`) where the annotator marked, NEGATIVE (weight 1) everywhere
else. So any real opening the annotator did not mark is actively taught as "not a connection". With
partial labelling that is unavoidable label noise, and it is the most plausible explanation for the
flat learning curve (going 77 -> 103 labelled parts did not improve anything).

mine_disagreements.py found the 91 places where the product emits a CP the annotator did not mark;
build_adjudication.py rendered them; the annotator answered one of:
    opening  -> a real opening they missed  => turn those vertices INTO positives (fixes a false negative)
    not_cp   -> a genuine distractor        => leave as a negative (already correct)
    unsure   -> cannot tell                 => write to <pid>.ignore.txt so the loss skips them entirely
                                               (a guess must not become a hard training signal)

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe apply_adjudication.py \
         --answers ~/Downloads/adjudication.json [--radius-mm 6.0] [--dry-run]
"""
import os, sys, glob, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import connector3d
from region_label_helper import load_obj

CE = int(connector3d.CABLE_ENTRY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="adjudication.json downloaded from the review page")
    ap.add_argument("--disagreements", default="results/disagreements.json")
    ap.add_argument("--radius-mm", type=float, default=6.0,
                    help="how far around the adjudicated point to mark (openings are ~5-8mm across)")
    ap.add_argument("--use-model-region", action="store_true",
                    help="adopt the MODEL's own segmented region for an adjudicated 'opening' instead "
                         "of a sphere of --radius-mm. More faithful: the verdict is 'the model was "
                         "right here', so its region is the label. A 6mm sphere marks ~357 verts per "
                         "opening where the annotator's own openings average ~212, i.e. it bleeds "
                         "onto surrounding housing.")
    ap.add_argument("--ckpts", nargs="+", default=[f"results/seg_extra/human77c_s{i}.pt" for i in (0, 1, 2)],
                    help="model whose regions are adopted for an adjudicated 'opening' -- must be the "
                         "SAME model that mined the disagreements, else the regions will not line up")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ans = json.load(open(os.path.expanduser(a.answers), encoding="utf-8"))
    models = None
    if a.use_model_region:
        import torch, diffusionnet as D, cp_openings
        from infer_step_cp import load_any
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        models = [load_any(c, dev=dev) for c in a.ckpts]
        print(f"  model bolgeleri kullanilacak (ensemble, {dev})", flush=True)
    dis = json.load(open(a.disagreements))
    by_part = {}
    for it in dis["items"]:
        by_part.setdefault(it["part_id"], []).append(it)

    n_pos = n_neg = n_ign = 0; touched = 0
    for pid, items in sorted(by_part.items()):
        picks = {i: ans.get(f"{pid}_{i}") for i in range(len(items))}
        if not any(picks.values()):
            continue
        d = os.path.join(items[0]["dir"], pid)
        of, lf = os.path.join(d, f"{pid}.obj"), os.path.join(d, f"{pid}.labels.txt")
        if not (os.path.exists(of) and os.path.exists(lf)):
            print(f"  {pid}: dosya yok, atlandi"); continue
        V, Fh = load_obj(of)
        L = np.array([int(x) for x in open(lf).read().split()], np.int64)
        if len(L) != len(V):
            print(f"  {pid}: vertex uyusmazligi, atlandi"); continue
        ig_path = os.path.join(d, f"{pid}.ignore.txt")
        ignore = np.zeros(len(V), bool)
        if os.path.exists(ig_path):
            old = np.array([int(x) for x in open(ig_path).read().split()], np.int64)
            ignore[old[(old >= 0) & (old < len(V))]] = True
        frags = None
        if models is not None:                    # the model's own regions on this part
            import torch, diffusionnet as D, cp_openings, connector3d as C3
            Vc = np.ascontiguousarray(V, np.float64); Fc = np.ascontiguousarray(Fh, np.int64)
            acc = None
            for model, meta, _ in models:
                _, pb = D.predict(model, meta, Vc, Fc, device="cuda" if __import__("torch").cuda.is_available() else "cpu",
                                  op_cache_dir="results/step_infer/ops", return_probs=True)
                pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
            probs = acc / len(models); plab = probs.argmax(-1)
            pc = probs[np.arange(len(plab)), plab]
            masked = plab.copy()
            masked[np.isin(plab, (CE, int(C3.CONTACT))) & (pc < 0.7)] = int(C3.HOUSING)
            frags = [(np.asarray(f.idx, int), V[np.asarray(f.idx, int)].mean(0))
                     for f in C3.build_fragments(Vc, Fc, masked, min_vertices=45)
                     if int(f.label) in (CE, int(C3.CONTACT))]
        changed = False
        for i, it in enumerate(items):
            pick = picks.get(i)
            if pick is None:
                continue
            p = np.asarray(it["point"], float)
            near = np.linalg.norm(V - p, axis=1) <= a.radius_mm
            if pick == "opening":
                sel = near
                if frags:                          # nearest model fragment to the adjudicated point
                    j = int(np.argmin([np.linalg.norm(c0 - p) for _, c0 in frags]))
                    if np.linalg.norm(frags[j][1] - p) <= a.radius_mm * 2:
                        sel = np.zeros(len(V), bool); sel[frags[j][0]] = True
                L[sel] = CE; ignore[sel] = False; n_pos += int(sel.sum()); changed = True
            elif pick == "unsure":
                ignore[near & (L != CE)] = True; n_ign += int((near & (L != CE)).sum()); changed = True
            else:
                n_neg += 1
        if changed and not a.dry_run:
            open(lf, "w").write("\n".join(map(str, L.tolist())))
            idx = np.flatnonzero(ignore)
            if len(idx):
                open(ig_path, "w").write("\n".join(map(str, idx.tolist())))
            elif os.path.exists(ig_path):
                os.remove(ig_path)
        touched += 1

    answered = sum(1 for v in ans.values() if v)
    print(f"\n{answered}/{dis['n_disagreements']} cevap islendi, {touched} parca guncellendi"
          + ("  [DRY RUN -- dosyalar yazilmadi]" if a.dry_run else ""))
    print(f"  POZITIFE cevrilen vertex : {n_pos}   (duzeltilen yanlis negatif)")
    print(f"  YOKSAY isaretlenen vertex: {n_ign}   (emin degilim -> kayiptan cikarildi)")
    print(f"  negatif birakilan bolge  : {n_neg}   (zaten dogruydu)")
    if not a.dry_run and (n_pos or n_ign):
        print("\n  Sonraki adim: ayni tarifle yeniden egit ve IKI olcumde de karsilastir")
        print("    (uretici hakemi + insan held-out). Kazanmazsa geri al -- git ile fark gorulebilir.")


if __name__ == "__main__":
    main()
