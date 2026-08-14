# -*- coding: utf-8 -*-
"""TEZ-ONERILI 3D BOUNDING-BOX TURETICI (Masterarbeit satir 2099 Ausblick).

Tez, 2D YOLOv6'yi Kontaktierung/Kabeleinfuehrung icin ACIKCA REDDEDIYOR (ortulme; "nicht ueber mehr
gelabelte Daten loesbar") ve YERINE sunu oneriyor:
  "ein 3D-basiertes YOLO-Netzwerk zu erproben ... auf Punktwolken ... Ein teilautomatisiertes Erstellen
   der 3D-Bounding-Box Labels ausgehend von den Scheitelpunkt-Labels ist ebenso denkbar."
Bu script tam o adimi yapar: MEVCUT vertex etiketlerinden (korpus 5-sinif + insan kismi etiketleri)
baglanti-sinifi (CableEntry|Contact) INSTANCE'larini mesh-bagliligiyla ayirir ve her instance icin
3D kutu (merkez, boyut, ana eksen, verteks sayisi) turetir. Ek etiketleme maliyeti YOK.

Cikti: results/boxes3d/<pid>.json + ozet istatistik (instance/parca, kutu boyut dagilimi) -- bir 3D
tespit agi icin anchor tasarimina ve egitim etiketlerine temel.
Neden bizim icin dogru hedef: bugunku zayifligimiz INSTANCE AYRIMI (fragment/kumeleme); tespit agi
N ayri CP'yi dogrudan tahmin eder, segmentasyon+clustering zincirini atlar.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import connector3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
OUT = "results/boxes3d"
MIN_V = 8          # bir instance sayilmak icin en az verteks (kucuk gurultu fragmentlerini eler)


def components(V, F, mask):
    """mask=True vertekslerin mesh-baglantili bilesenleri (face kenarlari uzerinden)."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    remap = -np.ones(len(V), np.int64); remap[idx] = np.arange(len(idx))
    e = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ea, eb = F[:, a], F[:, b]
        keep = mask[ea] & mask[eb]
        if keep.any():
            e.append(np.stack([remap[ea[keep]], remap[eb[keep]]], 1))
    if not e:
        return [np.array([i]) for i in idx]      # izole verteksler
    E = np.concatenate(e, 0)
    n = len(idx)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    return [idx[lab == c] for c in range(ncomp)]


def box_of(V, vids):
    P = V[vids]
    lo, hi = P.min(0), P.max(0)
    ctr = 0.5 * (lo + hi); ext = hi - lo
    # ana eksen (PCA) -- acikligin yonelimi hakkinda ipucu
    Q = P - P.mean(0)
    try:
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        axis = Vt[-1]                      # en kucuk varyans yonu ~ yuzey normali
    except Exception:
        axis = np.array([0.0, 0.0, 1.0])
    return {"center": [round(float(x), 4) for x in ctr],
            "extent": [round(float(x), 4) for x in ext],
            "axis": [round(float(x), 4) for x in axis],
            "n_verts": int(len(vids))}


def main():
    import train_seg_extra as T
    import scheffler_dataset as ds
    os.makedirs(OUT, exist_ok=True)
    samples = []
    for sp in ("train", "val"):
        for s in ds.load_split("wscad_corpus_scheffler_exact", sp, verify_hashes=False):
            samples.append(("corpus/" + sp, s))
    for d in ("_label_targets", "_label_targets_2", "_label_targets_3",
              "_label_targets_recall", "_label_targets_recall_hard", "_label_targets_recall_r3"):
        if os.path.isdir(d):
            for s in T.load_extra(d):
                samples.append((d, s))
    print(f"{len(samples)} etiketli parca (korpus + insan kismi)", flush=True)

    per_part = []; all_ext = []; n_written = 0
    for src, s in samples:
        try:
            V = np.asarray(s["verts"] if "verts" in s else s["V"], float)
            F = np.asarray(s["faces"] if "faces" in s else s["F"], np.int64)
            L = np.asarray(s["labels"] if "labels" in s else s["L"], np.int64)
            pid = s.get("part_id") or s.get("pid") or "?"
            if len(L) != len(V):
                continue
            mask = np.isin(L, (CE, CT))
            if not mask.any():
                per_part.append({"pid": pid, "src": src, "n_boxes": 0}); continue
            boxes = [box_of(V, c) for c in components(V, F, mask) if len(c) >= MIN_V]
            if boxes:
                json.dump({"part_id": pid, "source": src, "boxes": boxes},
                          open(os.path.join(OUT, f"{pid}.json"), "w"), indent=1)
                n_written += 1
                all_ext += [b["extent"] for b in boxes]
            per_part.append({"pid": pid, "src": src, "n_boxes": len(boxes)})
        except Exception:
            continue

    nb = np.array([p["n_boxes"] for p in per_part])
    E = np.array(all_ext, float) if all_ext else np.zeros((0, 3))
    print(f"\n=== 3D KUTU TURETIMI ===")
    print(f"  parca: {len(per_part)} | kutu yazilan parca: {n_written} | toplam instance: {int(nb.sum())}")
    print(f"  instance/parca: medyan {np.median(nb):.0f} | ort {nb.mean():.1f} | max {nb.max() if len(nb) else 0}")
    print(f"  0 instance'li parca: {(nb == 0).sum()}")
    if len(E):
        d = np.sort(E, axis=1)          # her kutunun kenarlari kucukten buyuge
        print(f"  kutu kenarlari (mm): en kisa medyan {np.median(d[:,0]):.1f} | orta {np.median(d[:,1]):.1f} | en uzun {np.median(d[:,2]):.1f}")
        print(f"  -> anchor tasarimi icin tipik CP kutusu ~ {np.median(d[:,0]):.1f} x {np.median(d[:,1]):.1f} x {np.median(d[:,2]):.1f} mm")
    json.dump(per_part, open("results/boxes3d_summary.json", "w"), indent=1)
    print(f"  -> {OUT}/  + results/boxes3d_summary.json")


if __name__ == "__main__":
    main()
