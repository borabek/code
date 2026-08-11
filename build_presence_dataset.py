# -*- coding: utf-8 -*-
"""G ADIMI 1/2: TESPIT veri seti -- hedefler AGIZDA, yuvada DEGIL.

C'NIN OLUMUNDEN OGRENILEN (2026-07-30, olculdu): `build_axis_dataset.py` hedefleri ureticinin
ConnectionPoint'ine 4mm yakin koselere koyuyor. Ama uretici CP'si bizim agiz noktamizdan eksende
**+7.79mm ICERIDE** (medyan; %91.7 tek yonlu) -- yani YUVADA. C o hedeflerle egitildi, sonra
AGIZDA sorgulandi ve kendi egitim parcalarinda bile %70.6 hata verdi. Ag yeteneksiz degildi;
yanlis yerde sorgulanmisti.

BU BETIK O HATAYI TEKRARLAMAZ: her uretici CP'si once `cp_geometry.seat_to_mouth` ile eksen
boyunca ILK YUZEYE (agza) tasinir, hedef koseler AGZIN cevresinde isaretlenir. Aday turetme de
yuzeydeki acikliklarda calistigi icin egitim ve kullanim AYNI YERDE olur.

Ek guvenlik: CP govdenin DISINDAYSA seat_to_mouth onu OYNATMAZ (parcaya gore degisir; olculdu
16/16 iceride vs 0/19 iceride). Offset ayrica kaydedilir ki dagilim denetlenebilsin.

Cikti: results/presence_dataset/<parca>.npz (idx, n_verts, offset_mm) + index.json
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BA_ALLOW_SEEN", "1")
OUT = "results/presence_dataset"
MESH_CACHE = "results/mesh_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target", type=int, default=6000)
    a = ap.parse_args()

    import thesis_remesh
    from cad_eval import align_frames
    from infer_step_cp import step_to_mesh
    from big_arbiter import eligible
    from cp_geometry import seat_to_mouth

    os.makedirs(OUT, exist_ok=True)
    done = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(OUT, "*.npz"))}
    parts = []
    for mfg, pid, jf, stp in eligible():
        try:
            n = len(json.load(open(jf, encoding="utf-8-sig")).get("ConnectionPoints") or [])
        except Exception:
            continue
        if n > 0:
            parts.append((mfg, pid, jf, stp, n))
    if a.limit:
        parts = parts[:a.limit]
    print(f"{len(parts)} parca | {len(done)} hazir | yaricap {a.radius}mm | hedefler AGIZDA",
          flush=True)

    index, nv, npart, offs = [], 0, 0, []
    for k, (mfg, pid, jf, stp, n) in enumerate(parts, 1):
        if pid in done:
            continue
        try:
            mc = os.path.join(MESH_CACHE, pid + ".npz")
            if os.path.exists(mc):
                m = np.load(mc)
                V = np.ascontiguousarray(m["V"], np.float64)
                F = np.ascontiguousarray(m["F"], np.int64)
                Vr, _ = step_to_mesh(stp)          # hizalama icin ham mesh sart
            else:
                Vr, Fr = step_to_mesh(stp)
                V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=a.target)
                V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            j = json.load(open(jf, encoding="utf-8-sig"))
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in j["ConnectionPoints"]], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in j["ConnectionPoints"]],
                          float)
            nn = np.linalg.norm(Gd, axis=1, keepdims=True)
            ok = nn[:, 0] > 1e-9
            G, Gd = G[ok], Gd[ok] / (nn[ok] + 1e-12)
            if not len(G):
                continue
            Vj = np.array([[q["X"], q["Y"], q["Z"]] for q in j["Graphic3d"]["Points"]], float)
            R, t, _ = align_frames(Vr, Vj)
            Gm = (G - t) @ R
            Gdm = Gd @ R

            # ASIL FARK: yuva -> AGIZ
            M, off = [], []
            for i in range(len(Gm)):
                mp, o = seat_to_mouth((V, F), Gm[i], Gdm[i], max_mm=45.0)
                M.append(mp); off.append(float(o))
            M = np.asarray(M, float)
            offs += off

            d = np.linalg.norm(V[:, None, :] - M[None, :, :], axis=-1)
            near = d.min(1) <= a.radius
            if int(near.sum()) < 8:
                continue
            idx = np.where(near)[0].astype(np.int32)
            np.savez(os.path.join(OUT, f"{pid}.npz"), idx=idx, n_verts=np.int32(len(V)),
                     offset_mm=np.asarray(off, np.float32), mfg=np.str_(mfg), stp=np.str_(stp))
            index.append({"part": pid, "mfg": mfg, "stp": stp,
                          "n_target_verts": int(len(idx)), "n_cp": int(len(M)),
                          "n_verts": int(len(V))})
            nv += len(idx); npart += 1
        except Exception:
            continue
        if k % 100 == 0:
            print(f"  {k}/{len(parts)}  hazir {npart} / {nv} hedef kose", flush=True)

    old = []
    ip = os.path.join(OUT, "index.json")
    if os.path.exists(ip):
        try:
            old = json.load(open(ip))["parts"]
        except Exception:
            old = []
    allp = {p["part"]: p for p in old}
    allp.update({p["part"]: p for p in index})
    json.dump({"radius_mm": a.radius, "remesh_target": a.target, "targets_at": "MOUTH",
               "n_parts": len(allp), "parts": list(allp.values())}, open(ip, "w"), indent=1)
    if offs:
        o = np.abs(np.asarray(offs))
        print(f"\nyuva->agiz kaydirmasi: medyan {np.median(o):.2f}mm  %90 {np.percentile(o,90):.2f}mm"
              f"  hic oynamayan %{100*(o<1e-6).mean():.0f}")
    print(f"bitti: {npart} yeni parca, {nv} hedef kose | toplam {len(allp)}")


if __name__ == "__main__":
    main()
