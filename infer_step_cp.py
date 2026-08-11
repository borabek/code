# -*- coding: utf-8 -*-
"""Thesis inference pipeline (§5.3.7) end-to-end on a RAW WSCAD STEP -> CP. This is the
"working product" path: does the 91-part model generalise to NEVER-SEEN parts?
  STEP --gmsh--> triangle mesh --thesis remesh--> ~6000 uniform verts --DiffusionNet--> 5-class
  segmentation --cp_openings--> CableEntry CP points + axis direction --> render for eyeball QA.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe infer_step_cp.py <part_id ...>
       (part_ids from benchmark_candidates.json; these are denylisted = truly unseen)
"""
import sys, os, json
import numpy as np
import gmsh
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import diffusionnet, connector3d, cp_openings, thesis_remesh

# PRODUCT MODEL (frozen 2026-07-20): self-trained (71 human + 122 confident pseudo-labels).
# Thesis-endorsed direction -- the thesis conclusion explicitly recommends feeding model predictions
# back as a labelling feedback loop. val human-GT: Connection IoU 0.676 / CableEntry 0.705 / mIoU
# 0.664 / acc 0.844 (thesis reference: CableEntry 0.472, mean Jaccard 0.514). Previous product
# refit91.pt is kept for provenance (Connection 0.672, CableEntry 0.649).
# PRODUCT 2026-07-22 = a SINGLE seed. The 3-seed ensemble was dropped after the 1811-CP manufacturer
# arbiter (242 PXC + 182 WEI parts, identical sets per variant) reversed the small measurements that
# had adopted it:  single .615/.229 -> COMBINED F1 .480   vs   ensemble .620/.157 -> .467.
# Averaging seeds is conservative: it flatters precision on a precision-limited test set (the 18-CP
# arbiter, the 82-CP human held-out) and costs RECALL, which is what collapses on an unseen
# manufacturer -- Weidmueller drops .229 -> .157 under ensembling.
CKPT = ["results/seg_extra/recall_hard_s2.pt"]
OP = "results/step_infer/ops"
CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
SEM = {int(connector3d.HOUSING): (0.72,0.72,0.72), CT: (0.15,0.45,0.95),
       int(connector3d.SNAP_POINT): (0.95,0.85,0.15), CE: (0.1,0.8,0.25), int(connector3d.LABEL_SURFACE): (0.95,0.6,0.15)}
# cp-v3 (2026-07-20): a CP can come from EITHER class (arbiter-selected -- see cp_openings.py
# module docstring). Distinguish them visually: CableEntry = green circle, Contact-as-CP = orange
# diamond (depth-gated: only real recessed connection openings, not flat pads).
CP_MARK = {CE: dict(c="limegreen", marker="o"), CT: dict(c="darkorange", marker="D")}


def step_to_mesh(path):
    """STEP -> ucgen mesh (gmsh ile yuzey mesh'i).

    KRITIK -- try/finally SART (2026-07-28'de bir gecelik is kaybettirdi):
    `gmsh.model.mesh.generate(2)` bazi STEP'lerde istisna atar (olculdu: WEI.2502750000 ->
    "Impossible to mesh periodic surface 65"). Eskiden `finalize()` fonksiyonun SONUNDAYDI, yani
    istisna halinde HIC CALISMIYORDU: gmsh baslatilmis ve kirli halde kaliyor, sonraki her parca
    birikmis durumun uzerine yukleniyordu. Sonuc: 220 parca normal hizda islendi, ilk hatadan sonra
    is surunmeye basladi ve 3 saat boyunca TEK SATIR ilerlemedi -- hata mesaji da yoktu, sadece
    "calisiyor" gibi gorundu. finalize() artik finally'de: hatali parca temiz sekilde atlanir.
    """
    # OCC-ONCE ROTASI (2026-08-05) -- ISTEGE BAGLI, VARSAYILAN KAPALI.
    # OLCULDU: SE.NTSXTB18000H (30 MB STEP) gmsh'te 25+ DAKIKA sonra hala bitmemisti;
    # OCC ayni parcayi 15.3s'de ucgenledi (9150v -> tez remesh 8889v). Yani cok buyuk
    # STEP'lerde gmsh'in 2B parametrik mesh'leyicisi YANLIS ARAC.
    #
    # NEDEN GENEL BIR BOYUT ESIGI KOYMUYORUZ: dondurulmus kumelerde de buyuk parcalar
    # var (194'lukte 2 adet >10MB, sinav kumesinde 7 adet, max 34.8 MB). Global bir esik
    # o parcalarin mesh KAYNAGINI degistirir ve mevcut olcumleri gecersiz kilar.
    # Bu yuzden rota YALNIZ acikca istendiginde (kurtarma turu) devreye girer; diger
    # tum kosular BIREBIR ayni davranir.
    if os.environ.get("MESH_OCC_ONCE") == "1":
        try:
            return _occ_mesh(path)
        except Exception:
            pass          # OCC de duserse asagidaki normal gmsh zinciri denenir
    try:
        return _gmsh_mesh(path, heal=False)
    except Exception:
        # OCC ONARIM YEDEGI (2026-07-28): bazi STEP'ler "Impossible to mesh periodic surface" ile
        # duser -- olculdu: 1650 parcanin 20'si (%1.2), yani sessiz korpus kaybi. OCC healing
        # (degenerate/kucuk kenar-yuz duzeltme + dikis + gevsek tolerans) UCUNU DE kurtardi
        # (2502750000, 1704350000, 1704360000 -> 8673/2787/2787 vertex).
        # SADECE YEDEK olarak: healing tum parcalara uygulanirsa calisan %98.8'in mesh'i de degisir
        # ve tum onbellekler/olcumler gecersiz olur. Normal yol aynen korunur.
        try:
            return _gmsh_mesh(path, heal=True)
        except Exception:
            # UCUNCU KADEME (2026-08-04): DataSet5 turetmesinde iki parca IKI kademeyi de gecemedi
            # ("1D mesh not forming a closed loop", "Could not fix wire in surface 265").
            # 7 ayar x 2 parca olculdu; kurtaran TEK ayar GEVSEK TOLERANS cikti (1e-3 -> 1e-2):
            # 5D.202.0055.6 -> 4321 tepe / 8638 yuz, HIC yuzey atmadan, tam mesh.
            # Yalniz ikinci kademe de DUSTUGUNDE calisir, yani calisan parcalarin mesh'i
            # ve tum onbellekler AYNEN korunur.
            try:
                return _gmsh_mesh(path, heal=True, tol=1e-2)
            except Exception:
                # DORDUNCU KADEME (2026-08-05, DataSet 6): uc gmsh kademesi de duseni
                # OCC'nin KENDI tessellatoru ile al. Butun gmsh kademeleri ayni yerde
                # kiriliyor -- OCC->gmsh ithalinde TEL/ILMEK topolojisi ("Could not fix
                # wire in surface 139", "1D mesh not forming a closed loop"). gmsh yuzeyi
                # 2B parametre uzayinda mesh'lemek icin KAPALI tel ister; BRepMesh ise
                # yuzeyi dogrudan ucgenler ve bozuk tele TAHAMMUL eder.
                #
                # TEZ ACISINDAN NOTR: bu yalnizca ILK tessellasyon kaynagi. Hemen ardindan
                # `thesis_remesh.remesh_uniform(V, F, target=6000)` calisir, yani parca
                # yine tezin ~6000 tepeli uniform izotropik mesh'i olur. Degisen tek sey,
                # aksi halde TAMAMEN KAYBEDILECEK parcanin kurtarilmasi.
                return _occ_mesh(path)


OCC_SAPMA = 0.25     # mm, lineer tessellasyon sapmasi (remesh zaten 3mm hedefinde)
OCC_ACI = 0.5        # rad, acisal sapma


def _occ_mesh(path):
    """DORDUNCU KADEME: OCC BRepMesh ile dogrudan tessellasyon (gmsh'siz).

    `_gmsh_mesh` ile AYNI tamlik disiplinine tabidir: kac yuzeyin ucgeni cikti sayilir,
    %95 altinda kalirsa YARIM MESH sayilip atilir. Yarim kabugu korpusa sokmak, parcayi
    durustce kaybetmekten KOTUDUR (bkz. `_tamlik_kontrol`: DST2.5_GY'de 202 yuzeyin 92'si).
    """
    from OCP.STEPControl import STEPControl_Reader
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    rd = STEPControl_Reader()
    if rd.ReadFile(path) != 1:
        raise RuntimeError(f"OCC STEP okunamadi: {path}")
    rd.TransferRoots()
    shape = rd.OneShape()
    BRepMesh_IncrementalMesh(shape, OCC_SAPMA, False, OCC_ACI, True)

    Vs, Fs, off, nyuzey, mesli = [], [], 0, 0, 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current()); exp.Next()
        nyuzey += 1
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        nn, nt = tri.NbNodes(), tri.NbTriangles()
        if nn < 3 or nt < 1:
            continue
        mesli += 1
        for i in range(1, nn + 1):
            p = tri.Node(i).Transformed(trsf)
            Vs.append((p.X(), p.Y(), p.Z()))
        # TERS YUZ (reversed face): OCC ucgen sirasini yuzun yonelimine gore verir.
        ters = f.Orientation() == 1          # TopAbs_REVERSED
        for i in range(1, nt + 1):
            a, b, c = tri.Triangle(i).Get()
            Fs.append((off + c - 1, off + b - 1, off + a - 1) if ters
                      else (off + a - 1, off + b - 1, off + c - 1))
        off += nn
    if nyuzey and mesli < MESH_TAMLIK * nyuzey:
        raise RuntimeError(f"YARIM MESH (OCC): {nyuzey} yuzeyden {mesli} ucgenlendi "
                           f"(%{100*mesli/nyuzey:.0f} < %{100*MESH_TAMLIK:.0f})")
    if not Fs:
        raise RuntimeError("OCC tessellasyonu bos")
    V = np.asarray(Vs, float)
    F = np.asarray(Fs, int)
    # Yuz yuz uretilen tepeler DIKISSIZ gelir: ayni fiziksel kose her yuzde ayri tepe.
    # thesis_remesh su gecirmez bir yuzey bekledigi icin kaynaklanir.
    V, F = _kaynakla(V, F)
    return V, F


def _kaynakla(V, F, tol=1e-6):
    """Ayni konumdaki tepeleri birlestir (yuz-yuz tessellasyonun dikisini kapat)."""
    q = np.round(V / max(tol, 1e-9)).astype(np.int64)
    _, ilk, ters = np.unique(q, axis=0, return_index=True, return_inverse=True)
    V2 = V[ilk]
    F2 = ters[F]
    F2 = F2[(F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])]
    # SIFIR ALANLI ucgen: kaynak sonrasi kose indisleri FARKLI kalir ama noktalar
    # dogrusal olabilir. Bunlar normal hesabinda 0'a bolunme -> NaN normal uretir
    # (diffusion_net/geometry.py:109 "invalid value encountered in divide").
    # Yalniz OCC yolunda temizlenir; gmsh ciktilari ve onbellekleri DEGISMEZ.
    if len(F2):
        e1 = V2[F2[:, 1]] - V2[F2[:, 0]]
        e2 = V2[F2[:, 2]] - V2[F2[:, 0]]
        alan2 = np.linalg.norm(np.cross(e1, e2), axis=1)
        F2 = F2[alan2 > 1e-12]
    return np.ascontiguousarray(V2, float), np.ascontiguousarray(F2, int)


MESH_TAMLIK = 0.95   # asagida OLCULDU; calisan parcalarda oran 1.00


def _tamlik_kontrol(nyuzey):
    """YARIM MESH'I KORPUSA SOKMA.

    NEDEN VAR (2026-08-04): ucuncu kademeyi yazarken "gmsh hata verse bile geriye kismi
    mesh birakiyor mu, onu kullansak mi" diye olctum. Birakiyor -- ama DST2.5_GY'de
    **202 yuzeyin yalnizca 92'si** mesh'lenmisti (%46). O mesh korpusa girseydi parca
    yarim kabuk halinde egitime ve olcume katilirdi: sahte agizlar, sahte CP'ler.
    Parcayi DURUSTCE kaybetmek, bozugunu sessizce eklemekten iyidir.

    Ayrica gmsh bazen istisna ATMADAN da yuzey atlayabilir (yalniz uyari verir); bu
    kontrol o sessiz durumu da yakalar.
    """
    mesli = sum(1 for _, t in gmsh.model.getEntities(2)
                if len(gmsh.model.mesh.getElements(2, t)[1]))
    if nyuzey and mesli < MESH_TAMLIK * nyuzey:
        raise RuntimeError(f"YARIM MESH: {nyuzey} yuzeyden {mesli} mesh'lendi "
                           f"(%{100*mesli/nyuzey:.0f} < %{100*MESH_TAMLIK:.0f})")


def _gmsh_mesh(path, heal=False, tol=1e-3):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0)
        if heal:
            for k, v in (("Geometry.OCCFixDegenerated", 1), ("Geometry.OCCFixSmallEdges", 1),
                         ("Geometry.OCCFixSmallFaces", 1), ("Geometry.OCCSewFaces", 1),
                         ("Geometry.Tolerance", tol)):
                gmsh.option.setNumber(k, v)
        gmsh.open(path)
        nyuzey = len(gmsh.model.getEntities(2))
        gmsh.model.mesh.generate(2)
        _tamlik_kontrol(nyuzey)
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        V = np.array(ncoords, float).reshape(-1, 3)
        tag2i = {int(t): i for i, t in enumerate(ntags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        F = []
        for et, en in zip(etypes, enodes):
            if int(et) == 2:
                tri = np.array(en, int).reshape(-1, 3)
                F = np.array([[tag2i[int(t)] for t in row] for row in tri])
        return V, np.asarray(F, int)
    finally:
        gmsh.finalize()


def render(V, F, plab, cps, out):
    thin = int(np.argmin(V.max(0) - V.min(0))); ev, az = {0:(0,0),1:(0,-90),2:(90,-90)}[thin]
    fig = plt.figure(figsize=(13, 6.5)); ctr = V.mean(0); rng = (V.max(0)-V.min(0)).max()
    # left: semantic; right: CP on grey
    for k, mode in enumerate(["semantic", "cp"], 1):
        ax = fig.add_subplot(1, 2, k, projection="3d")
        if mode == "semantic":
            fc = np.array([SEM.get(int(x),(0.6,0.6,0.6)) for x in plab])[F].mean(1)
            ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolors="none"))
        else:
            ax.add_collection3d(Poly3DCollection(V[F], facecolors=(0.62,0.62,0.62,0.25), edgecolors="none"))
            for c in cps:
                p = c["point"]; mk = CP_MARK.get(c["source_label"], dict(c="magenta", marker="s"))
                ax.scatter([p[0]],[p[1]],[p[2]], s=260, edgecolors="k", depthshade=False, zorder=10, **mk)
        for st,a,b in [(ax.set_xlim,ctr[0]-rng/2,ctr[0]+rng/2),(ax.set_ylim,ctr[1]-rng/2,ctr[1]+rng/2),(ax.set_zlim,ctr[2]-rng/2,ctr[2]+rng/2)]: st(a,b)
        try: ax.set_box_aspect(V.max(0)-V.min(0))
        except Exception: pass
        ax.view_init(elev=ev, azim=az); ax.set_axis_off()
        if mode == "semantic":
            ax.set_title("5-class segmentation")
        else:
            nce = sum(c["source_label"] == CE for c in cps); nct = len(cps) - nce
            ax.set_title(f"{len(cps)} CP (o green={nce} CableEntry, ◆ orange={nct} Contact)")
    plt.tight_layout(); plt.savefig(out, dpi=90, bbox_inches="tight"); plt.close()


def load_any(ckpt, dev="cuda"):
    """Load either the original refit91 checkpoint or a new {state,cfg,meta} product checkpoint."""
    import torch
    if os.path.basename(ckpt).startswith("refit") or ckpt.endswith("best.pt") and "seg_extra" not in ckpt:
        return diffusionnet.load_checkpoint(ckpt, device=dev)
    d = torch.load(ckpt, map_location=dev, weights_only=False)
    if "state" in d:
        meta = d["meta"]; meta.setdefault("n_eig", d["cfg"].get("n_eig", 48))
        # vote-head checkpoints (train_vote.py) emit n_seg_classes + n_offset channels; every older
        # product checkpoint has neither key and therefore still builds the plain 5-class head.
        nout = int(d["cfg"].get("n_seg_classes", 5)) + int(d["cfg"].get("n_offset", 0))
        m, _ = diffusionnet.build_diffusionnet(d["cfg"], n_classes=nout); m.load_state_dict(d["state"]); m.to(dev).eval()
        return m, meta, None
    return diffusionnet.load_checkpoint(ckpt, device=dev)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=CKPT, help="one checkpoint, or several -> seed ensemble (softmax-averaged)")
    ap.add_argument("--device", default="cuda"); ap.add_argument("parts", nargs="+")
    a = ap.parse_args()
    os.makedirs("results/step_infer", exist_ok=True); os.makedirs(OP, exist_ok=True)
    setf = "gen_dev_40.json" if os.path.exists("gen_dev_40.json") else "benchmark_candidates.json"
    cand = {c["part_id"]: c for c in json.load(open(setf))["candidates"]}
    models = [load_any(c, dev=a.device) for c in a.ckpt]
    tag = "+".join(os.path.basename(c).replace(".pt", "") for c in a.ckpt)
    for pid in a.parts:
        stp = cand[pid]["step_file"]
        Vr, Fr = step_to_mesh(stp)
        V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
        V = np.ascontiguousarray(V, dtype=np.float64); F = np.ascontiguousarray(F, dtype=np.int64)
        acc = None                                   # softmax-average the seeds (1 model = plain predict)
        for model, meta, _ in models:
            _, pb = diffusionnet.predict(model, meta, V, F, device=a.device, op_cache_dir=OP, return_probs=True)
            pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
        probs = acc / len(models); plab = probs.argmax(-1)
        # cp-v3.1 precision recipe (arbiter F1 0.388 -> 0.520, recall preserved at 0.722, FP 36->19):
        # CableEntry + depth-gated Contact, conf mask 0.7, min 60 verts, one-CP-per-terminal cluster 10mm.
        cps = cp_openings.connection_points(V, F, plab, min_v=30, probs=probs, vertex_conf=0.5, cluster_mm=5.0)
        counts = {connector3d.LABEL_NAMES[k]: int((plab==k).sum()) for k in SEM}
        render(V, F, plab, cps, f"results/step_infer/{pid}_infer.png")
        nce = sum(c["source_label"] == CE for c in cps); nct = len(cps) - nce
        print(f"  {pid}: raw {len(Vr)}v -> remesh {len(V)}v | seg {counts} | CP={len(cps)} ({nce} CableEntry, {nct} Contact) -> {pid}_infer.png")
    print("DONE (eyeball results/step_infer/*_infer.png -- does an UNSEEN part get sane segmentation + CPs?)")


if __name__ == "__main__":
    main()
