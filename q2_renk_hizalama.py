# -*- coding: utf-8 -*-
"""Q2: RENGI GEOMETRIYE baglayabiliyor muyuz? (tikanikligin gercek testi)

Q1 KANITLADI: parcalarin %83.8'inde >=2 ayrik renk var ve renkler metal/plastik ayrimini
tasiyor (gri-gumus metal, mavi/turuncu govde). Yani kanal BOS DEGIL -- kayitli "her parcada
gumusi" notu tek parcaya dayaniyormus.

TIKANIKLIK: gmsh rengi ATIYOR; STEP metnindeki koordinatlar ise parcanin YEREL cercevesinde
(eksen isinde bunu aci kaybi olarak olcmustuk: medyan 21mm sapma). XCAF renk tablosunu
yukluyor ama 0 alt-sekil etiketi veriyor. Onceki denemede yuzler SIRAYA gore eslestirilmis ve
null testte cakmisti (2/5).

YENI IMKAN: `brep_axes.cylinders` artik gmsh OCC'den KURESEL koordinatli silindirleri veriyor
(nokta, eksen, yaricap). STEP metninden ayni silindirleri YEREL koordinatlariyla ve RENKLERIYLE
ayristirabiliyoruz. Parcalar TEK KATI oldugundan (25/25 olculdu) ikisi arasindaki donusum TEK
bir rijit harekettir: yaricapla eslestirip Kabsch ile cozulebilir. Sira eslemesi DEGIL, GEOMETRI
eslemesi -- onceki denemeyi bu ayirir.

KILL (onceden yazildi): parcalarin %60'inda <1.0mm artik ile donusum kurtarilamazsa is KAPANIR.
NULL TEST zorunlu: ayni yordami RASTGELE eslemeyle kosuyoruz; o da basariyorsa test degersizdir.
"""
import os, sys, re, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- STEP metni ayristirma: yerel silindirler ---
ENT = re.compile(rb"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;", re.S)
REF = re.compile(rb"#(\d+)")


def parse(path):
    """#id -> (tip, ham argumanlar). Tek gecis, tum dosya."""
    with open(path, "rb") as fh:
        raw = fh.read()
    d = {}
    for m in ENT.finditer(raw):
        d[int(m.group(1))] = (m.group(2).decode("ascii", "replace"), m.group(3))
    return d


def yerel_silindirler(d):
    """CYLINDRICAL_SURFACE -> (merkez, eksen, yaricap) YEREL cerceve."""
    out = []
    for i, (t, a) in d.items():
        if t != "CYLINDRICAL_SURFACE":
            continue
        refs = [int(x) for x in REF.findall(a)]
        try:
            r = float(a.rsplit(b",", 1)[1])
        except ValueError:
            continue
        if not refs:
            continue
        ax = d.get(refs[0])
        if not ax or ax[0] != "AXIS2_PLACEMENT_3D":
            continue
        p = [int(x) for x in REF.findall(ax[1])]
        if len(p) < 2:
            continue
        try:
            c = [float(x) for x in d[p[0]][1].split(b"(")[1].split(b")")[0].split(b",")]
            n = [float(x) for x in d[p[1]][1].split(b"(")[1].split(b")")[0].split(b",")]
        except Exception:
            continue
        if len(c) == 3 and len(n) == 3:
            out.append((i, np.array(c), np.array(n), r))
    return out


def kabsch(A, B):
    """A -> B rijit donusum (R, t) ve artik (RMS)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    dsign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, dsign]) @ U.T
    t = cb - R @ ca
    res = float(np.sqrt(((A @ R.T + t - B) ** 2).sum(1).mean()))
    return R, t, res


def hizala(step, rastgele=False, rng=None):
    """Yerel (metin) silindirleri kuresel (gmsh) silindirlere yaricapla eslestir + Kabsch."""
    import brep_axes as B
    C, A, R_ = B.cylinders(step)
    if len(C) < 4:
        return None, 0, "gmsh<4 silindir"
    d = parse(step)
    loc = yerel_silindirler(d)
    if len(loc) < 4:
        return None, 0, "metin<4 silindir"
    lc = np.array([x[1] for x in loc]); lr = np.array([x[3] for x in loc])
    gc = np.asarray(C, float); gr = np.asarray(R_, float)
    # YARICAP ile eslestir: her yerel silindire, yaricapi 0.01mm icinde olan kuresel silindir
    pa, pb = [], []
    kul = set()
    order = rng.permutation(len(loc)) if rastgele else np.argsort(lr)
    for k in order:
        cand = [j for j in range(len(gc)) if j not in kul and abs(gr[j] - lr[k]) < 0.01]
        if not cand:
            continue
        j = int(rng.choice(cand)) if rastgele else cand[0]
        kul.add(j); pa.append(lc[k]); pb.append(gc[j])
    if len(pa) < 4:
        return None, len(pa), "eslesme<4"
    Rm, t, res = kabsch(np.array(pa), np.array(pb))
    return (Rm, t), len(pa), f"{res:.3f}mm"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    files = sorted(glob.glob("all_wscad_stp/*.stp"))
    rng = np.random.RandomState(0)
    sel = [files[i] for i in rng.choice(len(files), min(n, len(files)), replace=False)]
    print(f"{len(sel)} parcada donusum kurtarma testi\n", flush=True)
    print(f"{'parca':<16}{'esl':>5}{'GERCEK artik':>15}{'NULL (rastgele)':>18}")
    iyi = nul_iyi = tot = 0
    kayit = []
    for f in sel:
        pid = os.path.basename(f).split("_")[1]
        try:
            _, k, msg = hizala(f)
        except Exception as e:
            print(f"{pid:<16}{'-':>5}{'HATA ' + type(e).__name__:>15}")
            continue
        try:
            _, k2, msg2 = hizala(f, rastgele=True, rng=np.random.RandomState(7))
        except Exception:
            msg2 = "hata"
        tot += 1
        ok = msg.endswith("mm") and float(msg[:-2]) < 1.0
        nok = msg2.endswith("mm") and float(msg2[:-2]) < 1.0
        iyi += ok; nul_iyi += nok
        print(f"{pid:<16}{k:>5}{msg:>15}{msg2:>18}{'  <- GECTI' if ok else ''}", flush=True)
        kayit.append({"pid": pid, "eslesme": k, "artik": msg, "null": msg2, "gecti": bool(ok)})
    o = iyi / max(tot, 1); no = nul_iyi / max(tot, 1)
    print(f"\nGERCEK esleme: {iyi}/{tot} ({o:.3f}) <1.0mm")
    print(f"NULL  esleme : {nul_iyi}/{tot} ({no:.3f}) <1.0mm   "
          f"<- bu da yuksekse test DEGERSIZ")
    print(f"\nKILL: gercek oran <0.60 ya da null oran gercege yakinsa is KAPANIR -> "
          f"{'AC' if (o >= 0.60 and no < o - 0.3) else 'KAPAT'}")
    json.dump({"n": tot, "gercek_oran": o, "null_oran": no, "kayit": kayit,
               "karar": "AC" if (o >= 0.60 and no < o - 0.3) else "KAPAT"},
              open("results/q2_renk_hizalama.json", "w"), indent=1)
    print("makbuz -> results/q2_renk_hizalama.json")


if __name__ == "__main__":
    main()
