# -*- coding: utf-8 -*-
"""A1 SONDASI -- ISIN ATMA ile YON, GERCEK VERIDE

SENTETIK YETENEK GECTI (`isin_ekseni.kendini_dogrula`): ekseni bilinen uc
delikte sapma 0.0 derece, tup skoru 8.45 / zemin 1.00.

TESHIS (docs/OTOPSI_YOGUN_PARCA.md). NIT'te 6322 secenek = ~260 konum x 24
yon. Konum/GT orani 11:1 (iyi); fazlaligin TAMAMI yon coklugundan.
Yon konum basina TEK olsaydi gereken AUC 0.9968 -> 0.91.

ESLESME KUTUSU -- BUGUN IKI KEZ DUSTUGUM TUZAK. Aday-GT eslesmesi OKLID
mesafesiyle YAPILMAZ. Kabul kutusu carpimdir: GT yonune gore YANAL <= 2mm
ve EKSENEL <= 40mm. Oklid 2mm kullanmak `kahin`i 0.593'ten 0.0172'ye
dusuruyordu -- olculen sey mekanizma degil kusurdu.

KOLLAR (GT ile eslesen adaylarda, yon YALITILMIS):
  bugunku    : model skoru en yuksek secenek
  tup        : tup skoru en yuksek secenek (kendi isaretiyle)
  tup_disari : tup en yuksek EKSEN + isaret "govdeden disari"
  tup_x_skor : tup x model skoru
  kahin      : dogru yon adaylar arasinda VAR mi (ust sinir)

KAPI: NIT'te `tup` veya `tup_disari`, `bugunku`yu >= 0.05 asacak.
D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_tam4")
sys.path.insert(0, ".")
import isin_ekseni as IE           # noqa: E402
import kanonik_d7 as K             # noqa: E402
import p6_karar                    # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402

KUME = os.environ.get("IS_KUME", "d6")
KAT_MIN = int(os.environ.get("IS_KAT_MIN", "40"))
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))
MESH = {"d6": "results/_p1_olasilik",
        "tam": "results/_p1_olasilik_brepegit"}[KUME]
MAKS_KONUM = int(os.environ.get("IS_KONUM", "3"))   # GT basina konum tavani
MAKS_YON = int(os.environ.get("IS_YON", "24"))      # konum basina yon tavani
KOLLAR = ("bugunku", "tup", "tup_eksen", "tup_hava", "tup_disari",
          "tup_x_skor", "kahin")


def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def kutu_maskesi(Pk, g, gn, yanal=None, eksenel=None):
    """CARPIM kabul kutusu: GT yonune gore yanal ve eksenel AYRI."""
    yanal = K.YANAL if yanal is None else yanal
    eksenel = 40.0 if eksenel is None else eksenel
    v = Pk - g[None, :]
    al = v @ gn
    yan = np.linalg.norm(v - al[:, None] * gn[None, :], axis=1)
    return (yan <= yanal) & (np.abs(al) <= eksenel)


def main():
    t0 = time.time()
    import trimesh
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    for d in veri:
        d["y"] = np.asarray(d["y"], int)
        d["_M"] = temel(d)
    marka = collections.Counter(d["mfg"] for d in veri)
    katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"{len(veri)} parca | katlar {katlar} | konum<= {MAKS_KONUM} "
          f"yon<= {MAKS_YON}", flush=True)

    oof = [None] * len(veri)
    for b in katlar:
        ic = [i for i, d in enumerate(veri) if d["mfg"] != b]
        dis = [i for i, d in enumerate(veri) if d["mfg"] == b]
        n_s = sum(len(veri[i]["y"]) for i in ic)
        M = np.empty((n_s, veri[0]["_M"].shape[1]), np.float32)
        o = 0
        for i in ic:
            m_ = veri[i]["_M"]
            M[o:o + len(m_)] = m_
            o += len(m_)
        Y = np.concatenate([veri[i]["y"] for i in ic])
        rng = np.random.default_rng(0)
        poz, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
        sec = np.concatenate([poz, rng.choice(
            neg, min(len(neg), NEG_KAT * max(len(poz), 1)), replace=False)])
        m = HistGradientBoostingClassifier(
            max_iter=ITER, learning_rate=0.06, max_leaf_nodes=63,
            l2_regularization=1.0, random_state=0).fit(M[sec], Y[sec])
        del M
        for i in dis:
            oof[i] = m.predict_proba(veri[i]["_M"])[:, 1]
        print(f"  OOF {b} ({time.time() - t0:.0f} s)", flush=True)

    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    atlanan = collections.Counter()
    for d, s in zip(veri, oof):
        if s is None:
            continue
        mf = f"{MESH}/{d['pid']}.npz"
        if not os.path.exists(mf):
            atlanan["mesh_yok"] += 1
            continue
        z = np.load(mf)
        V, F = np.asarray(z["V"], float), np.asarray(z["F"], int)
        if len(F) < 100:
            atlanan["mesh_kucuk"] += 1
            continue
        ag = trimesh.Trimesh(vertices=V, faces=F, process=False)
        isinci = trimesh.ray.ray_triangle.RayMeshIntersector(ag)
        merkez = V.mean(0)

        G = np.asarray(d["G"], float)
        Gn = _birim(np.asarray(d["Gd"], float))
        P = np.asarray(d["P"], float)
        idx = np.asarray(d["idx"], int)
        YD = _birim(np.asarray(d["YD"], float))
        s = np.asarray(s, float)
        Pk = P[idx]

        say = {k: 0 for k in KOLLAR}
        for j in range(len(G)):
            m_ = kutu_maskesi(Pk, G[j], Gn[j])
            if not m_.any():
                continue
            se = np.where(m_)[0]
            # GT basina en cok MAKS_KONUM konum (yanal mesafeye gore)
            kon = np.unique(idx[se])
            if len(kon) > MAKS_KONUM:
                dk = np.linalg.norm(P[kon] - G[j], axis=1)
                kon = kon[np.argsort(dk)[:MAKS_KONUM]]
            se = se[np.isin(idx[se], kon)]
            if len(se) > MAKS_YON * MAKS_KONUM:
                se = se[np.argsort(-s[se])[:MAKS_YON * MAKS_KONUM]]

            aci = np.degrees(np.arccos(np.clip(YD[se] @ Gn[j], -1, 1)))
            if (aci <= K.ACI).any():
                say["kahin"] += 1
            if aci[int(np.argmax(s[se]))] <= K.ACI:
                say["bugunku"] += 1

            # --- TUP SKORU: her konum icin kendi kokunden
            tup = np.zeros(len(se))
            for ki in np.unique(idx[se]):
                yer = np.where(idx[se] == ki)[0]
                tup[yer] = IE.tup_skoru(isinci, P[ki], YD[se][yer])
            if aci[int(np.argmax(tup))] <= K.ACI:
                say["tup"] += 1
            karma = tup * s[se]
            if aci[int(np.argmax(karma))] <= K.ACI:
                say["tup_x_skor"] += 1

            # --- TESHIS: EKSEN mi yanlis, ISARET mi?
            # `tup` rastgeleden (1/24 ~ 0.04) bile kotu ciktiysa bu tesaduf
            # degil SISTEMATIK TERS ISARETTIR: agizda dururken tup ICERI
            # uzanir, GT yonu ise DISARI bakar. Asagidaki `tup_eksen`
            # ISARETSIZ acidir; yuksekse eksen dogru, sorun yalniz isarettir.
            en = int(np.argmax(tup))
            e = YD[se][en]
            eks_aci = np.degrees(np.arccos(
                np.clip(abs(float(e @ Gn[j])), -1, 1)))
            if eks_aci <= K.ACI:
                say["tup_eksen"] += 1

            # --- ISARET KURALI 1: HAVA TARAFI (isin verisinden dogrudan)
            # Disari olan taraf havaya cikan taraftir: o yonde isin hicbir
            # seye carpmaz (serbest yol buyuk), ic tarafta duvara carpar.
            kok = P[idx[se][en]]
            d_ci = IE.ilk_carpma(isinci, np.vstack([kok, kok]),
                                 np.vstack([e, -e]))
            v = e if d_ci[0] >= d_ci[1] else -e
            if np.degrees(np.arccos(
                    np.clip(float(v @ Gn[j]), -1, 1))) <= K.ACI:
                say["tup_hava"] += 1

            # --- ISARET KURALI 2: kuresel agirlik merkezinden disari
            r = kok - merkez
            v2 = e * (1.0 if (e @ r) >= 0 else -1.0)
            if np.degrees(np.arccos(
                    np.clip(float(v2 @ Gn[j]), -1, 1))) <= K.ACI:
                say["tup_disari"] += 1

        a = ist[d["mfg"]]
        a["gt"].append(len(G))
        for k_, v_ in say.items():
            a[k_].append(v_)
        n += 1
        if n % 20 == 0:
            print(f"  {n} parca ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{n} parca | atlanan {dict(atlanan)}")
    print("GT ile ESLESEN adaylarda yon dogrulugu (CARPIM kutusu)")
    print(f"{'marka':<7}{'GT':>7}" + "".join(f"{k:>13}" for k in KOLLAR))
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        g = max(sum(a["gt"]), 1)
        r = {k: sum(a[k]) / g for k in KOLLAR}
        r["gt"] = g
        out[m_] = r
        print(f"{m_:<7}{g:>7}" + "".join(f"{r[k]:>13.4f}" for k in KOLLAR))
    print("\n=== BUGUNKUYE GORE (yogun marka NIT) ===")
    if "NIT" in out:
        h = out["NIT"]["bugunku"]
        print(f"  {'tup_eksen':<12}{out['NIT']['tup_eksen']:.4f}   "
              f"(ISARETSIZ -- yuksekse eksen DOGRU, sorun isarette)")
        for k_ in ("tup", "tup_hava", "tup_disari", "tup_x_skor"):
            f = out["NIT"][k_] - h
            print(f"  {k_:<12}{out['NIT'][k_]:.4f}   {f:+.4f}"
                  + ("  <- KAPI GECTI" if f >= 0.05 else ""))
        print(f"  {'kahin':<12}{out['NIT']['kahin']:.4f}   (ust sinir)")
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME, "marka": out,
               "maks_konum": MAKS_KONUM, "maks_yon": MAKS_YON,
               "not": "Isin atma ile delik ekseni. CARPIM kabul kutusu "
                      "(yanal 2mm / eksenel 40mm) -- Oklid DEGIL. "
                      "Sentetik yetenek testi GECTI. D7'ye BAKILMADI."},
              open(f"results/isin_ekseni_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/isin_ekseni_{KUME}.json")


if __name__ == "__main__":
    main()
