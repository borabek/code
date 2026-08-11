# -*- coding: utf-8 -*-
"""MESH SEYRELTME KURALINI EGITIM MARKALARINDA SEC.

NEDEN BU BETIK VAR: kurali (uzamsal yaricap + ust sinir) ilk olarak D6'da
olcup sectim. D6 benim TEMIZ OKUMA kumem; oradan bir hiperparametre secmek onu
kirletir ve raporlanan D6 sayisi o olcude sisik olur.

Bu betik AYNI olcumu `tam` korpusunun markalarinda (TOGI/PXC/WEI/SIE/TE/...)
tekrarlar. Kural EGITIM tarafinda secilir; D6 ve D7 yalnizca OLCULUR.

Olcut: YALNIZ KONUM recall'u (yanal<=2mm, |eksenel|<=40mm). Yon bu asamada
sorulmaz -- yon secimi ayri bir problem ve `yon_bankasi` isi.
"""
import collections
import json
import os
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import connector3d          # noqa: E402
import d6_kayit             # noqa: E402
import havuz_seyrelt        # noqa: E402
import kanonik_d7 as K      # noqa: E402

OZ = "results/_tam_oz"
YANAL, EKSENEL = 2.0, 40.0
KURALLAR = {                      # ad -> (yaricap, taban, kat)
    "topP (eski) 250": (0.0, 250, 0),
    "uzamsal 4.0 / 2x120": (4.0, 120, 2),
    "uzamsal 3.0 / 3x200": (3.0, 200, 3),
    "uzamsal 2.5 / 4x250": (2.5, 250, 4),
    "uzamsal 2.0 / 6x400": (2.0, 400, 6),
    "HEPSI": (0.0, 10 ** 9, 0),
}


def konum_rec(P, G, Gd):
    if not len(P) or not len(G):
        return 0
    Gn = Gd / np.maximum(np.linalg.norm(Gd, axis=1, keepdims=True), 1e-12)
    df = np.asarray(P, float)[:, None, :] - np.asarray(G, float)[None, :, :]
    al = (df * Gn[None, :, :]).sum(-1)
    yan = np.linalg.norm(df - al[..., None] * Gn[None, :, :], axis=-1)
    return int(((yan <= YANAL) & (np.abs(al) <= EKSENEL)).any(0).sum())


def main():
    on = sys.argv[1] if len(sys.argv) > 1 else "tam"
    mesh_dizin = {"tam": "results/_p1_olasilik_brepegit",
                  "d6": "results/_p1_olasilik"}[on]
    fs = sorted(f for f in os.listdir(OZ)
                if f.startswith(on + "_") and f.endswith(".npz"))
    N = int(os.environ.get("SONDA_N", "0"))
    if N:
        fs = fs[:N]
    pidler = [f[len(on) + 1:-4] for f in fs]
    kay = K.yukle(pidler)
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    agg = {k: collections.Counter() for k in KURALLAR}
    print(f"{on}: {len(fs)} parca", flush=True)
    for i, (f, pid) in enumerate(zip(fs, pidler), 1):
        r = kay.get(pid)
        if r is None or not len(r.get("G", [])):
            continue
        mf = f"{mesh_dizin}/{pid}.npz"
        if not os.path.exists(mf):
            continue
        z = np.load(f"{OZ}/{f}")
        kk = np.asarray(z["kaynak"], int)
        P = np.asarray(z["P"], float)
        zz = np.load(mf)
        V = np.asarray(zz["V"], float)
        pb = np.mean([np.asarray(q, float) for q in zz["pbs"]], axis=0)
        pp = havuz_seyrelt.ppos(pb, connector3d.CABLE_ENTRY,
                                connector3d.CONTACT)
        m01 = np.isin(kk, (0, 1))
        i2 = np.where(kk == 2)[0]
        P2 = P[i2]
        s2 = (pp[np.argmin(np.linalg.norm(P2[:, None, :] - V[None, :, :],
                                          axis=-1), axis=1)]
              if len(P2) and len(P2) * len(V) < 6e7 else np.zeros(len(P2)))
        n01 = int(m01.sum())
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        for ad, (rr, tb, kt) in KURALLAR.items():
            if rr <= 0 and kt == 0 and tb >= 10 ** 9:
                sel = i2
            elif rr <= 0:
                sel = i2[np.argsort(-s2)[:tb]] if len(P2) else i2
            else:
                sel = i2[havuz_seyrelt.seyrelt(P2, s2, n01, rr, tb, kt)] \
                    if len(P2) else i2
            Pu = np.vstack([P[m01], P[sel]]) if len(sel) else P[m01]
            a = agg[ad]
            a["gt"] += len(G)
            a["tut"] += konum_rec(Pu, G, Gd)
            a["n"] += len(Pu)
            a["p"] += 1
        if i % 250 == 0:
            print(f"  {i}/{len(fs)}", flush=True)

    print(f"\n{'kural':<24}{'KONUM recall':>14}{'aday/parca':>13}")
    out = {}
    for ad in KURALLAR:
        a = agg[ad]
        rc = a["tut"] / max(a["gt"], 1)
        out[ad] = {"konum_recall": rc, "aday_parca": a["n"] / max(a["p"], 1),
                   "n_parca": a["p"], "n_gt": a["gt"]}
        print(f"{ad:<24}{rc:>14.4f}{a['n'] / max(a['p'], 1):>13.0f}")
    json.dump({"damga": makbuz_hash.damga(), "kume": on, "sonuc": out,
               "not": "Mesh seyreltme kuralinin YALNIZ KONUM recall'u. Kural "
                      "EGITIM markalarinda secilir; D6/D7 yalnizca olculur."},
              open(f"results/seyreltme_kurali_{on}.json", "w"), indent=1)
    print(f"makbuz -> results/seyreltme_kurali_{on}.json")


if __name__ == "__main__":
    main()
