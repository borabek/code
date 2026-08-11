# -*- coding: utf-8 -*-
"""HAVUZ TAVANI: bir secenek onbelleginin YONLU recall'u ve F1 tavani.

KAPI A'nin olcusu. `P6_DIZIN` ile hangi onbellegin olculecegi secilir; boylece
"eski havuz vs uc kaldirac acik havuz" tek degiskenli kiyaslanir.

Olcut: mukemmel secici varsayimiyla ulasilabilecek en yuksek F1.
  yonlu recall r  ->  F1 tavani = 2r / (1+r)
Kabul kutusu urun metrigiyle BIREBIR: yanal<=2mm, |eksenel|<=40mm,
ISARETLI aci<=10 derece.
"""
import collections
import json
import os
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import d6_kayit               # noqa: E402
import kanonik_d7 as K        # noqa: E402
import yon_bankasi as YB      # noqa: E402

DIZ = os.environ.get("P6_DIZIN", "results/_p6_oz_u25")
ONLER = os.environ.get("HT_ONLER", "d6").split(",")
YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0


def rec(P, D, G, Gd, yon=True):
    if not len(P) or not len(G):
        return 0
    Gn = YB.birim(Gd)
    df = np.asarray(P, float)[:, None, :] - np.asarray(G, float)[None, :, :]
    al = (df * Gn[None, :, :]).sum(-1)
    yan = np.linalg.norm(df - al[..., None] * Gn[None, :, :], axis=-1)
    ok = (yan <= YANAL) & (np.abs(al) <= EKSENEL)
    if yon:
        an = np.degrees(np.arccos(np.clip(YB.birim(D) @ Gn.T, -1.0, 1.0)))
        ok = ok & (an <= ACI)
    return int(ok.any(0).sum())


def main():
    kay = K.yukle()
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    agg = collections.defaultdict(collections.Counter)
    top = collections.Counter()
    for on in ONLER:
        fs = sorted(f for f in os.listdir(DIZ) if f.startswith(on + "_"))
        for f in fs:
            pid = f[len(on) + 1:-4]
            r = kay.get(pid)
            if r is None or not len(r.get("G", [])):
                continue
            z = np.load(f"{DIZ}/{f}")
            idx = np.asarray(z["idx"], int)
            P = np.asarray(z["P"], float)
            YD = np.asarray(z["YD"], float)
            G = np.asarray(r["G"], float)
            Gd = np.asarray(r["Gd"], float)
            a = agg[r["mfg"]]
            a["parca"] += 1
            a["gt"] += len(G)
            a["konum"] += rec(P[idx], YD, G, Gd, yon=False)
            a["yonlu"] += rec(P[idx], YD, G, Gd, yon=True)
            a["aday"] += len(P)
            a["secenek"] += len(idx)
            for k_ in ("parca", "gt", "konum", "yonlu", "aday", "secenek"):
                top[k_] += a[k_] - top.get("_", 0) * 0   # (toplam asagida)
    # toplami markalardan topla (yukaridaki dongu icinde birikim yanlis olurdu)
    top = collections.Counter()
    for a in agg.values():
        for k_, v in a.items():
            top[k_] += v

    print(f"DIZIN {DIZ} | kume(ler) {ONLER}")
    print(f"{'marka':<7}{'parca':>6}{'GT':>7}{'KONUM':>9}{'YONLU':>9}"
          f"{'F1 tavani':>11}{'aday/p':>8}{'sec/p':>8}")
    out = {}
    for m, a in sorted(agg.items(), key=lambda x: -x[1]["gt"]):
        g = max(a["gt"], 1)
        p = max(a["parca"], 1)
        ry = a["yonlu"] / g
        out[m] = {"parca": a["parca"], "gt": a["gt"],
                  "konum_recall": a["konum"] / g, "yonlu_recall": ry,
                  "f1_tavani": 2 * ry / (1 + ry),
                  "aday_parca": a["aday"] / p, "secenek_parca": a["secenek"] / p}
        print(f"{m:<7}{a['parca']:>6}{a['gt']:>7}{a['konum'] / g:>9.4f}"
              f"{ry:>9.4f}{2 * ry / (1 + ry):>11.4f}"
              f"{a['aday'] / p:>8.0f}{a['secenek'] / p:>8.0f}")
    g = max(top["gt"], 1)
    p = max(top["parca"], 1)
    ry = top["yonlu"] / g
    T = {"parca": top["parca"], "gt": top["gt"],
         "konum_recall": top["konum"] / g, "yonlu_recall": ry,
         "f1_tavani": 2 * ry / (1 + ry),
         "aday_parca": top["aday"] / p, "secenek_parca": top["secenek"] / p}
    print(f"{'TOPLAM':<7}{top['parca']:>6}{top['gt']:>7}"
          f"{T['konum_recall']:>9.4f}{ry:>9.4f}{T['f1_tavani']:>11.4f}"
          f"{T['aday_parca']:>8.0f}{T['secenek_parca']:>8.0f}")
    # TAVAN DOYGUNLUGU UYARISI (2026-08-12). Aday basina secenek sayisi
    # `yon_bankasi.MAX_SEC` tavanina dayanmissa, yon KAYNAKLARINI zenginlestirmek
    # (ornegin yelpaze cozunurlugunu artirmak) recall'u ARTIRAMAZ: yeni yonler
    # tavana takilip mevcutlarin yerine geciyordur. Bu, olcumu yorumlarken
    # kolayca gozden kacan bir kisittir -- once tavan buyutulmelidir.
    try:
        import yon_bankasi as _YB
        _sp = T["secenek_parca"] / max(T["aday_parca"], 1e-9)
        if _sp >= 0.9 * _YB.MAX_SEC:
            print(f"\n!! TAVAN DOYGUN: aday basina {_sp:.1f} secenek, "
                  f"MAX_SEC={_YB.MAX_SEC}. Yon kaynagi eklemek recall'u "
                  f"ARTIRMAZ; once tavan buyutulmeli (sonda_max_sec.py).")
    except Exception:
        pass
    print(f"\nKAPI A: yonlu recall >= 0.85 mi -> "
          f"{'GECTI' if ry >= 0.85 else 'GECMEDI'} ({ry:.4f})")
    json.dump({"damga": makbuz_hash.damga(), "dizin": DIZ, "kume": ONLER,
               "toplam": T, "marka": out,
               "kapi_a_gecti": bool(ry >= 0.85),
               "not": "Havuz TAVANI (mukemmel secici). Kabul kutusu urun "
                      "metrigiyle birebir. D7'ye BAKILMADI."},
              open(f"results/havuz_tavani_{os.path.basename(DIZ)}.json", "w"),
              indent=1)
    print(f"makbuz -> results/havuz_tavani_{os.path.basename(DIZ)}.json")


if __name__ == "__main__":
    main()
