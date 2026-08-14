# -*- coding: utf-8 -*-
"""HAVUZ mu SKOR mu — kacanlarin sebebini AYIRAN olcum

Hata otopsisi: kacanlarin %94'unun yakininda HIC tahmin yok. Ama bu iki
BAMBASKA sey olabilir:
  (a) HAVUZDA aday yok      -> aday uretimi sorunu (PAHALI onarim)
  (b) Aday VARDI, elendi    -> gate/skor sorunu (UCUZ onarim olabilir)

Bu betik ikisini AYIRIR: ayni parcalarda HAVUZ recall'u ile CIKTI recall'u
yan yana olculur.
  havuz_recall : GT'nin kacinin KABUL KUTUSUNDA en az bir HAM aday var
  cikti_recall : GT'nin kacinin kutusunda nihai CIKTI var
  fark         = gate/skorun ATTIGI dogru adaylar

`havuz_recall` da dusukse onarim ADAY URETIMINDE; yuksekse SKORDA.
D7'ye BAKILMAZ.
"""
import json, os, sys
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K

DOKUM = os.environ.get("HV_DOKUM", "results/_dokum_taban.json")
YOL = os.environ.get("HV_YOL", "saha")

def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def kutuda(P, D, G, Gd, isaretli):
    """her GT icin kutusunda uygun aday VAR mi (bool dizi)."""
    if not len(P):
        return np.zeros(len(G), bool)
    v = P[:, None, :] - G[None, :, :]
    al = np.einsum("pgc,gc->pg", v, Gd)
    yan = np.linalg.norm(v - al[..., None] * Gd[None, :, :], axis=-1)
    ok = (yan <= K.YANAL) & (np.abs(al) <= 40.0)
    if isaretli:
        aci = np.degrees(np.arccos(np.clip(D @ Gd.T, -1, 1)))
        ok &= (aci <= K.ACI)
    return ok.any(0)

def main():
    d = [r for r in json.load(open(DOKUM)) if r["yol"] == YOL]
    if not d:
        sys.exit(f"{DOKUM} icinde '{YOL}' yok")
    if "havuz_P" not in d[0]:
        sys.exit("dokumde havuz YOK -- sondayi havuz dokumuyle yeniden kos")
    top = {k: 0 for k in ("gt", "hav_konum", "hav_yonlu",
                          "cik_konum", "cik_yonlu")}
    for r in d:
        G = np.asarray(r["G"], float); Gd = _birim(np.asarray(r["Gd"], float))
        if not len(G): continue
        HP = np.asarray(r["havuz_P"], float).reshape(-1, 3)
        HD = _birim(np.asarray(r["havuz_D"], float).reshape(-1, 3))
        CP = np.asarray(r["P"], float).reshape(-1, 3)
        CD = _birim(np.asarray(r["D"], float).reshape(-1, 3))
        top["gt"] += len(G)
        top["hav_konum"] += int(kutuda(HP, HD, G, Gd, False).sum())
        top["hav_yonlu"] += int(kutuda(HP, HD, G, Gd, True).sum())
        top["cik_konum"] += int(kutuda(CP, CD, G, Gd, False).sum())
        top["cik_yonlu"] += int(kutuda(CP, CD, G, Gd, True).sum())
    g = max(top["gt"], 1)
    print(f"{len(d)} parca | GT {top['gt']} | yol={YOL}\n")
    print(f"{'olcu':<28}{'recall':>9}")
    for ad, k in (("HAVUZ recall (konum)", "hav_konum"),
                  ("HAVUZ recall (yonlu)", "hav_yonlu"),
                  ("CIKTI recall (konum)", "cik_konum"),
                  ("CIKTI recall (yonlu)", "cik_yonlu")):
        print(f"{ad:<28}{top[k]/g:>9.4f}")
    kayip = (top["hav_yonlu"] - top["cik_yonlu"]) / g
    print(f"\nGATE/SKORUN ATTIGI DOGRU ADAY: {kayip:.4f}")
    print("OKUMA:")
    print("  havuz_yonlu DUSUK  -> onarim ADAY URETIMINDE (pahali)")
    print("  fark BUYUK         -> onarim SKOR/GATE'te (ucuz olabilir)")
    json.dump({"yol": YOL, "n_parca": len(d),
               "recall": {k: top[k]/g for k in top if k != "gt"},
               "gate_attigi": kayip,
               "not": "Havuz recall vs cikti recall. Kacanlarin sebebini "
                      "AYIRIR. D7'ye BAKILMADI."},
              open(f"results/havuz_vs_cikti_{YOL}.json", "w"), indent=1)
    print(f"makbuz -> results/havuz_vs_cikti_{YOL}.json")

if __name__ == "__main__":
    main()
