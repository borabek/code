# -*- coding: utf-8 -*-
"""HAVUZ mu SKOR mu -- DOKUMDEN teshis (yeni cikarim GEREKTIRMEZ).

Onceki olcum GECERSIZ cikmisti: cikti recall'u (0.4682) HAVUZ recall'undan
(0.4242) BUYUK gorunuyordu -- yapisal olarak IMKANSIZ, cunku cikti havuzun
alt kumesidir. Bu betik once o CELISKIYI teshis eder (havuz gercekten
ciktinin ust kumesi mi), sonra dogru soruyu sorar:

  KACIRILAN GT'lerin kaci HAVUZDA ZATEN VAR (yani sorun SKOR/GATE),
  kaci havuzda HIC YOK (yani sorun TEMSIL/ADAY URETIMI)?

Havuz kapsamasi icin Macar eslestirme DEGIL, **GT basina kapsama** olculur
(o GT'yi kabul kutusunda karsilayan HERHANGI bir aday var mi) -- havuz
soruşturmasinin dogru olcutu budur.
"""
import json
import sys

import numpy as np

DOKUM = sys.argv[1] if len(sys.argv) > 1 else "results/_dokum_taban.json"
YOL = sys.argv[2] if len(sys.argv) > 2 else "olculen"
YANAL, EKSEN, ACI = 2.0, 40.0, 10.0


def _birim(v):
    v = np.asarray(v, float).reshape(-1, 3)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def kapsama(P, D, G, Gd, isaretli):
    """(n_GT,) bool: her GT icin kabul kutusunda EN AZ BIR aday var mi."""
    if not len(P) or not len(G):
        return np.zeros(len(G), bool)
    diff = P[:, None, :] - G[None, :, :]
    al = (diff * Gd[None, :, :]).sum(-1)
    pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
    c = D @ Gd.T
    an = np.degrees(np.arccos(np.clip(c if isaretli else np.abs(c), -1, 1)))
    kabul = (np.abs(al) <= EKSEN) & (an <= ACI) & (pe <= YANAL)
    return kabul.any(0)


def main():
    kayit = [r for r in json.load(open(DOKUM)) if r.get("yol") == YOL]
    print(f"{DOKUM} / yol={YOL} -> {len(kayit)} parca")
    ust_kume_ihlali = 0
    havuz_bos = 0
    say = {k: 0 for k in ("gt", "cikti", "havuz", "havuz_var_cikti_yok",
                          "havuz_yok")}
    for isaretli in (False, True):
        for k in say:
            say[k] = 0
        ust_kume_ihlali = havuz_bos = 0
        for r in kayit:
            G = np.asarray(r["G"], float).reshape(-1, 3)
            if not len(G):
                continue
            Gd = _birim(r["Gd"])
            P = np.asarray(r["P"], float).reshape(-1, 3)
            D = _birim(r["D"]) if len(P) else np.zeros((0, 3))
            hP = np.asarray(r.get("havuz_P") or [], float).reshape(-1, 3)
            hD = _birim(r["havuz_D"]) if len(hP) else np.zeros((0, 3))
            if not len(hP):
                havuz_bos += 1
            c_cik = kapsama(P, D, G, Gd, isaretli)
            c_hav = kapsama(hP, hD, G, Gd, isaretli)
            # YAPISAL KONTROL: cikti havuzun alt kumesiyse, ciktinin
            # kapsadigi her GT'yi havuz da kapsamali.
            ust_kume_ihlali += int((c_cik & ~c_hav).sum())
            say["gt"] += len(G)
            say["cikti"] += int(c_cik.sum())
            say["havuz"] += int(c_hav.sum())
            say["havuz_var_cikti_yok"] += int((c_hav & ~c_cik).sum())
            say["havuz_yok"] += int((~c_hav).sum())
        ad = "ISARETLI" if isaretli else "isaretsiz"
        gt = max(say["gt"], 1)
        print(f"\n--- {ad} kabul kutusu (yanal<={YANAL} eksen<={EKSEN} "
              f"aci<={ACI}) ---")
        print(f"  GT toplam                        : {say['gt']}")
        print(f"  HAVUZ kapsamasi (tavan)          : {say['havuz']:5d} "
              f"({say['havuz']/gt:.4f})")
        print(f"  CIKTI kapsamasi                  : {say['cikti']:5d} "
              f"({say['cikti']/gt:.4f})")
        print(f"  havuzda VAR ama ciktida YOK      : "
              f"{say['havuz_var_cikti_yok']:5d} "
              f"({say['havuz_var_cikti_yok']/gt:.4f})  <- SKOR/GATE kaybi")
        print(f"  havuzda HIC YOK                  : {say['havuz_yok']:5d} "
              f"({say['havuz_yok']/gt:.4f})  <- TEMSIL kaybi")
        print(f"  [yapisal kontrol] ciktida var/havuzda yok: "
              f"{ust_kume_ihlali}  (0 OLMALI)")
        if havuz_bos:
            print(f"  UYARI: {havuz_bos} parcada havuz BOS kaydedilmis")
        kayip = say["gt"] - say["cikti"]
        if kayip:
            print(f"  => kacan {kayip} GT'nin "
                  f"%{100*say['havuz_var_cikti_yok']/kayip:.1f}'i SKOR, "
                  f"%{100*say['havuz_yok']/kayip:.1f}'i TEMSIL")


if __name__ == "__main__":
    main()
