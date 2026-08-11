# -*- coding: utf-8 -*-
"""URUNUN TAM POZ ZINCIRI -- olcum betikleri bunu cagirir.

BULUNAN KUSUR (2026-08-07): butun D6/D7 olcumlerim `adaylari_uret` + gate ile duruyordu.
Oysa urunun gercek yolu (`robot_cp.extract`) gate'ten SONRA iki adim daha kosuyor:

    pose_duzelt      yanal sapma duzeltmesi   -> robot +0.0148 (D6'da olculdu)
    yon_sozluk_sec   fiziksel yon SOZLUGU     -> robot +0.0218 (ustune)

Yani olctugum 0.2103, urunun gercek 0.2469'unun altindaydi. Bu modul o farki kapatir:
her olcum betigi ayni fonksiyonu cagirir, boylece "olculen sey urunun YAPTIGI sey" olur
([[olcum-zaafiyetleri-kapatildi]] ile ayni ilke).

TEZ DEGISMEZ: iki adim da SON ISLEM; `v_o` turetmesi, 5 sinif ve ~6000 remesh aynen kalir.
Ham `v_o` istenirse `duzelt=False` ile alinir ve yan yana raporlanabilir.
"""
import numpy as np


def tam_poz(V, F, avg_probs, P, D, step_path=None, uyeler=None, cfg=None,
            pose_head=True, yon_sozluk=True):
    """Gate'ten gecmis (P, D) ciftlerine urunun poz zincirini uygula.

    Doner: (P2, D2). Herhangi bir adim duserse GIRDI aynen doner -- uydurma yok.
    """
    import connector3d
    import wire_gate as _wg
    CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
    if not len(P):
        return P, D
    cps = [{"point": np.asarray(P[i], float).tolist(),
            "direction": np.asarray(D[i], float).tolist(),
            "confidence": 0.8, "cls": CE, "_votes": 2} for i in range(len(P))]
    try:
        Xp = _wg.feats_for(V, F, avg_probs, cps, CE, CT, step_path=step_path)
    except Exception:
        return P, D
    if pose_head:
        try:
            cps = _wg.pose_duzelt(Xp, cps)
        except Exception:
            pass
    if yon_sozluk:
        try:
            cps = _wg.yon_sozluk_sec(Xp, cps, V, step_path=step_path, uyeler=uyeler)
        except Exception:
            pass
    P2 = np.asarray([c["point"] for c in cps], float)
    D2 = np.asarray([c["direction"] for c in cps], float)
    return P2, D2
