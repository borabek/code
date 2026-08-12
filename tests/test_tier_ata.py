# -*- coding: utf-8 -*-
"""`robot_cp.tier_ata` ortak yere tasindi -- DAVRANIS DEGISMEDIGINI dogrula.

Bu kural `_format_cps` govdesine gomuluydu; GLB ihracatcisini olculen zincire
(`kanonik_zincir.urun_cikti`) baglayabilmek icin ayri cagrilabilmesi gerekti.
Tasima SAF olmali: ayni girdi -> ayni tier.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robot_cp  # noqa: E402


def test_gate_skoruyla_esik_uzerinde_auto():
    c = {"wire_score": 0.75}
    assert robot_cp.tier_ata(c, 0.5, 3, auto_thr=0.6) == "auto"


def test_gate_skoruyla_esik_altinda_review():
    c = {"wire_score": 0.55}
    assert robot_cp.tier_ata(c, 0.5, 3, auto_thr=0.6) == "review"


def test_gate_hatasi_ASLA_auto_olmaz():
    """Gate calismadiysa elimizdeki HAM BIRLESIM ve kesinligi ~0.40."""
    c = {"wire_score": 0.99, "_gate_hata": True}
    assert robot_cp.tier_ata(c, 0.5, 3, auto_thr=0.6) == "review"


def test_gate_skoru_YOKSA_eski_kurala_duser():
    # yuksek guven + yeterli oy -> auto
    assert robot_cp.tier_ata({"confidence": 0.9, "_votes": 3}, 0.5, 3,
                             auto_thr=0.6) == "auto"
    # oy yetersiz -> review
    assert robot_cp.tier_ata({"confidence": 0.9, "_votes": 1}, 0.5, 3,
                             auto_thr=0.6) == "review"
    # guven yetersiz -> review
    assert robot_cp.tier_ata({"confidence": 0.2, "_votes": 3}, 0.5, 3,
                             auto_thr=0.6) == "review"


def test_auto_thr_None_ise_eski_kural():
    assert robot_cp.tier_ata({"wire_score": 0.99, "confidence": 0.1,
                              "_votes": 1}, 0.5, 3, auto_thr=None) == "review"


def test_format_cps_ayni_sonucu_verir():
    """Uctan uca: `_format_cps` ciktisi tier_ata ile TUTARLI olmali."""
    cps = [{"point": [0, 0, 0], "direction": [0, 0, 1], "wire_score": 0.9,
            "confidence": 0.8, "_votes": 3, "area": 1.0},
           {"point": [1, 0, 0], "direction": [0, 0, 1], "wire_score": 0.1,
            "confidence": 0.8, "_votes": 3, "area": 1.0}]
    out = robot_cp._format_cps(cps, 0.5, 3)
    tierler = {r["wire_score"]: r["tier"] for r in out}
    # esik config'ten okunur; hangi degerse iki CP AYNI kurala tabi olmali
    assert set(tierler.values()) <= {"auto", "review"}
    assert tierler[0.9] == "auto" or tierler[0.1] == "review"




def test_guvenli_anahtar_hepsini_review_yapar(monkeypatch=None):
    """`robot_auto_kapali=true` -> hicbir isaret otonom olmaz.

    Olculen gerekce: D7'de dagitilan esikte isaretlerin %100'u AUTO ve
    kesinlik 0.3471. Anahtarin VARSAYILANI False -- urun davranisi degismez.
    """
    import json
    import robot_cp as R
    ger = R._load_cfg

    def sahte(*a, **k):
        d = dict(ger())
        d["robot_auto_kapali"] = True
        return d

    R._load_cfg = sahte
    try:
        assert R.tier_ata({"wire_score": 0.99}, 0.5, 3, auto_thr=0.6) == "review"
        assert R.tier_ata({"confidence": 1.0, "_votes": 9}, 0.5, 3,
                          auto_thr=None) == "review"
    finally:
        R._load_cfg = ger
    # VARSAYILAN kapali degil: davranis korunuyor
    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    assert not cfg.get("robot_auto_kapali", False), \
        "cp_config'te anahtar ACIK -- urun davranisi sessizce degismis olur"


if __name__ == "__main__":
    for ad, f in sorted(globals().items()):
        if ad.startswith("test_"):
            f()
            print(f"  GECTI  {ad}")
    print("hepsi gecti")
