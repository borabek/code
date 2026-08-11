# -*- coding: utf-8 -*-
"""EGITIM <-> RUNTIME PARITESI.

2026-08-01 DENETIM BULGUSU: gate egitim verisi adaylari AYRI bir kod yolundan uretiyordu ve
iki yerde sapiyordu:
  * `step_path` GECILMIYORDU -> B-rep analitik ekseni ve fiziksel ozellikler egitimde YOKTU,
    uründe VARDI. Gate, gercekte gordugunden FARKLI adaylarla egitiliyordu.
  * `f1_sweep.union_all` kullaniliyordu, urun ise `robot_cp._vote2` (BENZERSIZ model sayimi).
    Sonuc: egitim npz'sinde `votes` 12'ye kadar cikiyordu, uründe tavan MODEL SAYISI (4).

Mevcut parite testi (`test_parite.py`) YALNIZ cikarim tarafina bakiyordu; egitim yolu
denetlenmiyordu. Bu dosya o acigi kapatir.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_aday_uretimi_TEK_FONKSIYONDAN():
    """Hem urun hem gate egitimi `robot_cp.adaylari_uret` cagirmali."""
    import robot_cp
    assert hasattr(robot_cp, "adaylari_uret"), "ortak aday ureticisi YOK"
    with open(os.path.join(KOK, "gate_regrow.py"), encoding="utf-8") as f:
        gr = f.read()
    assert "adaylari_uret" in gr, "gate_regrow ortak ureticiyi cagirmiyor"
    assert "f1_sweep.union_all(per)" not in gr, \
        "gate_regrow hala union_all ile aday uretiyor (urun _vote2 kullanir)"


def test_vote2_BENZERSIZ_model_sayar_ve_tavan_MODEL_SAYISI():
    """`votes`, model sayisini ASAMAZ. Egitim verisinde 12 gorulmustu (4 model varken)."""
    import robot_cp
    p = np.array([0.0, 0.0, 0.0])
    # ayni modelden IKI yakin aday + baska modelden bir tane
    listeler = [[{"point": p, "direction": np.array([0, 0, 1.0]), "confidence": 0.9},
                 {"point": p + 0.1, "direction": np.array([0, 0, 1.0]), "confidence": 0.8}],
                [{"point": p + 0.2, "direction": np.array([0, 0, 1.0]), "confidence": 0.7}]]
    out = robot_cp._vote2(listeler, cluster_mm=5.0, min_votes=1)
    assert len(out) == 1, "yakin adaylar tek CP'de birlesmeliydi"
    assert out[0]["_votes"] == 2, f"benzersiz MODEL sayilmiyor: {out[0]['_votes']}"
    assert out[0]["_votes"] <= len(listeler), "votes model sayisini ASIYOR"


def test_egitim_verisi_VOTES_tavanina_uyuyor():
    """Dagitilan egitim npz'sinde votes, urun tavanini asmamali.

    ESKI VERI ICIN BEKLENEN BASARISIZLIK: parite duzeltmesinden ONCE uretilmis npz'de
    votes 12'ye kadar cikiyor. Bu test, korpus YENIDEN URETILDIKTEN sonra yesillenir;
    o zamana kadar acik bir borcu isaret eder."""
    # DAGITILAN egitim verisi (config'ten) -- sabit dosya adi degil.
    with open(os.path.join(KOK, "cp_config.json"), encoding="utf-8") as f:
        _c = json.load(f)
    _yol = _c.get("current_product", {}).get("wire_gate", {}).get(
        "egitim_verisi", "results/gate_regrow_data_topo.npz")
    npz = os.path.join(KOK, _yol)
    if not os.path.exists(npz):
        pytest.skip("egitim verisi yok")
    d = np.load(npz, allow_pickle=True)
    with open(os.path.join(KOK, "cp_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    n_model = len(cfg.get("robot_vote2_checkpoints", []) or [])
    if not n_model:
        pytest.skip("model sayisi bilinmiyor")
    v = np.asarray(d["votes"] if "votes" in d.files else d["X"][:, 11])
    if v.max() > n_model:
        pytest.xfail(f"PARITE BORCU: egitim verisinde votes {v.max():.0f} > model sayisi "
                     f"{n_model}. Kod duzeltildi; korpus henuz yeniden uretilmedi.")
    assert v.max() <= n_model
