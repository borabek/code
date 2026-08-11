# -*- coding: utf-8 -*-
"""TEK KANONIK ZINCIR (P0-c). Her olcum BURADAN gecer.

NEDEN (2026-08-09'da pahaliya ogrenildi): ayni yigin uc farkli olcum yolunda
0.2344 / 0.0935 / 0.0666 verdi. Sebep, her betigin gate'i KENDI usulunce
uygulamasiydi -- urun `wire_gate.apply` ile MUTLAK esik (0.30, rejim-kosullu)
kullaniyor, benim olcum betiklerim ise elle yazilmis GORELI esik (maske 0.40/0.30).
Farkli politika, farkli sayi.

Bu modul `robot_cp.extract`'in CIKARIM SONRASI yarisini AYNEN tekrarlar; tek fark
olasiliklarin onbellekten gelmesi (yeniden cikarim yapmamak icin). Boylece bir
olcum, urunun YAPTIGI seyi olcer.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robot_cp


def urun_cikti(V, F, pbs, step_path, cfg=None, cp_count=None):
    """URUNUN adaylari + gate'i, `extract` ile AYNI sirayla. Doner: cps listesi."""
    _cfg = cfg if cfg is not None else robot_cp._load_cfg()
    cps, avg, is_hi, _u = robot_cp.adaylari_uret(V, F, pbs, step_path, cfg=_cfg)
    if not cps:
        return []
    # GENISLETILMIS YOL (2026-08-11). Olculdu (D7 835 parca marka-disi, TAM
    # ZINCIR, MIKRO, esik D6'da secildi): robot 0.2029 -> 0.3090, tespit
    # 0.4523 -> 0.4813, makro 0.2146 -> 0.3142; 10/12 marka artida.
    # Kol calisamazsa (model/STEP/B-rep yok) None doner ve ESKI yol surer --
    # sessiz bozuk cikti YOK. Kapatmak: cp_config `robot_genis_havuz=false`
    # ya da `URUN_GENIS=0`.
    # P6 ORTAK YOL (2026-08-11). Her adaya `yon_bankasi` secenekleri takilir ve
    # (konum, yon) cifti TEK skorla siralanir -- yon artik SECILIR. Genis yoldan
    # ONCE denenir; kol calisamazsa (model yok / STEP yok) None doner ve
    # ASAGIDAKI genis yol aynen surer. Kapatmak: `robot_p6_ortak=false` ya da
    # `URUN_P6=0`.
    import urun_p6
    if urun_p6.ACIK:                      # cevre degiskeni VE config, tek yerde
        _p = urun_p6.cikti(V, F, avg, cps, step_path,
                           robot_cp.CE, robot_cp.CT)
        if _p is not None:
            return _p
    if _cfg.get("robot_genis_havuz", False):
        import urun_genis
        if urun_genis.ACIK:
            _g = urun_genis.cikti(V, F, avg, cps, step_path,
                                  robot_cp.CE, robot_cp.CT)
            if _g is not None:
                return _g
    if _cfg.get("robot_wire_gate", True):
        import wire_gate
        thr = float(_cfg.get("robot_wire_gate_threshold", 0.30))
        if is_hi:
            thr = float(_cfg.get("robot_wire_gate_threshold_highcp", thr))
        cps = wire_gate.apply(V, F, avg, cps, robot_cp.CE, robot_cp.CT,
                              threshold=thr, top_n=cp_count, step_path=step_path)
    return cps


def poz_ver(cps):
    """cps -> (P, D). Anahtarlar `point` / `direction` -- `dir` DEGIL."""
    if not cps:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (np.asarray([c["point"] for c in cps], float),
            np.asarray([c["direction"] for c in cps], float))
