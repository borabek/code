# -*- coding: utf-8 -*-
"""TEK KANONIK ZINCIR (P0-c). Her measurement BURADAN gecer.

WHY (2026-08-09'da pahaliya ogrenildi): same yigin three different measurement yolunda
0.2344 / 0.0935 / 0.0666 verdi. Sebep, each betigin gate'i KENDI usulunce
uygulamasiydi -- urun `wire_gate.apply` with MUTLAK threshold (0.30, regime-kosullu)
kullaniyor, benim measurement betiklerim whereas elle yazilmis GORELI threshold (maske 0.40/0.30).
Farkli politika, different number.

Bu modul `robot_cp.extract`'in CIKARIM SONRASI yarisini AYNEN tekrarlar; single difference
olasiliklarin onbellekten gelmesi (yeniden inference yapmamak for). Boylece a
measurement, urunun YAPTIGI seyi olcer.
"""
import os ,sys 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import robot_cp 


def product_output (V ,F ,pbs ,step_path ,cfg =None ,cp_count =None ):
    """URUNUN adaylari + gate'i, `extract` with AYNI sirayla. Doner: cps listesi."""
    _cfg =cfg if cfg is not None else robot_cp ._load_cfg ()
    cps ,avg ,is_hi ,_u =robot_cp .derive_candidates (V ,F ,pbs ,step_path ,cfg =_cfg )
    if not cps :
        return []
        # GENISLETILMIS YOL (2026-08-11). Olculdu (D7 835 part brand-disi, TAM
        # ZINCIR, MIKRO, threshold D6'da secildi): robot 0.2029 -> 0.3090, tespit
        # 0.4523 -> 0.4813, makro 0.2146 -> 0.3142; 10/12 brand artida.
        # Kol calisamazsa (model/STEP/B-rep absent) None returns and ESKI path surer --
        # silent bozuk output YOK. Kapatmak: cp_config `robot_genis_havuz=false`
        # ya da `URUN_GENIS=0`.
        # P6 ORTAK YOL (2026-08-11). Her adaya `direction_bank` secenekleri takilir and
        # (konum, direction) cifti TEK skorla siralanir -- direction residual SECILIR. Genis yoldan
        # ONCE denenir; arm calisamazsa (model absent / STEP absent) None returns and
        # ASAGIDAKI genis path aynen surer. Kapatmak: `robot_p6_ortak=false` ya da
        # `URUN_P6=0`.
    import product_p6 
    if product_p6 .ACIK :# cevre degiskeni VE config, single places
        _p =product_p6 .out_ (V ,F ,avg ,cps ,step_path ,
        robot_cp .CE ,robot_cp .CT )
        if _p is not None :
            return _p 
    if _cfg .get ("robot_genis_havuz",False ):
        import product_genis 
        if product_genis .ACIK :
            _g =product_genis .out_ (V ,F ,avg ,cps ,step_path ,
            robot_cp .CE ,robot_cp .CT )
            if _g is not None :
                return _g 
    if _cfg .get ("robot_wire_gate",True ):
        import wire_gate 
        thr =float (_cfg .get ("robot_wire_gate_threshold",0.30 ))
        if is_hi :
            thr =float (_cfg .get ("robot_wire_gate_threshold_highcp",thr ))
        cps =wire_gate .apply (V ,F ,avg ,cps ,robot_cp .CE ,robot_cp .CT ,
        threshold =thr ,top_n =cp_count ,step_path =step_path )
    return cps 


def poz_ver (cps ):
    """cps -> (P, D). Anahtarlar `point` / `direction` -- `dir` DEGIL."""
    if not cps :
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
    return (np .asarray ([c ["point"]for c in cps ],float ),
    np .asarray ([c ["direction"]for c in cps ],float ))
