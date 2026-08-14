# -*- coding: utf-8 -*-
"""PARCA-ICI donusum testleri (wire_gate.within_part).

Bu donusum GORULMEMIS URETICI for deployed (most kotu manufacturer-disi 0.4832 -> 0.5702). Kazanci
saglayan OZELLIGIN kendisi test ediliyor: part-ici z-score, a sutunun PARCA CAPINDA kaymasina
and olceklenmesine BAGISIK must be. Ureticiden ureticiye changed full as this -- dagilimin yeri
and genisligi. Ozellik bozulursa transfer kazanci sessizce kaybolurdu and F1 tablosu bunu a
sonraki full olcume up to gostermezdi.
"""
import os 
import sys 

import numpy as np 
import pytest 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))


def _wg ():
    import wire_gate 
    return wire_gate 


def test_donusum_yoksa_matris_aynen_doner ():
    wg =_wg ()
    X =np .arange (12 ,dtype =float ).reshape (4 ,3 )
    assert wg .within_part (X ,None )is X 
    assert wg .within_part (X ,"").shape ==(4 ,3 )


def test_zskor_genisligi_ikiye_katlar_ve_ham_sutunlari_KORUR ():
    """Ham sutunlar KORUNMALI: only goreli kullanmak measured and familiar veride -0.090."""
    wg =_wg ()
    X =np .random .default_rng (0 ).normal (size =(6 ,5 ))*3.0 +7.0 
    M =wg .within_part (X ,"zskor")
    assert M .shape ==(6 ,10 )
    assert np .allclose (M [:,:5 ],X ),"ham sutunlar degismis"
    z =M [:,5 :]
    assert np .allclose (z .mean (0 ),0.0 ,atol =1e-9 )
    assert np .allclose (z .std (0 ),1.0 ,atol =1e-9 )


@pytest .mark .parametrize ("kaydir,olcek",[(100.0 ,1.0 ),(0.0 ,50.0 ),(-13.5 ,0.2 )])
def test_zskor_parca_ici_KAYMA_ve_OLCEGE_bagisik (kaydir ,olcek ):
    """ISIN OZU: ureticiden ureticiye kayan sey dagilimin yeri/genisligi. z-kismi degismemeli."""
    wg =_wg ()
    X =np .random .default_rng (1 ).normal (size =(7 ,4 ))
    z0 =wg .within_part (X ,"zskor")[:,4 :]
    z1 =wg .within_part (X *olcek +kaydir ,"zskor")[:,4 :]
    assert np .allclose (z0 ,z1 ,atol =1e-9 ),"z-kismi part-ici kayma/olcege bagisik not"


def test_sabit_sutun_sifir_verir_NaN_uretmez ():
    """sd=0 -> split absent. NaN uretilirse RandomForest patlar ya da sessizce sacmalar."""
    wg =_wg ()
    X =np .ones ((5 ,3 ))
    X [:,1 ]=np .arange (5 )
    M =wg .within_part (X ,"zskor")
    assert np .isfinite (M ).all ()
    assert np .allclose (M [:,3 ],0.0 )and np .allclose (M [:,5 ],0.0 )


def test_tek_aday_NaN_uretmez ():
    wg =_wg ()
    for d in ("zskor","sira"):
        M =wg .within_part (np .array ([[1.0 ,2.0 ,3.0 ]]),d )
        assert M .shape ==(1 ,6 )and np .isfinite (M ).all (),d 


def test_bilinmeyen_donusum_SESSIZ_gecmez ():
    """Yanlis yazilmis a donusum adi sessizce 'donusum absent'a dusmemeli."""
    wg =_wg ()
    with pytest .raises (ValueError ):
        wg .within_part (np .zeros ((3 ,2 )),"zkor")


def test_dagitilan_model_donusumu_KENDI_tasiyor ():
    """Geriye donuk uyum tasarimi: donusum modelde, config'te DEGIL.

    cp_config alani only belge; model degistirilmeden bayrak cevrilirse gate wrong
    genislikte beslenirdi. Kaynak TEK must be."""
    import pickle 
    path =os .path .join (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))),
    "results","wire_gate.pkl")
    if not os .path .exists (path ):
        pytest .skip ("dagitilmis gate none")
    with open (path ,"rb")as f :
        m =pickle .load (f )
    wg =_wg ()
    don =m .get ("donusum")or m .get ("donusum_z")
    if not don :
        pytest .skip ("dagitilan gate donusumsuz")
    if m .get ("clf_z")is not None :
    # YONLENDIRMELI model: ham width n_feat, z modeli onun two kati bekler.
        assert wg .within_part (np .zeros ((3 ,m ["n_feat"])),don ).shape [1 ]==2 *m ["n_feat"]
    else :
        ham =len (m ["feat_names"])//2 
        assert wg .within_part (np .zeros ((3 ,ham )),don ).shape [1 ]==m ["n_feat"]


def test_yaricap_uyusmazligi_SESSIZ_gecmez ():
    """MAYIN (2026-08-01'de acildi, same gun kapatildi): TOPO_R ayarlanabilir yapildi but model
    onu tasimiyordu. R=6'da egitilmis gate R=8 ozellikleriyle beslenseydi SUTUN SAYISI AYNI
    KALDIGI ICIN error vermezdi -- gate sessizce kotulesirdi.

    `n_feat` this hatayi YAKALAYAMAZ: sutunlarin count not ANLAMI degisir. Bu yuzden ayarin
    kendisi modelde saklanir and calisma aninda karsilastirilir."""
    import importlib 
    os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1"
    os .environ ["WG_TOPO_R"]="6.0"
    import wire_gate 
    importlib .reload (wire_gate )
    m6 ={"topo_r":6.0 }
    wire_gate ._dogrula_uyum (m6 )# same radius: sorun absent
    wire_gate ._dogrula_uyum ({})# old model (kayitsiz): engellenmez
    wire_gate ._dogrula_uyum (None )
    os .environ ["WG_TOPO_R"]="8.0"
    importlib .reload (wire_gate )
    with pytest .raises (ValueError ,match ="yaricap"):
        wire_gate ._dogrula_uyum (m6 )
    os .environ .pop ("WG_TOPO_R",None )
    importlib .reload (wire_gate )


def test_dagitilan_model_yaricabini_TASIYOR ():
    """Dagitilan gate topoloji kullaniyorsa `topo_r` yazili OLMALI -- otherwise mayin geri acilir."""
    import json ,pickle 
    kok =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
    with open (os .path .join (kok ,"cp_config.json"),encoding ="utf-8")as f :
        cfg =json .load (f )
    if not cfg .get ("gate_topo_feats"):
        pytest .skip ("topoloji kapali")
    path =os .path .join (kok ,"results","wire_gate.pkl")
    if not os .path .exists (path ):
        pytest .skip ("dagitilmis gate none")
    with open (path ,"rb")as f :
        m =pickle .load (f )
    assert m .get ("topo_r")is not None ,"dagitilan gate topoloji kullaniyor but egitildigi yaricapi TASIMIYOR"
    assert float (m ["topo_r"])==float (cfg .get ("gate_topo_r",6.0 )),f"model {m ['topo_r']} mm with egitildi, config {cfg .get ('gate_topo_r',6.0 )} mm diyor"


def test_cokus_yonlendirme_ESIGE_gore_model_degistirir ():
    """Yonlendirme SADECE cokuste tetiklenmeli.

    Bu, urunun most ince davranisi: z-score modeli each parcaya uygulanirsa saglikli ureticide
    vergi odenir (measured: PXC-disi -0.0375). Esigin YANLIS yonde calismasi this vergiyi geri
    getirirdi and total F1'de small gorunecegi for gozden kacardi."""
    wg =_wg ()

    class _Sahte :
        def __init__ (self ,val_ ):self .val_ =val_ 
        def predict_proba (self ,X ):
            import numpy as _np 
            return _np .column_stack ([1 -_np .full (len (X ),self .val_ ),
            _np .full (len (X ),self .val_ )])

    m ={"clf_z":_Sahte (0.77 ),"esik_cokus":0.50 ,"donusum_z":"zskor"}
    X =np .random .default_rng (0 ).normal (size =(4 ,3 ))
    saglikli =np .array ([0.9 ,0.2 ,0.1 ,0.1 ])# maks 0.9 >= 0.50 -> HAM kalmali
    assert np .allclose (wg ._cokus_yonlendir (m ,X ,saglikli ),saglikli )
    cokus =np .array ([0.3 ,0.2 ,0.1 ,0.1 ])# maks 0.3 < 0.50 -> Z modele gecmeli
    assert np .allclose (wg ._cokus_yonlendir (m ,X ,cokus ),0.77 )


def test_yonlendirme_yoksa_skor_AYNEN_doner ():
    """Eski (yonlendirmesiz) models degismeden calismali."""
    wg =_wg ()
    s =np .array ([0.4 ,0.9 ])
    X =np .zeros ((2 ,3 ))
    for m in ({},{"clf_z":None ,"esik_cokus":0.5 },{"clf_z":object (),"esik_cokus":None }):
        assert np .allclose (wg ._cokus_yonlendir (m ,X ,s ),s )
    assert len (wg ._cokus_yonlendir ({"clf_z":object (),"esik_cokus":0.5 },X ,np .array ([])))==0 


def test_dagitilan_model_yonlendirme_esigini_TASIYOR ():
    import json ,pickle 
    kok =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
    with open (os .path .join (kok ,"cp_config.json"),encoding ="utf-8")as f :
        cfg =json .load (f )
    if "yonlendir"not in str (cfg .get ("gate_parca_ici","")):
        pytest .skip ("yonlendirme dagitilmamis")
    with open (os .path .join (kok ,"results","wire_gate.pkl"),"rb")as f :
        m =pickle .load (f )
    assert m .get ("clf_z")is not None ,"yonlendirme dagitik but z modeli YOK"
    assert 0.0 <float (m ["esik_cokus"])<1.0 ,m .get ("esik_cokus")
