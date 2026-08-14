# -*- coding: utf-8 -*-
"""OLCUM TUZAKLARI -- each biri gercekten yasandi, each biri wrong a hukme path acti.

Bunlar 'kod calisiyor mu' testi not; 'olctugum sey olcmek istedigim sey mi' testi.
"""
import os 
import sys 

import numpy as np 
import pytest 

KOK =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
from sina_cluster import W ,f1_rejim ,f1w 


def _satir (regime ,tp ,fp ,fn ,n =1 ):
    return [(regime ,tp ,fp ,fn )]*n 


def test_f1w_TEK_REJIMDE_o_rejimin_F1ini_verir ():
    """DUZELTILDI 2026-08-03. Bu test ONCE ters yonde civiliyordu.

    Eski hali `f1w(very) == W["very"] * correct` diye ISRAR ediyor and yorumunda "this davranis
    YANLIS DEGIL but TRAP" yaziyordu. Denetimde bit-duzeyinde gorulduki YANLIS: headline,
    DOGAL OLARAK single rejimli bolmelerde (that parcalarin all of them low-CP) same carpani yiyordu.
    ATANMAMIS 0.6635 raporlandi, gercegi 0.7413; seri 16 0.8333 -> 0.9310; seri 17
    0.6531 -> 0.7297. Ucunde de rapor/0.895 == dusuk_CP_F1 TAM tutuyordu.

    DOGRU DAVRANIS: only a rejimin verisi varsa corpus-agirlikli a number RAPORLANAMAZ;
    that bolmenin F1'i, present which is rejimin F1'idir. Agirliklar VAR OLAN rejimler uzerinden
    yeniden normalize edilir.
    """
    cok =_satir ("very",3 ,1 ,2 ,10 )
    dogru =f1_rejim (cok )["F1"]["very"]
    assert abs (f1w (cok )-dogru )<1e-9 ,"tek rejimde f1w o rejimin F1'ini vermeli"
    dusuk =_satir ("dusuk",8 ,2 ,1 ,10 )
    assert abs (f1w (dusuk )-f1_rejim (dusuk )["F1"]["dusuk"])<1e-9 


def test_f1_rejim_dogru_kirilim_ve_agirlikli_toplam_verir ():
    rows =_satir ("very",3 ,1 ,2 ,10 )+_satir ("dusuk",8 ,2 ,1 ,10 )
    r =f1_rejim (rows )
    assert r ["n"]=={"dusuk":10 ,"very":10 }
    assert abs (r ["agirlikli_F1"]-f1w (rows ))<1e-9 ,"agirlikli toplam f1w with tutmuyor"
    for k in ("dusuk","very"):
        tek =f1_rejim ([x for x in rows if x [0 ]==k ])["F1"][k ]
        assert abs (r ["F1"][k ]-tek )<1e-9 ,f"{k } kirilimi alt kumeyle tutmuyor"


def test_f1_rejim_tek_rejimde_agirligi_YENIDEN_NORMALLESTIRIR ():
    """Alt cluster single regime iceriyorsa agirlikli value that rejimin own F1'i must be (0.105 with
    carpilmis hali not) -- f1w'nin tuzagina dusmeyen davranis."""
    cok =_satir ("very",3 ,1 ,2 ,10 )
    r =f1_rejim (cok )
    assert abs (r ["agirlikli_F1"]-r ["F1"]["very"])<1e-9 
    assert r ["F1"]["dusuk"]is None 


def test_carpik_kumede_HAM_ve_AGIRLIKLI_PR_ayrisir ():
    """Puanlanan kumede very-CP %30, korpusta %10.5. Ham P/R korpusu temsil etmez; same tabloda
    agirlikli F1 with ham P/R yan yana konursa IKI FARKLI evren raporlanmis becomes."""
    rows =_satir ("very",1 ,9 ,9 ,30 )+_satir ("dusuk",9 ,1 ,1 ,70 )
    r =f1_rejim (rows )
    assert r ["ham_kesinlik"]<r ["agirlikli_kesinlik"],"carpiklik etkisi kaybolmus"
    assert abs (r ["agirlikli_kesinlik"]-(W ["dusuk"]*0.9 +W ["very"]*0.1 ))<1e-9 


def test_rejim_agirliklari_korpusla_tutuyor ():
    """W, corpus oranlarini temsil eder (low 0.895 / very 0.105). Degisirse headline degisir."""
    assert abs (sum (W .values ())-1.0 )<1e-9 
    assert abs (W ["dusuk"]-0.895 )<1e-6 and abs (W ["very"]-0.105 )<1e-6 


def test_LOCKED_kirliligi_CIKARILANLARI_da_kapsar ():
    """2026-08-02 DENETIM BULGUSU: dogrudan kullanilip puanlamadan CIKARILAN LOCKED parts
    yeniden 'temiz' sayiliyordu.

    Sebep: `kullanilan_geo` only KALAN parcalardan hesaplaniyordu, i.e. cikarilan parcanin
    own grubu "dokunulmamis" gorunuyordu. Ama that part KULLANILDI -- measurement onbelleginde present.
    Bir times dokunulan part KALICI as kirlidir; aksi halde single-atislik exam kirlenir and
    bunu FARK ETMEYIZ.

    Etkisi measured: "temiz LOCKED 98" -> DOGRUSU 95."""
    import measure_set 
    import os 
    if not os .path .exists (os .path .join (
    os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))),
    "results","_der_tam.pkl")):
        import pytest 
        pytest .skip ("measurement onbellegi none")
    _ ,rap =measure_set .cluster ("results/_der_tam.pkl")
    atilan =set (rap ["atilan_locked"])
    assert atilan ,"this testin anlamli olmasi for at least a LOCKED cikarilmali"
    temiz =set (rap ["locked_temiz"])
    assert not (atilan &temiz ),f"DOGRUDAN kullanilan LOCKED parts TEMIZ sayiliyor: {sorted (atilan &temiz )}"
    assert rap ["locked_temiz_n"]<=100 -len (atilan )


def test_SINAV_egitim_maskesi_LOCKED_gruplarini_atar ():
    """FINAL SINAVI for training maskesi LOCKED'in GEOMETRI GRUPLARINI atmali.

    2026-08-03 BULGUSU: dagitilan training verisi (zengin_parite_w2.npz) 95 grup-temiz
    LOCKED parcasinin 77'sini iceriyordu (grup uzerinden 80). Sinav dagitilan gate'le
    kosulsaydi 95 parcanin 80'i KIRLI olurdu -> single atislik hak bosa giderdi.
    Bu test, maskenin gercekten temizledigini garanti eder.
    """
    import numpy as np 
    import measure_set 
    npz =os .path .join (KOK ,"results","zengin_parite_w2.npz")
    if not os .path .exists (npz ):
        pytest .skip ("training npz none")
    d =np .load (npz ,allow_pickle =True )
    pids =np .array ([str (x )for x in d ["pids"]])
    m =measure_set .sinav_egitim_maskesi (pids )
    assert m .sum ()>0 and m .sum ()<len (m ),"maske ya hepsini atiyor ya hicbirini"
    _ ,rap =measure_set .cluster ("results/_der_tam.pkl")
    gk =measure_set .geo_anahtarlari ()
    kalan ={gk .get (x ,"none:"+x )for x in np .unique (pids [m ])}
    ihlal =[x for x in rap ["locked_temiz"]if gk .get (x ,"none:"+x )in kalan ]
    assert not ihlal ,f"maskeden sonra hala {len (ihlal )} LOCKED grubu egitimde: {ihlal [:5 ]}"


def test_f1w_TEK_REJIMLI_bolmede_agirligi_YENIDEN_NORMALLESTIRIR ():
    """Tek rejimli a bolmede f1w, missing rejimi F1=0 sayip 0.895 with CARPMAMALI.

    2026-08-03 denetiminde bit-duzeyinde yakalandi: ATANMAMIS 0.6635 raporlaniyordu,
    gercegi 0.7413 idi (rapor/0.895 == dusuk_CP_F1 TAM tutuyordu). Ayni sekilde
    seri 16 (0.8333 -> 0.9310) and seri 17 (0.6531 -> 0.7297).
    HAVUZLANMIS and manufacturer-disi bolmeler etkilenmemisti (ikisinde de each two regime present).
    """
    from sina_cluster import f1w 
    # only low-CP parts: TP=8 FP=2 FN=2 -> P=0.8 R=0.8 F1=0.8
    tek =[("dusuk",8 ,2 ,2 )]
    assert abs (f1w (tek )-0.8 )<1e-6 ,f"tek rejimde F1 0.8 olmali, {f1w (tek )} geldi"
    # only very-CP parts da same sekilde
    tek_cok =[("very",8 ,2 ,2 )]
    assert abs (f1w (tek_cok )-0.8 )<1e-6 ,f"tek rejimde (cok) 0.8 olmali, {f1w (tek_cok )}"
    # IKI regime varsa AGIRLIKLI mean: 0.895*0.8 + 0.105*0.5
    iki =[("dusuk",8 ,2 ,2 ),("very",5 ,5 ,5 )]
    bek =0.895 *0.8 +0.105 *0.5 
    assert abs (f1w (iki )-bek )<1e-6 ,f"iki rejimde {bek } olmali, {f1w (iki )} geldi"
