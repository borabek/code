# -*- coding: utf-8 -*-
"""D5-2b: 4 VARDIYANIN CIKTISINI BIRLESTIR + KAYIP PARCALARI SAY (D5-5b'nin girdisi).

Turetme 4 paralel vardiyada kostu (`d5_derive_new.py --vardiya k --total 4`); each biri
own `_der_yeni_<k>.pkl` and `_geo_yeni_<k>.json` dosyasina yazdi. Bu betik onlari TEK
dosyada toplar and **kimin dustugunu** removes.

WHY KAYIP LISTESI AYRI CIKARILIYOR: turetme betigi hatali parcalari only EKRANA
yaziyordu (first 5'ini). Kosan surecleri bozmamak for betigi degistirmedim; loss listesi
already `eligible()` with turetilenlerin FARKI as full and conclusive sekilde geri alinabilir.

TEKILLIK: same pid two vardiyada olmamali (indeks % 4 with bolundu). Yine de kontrol edilir
and cakisma varsa YUKSELTILIR -- silent double-weight [[geometry-twin-leakage]] up to sinsi.
"""
import collections 
import glob 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CIKTI ="results/_der_yeni.pkl"
GEO ="results/_geo_new.json"
KAYIP ="results/_d5_kayip.json"


def main ():
    import protocol 
    protocol .tez_dogrula ()
    from big_arbiter import eligible 

    OUT ,GEOD ,nerede =[],{},{}
    kirpilan =0 
    kopya_top =[0 ]
    # VARDIYA SAYISI SABIT DEGIL (2026-08-05): `range(4)` yaziliydi; that gun turetme before
    # 4, after 8 vardiyayla, ardindan two ayri KURTARMA turuyla (`_dev_*`) was run.
    # Sabit 4 with birlestirme shard'larin YARISINDAN COGUNU sessizce dusururdu -- error
    # vermeden, only "data kolu whereas yaramadi" like gorunerek. Artik diskte NE VARSA
    # that okunur; `nerede` sozlugu kopyalari already eliyor.
    kaynaklar =sorted (glob .glob ("results/_der_yeni_*.pkl"))
    kaynaklar =[f for f in kaynaklar if "_g6"not in os .path .basename (f )]
    print (f"BIRLESTIRILECEK SHARD: {len (kaynaklar )}")
    for f in kaynaklar :
        k =os .path .basename (f )[len ("_der_yeni_"):-len (".pkl")]
        if not os .path .exists (f ):
            print (f"  vardiya {k }: DOSYA YOK");continue 
        with open (f ,"rb")as h :
            R =pickle .load (h )
            # CAKISMA: onceden `assert` idi and GOREVINI YAPTI -- 2026-08-04'te gercekten atesledi.
            # Sebep: vardiya split hatasi duzeltilmeden ONCEKI kosularda each vardiya only own
            # ciktisini "already present" saydigi for lists farklilasmis and some parts IKI
            # vardiyada turetilmisti. Kayitlar wrong DEGIL (same part, same boru hatti) but
            # kopya kayit egitimde CIFT AGIRLIK yaratir -- [[geometry-twin-leakage]] up to sinsi.
            # Artik: first gelen tutulur, kopya ATILIR and count GURULTULU sekilde raporlanir.
        cak =[r ["pid"]for r in R if r ["pid"]in nerede ]
        if cak :
            print (f"    KOPYA: {len (cak )} part already baska vardiyada present, atiliyor "
            f"(ornek {cak [:3 ]})")
            kopya_top [0 ]+=len (cak )
            R =[r for r in R if r ["pid"]not in nerede ]
        for r in R :
            nerede [r ["pid"]]=k 
            # OZNITELIK GENISLIGI NORMALIZASYONU (2026-08-04'te yakalandi):
            # `d5_derive_new.py` WG_ZENGIN=1 with kostugu for `feats_for` zengin sutunlari
            # da X'e EKLIYOR -> X 58 column cikiyor. Oysa measurement korpusu (`_der_tam.pkl`) and
            # DAGITILAN gate 22 bekliyor (n_feat 116 = (22+36)*2). Normalize edilmezse new
            # corpus ne eskisiyle birlestirilebilir ne de puanlanabilir.
            # MEASURED: 984/984 kayitta `X[:, 22:]` with `XR` BIREBIR AYNI (maks difference 0.0),
            # i.e. 58 column 22'nin UST KUMESI and dilimleme KAYIPSIZ -- yeniden turetme YOK.
            X =r .get ("X")
            if X is not None and X .shape [1 ]>22 :
                assert X .shape [1 ]==22 +r ["XR"].shape [1 ],(
                f"{r ['pid']}: beklenmeyen genislik {X .shape [1 ]}")
                assert np .abs (X [:,22 :]-r ["XR"]).max ()<1e-9 ,(
                f"{r ['pid']}: X kuyrugu XR with same DEGIL -- dilimleme KAYIPLI olur")
                r ["X"]=np .ascontiguousarray (X [:,:22 ])
                kirpilan +=1 
        OUT +=R 
        g =f"results/_geo_yeni_{k }.json"
        if os .path .exists (g ):
            GEOD .update (json .load (io .open (g ,encoding ="utf-8")))
        print (f"  vardiya {k }: {len (R )} part")

    with open (CIKTI ,"wb")as f :
        pickle .dump (OUT ,f )
    with io .open (GEO ,"w",encoding ="utf-8")as f :
        json .dump (GEOD ,f )
    print (f"\nBIRLESIK: {len (OUT )} part -> {CIKTI }")
    print (f"  geometri anahtari: {len (GEOD )} part -> {GEO }")

    # --- KAYIP: hedeflenmis but turetilememis parts
    present =set ()
    d =np .load ("results/zengin_parite_w2.npz",allow_pickle =True )
    present |={str (x )for x in d ["pids"]}
    for y in ("results/_der_tam.pkl","results/_der_kontrol.pkl"):
        if os .path .exists (y ):
            with open (y ,"rb")as f :
                present |={r ["pid"]for r in pickle .load (f )}
    turetilen ={r ["pid"]for r in OUT }
    E =eligible ()
    loss =[(m ,p ,s )for m ,p ,_ ,s in E if p not in present and p not in turetilen ]
    c =collections .Counter (m for m ,_ ,_ in loss )
    hedef =len ([1 for m ,p ,_ ,s in E if p not in present ])
    print (f"\nKAYIP: {len (loss )} / {hedef } hedef part "
    f"(%{100 *len (loss )/max (hedef ,1 ):.1f})")
    print (f"  manufacturer: {dict (c .most_common (8 ))}")
    with io .open (KAYIP ,"w",encoding ="utf-8")as f :
        json .dump ([{"mfg":m ,"pid":p ,"stp":s }for m ,p ,s in loss ],f ,indent =1 )
    print (f"  list -> {KAYIP }  (D5-5b bunu ucuncu mesh kademesiyle yeniden dener)")

    cm =collections .Counter (r ["mfg"]for r in OUT )
    print (f"\n  manufacturer dagilimi: {dict (cm .most_common (10 ))}")
    print (f"  GT total {sum (r ['n']for r in OUT )} | candidate {sum (len (r ['P'])for r in OUT )}")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
