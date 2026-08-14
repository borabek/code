"""g10 KILL KAPISI: sig boyama + duzeltilmis secim sinyali, g7'yi geciyor mu?

Olcut: ADAY KAHINI -- gate'ten ONCE, uretilen candidate havuzunun GT'yi ne up to
kapsadigi. Bire-a Macar, TESPIT toleransi, angle serbest. (robot_cp.derive_candidates
notundaki olcutle same: "candidate kahini one-to-one Macar".)

WHY KAHIN, WHY F1 DEGIL: g10 only SEGMENTASYONU degistirdi. Gate and p3c
old dagilimda egitildi; uctan uca F1 dusukse this g10'un kotu oldugunu DEGIL,
gate'in yeniden fit edilmedigini gosterir. Kahin, sonraki katmanlardan bagimsiz
as temsilin iyilesip iyilesmedigini olcer.

KILL: g10 kahini g7'yi GECMEZSE sig boyama GERI ALINIR.
"""
import os 
import sys 
import json 
import glob 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")

import d6_record # noqa: E402
import robot_cp # noqa: E402
from sina_cluster import match_hungarian ,f1w # noqa: E402
from corpus_identity import step_kimlik as SK # noqa: E402

KOLLAR ={"g10":"results/_p1_olasilik_g10","b9":"results/_p1_olasilik_b9"}
CIKTI ="results/b9_kill_kapisi.json"


def main ():
    sv =d6_record .exam ()
    rec_ =d6_record .yukle (set (sv ["pidler"]))
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}

    # SADECE each two onbellekte de bulunan parts -- kiyas same kumede must be
    ortak =None 
    for ad ,yol in KOLLAR .items ():
        p ={f [:-4 ]for f in os .listdir (yol )if f .endswith (".npz")}
        ortak =p if ortak is None else (ortak &p )
    ortak =sorted (ortak &set (rec_ ))
    print (f"ortak part: {len (ortak )}")

    res_ ={}
    for ad ,yol in KOLLAR .items ():
        T ,regime =[],{"dusuk":[],"very":[]}
        error =0 # residual only raporlanir; no istisna yutulmaz
        for pid in ortak :
            r =rec_ [pid ]
            G =np .asarray (r ["G"],float )
            Gd =np .asarray (r ["Gd"],float )
            if not len (G ):
                continue 
                # CIPLAK `except Exception` YOK. Ilk denemede vardi and a unpack
                # hatasini (derive_candidates 4 value returns, 3 not -- docstring bayat)
                # 468/468 "error" as yutup ekrana SAHTE a "KILL" karari bastirdi.
                # Bir measurement betigi, olcemedigi zaman DECISION URETMEMELI: patlamali.
            d =np .load (f"{yol }/{pid }.npz")
            V =np .ascontiguousarray (d ["V"],np .float64 )
            F =np .ascontiguousarray (d ["F"],np .int64 )
            pbs =[np .asarray (q ,float )for q in d ["pbs"]]
            cps ,_op ,_cok ,_per =robot_cp .derive_candidates (V ,F ,pbs ,S .get (pid ))
            P =np .asarray ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            D =np .asarray ([c .get ("dir",[0 ,0 ,1 ])for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            oge =(len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ]
            T .append (oge )
            (regime ["very"]if len (G )>=8 else regime ["dusuk"]).append (oge )
        res_ [ad ]={
        "kahin_F1":f1w (T ),
        "dusuk_CP":f1w (regime ["dusuk"])if regime ["dusuk"]else None ,
        "cok_CP":f1w (regime ["very"])if regime ["very"]else None ,
        "n_parca":len (T ),"error":error ,
        }
        s =res_ [ad ]
        _f =lambda v :"  --  "if v is None else f"{v :.4f}"# noqa: E731
        print (f"{ad :>4}: kahin {_f (s ['kahin_F1'])} | dusuk-CP "
        f"{_f (s ['dusuk_CP'])} | cok-CP {_f (s ['cok_CP'])} "
        f"({s ['n_parca']} part, {error } error)",flush =True )

    fark =res_ ["b9"]["kahin_F1"]-res_ ["g10"]["kahin_F1"]
    gecti =fark >0 
    print (f"\nFARK (g10 - g7): {fark :+.4f}")
    print (f"KARAR: {'GECTI -- augmentation KALIR'if gecti else 'KALDI -- augmentation GERI ALINIR'}")
    with open (CIKTI ,"w")as f :
        json .dump ({"sonuc":res_ ,"fark":fark ,"gecti":bool (gecti ),
        "criterion":"candidate kahini, bire-a Macar, tespit toleransi, aci serbest",
        "not":"gate/p3c YENIDEN FIT EDILMEDI; uctan uca F1 this kapiyla "
        "olculmez"},f ,indent =1 )
    print (f"receipt -> {CIKTI }")


if __name__ =="__main__":
    main ()
