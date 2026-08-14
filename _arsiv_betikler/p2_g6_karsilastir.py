# -*- coding: utf-8 -*-
"""P2 DECISION OLCUMU: 2x2 -- AG (g5 vs g6) x YARICAP (mevcut 3/10/5 vs P1 kazanani 1/2/2).

WHY 2x2: ikisini same anda degistirip "iyilesti" demek, hangisinin whereas yaradigini
GIZLER. Ayrica g6 dogrulamada g5'ten DUSUK output (0.5489 vs 0.5584) -- 20 parcalik
val kumesi karar for yetersiz, real soru D6'da candidate KAPSAMASI.

OLCU: ADAY KAHINI (mukemmel gate, one-to-one Macar) + real tespit (dagitilan gate v5).
Kahin gate'ten BAGIMSIZ oldugu for agin temsil gucunu dogrudan gosterir.
"""
import collections ,glob ,io ,json ,os ,sys 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import d6_record 
from p1_birlesme_tarama import pool ,kahin 

MAKBUZ ="results/p2_g6_karsilastir.json"


def main ():
    import protocol 
    protocol .tez_dogrula ()
    from korpus_kimlik import step_kimlik as SK 
    from sina_cluster import f1w 

    sv =d6_record .exam ()
    kayit =d6_record .yukle (set (sv ["pidler"]))
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    sonuc ={}
    print (f"{'ag':<6}{'yaricap':<10}{'KAHIN':>9}{'dusuk':>9}{'cok':>9}{'candidate':>8}{'part':>7}")
    for ag ,kok in (("g5","results/_p1_olasilik"),("g6","results/_p1_olasilik_g6")):
        dosyalar =sorted (glob .glob (os .path .join (kok ,"*.npz")))
        for ad ,(cl ,de ,oy )in (("3/10/5",(3.0 ,10.0 ,5.0 )),("1/2/2",(1.0 ,2.0 ,2.0 ))):
            satir ,n_aday ,n_p =[],0 ,0 
            for f in dosyalar :
                pid =os .path .splitext (os .path .basename (f ))[0 ]
                r =kayit .get (pid )
                if r is None or pid not in S :
                    continue 
                d =np .load (f )
                V =np .ascontiguousarray (d ["V"],np .float64 )
                Fq =np .ascontiguousarray (d ["F"],np .int64 )
                pbs =[np .asarray (q ,float )for q in d ["pbs"]]
                try :
                    cps =pool (V ,Fq ,pbs ,S [pid ],cl ,de ,oy )
                except Exception :
                    continue 
                P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
                n_aday +=len (P );n_p +=1 
                G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
                rj ="cok"if r ["n"]>=8 else "dusuk"
                tp =kahin (P ,G ,Gd ,r ["diag"])if len (G )else 0 
                satir .append ((rj ,tp ,0 ,len (G )-tp ))
            v =f1w (satir )
            dl =f1w ([s for s in satir if s [0 ]=="dusuk"])
            ck =f1w ([s for s in satir if s [0 ]=="cok"])
            print (f"{ag :<6}{ad :<10}{v :>9.4f}{dl :>9.4f}{ck :>9.4f}{n_aday :>8}{n_p :>7}")
            sonuc [f"{ag }|{ad }"]={"kahin":v ,"dusuk":dl ,"cok":ck ,"candidate":n_aday ,
            "part":n_p }
    a =sonuc .get ("g5|3/10/5");b =sonuc .get ("g6|3/10/5")
    c =sonuc .get ("g5|1/2/2");e =sonuc .get ("g6|1/2/2")
    if a and b :
        print (f"\nAG ETKISI (yaricap sabit 3/10/5): g5 {a ['kahin']:.4f} -> g6 "
        f"{b ['kahin']:.4f} = {b ['kahin']-a ['kahin']:+.4f} "
        f"(cok-CP {b ['cok']-a ['cok']:+.4f})")
    if a and c :
        print (f"YARICAP ETKISI (ag sabit g5): {c ['kahin']-a ['kahin']:+.4f} "
        f"(cok-CP {c ['cok']-a ['cok']:+.4f})")
    if a and e :
        print (f"IKISI BIRDEN: {e ['kahin']-a ['kahin']:+.4f} (cok-CP {e ['cok']-a ['cok']:+.4f})")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump (sonuc ,f ,indent =1 )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
