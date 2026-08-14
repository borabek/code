# -*- coding: utf-8 -*-
"""TIER COKUSU: "GLB'deki kirmizi (AUTO) isaretlerin kaci correct?"

FINDING (2026-08-12). Dagitilan AUTO esiginde (`cp_config.robot_auto_gate_threshold`
= 0.6) GORULMEMIS MARKADA isaretlerin **%100'u AUTO** oluyor; REVIEW katmani
BOS kaliyor. Yani two katmanli guvenlik mekanizmasi ATIL: robot each isarete
own basina guveniyor, oysa isaretlerin however ucte biri correct.

WHY COKUYOR. Secim kurali with tier esigi AYNI skoru kullaniyor. Secim kurali
already goreli (part-maksimumunun %85'i) oldugu for hayatta kalan each tahminin
skoru high; tier esigi no seyi elemiyor. Esik "baglamiyor".

TARIHCE. Ayni cokus 2026-07-29'da a times yasanmis and duzeltilmisti (that zaman
tier SEGMENTASYON guvenine bakiyordu, REVIEW empty cikmisti, precision 0.7735).
Duzeltme skoru degistirdi but COKUS BICIMI geri geldi -- this sefer gorulmemis
brand kosulunda and very more low kesinlikle.

BU BETIK YENI BIR D7 OKUMASI DEGILDIR: harcanmis olcumun makbuzlarini yeniden
reads, model secimi/ayar yapmaz.

Kullanim:  python probe_tier_cokusu.py
"""
import glob 
import json 
import os 
import sys 

import numpy as np 

ESIKLER =(0.0 ,0.3 ,0.5 ,0.6 ,0.66 ,0.7 ,0.8 ,0.9 ,0.95 )


def cift (y ):
    """receipt -> (skorlar, correct, GT count)."""
    try :
        d =json .load (open (y ,encoding ="utf-8"))
    except Exception :
        return None 
    s =d .get ("sonuc",{})
    kir =s .get ("parca_kirilim")or s .get ("parca_tp_fp_fn")
    if not kir :
        return None 
    S ,Y ,gt =[],[],0 
    gercek =True 
    for v in kir .values ():
        gt +=v ["rob"][0 ]+v ["rob"][2 ]
        sk =v .get ("skor")or []
        dg =v .get ("dogru")or []
        # `skor_gercek` new makbuzlarda present; old makbuzlarda YOK and orada
        # dejenere distribution kontrolu devreye girer.
        if sk and not v .get ("skor_gercek",True ):
            gercek =False 
        if sk and len (sk )==len (dg ):
            S +=list (sk )
            Y +=list (dg )
    if not S :
        return None 
    return np .asarray (S ,float ),np .asarray (Y ,bool ),gt ,len (kir ),gercek 


def tablo (ad ,S ,Y ,gt ):
    print (f"\n### {ad }   ({len (S )} sign, {gt } GT)")
    print (f"{'threshold':>6}{'AUTO':>8}{'AUTO pay':>10}{'precision':>10}"
    f"{'GT kapsama':>12}")
    sat =[]
    for t in ESIKLER :
        m =S >=t 
        if not m .sum ():
            continue 
        r ={"threshold":t ,"n":int (m .sum ()),"pay":float (m .mean ()),
        "precision":float (Y [m ].mean ()),
        "gt_kapsama":float (Y [m ].sum ()/max (gt ,1 ))}
        sat .append (r )
        print (f"{t :>6.2f}{r ['n']:>8d}{r ['pay']:>10.4f}{r ['precision']:>10.4f}"
        f"{r ['gt_kapsama']:>12.4f}")
    return sat 


def main ():
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    dag =float (cfg .get ("robot_auto_gate_threshold",0.6 ))
    print (f"DAGITILAN AUTO ESIGI = {dag }")
    out ={"dagitilan_esik":dag ,"kumeler":{}}
    for y in sorted (glob .glob ("results/d7_p6.json")+
    glob .glob ("results/d7_taban.json")):
        r =cift (y )
        if not r :
            continue 
        S ,Y ,gt ,npar ,gercek =r 
        ad =os .path .basename (y ).replace (".json","")
        if not gercek :
            print (f"\n### {ad }  ({npar } part) -- OLCULMEMIS")
            print ("  receipt `skor_gercek=false` diyor: zincir wire_score "
            "uretmemis.")
            out ["kumeler"][ad ]={"n_parca":npar ,"n_isaret":int (len (S )),
            "durum":"OLCULMEMIS_skor_gercek_false"}
            continue 
            # VARSAYILAN DOLGU TUZAGI: `probe_dagitim_verify` skoru
            # `c.get("wire_score", 1.0)` with okuyor. Zincir wire_score URETMIYORSA
            # each tahmine 1.0 yazilir and tablo "each esikte %100 AUTO" like gorunur.
            # Bu a FINDING DEGIL, measurement bosllugudur -- ayirt edilmezse tier cokusu
            # diye raporlanir.
        if len (np .unique (S ))==1 and float (S [0 ])==1.0 :
            print (f"\n### {ad }  ({npar } part) -- OLCULMEMIS")
            print (f"  {len (S )} tahminin HEPSI tam 1.0: zincir wire_score "
            f"uretmemis, probe varsayilani yazmis. Bu kumede tier "
            f"davranisi hakkinda HICBIR SEY SOYLENEMEZ.")
            out ["kumeler"][ad ]={"n_parca":npar ,"n_isaret":int (len (S )),
            "durum":"OLCULMEMIS_varsayilan_dolgu"}
            continue 
        sat =tablo (f"{ad }  ({npar } part)",S ,Y ,gt )
        print (f"  skor dagilimi: min {S .min ():.4f}  medyan "
        f"{np .median (S ):.4f}  maks {S .max ():.4f}")
        if S .min ()>=dag :
            print (f"  !! DAGITILAN ESIK ({dag }) SKOR TABANININ ALTINDA "
            f"({S .min ():.4f}) -- threshold hicbir seyi elemiyor.")
        m =S >=dag 
        out ["kumeler"][ad ]={
        "n_parca":npar ,"n_isaret":int (len (S )),"gt":int (gt ),
        "dagitilan_esikte":{
        "auto_pay":float (m .mean ()),
        "precision":float (Y [m ].mean ())if m .sum ()else None ,
        "review_bos":bool (m .mean ()>=0.999 )},
        "egri":sat }
        if m .mean ()>=0.999 :
            print (f"  !! REVIEW KATMANI BOS: isaretlerin %100'u AUTO, "
            f"precision {Y [m ].mean ():.4f}")

    json .dump (out ,open ("results/tier_cokusu_d7.json","w"),indent =1 )
    print ("\nmakbuz -> results/tier_cokusu_d7.json")
    print ("\nNOT: yeni D7 OKUMASI DEGIL -- harcanmis olcumun yeniden analizi.")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
