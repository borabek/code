# -*- coding: utf-8 -*-
"""DAGITIM DOGRULAMASI: baglanan path URUNUN ZINCIRINDEN gecince ne veriyor?

Kod yazip olcmeden "deployed" denmez. Bu betik `canonical_chain.product_output`'yi
-- i.e. urunun TEK kanonik zincirini -- cagirir, uzerine poz kafasini uygular
and D7'de olcer. Beklenen: robot ~0.3090 (ayri betiklerde measured_path value).

Fark cikarsa entegrasyonda a sey ayrisiyor demektir and DAGITIM YAPILMAZ.

Iki arm: `URUN_GENIS=0` (old path, beklenen ~0.2029) and `URUN_GENIS=1`.
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

import receipt_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")

OB ="results/_p1_olasilik_d7"
N =int (os .environ .get ("DOG_N","0"))# 0 = tum D7
# POZ KAFASI: dagitilan yolda OPEN (default). P6 kolu yonu KENDISI sectigi
# for poz kafasinin uzerine yazip yazmadigi AYRI olculmelidir -> DOG_POZ=0.
POZ =os .environ .get ("DOG_POZ","1")!="0"


def main ():
    import canonical_d7 as K 
    import canonical_chain 
    import product_wide 
    import product_chain 
    from sina_cluster import match_hungarian 

    S =K .step_map ()
    pids =sorted (f [:-4 ]for f in os .listdir (OB )if f .endswith (".npz"))
    kay =K .yukle (pids )
    secili =[p for p in pids if p in kay and len (kay [p ].get ("G",[]))]
    if N :
        secili =secili [:N ]
        # PAYLI KOSU: `DOG_SHARD=i/n`. P6 kolu part basina saniyeler suruyor
        # (B-rep + ~2000 option for isin atisi); 835 part single islemde saatler
        # takes. Paylar `merge_receipt.py` with birlestirilir; mikro F1 part
        # basina TP/FP/FN toplami oldugu for birlestirme KAYIPSIZDIR.
    sh =os .environ .get ("DOG_SHARD")
    if sh :
        i_ ,n_ =(int (x )for x in sh .split ("/"))
        secili =[p for k ,p in enumerate (secili )if k %n_ ==i_ ]
        print (f"PAY {i_ }/{n_ }",flush =True )
    print (f"D7 {len (secili )} part | genis arm {'ACIK'if product_wide .ACIK else 'KAPALI'}",
    flush =True )
    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[]
    kolsuz =0 
    # PARCA BAZINDA kirilim: bootstrap confidence araligi olcumu YENIDEN KOSMADAN
    # cikarilabilsin diye. Manset mikro F1'dir; CI same sayidan uretilir.
    parca_kirilim ={}
    for i ,pid in enumerate (secili ,1 ):
        r =kay [pid ]
        z =np .load (f"{OB }/{pid }.npz")
        V =np .ascontiguousarray (z ["V"],np .float64 )
        F =np .ascontiguousarray (z ["F"],np .int64 )
        pbs =[np .asarray (q ,float )for q in z ["pbs"]]
        cps =canonical_chain .product_output (V ,F ,pbs ,S .get (pid ))
        P ,D =canonical_chain .poz_ver (cps )
        # TAHMIN BASINA SKOR: confidence kapili GLB'nin kalibrasyonu this makbuzdan
        # cikar. Boylece precision-kapsama egrisi for AYRI a D7 okumasi
        # gerekmez -- single okuma two cevap gives.
        # !! VARSAYILAN DOLGU TUZAGI (2026-08-12'de yakalandi): asagidaki
        # `1.0` varsayilani, zincir wire_score URETMEDIGINDE each tahmine 1.0
        # writes. Sonuc makbuzda real score like durur and precision-kapsama
        # analizinde "each esikte %100 AUTO" diye okunur -- i.e. OLCUM BOSLUGU
        # FINDING sanilir. (d7_baseline.json'da 2001 skorun all of them full 1.0 boyle
        # olusmustu.) Skorun GERCEK olup olmadigi residual also yaziliyor.
        score =[float (c .get ("wire_score",1.0 ))for c in (cps or [])]
        skor_gercek =bool (cps )and all ("wire_score"in c for c in cps )
        if len (P )and POZ :
            P ,D =product_chain .tam_poz (V ,F ,np .mean (pbs ,axis =0 ),P ,D ,
            step_path =S .get (pid ))
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        dg =float (r ["diag"])
        tp ,fp ,fn ,bilgi =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
        signed =True )
        a =rob [r ["mfg"]]
        a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
        t_ =match_hungarian (P ,D ,G ,Gd ,dg ,max (3.0 ,0.06 *dg ),180.0 ,True )[:3 ]
        tes .append ((len (G ),)+t_ )
        matched ={int (e [0 ])for e in bilgi .get ("eslesme",[])}
        parca_kirilim [pid ]={
        "mfg":r ["mfg"],"rob":[tp ,fp ,fn ],
        "tes":[int (x )for x in t_ ],
        # (score, dogru_mu) ciftleri -> precision-kapsama egrisi
        "score":[round (s ,5 )for s in score [:len (P )]],
        "skor_gercek":skor_gercek ,
        "correct":[int (j in matched )for j in range (len (P ))]}
        if i %100 ==0 :
            print (f"  {i }/{len (secili )}",flush =True )
    pm ={m :2 *v [0 ]/max (2 *v [0 ]+v [1 ]+v [2 ],1 )for m ,v in rob .items ()}
    mi =float (2 *sum (v [0 ]for v in rob .values ())/
    max (sum (2 *v [0 ]+v [1 ]+v [2 ]for v in rob .values ()),1 ))
    import product_p6 
    out ={"robot":mi ,"detection":K .mikro (tes ),
    "makro":float (np .mean (list (pm .values ()))),
    "en_kotu":float (min (pm .values ())),"brand":pm ,
    "n_parca":len (secili ),"genis_acik":bool (product_wide .ACIK ),
    "p6_acik":bool (product_p6 .ACIK ),"poz_kafasi":bool (POZ ),
    "p6_sayac":dict (product_p6 .SAYAC ),
    "parca_tp_fp_fn":{p :v for p ,v in parca_kirilim .items ()}}
    print (f"\nURUN ZINCIRI robot {mi :.4f} | detection {out ['detection']:.4f} | "
    f"makro {out ['makro']:.4f} | en kotu {out ['en_kotu']:.4f}")
    print (f"BEKLENEN: genis ACIK ~0.3090 | genis KAPALI ~0.2029")
    json .dump ({"damga":receipt_hash .damga (),"sonuc":out ,
    "not":"Urunun TEK kanonik zinciri (`canonical_chain.product_output`) "
    "+ poz kafasi. D7 brand-disi, MIKRO."},
    open (os .environ .get ("DOG_CIKTI",
    "results/dagitim_dogrula.json"),"w"),indent =1 )
    print ("receipt yazildi")


if __name__ =="__main__":
    main ()
