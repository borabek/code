# -*- coding: utf-8 -*-
"""MANSET: a measurement makbuzundan DURUST rapor produces.

Kullanim:  python manset_050.py results/<receipt>.json [results/<baseline>.json]

Tek a number vermez. Bir sayinin sisik olup olmadigini anlamak for gereken
each seyi yan yana koyar:

  1. MIKRO robot F1 (headline) and tespit -- havuzlanmis TP/FP/FN uzerinden.
     `f1w` (regime-agirlikli) KULLANILMAZ: same veride 0.0838 vs 0.1956 gives.
  2. PARCA BOOTSTRAP confidence araligi (%95). Parca ornekler, because mikro F1 part
     basina TP/FP/FN toplamidir.
  3. MARKA BAZINDA kirilim + makro + most kotu brand. Tek a markanin very GT'si
     mikro'yu tasiyabilir (D6'da NIT GT'nin %46'si); brand tablosu bunu gorunur
     kilar.
  4. TEMIZ ALT KUME duyarliligi: `results/bolme_denetimi.json` icindeki, no
     training parcasiyla kaba iz (kutu 0.5mm + GT count) paylasmayan D7 parcalari.
     Manset with this lower cluster arasindaki difference buyukse number supheli demektir.
  5. TABAN verilirse FARK and markalarin kaci artida.

Makbuz `parca_kirilim` alanini icermelidir (pid -> {mfg, rob:[tp,fp,fn],
tes:[tp,fp,fn]}). `probe_dagitim_verify.py` and `probe_d6_product.py` writes.
"""
import json 
import os 
import sys 

import numpy as np 


def f1 (tp ,fp ,fn ):
    return 2 *tp /max (2 *tp +fp +fn ,1 )


def topla (kir ,pidler =None ,alan ="rob"):
    tp =fp =fn =0 
    for p ,v in kir .items ():
        if pidler is not None and p not in pidler :
            continue 
        a =v [alan ]
        tp +=a [0 ];fp +=a [1 ];fn +=a [2 ]
    return tp ,fp ,fn 


def bootstrap (kir ,pidler =None ,alan ="rob",n =4000 ,seed =0 ):
    """PARCA bootstrap. Grup bootstrap'a gerek absent: split denetimi D7 inside
    kaba-iz ikiz oranini %22.3 olcmustu and ikizler AYNI markadadir; part
    ornekleme here confidence araligini daraltmaz, because unit already parcadir."""
    ps =sorted (kir )if pidler is None else sorted (pidler )
    A =np .asarray ([kir [p ][alan ]for p in ps ],float )
    rng =np .random .default_rng (seed )
    v =[]
    for _ in range (n ):
        i =rng .integers (0 ,len (A ),len (A ))
        s =A [i ].sum (0 )
        v .append (f1 (s [0 ],s [1 ],s [2 ]))
    v =np .asarray (v )
    return float (np .percentile (v ,2.5 )),float (np .percentile (v ,97.5 ))


def marka_tablo (kir ,pidler =None ,alan ="rob"):
    d ={}
    for p ,v in kir .items ():
        if pidler is not None and p not in pidler :
            continue 
        a =d .setdefault (v ["mfg"],[0 ,0 ,0 ,0 ])
        a [0 ]+=v [alan ][0 ];a [1 ]+=v [alan ][1 ];a [2 ]+=v [alan ][2 ];a [3 ]+=1 
    return {m :{"f1":f1 (*a [:3 ]),"n":a [3 ],"TP":a [0 ],"FP":a [1 ],
    "FN":a [2 ]}for m ,a in d .items ()}


def main ():
    if len (sys .argv )<2 :
        sys .exit (__doc__ )
    yol =sys .argv [1 ]
    m =json .load (open (yol ))
    kir =m ["sonuc"].get ("parca_kirilim")or m ["sonuc"].get ("parca_tp_fp_fn")
    if not kir :
        sys .exit ("makbuzda `parca_kirilim` absent -- olcumu yeniden kos.")
    baseline =None 
    if len (sys .argv )>2 and os .path .exists (sys .argv [2 ]):
        t =json .load (open (sys .argv [2 ]))
        baseline =t ["sonuc"].get ("parca_kirilim")or t ["sonuc"].get ("parca_tp_fp_fn")

    print (f"MAKBUZ  {yol }")
    print (f"arm     P6 {m ['sonuc'].get ('p6_acik')} | genis "
    f"{m ['sonuc'].get ('genis_acik')} | poz kafasi "
    f"{m ['sonuc'].get ('poz_kafasi')}")
    print (f"part   {len (kir )}")
    sy =m ["sonuc"].get ("p6_sayac")or {}
    if sy .get ("cagri"):
        p6 =sy .get ("p6",0 )
        print (f"P6 kolu {p6 }/{sy ['cagri']} parcada CALISTI "
        f"({100 *p6 /sy ['cagri']:.1f}%) | regime disi "
        f"{sy .get ('rejim_disi',0 )} | tablo yok {sy .get ('tablo_yok',0 )}")
        if p6 ==0 :
            print ("!! P6 HIC CALISMAMIS -- this number TABAN sayisidir.")
    print ()

    for alan ,ad in (("rob","ROBOT (lateral<=2mm, signed angle<=10)"),
    ("tes","TESPIT (konum, direction serbest)")):
        tp ,fp ,fn =topla (kir ,alan =alan )
        a ,b =bootstrap (kir ,alan =alan )
        print (f"{ad }")
        print (f"  MIKRO F1 {f1 (tp ,fp ,fn ):.4f}   %95 GA [{a :.4f}, {b :.4f}]"
        f"   TP {tp }  FP {fp }  FN {fn }"
        f"   recall {tp /max (tp +fn ,1 ):.4f}"
        f"   precision {tp /max (tp +fp ,1 ):.4f}")

    mt =marka_tablo (kir )
    print (f"\n{'brand':<8}{'n':>5}{'robot F1':>10}{'TP':>7}{'FP':>7}{'FN':>7}"
    +("{:>10}".format ("baseline")if baseline else ""))
    tb =marka_tablo (baseline )if baseline else {}
    art =0 
    for k ,v in sorted (mt .items (),key =lambda x :-x [1 ]["FN"]-x [1 ]["TP"]):
        line_ =(f"{k :<8}{v ['n']:>5}{v ['f1']:>10.4f}{v ['TP']:>7}{v ['FP']:>7}"
        f"{v ['FN']:>7}")
        if baseline and k in tb :
            line_ +=f"{tb [k ]['f1']:>10.4f}"
            art +=int (v ["f1"]>tb [k ]["f1"])
        print (line_ )
    print (f"{'MAKRO':<8}{'':>5}{np .mean ([v ['f1']for v in mt .values ()]):>10.4f}")
    print (f"{'EN KOTU':<8}{'':>5}{min (v ['f1']for v in mt .values ()):>10.4f}")
    if baseline :
        t_tp ,t_fp ,t_fn =topla (baseline )
        n_tp ,n_fp ,n_fn =topla (kir )
        print (f"\nTABAN {sys .argv [2 ]}")
        print (f"  robot {f1 (t_tp ,t_fp ,t_fn ):.4f} -> {f1 (n_tp ,n_fp ,n_fn ):.4f}"
        f"   ({f1 (n_tp ,n_fp ,n_fn )-f1 (t_tp ,t_fp ,t_fn ):+.4f})"
        f"   {art }/{len (mt )} markada ARTI")

        # --- TEMIZ ALT KUME duyarliligi ---------------------------------------
    bd ="results/bolme_denetimi.json"
    if os .path .exists (bd ):
        temiz =set (json .load (open (bd ))["sonuc"].get ("d7_temiz",[]))
        ort =temiz &set (kir )
        if len (ort )>20 :
            tp ,fp ,fn =topla (kir ,ort )
            a ,b =bootstrap (kir ,ort )
            tam =f1 (*topla (kir ))
            print (f"\nTEMIZ ALT KUME ({len (ort )} part -- hicbir training "
            f"parcasiyla kaba iz paylasmayan)")
            print (f"  robot {f1 (tp ,fp ,fn ):.4f}   %95 GA [{a :.4f}, {b :.4f}]"
            f"   headline farki {f1 (tp ,fp ,fn )-tam :+.4f}")
            print ("  (difference buyukse headline geometri benzerliginden besleniyor "
            "demektir)")


if __name__ =="__main__":
    main ()
