# -*- coding: utf-8 -*-
"""PAYLI OLCUM MAKBUZLARINI BIRLESTIR.

Kullanim:  python birlestir_makbuz.py <output.json> <pay1.json> <pay2.json> ...

Mikro F1 part basina TP/FP/FN toplami oldugu for birlestirme KAYIPSIZDIR;
paylara bolup after toplamak single islemde kosmakla AYNI sayiyi gives. Bu betik
also same parcanin two payda birden puanlanip puanlanmadigini CHECK EDER --
oyle a cakisma sessizce sayilari sisirirdi.
"""
import collections 
import json 
import sys 

import numpy as np 


def f1 (tp ,fp ,fn ):
    return 2 *tp /max (2 *tp +fp +fn ,1 )


def main ():
    if len (sys .argv )<3 :
        sys .exit (__doc__ )
    cik ,paylar =sys .argv [1 ],sys .argv [2 :]
    kir ={}
    cakisan =[]
    damga =None 
    bayrak ={}
    sayac =collections .Counter ()
    for y in paylar :
        m =json .load (open (y ))
        damga =damga or m .get ("damga")
        s =m ["sonuc"]
        for k in ("p6_acik","genis_acik","poz_kafasi"):
            if k in s :
                bayrak .setdefault (k ,s [k ])
                if bayrak [k ]!=s [k ]:
                    sys .exit (f"PAYLAR FARKLI AYARDA: {k } {bayrak [k ]} vs {s [k ]}")
        sayac .update (s .get ("p6_sayac")or {})
        for p ,v in (s .get ("parca_kirilim")or s .get ("parca_tp_fp_fn")or 
        {}).items ():
            if p in kir :
                cakisan .append (p )
            kir [p ]=v 
    if cakisan :
        print (f"UYARI: {len (cakisan )} part birden fazla payda puanlanmis "
        f"(ilk 5: {cakisan [:5 ]}). Paylar ORTUSUYOR.")

    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[0 ,0 ,0 ]
    for p ,v in kir .items ():
        a =rob [v ["mfg"]]
        for i in range (3 ):
            a [i ]+=v ["rob"][i ]
            tes [i ]+=v ["tes"][i ]
    pm ={m :f1 (*a )for m ,a in rob .items ()}
    T =[sum (a [i ]for a in rob .values ())for i in range (3 )]
    out ={"robot":f1 (*T ),"tespit":f1 (*tes ),
    "makro":float (np .mean (list (pm .values ()))),
    "en_kotu":float (min (pm .values ())),"brand":pm ,
    "TP":T [0 ],"FP":T [1 ],"FN":T [2 ],
    "n_parca":len (kir ),"parca_kirilim":kir ,
    "p6_sayac":dict (sayac ),**bayrak }
    json .dump ({"damga":damga ,"sonuc":out ,
    "not":f"{len (paylar )} paydan birlestirildi. Mikro F1 part "
    "basina total oldugu for birlestirme kayipsizdir."},
    open (cik ,"w"),indent =1 )
    print (f"{len (kir )} part | robot {out ['robot']:.4f} | tespit "
    f"{out ['tespit']:.4f} | makro {out ['makro']:.4f} | "
    f"TP {T [0 ]} FP {T [1 ]} FN {T [2 ]}")
    if sayac .get ("cagri"):
        p6 =sayac .get ("p6",0 )
        print (f"P6 kolu: {p6 }/{sayac ['cagri']} parcada CALISTI "
        f"({100 *p6 /sayac ['cagri']:.1f}%)"
        +(f" | dusenler: {dict ((k ,v )for k ,v in sayac .items ()if k .endswith ('_yok')and v )}"
        if p6 <sayac ["cagri"]else ""))
        if p6 ==0 and bayrak .get ("p6_acik"):
            print ("!! P6 OPEN but HIC CALISMAMIS -- this number TABAN sayisidir.")
    print (f"receipt -> {cik }")


if __name__ =="__main__":
    main ()
