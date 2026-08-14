# -*- coding: utf-8 -*-
"""BOLME DENETIMI: training / dev / exam between ne sizmis?

Sonuc a SAYI not, three ayri olcekte cevap -- because "ikiz" tanimi olcege bagli
and single a number vermek yaniltir:

  1. PARCA KIMLIGI    full n d6 n d7 kesisimi
  2. MARKA            brand kumeleri kesisimi
  3. TAM GEOMETRI     vertex count + face count + kutu (0.1mm)
                      -> remesh SURECLER ARASI oynadigi for ikizleri EKSIK sayar
  4. KABA IZ          siralanmis kutu (0.5mm) + GT count
                      -> remesh'ten bagimsiz, but DIN klemensleri STANDART olculu
                         oldugu for FAZLA sayar (different markanin same boy parcasi)

Gercek leakage 3 with 4 ARASINDADIR. Bu betik ikisini de writes and 4'e according to
"kaba iz eslesmesi OLMAYAN" D7 lower kumesini produces; headline next to this lower kumede
de olcup two sayiyi yan yana vermek, single sayiya guvenmekten durusttur.

Cikti: results/bolme_denetimi.json  (+ temiz D7 pid listesi)
"""
import json 
import os 
import sys 

import numpy as np 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402

OZ ="results/_tam_oz"
MESH ={"tam":"results/_p1_olasilik_brepegit","d6":"results/_p1_olasilik",
"d7":"results/_p1_olasilik_d7"}


def pidler (on ):
    return sorted (f [len (on )+1 :-4 ]for f in os .listdir (OZ )
    if f .startswith (on +"_")and f .endswith (".npz"))


def izler (on ,ps ,kay ):
    """Doner: (tam_iz, kaba_iz) sozlukleri."""
    tam ,kaba ={},{}
    for p in ps :
        f =os .path .join (MESH [on ],p +".npz")
        if not os .path .exists (f ):
            continue 
        z =np .load (f )
        V =np .asarray (z ["V"],float )
        b =V .max (0 )-V .min (0 )
        tam [p ]=(len (V ),len (z ["F"]))+tuple (np .round (np .sort (b ),1 ))
        n_gt =len (kay [p ].get ("G",[]))if p in kay else -1 
        kaba [p ]=tuple (np .round (np .sort (b )/0.5 )*0.5 )+(n_gt ,)
    return tam ,kaba 


def main ():
    ps ={on :pidler (on )for on in ("tam","d6","d7")}
    hepsi =[p for v in ps .values ()for p in v ]
    kay =K .yukle (hepsi )
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    mfg ={on :sorted ({kay [p ]["mfg"]for p in v if p in kay })
    for on ,v in ps .items ()}
    T ,Kb ={},{}
    for on in ps :
        T [on ],Kb [on ]=izler (on ,ps [on ],kay )

    def kes (a ,b ,d ):
        return len (set (d [a ].values ())&set (d [b ].values ()))

    d7_kirli =sorted (p for p ,v in Kb ["d7"].items ()
    if v in set (Kb ["tam"].values ()))
    out ={
    "n_parca":{k :len (v )for k ,v in ps .items ()},
    "brand":mfg ,
    "marka_kesisim":{
    "tam-d7":sorted (set (mfg ["tam"])&set (mfg ["d7"])),
    "tam-d6":sorted (set (mfg ["tam"])&set (mfg ["d6"])),
    "d6-d7":sorted (set (mfg ["d6"])&set (mfg ["d7"]))},
    "pid_kesisim":{
    "tam-d7":len (set (ps ["tam"])&set (ps ["d7"])),
    "tam-d6":len (set (ps ["tam"])&set (ps ["d6"])),
    "d6-d7":len (set (ps ["d6"])&set (ps ["d7"]))},
    "tam_geometri_kesisim":{
    "tam-d7":kes ("tam","d7",T ),"tam-d6":kes ("tam","d6",T ),
    "d6-d7":kes ("d6","d7",T )},
    "kaba_iz_kesisim":{
    "tam-d7":kes ("tam","d7",Kb ),"tam-d6":kes ("tam","d6",Kb ),
    "d6-d7":kes ("d6","d7",Kb )},
    "ic_ikiz_orani":{on :1 -len (set (Kb [on ].values ()))/max (len (Kb [on ]),1 )
    for on in ps },
    "d7_kaba_eslesen_parca":len (d7_kirli ),
    "d7_temiz_n":len (Kb ["d7"])-len (d7_kirli ),
    "d7_temiz":sorted (set (Kb ["d7"])-set (d7_kirli )),
    }
    print (f"part      : {out ['n_parca']}")
    print (f"brand kesisim: {out ['marka_kesisim']}")
    print (f"pid kesisim  : {out ['pid_kesisim']}")
    print (f"TAM geometri : {out ['tam_geometri_kesisim']}")
    print (f"KABA iz      : {out ['kaba_iz_kesisim']}")
    print (f"ic ikiz orani: "+
    ", ".join (f"{k } {v :.1%}"for k ,v in out ["ic_ikiz_orani"].items ()))
    print (f"D7: kaba eslesen {out ['d7_kaba_eslesen_parca']} / "
    f"TEMIZ {out ['d7_temiz_n']}")
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,
    "not":"Gercek leakage TAM GEOMETRI (alt sinir) with KABA IZ "
    "(ust sinir) ARASINDADIR. Manset yaninda `d7_temiz` alt "
    "kumesinde de olculur."},
    open ("results/bolme_denetimi.json","w"),indent =1 )
    print ("receipt -> results/bolme_denetimi.json")


if __name__ =="__main__":
    main ()
