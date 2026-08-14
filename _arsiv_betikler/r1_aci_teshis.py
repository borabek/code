# -*- coding: utf-8 -*-
"""R1: ACIDAKI ~90 DERECE POPULASYONU -- kim, nerede, why?

R0 bulgusu: angle medyani 0.00 deg (i.e. cogunlukla MUKEMMEL) but %90'lik dilim 89.73 and maksimum
90.00. Acidan kaybedilen 162 noktanin %66'si 45-90 between and 90'da YIGILIYOR.

Bu a kestirim gurultusu DEGIL: 90 derece, eksenin DIK secilmesi demektir. Halka bicimli a
agizda normal-kovaryansin most KUCUK ozvektoru eksendir; most BUYUGU alinirsa full 90 derece cikar.
Ama kod correct ozvektoru aliyor -- demek ki ariza belirli a ALT KUMEDE.

SORULAR (all of them olculur, none of them varsayilmaz):
  1. 90-derecelikler belirli PARCALARDA mi toplaniyor, otherwise each yere mi dagilmis?
  2. Ayni parcada hem 0 hem 90 derece present mi? (varsa part genelinde a cerceve sorunu DEGIL)
  3. GT ekseni (Gd) part eksenlerine according to nasil duruyor? Tahmin (Pd) nasil?
  4. 90-derecelikler dar/derin kanallarda mi, sig agizlarda mi? (candidate ozellikleri with bak)
  5. Iki axis DIK whereas, TAHMINI 90 dondurmek "fixes" mi -- i.e. sistematik a axis
     karisikligi mi (a<->b), otherwise rastgele mi?
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def main ():
    with open ("results/r0_kayit.pkl","rb")as f :
        K =pickle .load (f )
    ac =np .array ([e ["angle"]for e in K ])
    kotu =ac >10.0 
    dik =ac >45.0 
    print (f"{len (K )} eslesme | aci>10: {int (kotu .sum ())} | aci>45: {int (dik .sum ())} "
    f"({dik .sum ()/max (kotu .sum (),1 ):.1%} kotulerin)")

    print ("\n1) 90-DERECELIKLER PARCALARDA TOPLANIYOR MU?")
    pp =collections .defaultdict (lambda :[0 ,0 ])
    for e ,b in zip (K ,dik ):
        pp [e ["pid"]][0 ]+=1 ;pp [e ["pid"]][1 ]+=int (b )
    etkilenen ={p :v for p ,v in pp .items ()if v [1 ]}
    print (f"  {len (etkilenen )}/{len (pp )} parcada en az bir dik axis var")
    tam =[p for p ,v in etkilenen .items ()if v [1 ]==v [0 ]]
    kismi =[p for p ,v in etkilenen .items ()if 0 <v [1 ]<v [0 ]]
    print (f"  TAMAMI dik olan part : {len (tam )}  (part genelinde axis sorunu)")
    print (f"  KISMEN dik olan part : {len (kismi )}  (part ici karisik -> cerceve sorunu DEGIL)")
    print (f"  ornek tam-dik parts: {tam [:8 ]}")

    print ("\n2) ACI DAGILIMI (kovalar)")
    for lo ,hi in ((0 ,1 ),(1 ,5 ),(5 ,10 ),(10 ,30 ),(30 ,60 ),(60 ,85 ),(85 ,90.1 )):
        i =(ac >=lo )&(ac <hi )
        print (f"  [{lo :>3}, {hi :>5}) : {int (i .sum ()):>4} ({i .mean ():>6.1%})")

    print ("\n3) DIK OLANLAR nerede? (regime / manufacturer / cluster)")
    for alan in ("regime","mfg","cluster"):
        c =collections .Counter (e [alan ]for e ,b in zip (K ,dik )if b )
        t =collections .Counter (e [alan ]for e in K )
        print ("  "+alan +": "+" | ".join (
        f"{k } {c .get (k ,0 )}/{t [k ]} ({c .get (k ,0 )/t [k ]:.1%})"for k in sorted (t )))

    print ("\n4) DIK OLANLARIN YANAL MESAFESI different mi?")
    ya =np .array ([e ["lateral"]for e in K ])
    print (f"  dik olanlar   : medyan {np .median (ya [dik ]):.2f}mm (n={int (dik .sum ())})")
    print (f"  digerleri     : medyan {np .median (ya [~dik ]):.2f}mm (n={int ((~dik ).sum ())})")
    print ("  -> yanali da kotuyse same kok why; iyiyse SADECE axis sorunu")

    print ("\n5) GT EKSENLERI: dik-eslesmelerde GT ekseni ozel mi?")
    with open ("results/_u4_der.pkl","rb")as f :
        DER ={r ["pid"]:r for r in pickle .load (f )}
    eks_dik ,eks_iyi =[],[]
    for e ,b in zip (K ,dik ):
        r =DER .get (e ["pid"])
        if r is None or not len (r ["Gd"]):
            continue 
            # GT eksenlerinin part icindeki cesitliligi: all of them same yone mi bakiyor?
        c =np .abs (r ["Gd"]@r ["Gd"].T )
        (eks_dik if b else eks_iyi ).append (float (np .median (c )))
    if eks_dik and eks_iyi :
        print (f"  dik-eslesmelerin parcasinda GT axis benzerligi (medyan |cos|): "
        f"{np .median (eks_dik ):.3f}")
        print (f"  iyi eslesmelerin parcasinda                                  : "
        f"{np .median (eks_iyi ):.3f}")
        print ("  -> 1.0'a yakin = parcadaki tum CP'ler same yone bakiyor")

    print ("\n6) TEK PARCA ORNEGI (at most dik eksenli part)")
    if etkilenen :
        pid =max (etkilenen ,key =lambda p :etkilenen [p ][1 ])
        alt =[e for e in K if e ["pid"]==pid ]
        r =DER .get (pid )
        print (f"  {pid } | {etkilenen [pid ][1 ]}/{etkilenen [pid ][0 ]} dik | "
        f"regime {alt [0 ]['regime']} | diag {alt [0 ]['diag']:.0f}mm")
        print (f"  acilar: {sorted (round (e ['angle'],1 )for e in alt )}")
        if r is not None and len (r ["Gd"]):
            print (f"  GT eksenleri (ilk 4): {np .round (r ['Gd'][:4 ],3 ).tolist ()}")
            print (f"  tahmin eksenleri (ilk 4): {np .round (r ['Pd'][:4 ],3 ).tolist ()}")

    with open ("results/r1_aci_teshis.json","w",encoding ="utf-8")as f :
        json .dump ({"n":len (K ),"aci_10_ustu":int (kotu .sum ()),"aci_45_ustu":int (dik .sum ()),
        "etkilenen_parca":len (etkilenen ),"tam_dik_parca":len (tam ),
        "kismi_dik_parca":len (kismi ),
        "dik_yanal_medyan":float (np .median (ya [dik ]))if dik .any ()else None ,
        "iyi_yanal_medyan":float (np .median (ya [~dik ]))},f ,indent =1 )
    print ("\nmakbuz -> results/r1_aci_teshis.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
