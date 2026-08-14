# -*- coding: utf-8 -*-
"""D20 ORNEKLEME: disk sinirina sigan, AGIZ KALITESI DUSUK oncelikli lower cluster.

MEASURED (2026-08-09): operator onbellegi part basina ~2.8 MB (k_eig=96).
5391 part -> ~15 GB; makinede sigmadi. Bu betik D20'den ~N part selects.

SECIM RASTGELE DEGIL: lateral error AGIZ KALITESIYLE aciklaniyor
([[lateral-error-segmentasyon-kalitesiyle-aciklanir]]: low ceyrekte <=2mm %25.8,
high ceyrekte %75.9). Kazanc DUSUK KALITELI agizlarda; ogrenilecek sey orada.

VEKIL OLCUT (etiketten is computed, modele ihtiyac YOK):
  boyanan vertex count / CP count  -- low = mouth zayif yakalanmis
  + part basina CP count (high-CP oncelikli, plan boyle diyor)
MARKA DENGESI korunur: each markadan payina dusen up to.
"""
import collections ,io ,json ,os ,shutil ,sys 
import numpy as np 
sys .path .insert (0 ,".")
import connector3d 
CE =int (connector3d .CABLE_ENTRY )

KAYNAK ="_label_ds1_obj"
HEDEF ="_label_ds1_obj_SECILI"
N =int (os .environ .get ("D20_N","1500"))


def main ():
    adlar =[d for d in os .listdir (KAYNAK )if os .path .isdir (os .path .join (KAYNAK ,d ))]
    print (f"kaynak {len (adlar )} part | hedef ~{N }")
    kayit =[]
    for ad in adlar :
        lf =os .path .join (KAYNAK ,ad ,ad +".labels.txt")
        if not os .path .exists (lf ):
            continue 
        L =np .fromstring (io .open (lf ,encoding ="utf-8").read (),sep =" ",dtype =int )
        if not len (L ):
            continue 
        n_ce =int ((L ==CE ).sum ())
        if n_ce ==0 :
            continue 
        mfg =ad .split (".",1 )[0 ]
        kayit .append ({"ad":ad ,"mfg":mfg ,"n_ce":n_ce ,"n_v":len (L ),
        "ratio":n_ce /max (len (L ),1 )})
    print (f"okunabilen {len (kayit )}")
    # DUSUK 'ratio' = mouth zayif yakalanmis -> ONCELIKLI
    say =collections .Counter (k ["mfg"]for k in kayit )
    pay ={m :max (1 ,int (N *c /len (kayit )))for m ,c in say .items ()}
    grup =collections .defaultdict (list )
    for k in kayit :
        grup [k ["mfg"]].append (k )
    sec =[]
    for m ,lst in grup .items ():
        lst .sort (key =lambda x :x ["ratio"])# low ratio ONCE
        sec +=lst [:pay [m ]]
    os .makedirs (HEDEF ,exist_ok =True )
    for k in sec :
        h =os .path .join (HEDEF ,k ["ad"])
        if not os .path .exists (h ):
            shutil .copytree (os .path .join (KAYNAK ,k ["ad"]),h )
    d =collections .Counter (k ["mfg"]for k in sec )
    print (f"SECILDI {len (sec )} part -> {HEDEF }")
    print (f"brand dagilimi: {dict (d )}")
    print (f"secilen medyan mouth orani {np .median ([k ['ratio']for k in sec ]):.4f} | "
    f"TUM corpus medyani {np .median ([k ['ratio']for k in kayit ]):.4f}")
    json .dump ({"n":len (sec ),"brand":dict (d ),"N_hedef":N ,
    "criterion":"mouth orani (boyanan tepe / toplam tepe) DUSUK oncelikli"},
    io .open ("results/d20_ornek.json","w",encoding ="utf-8"),indent =1 )


if __name__ =="__main__":
    main ()
