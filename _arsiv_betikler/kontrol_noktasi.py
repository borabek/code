# -*- coding: utf-8 -*-
"""CHECK NOKTASI kurucusu -- "each an full this ana donebilelim" sartinin yarisi.

Git kodu and makbuzlari keeps; model agirliklari (pkl/pt, ~650 MB) git'e girmez.
Bu betik onlari `_KN_<label>/dosyalar/` altina FIZIKSEL as kopyalar and
tum risky girdilerin SHA-256'sini manifestoya writes.

Iki cluster present:
  KOPYALA  -- kanonik zincirin OKUDUGU and benim UZERINE YAZABILECEGIM dosyalar.
              Bunlar geri alinabilir.
  DAMGALA  -- kopyalamak for extra large olanlar (olasilik onbellekleri).
              Bunlar geri ALINAMAZ; degisirlerse `rollback.py` BAGIRIR.
              Kural: this betikten after this dosyalara YAZMA.

Kullanim:  python kontrol_noktasi.py [label]
"""
import glob 
import hashlib 
import json 
import os 
import shutil 
import subprocess 
import sys 
import time 

KOK =os .path .dirname (os .path .abspath (__file__ ))

# --- kanonik zincirin okudugu, uzerine yazilabilecek girdiler -------------
KOPYALA =[
"cp_config.json",
"results/metrik_dondurulmus.json",
"results/split3.json",
"results/wire_gate_v5.pkl",
"results/wire_gate_v6.pkl",
"results/wire_gate_v7.pkl",
"results/p3c_axis_selector.pkl",
"results/kazanan_hgb_derin.pkl",
"results/_der_yeni_G7BIRLESIK.pkl",
"results/seg_g7/g7_s0.pt",
"results/seg_g10/g10_s0.pt",
]
# --- kopyalanmayan but degisimi yakalanmasi gereken girdiler --------------
DAMGALA_GLOB =["results/*.pkl","results/seg_*/*.pt"]
# --- only parmak izi (file count + total bayt + name/size ozeti) ----
PARMAK_IZI_DIZIN =[
"results/_p1_olasilik_d7",
"results/_p1_olasilik",
"results/_p1_olasilik_d7g10",
"results/_p1_olasilik_g10",
]


def sha (yol ,blok =1 <<20 ):
    h =hashlib .sha256 ()
    with open (yol ,"rb")as f :
        while True :
            b =f .read (blok )
            if not b :
                break 
            h .update (b )
    return h .hexdigest ()


def dizin_parmak_izi (d ):
    """Icerigi okumadan name+size ozeti. Kazara degisimi yakalar, ucuzdur."""
    if not os .path .isdir (d ):
        return None 
    ad =[]
    top =0 
    for f in sorted (os .listdir (d )):
        p =os .path .join (d ,f )
        if os .path .isfile (p ):
            n =os .path .getsize (p )
            top +=n 
            ad .append (f"{f }:{n }")
    return {"n":len (ad ),"bayt":top ,
    "ozet":hashlib .sha256 ("|".join (ad ).encode ()).hexdigest ()[:32 ]}


def main ():
    etiket =sys .argv [1 ]if len (sys .argv )>1 else time .strftime ("%Y-%m-%d")
    kn =os .path .join (KOK ,f"_KN_{etiket }")
    dos =os .path .join (kn ,"dosyalar")
    os .makedirs (dos ,exist_ok =True )

    man ={"etiket":etiket ,"zaman":time .strftime ("%Y-%m-%d %H:%M:%S"),
    "kopya":{},"damga":{},"parmak_izi":{},"eksik":[]}
    try :
        man ["git"]=subprocess .check_output (
        ["git","rev-parse","HEAD"],cwd =KOK ,text =True ).strip ()
    except Exception as e :# pragma: no cover
        man ["git"]=f"YOK ({e })"

    top =0 
    for rel in KOPYALA :
        src =os .path .join (KOK ,rel )
        if not os .path .exists (src ):
            man ["eksik"].append (rel )
            print (f"  ! yok: {rel }")
            continue 
        dst =os .path .join (dos ,rel .replace ("/",os .sep ))
        os .makedirs (os .path .dirname (dst ),exist_ok =True )
        shutil .copy2 (src ,dst )
        n =os .path .getsize (src )
        top +=n 
        man ["kopya"][rel ]={"sha256":sha (src ),"bayt":n }
        print (f"  + {rel }  ({n /1048576 :.1f} MB)")

    for g in DAMGALA_GLOB :
        for src in sorted (glob .glob (os .path .join (KOK ,g ))):
            rel =os .path .relpath (src ,KOK ).replace (os .sep ,"/")
            if rel in man ["kopya"]:
                continue 
            man ["damga"][rel ]={"sha256":sha (src ),
            "bayt":os .path .getsize (src )}

    for d in PARMAK_IZI_DIZIN :
        fi =dizin_parmak_izi (os .path .join (KOK ,d ))
        if fi :
            man ["parmak_izi"][d ]=fi 

    json .dump (man ,open (os .path .join (kn ,"manifest.json"),"w"),indent =1 )
    print (f"\nKONTROL NOKTASI: {kn }")
    print (f"  git      : {man ['git']}")
    print (f"  kopya    : {len (man ['kopya'])} dosya / {top /1048576 :.0f} MB")
    print (f"  damga    : {len (man ['damga'])} dosya (kopyasiz, degisim yakalanir)")
    print (f"  parmak iz: {len (man ['parmak_izi'])} dizin")
    if man ["eksik"]:
        print (f"  EKSIK    : {man ['eksik']}")
    print ("\nGeri donus:  python rollback.py "+etiket )


if __name__ =="__main__":
    main ()
