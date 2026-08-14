# -*- coding: utf-8 -*-
"""DS6-5b: TURETILEMEYEN parcalari bul and DORDUNCU KADEME with kurtar.

WHY LOG AYRISTIRMIYORUZ: log only ISTISNA ATAN parcayi writes. Bir part baska
sebeplerle de dusebilir (process oldurulmus, vardiya bolmesi kaymis, kayit yazilmadan
kesilmis). DOGRU tanim this: `eligible()` eligible diyor AMA no shard'da kaydi YOK.
Boylece loss part, sebebi ne olursa olsun yakalanir.

Cikti: `results/_ds6_kurtarilacak.txt` -- dogrudan `d5_derive_new.py --pids-file` girdisi.
"""
import glob 
import io 
import os 
import pickle 
import sys 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CIKTI ="results/_ds6_kurtarilacak.txt"


def turetilmis ():
    """TUM shard'lardaki pid'ler + onceden turetilmis kaynaklar."""
    present =set ()
    for f in glob .glob ("results/_der_yeni*.pkl"):
        try :
            with open (f ,"rb")as h :
                present |={r ["pid"]for r in pickle .load (h )}
        except (OSError ,ValueError ,EOFError ,pickle .UnpicklingError ):
            print (f"  UYARI: {f } okunamadi")
    for y in ("results/_der_tam.pkl","results/_der_kontrol.pkl"):
        if os .path .exists (y ):
            with open (y ,"rb")as f :
                present |={r ["pid"]for r in pickle .load (f )}
    try :
        import numpy as np 
        d =np .load ("results/zengin_parite_w2.npz",allow_pickle =True )
        present |={str (x )for x in d ["pids"]}
    except OSError :
        pass 
    return present 


def main ():
    import collections 
    from big_arbiter import eligible 
    E =eligible ()
    present =turetilmis ()
    missing =[(m ,p )for m ,p ,_jf ,_s in E if p not in present ]
    skip =set ()
    if os .path .exists ("_d5_atla.txt"):
        skip ={x .strip ()for x in io .open ("_d5_atla.txt",encoding ="utf-8")if x .strip ()}
    eksik_a =[(m ,p )for m ,p in missing if p not in skip ]
    print (f"eligible corpus {len (E )} | turetilmis {len (present )} | EKSIK {len (missing )}"
    f" (kilitli list disi: {len (eksik_a )})")
    c =collections .Counter (m for m ,_ in eksik_a )
    for m ,n in c .most_common (15 ):
        print (f"  {m :<8}{n :>5}")
    with io .open (CIKTI ,"w",encoding ="utf-8")as f :
        f .write ("\n".join (p for _m ,p in eksik_a )+"\n")
    print (f"\n-> {CIKTI } ({len (eksik_a )} pid)")
    print ("   d5_derive_new.py --pids-file "+CIKTI +" --vardiya K --total N")


if __name__ =="__main__":
    main ()
