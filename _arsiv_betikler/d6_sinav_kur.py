# -*- coding: utf-8 -*-
"""D6-6: TEMIZ GORULMEMIS-URETICI SINAV KUMESI.

WHY YENI BIR KUME (2026-08-05 denetimi):
Eski cluster (`d5_4_sinav_kumesi.json`, 250 part) "gorulmemis URETICI sinavi" adiyla
kurulmustu but dislama only PARCA + IKIZ duzeyindeydi. Korpus buyudukce same
ureticilerin baska parcalari egitime input. MEASURED:

  * seg oto-label korpusu (`_label_auto_obj`): exam kumesinin 250 parcasindan
    **169'u DOGRUDAN inside** (%68) + exam ureticilerinden 958 part
  * gate korpusu (`zengin_parite_v3`): exam ureticilerinden 1502 tekil part

Yani that kumede olculen "gorulmemis manufacturer" sayilari SISIK. Eski kumede kurtarilacak
temiz manufacturer de kalmadi (13'un 12'si seg egitiminde).

COZUM: DataSet 6 with gelen ureticiler HICBIR training yapitinda absent -- bugun geldiler.
Onlardan kurulan cluster, MEVCUT ckpt'ler for gercekten gorulmemis manufacturer sinavidir
and yeniden training GEREKTIRMEZ.

UC KATMANLI TEMIZLIK (old kumeden aynen devralindi):
  1. URETICI: no training yapitinda (w2, v3, _label_auto_obj, _label_targets*) YOK
  2. GEOMETRI GRUBU: anahtari training / 194'luk measurement / LOCKED anahtarlariyla carpismaz
  3. GRUP-ICI TEKILLIK: same geometri grubundan single part

URETICI KAPISI n>=5 ([[gorulmemis-manufacturer-sinavi-first-measurement]] WAGO dersi: single parcali
manufacturer gurultuden ibaret).

TEZ DEGISMEZ: no sey egitilmez, this a SECIM betigidir.
"""
import collections 
import glob 
import hashlib 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CIKTI ="results/d6_sinav_kumesi.json"
GEO_YASAK ="results/_geo_yasak.json"
MIN_URETICI =5 
KOVA =((1 ,3 ),(4 ,7 ),(8 ,10 **9 ))


def kova_adi (n ):
    for lo ,hi in KOVA :
        if lo <=n <=hi :
            return f"{lo }-{hi if hi <10 **9 else '8+'}"
    return "0"


def egitimde_gorulen_ureticiler ():
    """HER training yapitindaki ureticiler. Biri bile atlanirsa cluster KIRLENIR."""
    gor =set ()
    src_ ={}
    for f in ("results/zengin_parite_w2.npz","results/zengin_parite_v3.npz",
    "results/zengin_parite_v5.npz"):
        if os .path .exists (f ):
            d =np .load (f ,allow_pickle =True )
            u ={str (m )for m in np .asarray (d ["mfg"])}
            gor |=u ;src_ [f ]=len (u )
            # seg training korpuslari: pid -> manufacturer esleme is required
    import big_arbiter 
    mp ={p :m for m ,p ,_jf ,_s in big_arbiter .eligible ()}
    for d in sorted (glob .glob ("_label_*")):
        if not os .path .isdir (d ):
            continue 
        pids =[os .path .basename (os .path .normpath (p ))for p in glob .glob (d +"/*/")]
        u ={mp .get (p )for p in pids }-{None }
        if u :
            gor |=u ;src_ [d ]=len (u )
    return gor ,src_ 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    from big_arbiter import eligible 

    gor ,src_ =egitimde_gorulen_ureticiler ()
    print ("EGITIMDE GORULEN URETICILER (source basina count):")
    for k ,n in sorted (src_ .items ()):
        print (f"  {k :<42}{n :>4}")
    print (f"  TOPLAM tekil: {len (gor )}\n")

    E =eligible ()
    mfg_n =collections .Counter (m for m ,_p ,_jf ,_s in E )
    aday_u =sorted (m for m ,n in mfg_n .items ()if m not in gor and n >=MIN_URETICI )
    kucuk =sorted (m for m ,n in mfg_n .items ()if m not in gor and n <MIN_URETICI )
    print (f"SINAV ADAYI URETICI (n>={MIN_URETICI }): {aday_u }")
    print (f"  n<{MIN_URETICI } oldugu icin EGITIME birakilanlar: "
    f"{[(m ,mfg_n [m ])for m in kucuk ]}\n")
    if not aday_u :
        print ("HIC temiz manufacturer absent -- cluster kurulamaz.");return 

        # turetme kayitlari (GT + geometri anahtari for)
    R =[]
    for f in sorted (glob .glob ("results/_der_yeni*.pkl")):
        try :
            with open (f ,"rb")as h :
                R +=pickle .load (h )
        except (OSError ,ValueError ,EOFError ,pickle .UnpicklingError ):
            print (f"  UYARI: {f } okunamadi")
    rec_ ={}
    for r in R :# same pid birden very shard'da may be
        rec_ .setdefault (r ["pid"],r )
    print (f"turetme kaydi: {len (rec_ )} part")

    YASAK =set ()
    if os .path .exists (GEO_YASAK ):
        YASAK =set (json .load (io .open (GEO_YASAK ,encoding ="utf-8")).values ())
    print (f"yasak geometri anahtari (training/measurement/LOCKED ikizleri): {len (YASAK )}")

    secili ,gruplar =[],set ()
    atlanan =collections .Counter ()
    for m ,p ,_jf ,_s in sorted (E ,key =lambda t :(t [0 ],t [1 ])):
        if m not in aday_u :
            continue 
        r =rec_ .get (p )
        if r is None :
            atlanan ["turetme_yok"]+=1 ;continue 
            # `r["G"]` numpy dizisi may be -- `or` with bosluk kontrolu ValueError gives
        _g =r .get ("G")
        G =np .asarray ([]if _g is None else _g ,float )
        if not len (G ):
            atlanan ["gt_yok"]+=1 ;continue 
        anah =r .get ("geo")
        if anah and anah in YASAK :
            atlanan ["yasak_anahtar"]+=1 ;continue # training/measurement IKIZI
        if anah and anah in gruplar :
            atlanan ["grup_ici_tekrar"]+=1 ;continue # same geometri grubu
        if anah :
            gruplar .add (anah )
        secili .append ({"pid":p ,"mfg":m ,"cp":int (len (G )),
        "kova":kova_adi (len (G )),"geo":anah })
    print (f"\nATLANAN: {dict (atlanan )}")

    up =collections .Counter (s ["mfg"]for s in secili )
    kv =collections .Counter (s ["kova"]for s in secili )
    pidler =sorted (s ["pid"]for s in secili )
    muhur =hashlib .sha256 ("\n".join (pidler ).encode ()).hexdigest ()[:16 ]
    out ={"n_parca":len (secili ),"sha16":muhur ,"manufacturer":dict (up ),
    "cp_kovasi":dict (kv ),"gt_toplam":sum (s ["cp"]for s in secili ),
    "pidler":pidler ,"parts":secili ,
    "min_uretici":MIN_URETICI ,
    "not":("TEMIZ gorulmemis-URETICI sinavi (D6). Bu ureticiler no training "
    "yapitinda YOK. Eski d5_4 kumesi URETICI duzeyinde kirlenmisti "
    "(seg korpusu sinavin 169/250 parcasini iceriyordu) and onunla "
    "KIYASLANAMAZ. 194'luk measurement kumesi AYRI and dondurulmus kalir.")}
    with io .open (CIKTI ,"w",encoding ="utf-8")as f :
        json .dump (out ,f ,indent =1 ,ensure_ascii =False )
    print (f"\nSINAV KUMESI: {len (secili )} part | {len (up )} manufacturer | "
    f"GT {out ['gt_toplam']} CP | muhur {muhur }")
    print (f"  manufacturer: {dict (sorted (up .items (),key =lambda kv :-kv [1 ]))}")
    print (f"  CP kovasi: {dict (kv )}")
    print (f"-> {CIKTI }")


if __name__ =="__main__":
    main ()
