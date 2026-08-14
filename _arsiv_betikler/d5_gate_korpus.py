# -*- coding: utf-8 -*-
"""D5-3b: GATE EGITIM KORPUSU v3 -- old corpus + new turetme, SIZINTI FILTRELI.

YENIDEN TURETME YOK: `d5_turet_yeni.py` each parcanin `X` (22 column) and `XR` (36 column)
matrislerini already sakladi; `d5_birlestir.py` de X'i 58 -> 22'ye normalize etti (kayipsiz
oldugu 984/984 kayitta measured). Burada only ETIKET uretilir and two corpus birlestirilir.

ETIKET (`y`) `build_zengin_parite.py` with BIREBIR AYNI kuralla uretilir:
  tol = max(3.0, 0.06*diag) | axial |al| <= 40 | TEKIL greedy esleme
Baska a rule kullanilirsa new satirlar eskilerden FARKLI a hedefe egitilir and gate
sessizce bozulur -- that is why rule here TEK YERDE and yorumlu duruyor.

UC KATMANLI SIZINTI FILTRESI:
  1. `protocol.dogrula()`        -- LOCKED + measurement gruplari (ESKI anahtar uzayi)
  2. YASAK ANAHTAR capraz kontrolu -- new parcalarin anahtarlari `_geo_yasak.json` with
     karsilastirilir. **Bu sart because two anahtar uzayi string as ASLA eslesmez**:
     old `_strict_geometry_keys.json` bicimi with `geometri_anahtar.anahtar()` bicimi
     different. 2026-08-04 olcumu: **126 new part** yasak anahtarla carpisiyor -- i.e.
     measurement/LOCKED parcalarinin IKIZLERI. Bu filtre olmadan egitime girip mansedi
     sessizce sisirirlerdi.
  3. Aday uretilememis parts (X is None) atlanir.

TEZ DEGISMEZ: no network egitilmez, no threshold does not change, `v_o` turetmesi aynen kalir.
Degisen TEK sey corpus buyuklugu and manufacturer cesitliligi -- F2-12'nin sarti budur.

!!! F2-12 ICIN KRITIK WARNING -- FILTRE KARISTIRMASI !!!
Bu betik protocol bekcisini `olcum_da=True` with uygular and ESKI korpustan da part removes
(measured: 7983 candidate / 519 part). Yani v3, w2'nin "buyutulmus" hali DEGIL; **different
filtreden gecmis** a corpus. Dagitilan gate w2 with egitildiyse, v3'le egitilmis a
gate'i onunla kiyaslamak "data artti"yi DEGIL "filtre degisti"yi olcer.

F2-12 DOGRU TASARIM: IKI gate AYNI filtreyle egitilir --
    (a) only ESKI parts + same protocol filtresi
    (b) old + YENI parts + same protocol filtresi
Fark however that zaman SADECE VERIDIR. `--only-old` bayragi (a)'yi produces.
[[gate-refit-minv4]] dersi birebir this: two gate AYNI dagilimda egitilmeli.
"""
import collections 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

ESKI ="results/zengin_parite_w2.npz"
YENI =os .environ .get ("D5_YENI","results/_der_yeni.pkl")
GEO_YENI ="results/_geo_yeni.json"
GEO_YASAK ="results/_geo_yasak.json"
SINAV ="results/d5_4_sinav_kumesi.json"# D5-4 gorulmemis manufacturer sinavi -- EGITIME GIRMEZ
# CIKTI da ORTAM DEGISKENI (2026-08-08): g10 with yeniden turetilen corpus AYRI a
# dosyaya yazilmali. Sabit kalsaydi `zengin_parite_v3` -- i.e. DAGITILAN gate'in
# egitildigi corpus -- uzerine yazilir and A/B kiyasi imkansizlasirdi.
CIKTI =os .environ .get ("D5_CIKTI","results/zengin_parite_v3.npz")
CIKTI_TABAN =CIKTI .replace (".npz","_taban.npz")
MAKBUZ ="results/d5_gate_korpus.json"

VOTES_SUTUN =11 # X22 inside `votes` sutunu -- %100 dogrulandi (X22[:,11] == votes)


def etiketle (P ,G ,Gd ,diag ):
    """build_zengin_parite with BIREBIR AYNI tekil esleme -> candidate basina 0/1."""
    n =len (P )
    y =np .zeros (n ,int )
    if not n or not len (G ):
        return y 
    tol =max (3.0 ,0.06 *float (diag ))
    d =P [:,None ,:]-G [None ,:,:]
    al =(d *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
    pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
    up ,ug =set (),set ()
    for dd ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
    for a in range (n )for b in range (len (G ))):
        if dd >tol or a_ in up or b_ in ug :
            continue 
        up .add (a_ );ug .add (b_ );y [a_ ]=1 
    return y 


def main ():
    import argparse 
    ap =argparse .ArgumentParser ()
    # F2-12'nin (a) kolu: new parcalari EKLEMEDEN, AYNI filtreyle baseline corpus uret.
    # Boylece (a) with (b) arasindaki TEK difference VERI becomes (see. yukaridaki kritik uyari).
    ap .add_argument ("--yalniz-eski",action ="store_true",
    help ="yeni parcalari EKLEME -- F2-12 baseline kolu")
    a =ap .parse_args ()
    import protocol 
    protocol .tez_dogrula ()

    d =np .load (ESKI ,allow_pickle =True )
    X22 =[d ["X22"]];XR =[d ["XR"]];Y =[d ["y"]]
    PID =[np .asarray (d ["pids"],str )];MFG =[np .asarray (d ["mfg"],str )]
    VOT =[d ["votes"]]
    eski_parca =len (set (map (str ,d ["pids"])))
    print (f"ESKI corpus: {len (d ['y'])} candidate / {eski_parca } part")

    if a .yalniz_eski :
        R =[]
        print ("YALNIZ ESKI: yeni parts eklenmiyor (F2-12 baseline kolu)")
    else :
        with open (YENI ,"rb")as f :
            R =pickle .load (f )
    GY =json .load (io .open (GEO_YENI ,encoding ="utf-8"))if os .path .exists (GEO_YENI )else {}
    YASAK =set (json .load (io .open (GEO_YASAK ,encoding ="utf-8")).values ())
    # D5-4 SINAV KUMESI EGITIME GIREMEZ. Kume anahtarlari da yasak listesine eklenir --
    # part kimligi yetmez, IKIZI de girmemeli ([[geometry-twin-leakage]]).
    SINAV_PID ,SINAV_MFG =set (),set ()
    # IKI SINAV KUMESI birden dislanir (2026-08-05): old d5_4 kumesi karsilastirilabilirlik
    # for duruyor, YENI d6 kumesi whereas real gorulmemis-manufacturer sinavi. Biri atlanirsa that
    # kumede olculen each number sisik becomes.
    for _sf in (SINAV ,"results/d6_sinav_kumesi.json"):
        if not os .path .exists (_sf ):
            continue 
        _sv =json .load (io .open (_sf ,encoding ="utf-8"))
        SINAV_PID |=set (_sv .get ("pidler")or [])
        SINAV_MFG |=set (_sv .get ("manufacturer")or {})
        print (f"  SINAV KUMESI {_sf }: {len (_sv .get ('pidler')or [])} part, "
        f"{len (_sv .get ('manufacturer')or {})} manufacturer (muhur {_sv .get ('sha16')})")
    if os .path .exists (SINAV ):
        sv =json .load (io .open (SINAV ,encoding ="utf-8"))
        # NOTE -- here a zamanlar `SINAV_MFG = set(...)` yaziyordu and yukaridaki
        # IKI-KUMELI birlesimi EZIYORDU: d6'nin 8 ureticisi sessizce egitime giriyordu
        # (belirti: korpusta SE 74 / UTL 36 / SUPU 24 candidate). Atama DEGIL, BIRLESIM.
        #
        # URETICI DUZEYI DISLAMA (2026-08-05'te BULUNAN KIRLILIK):
        # Dislama only PARCA + IKIZ duzeyindeydi. Sinav kumesi "gorulmemis URETICI"
        # sinavi as tanimlandigi halde, corpus buyudukce same ureticilerin BASKA
        # parcalari egitime giriyordu. Olculdu: `zengin_parite_v3` exam ureticilerinden
        # **1502 tekil part** iceriyordu (CWT 691, A-B 358, WIE 226, ...). Sinav pid'leri
        # temizdi, but URETICI residual gorulmemis DEGILDI -- i.e. kumenin ADI yalan olmustu.
        # ([[locked-exam-kirliligi-yakalandi]] with same desen.)
        SINAV_MFG |=set (sv .get ("manufacturer")or {})
        print (f"SINAV (iki cluster): {len (SINAV_PID )} part training disi")
        print (f"  URETICI DUZEYI dislama ({len (SINAV_MFG )}): {sorted (SINAV_MFG )}")
    print (f"YENI turetme: {len (R )} part | yasak anahtar {len (YASAK )}")

    SINAV_ANAH ={r ["geo"]for r in R if r ["pid"]in SINAV_PID and r .get ("geo")}
    atlanan =collections .Counter ()
    n_yeni =0 
    for r in R :
        if r .get ("X")is None or r .get ("XR")is None :
            atlanan ["aday_yok"]+=1 ;continue 
        if r ["X"].shape [1 ]!=22 or r ["XR"].shape [1 ]!=36 :
            atlanan ["genislik"]+=1 ;continue 
            # GT GECERLILIK (2026-08-04): `eligible()` kapisi turetmeden SONRA eklendi, i.e.
            # `_der_yeni.pkl` still dejenere GT'li parcalari iceriyor (AL vakasi: direction (0,0,0),
            # tum CP same noktada). Onlarda HICBIR candidate eslesemez -> all of them y=0 with egitime
            # girer and gate'e "gecerli agizlari REDDET" ogretir. Burada da kapatilir.
        Gq =np .asarray (r ["G"],float );Dq =np .asarray (r ["Gd"],float )
        if (len (Dq )and (np .linalg .norm (Dq ,axis =1 )<1e-6 ).all ())or (len (Gq )>1 and float (np .abs (Gq .max (0 )-Gq .min (0 )).max ())<1e-6 ):
            atlanan ["gt_dejenere"]+=1 ;continue 
        if r ["pid"]in SINAV_PID :
            atlanan ["sinav_kumesi"]+=1 ;continue 
        if str (r ["mfg"])in SINAV_MFG :
            atlanan ["sinav_ureticisi"]+=1 ;continue 
        anah =r .get ("geo")or GY .get (r ["pid"])
        if anah in SINAV_ANAH :
            atlanan ["sinav_ikizi"]+=1 ;continue 
        if anah in YASAK :
        # OLCUM ya da LOCKED parcasinin IKIZI -- egitime GIREMEZ
            atlanan ["yasak_anahtar"]+=1 ;continue 
        P =np .asarray (r ["P"],float )
        y =etiketle (P ,np .asarray (r ["G"],float ),np .asarray (r ["Gd"],float ),r ["diag"])
        X22 .append (r ["X"]);XR .append (r ["XR"]);Y .append (y )
        PID .append (np .array ([str (r ["pid"])]*len (P )))
        MFG .append (np .array ([str (r ["mfg"])]*len (P )))
        VOT .append (np .asarray (r ["X"],float )[:,VOTES_SUTUN ])
        n_yeni +=1 

    print (f"\nATLANAN: {dict (atlanan )}")
    print (f"  -> yasak_anahtar = measurement/LOCKED IKIZI, egitime ALINMADI")

    X22 =np .vstack (X22 );XR =np .vstack (XR );Y =np .concatenate (Y )
    PID =np .concatenate (PID );MFG =np .concatenate (MFG );VOT =np .concatenate (VOT )

    # KATMAN 1: old anahtar uzayinda protocol bekcisi (LOCKED + measurement gruplari)
    m =protocol .dogrula (PID .tolist (),ad ="gate_korpus_v3",olcum_da =True ,sert =False )
    if (~m ).any ():
        atl =len (set (PID [~m ].tolist ()))
        print (f"  protocol bekcisi: {int ((~m ).sum ())} candidate / {atl } part CIKARILDI")
        X22 ,XR ,Y ,PID ,MFG ,VOT =(a [m ]for a in (X22 ,XR ,Y ,PID ,MFG ,VOT ))

        # CIKTI and RECEIPT IKISI DE MODA BAGLI. Ilk surumde receipt sabitti and baseline kosusu
        # v3'un makbuzunun UZERINE yaziyordu; ekrana da wrong file adi basiyordu. Iki kolun
        # makbuzu karisirsa F2-12 karsilastirmasi sessizce wrong sayilara dayanir.
    out =CIKTI_TABAN if a .yalniz_eski else CIKTI 
    mak =MAKBUZ .replace (".json","_taban.json")if a .yalniz_eski else MAKBUZ 
    np .savez (out ,X22 =X22 ,XR =XR ,y =Y ,pids =PID ,mfg =MFG ,votes =VOT ,
    zengin_ad =d ["zengin_ad"])
    part =len (set (PID .tolist ()))
    c =collections .Counter (MFG .tolist ())
    print (f"\nKORPUS {'TABAN (yalniz eski)'if a .yalniz_eski else 'v3'} -> {out }")
    # ORAN HANGI TABANA GORE (2026-08-04'te duzeltildi): first version v3'u FILTRELENMEMIS
    # w2 (1709 part) with kiyasliyordu and "x1.73" yaziyordu. Ama F2-12'nin kontrol kolu
    # TABAN korpusudur (same protocol filtresinden gecmis, 1190 part) -- correct ratio
    # ona according to x2.48. Iki different tabana according to two different number yazmak, kolun buyuklugunu
    # oldugundan KUCUK gosteriyordu.
    tb =0 
    if os .path .exists (CIKTI_TABAN ):
        tb =len (set (np .load (CIKTI_TABAN ,allow_pickle =True )["pids"].tolist ()))
    print (f"  {len (Y )} candidate / {part } part / {len (c )} manufacturer")
    print (f"  buyume: filtresiz w2 ({eski_parca }) -> x{part /max (eski_parca ,1 ):.2f}  |  "
    f"TABAN ({tb }) -> x{part /max (tb ,1 ):.2f}   <- F2-12 icin GECERLI olan bu")
    print (f"  pozitif %{100 *Y .mean ():.1f} | X22 {X22 .shape } | XR {XR .shape }")
    print (f"  manufacturer: {dict (c .most_common (10 ))}")
    with io .open (mak ,"w",encoding ="utf-8")as f :
        json .dump ({"candidate":int (len (Y )),"part":int (part ),"manufacturer":len (c ),
        "eski_parca":eski_parca ,"yeni_parca_eklendi":n_yeni ,
        "atlanan":dict (atlanan ),"pozitif_oran":float (Y .mean ()),
        "uretici_dagilimi":dict (c )},f ,indent =1 ,ensure_ascii =False )
    print (f"  receipt -> {mak }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
