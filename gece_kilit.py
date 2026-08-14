# -*- coding: utf-8 -*-
"""T1: HAKIKAT KILIDI -- gecenin baslangic durumunu DONDUR and celiskileri gider.

WHY: gece along dagitilan artefaktlar degisecek. Sabah "ne degisti" sorusunun single
cevabi this receipt must be. Ayrica belgeler with config arasindaki bilinen celiskiler
(LOCKED 98 vs 95) simdi kapatilir ki gece along wrong number dolasmasin.

DISK BEKCISI: disk this an ~22GB. Tek agir writes works; 12GB'in altina inerse operasyon
DURUR (gece along two times %100 doldu and kosulari oldurdu).
"""
import glob 
import hashlib 
import io 
import json 
import os 
import re 
import shutil 
import sys 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
OUT ="results/gece_kilit.json"
ESIK_GB =12.0 


def bos_gb ():
    t ,u ,f =shutil .disk_usage (os .path .abspath ("."))
    return f /(1024 **3 )


def guard (nerede =""):
    """Disk esigin altina indiyse KOSUYU DURDUR."""
    g =bos_gb ()
    if g <ESIK_GB :
        raise SystemExit (f"DISK BEKCISI: {g :.1f} GB < {ESIK_GB } GB -> operasyon durduruldu ({nerede })")
    return g 


def md5 (p ):
    h =hashlib .md5 ()
    with open (p ,"rb")as f :
        for b in iter (lambda :f .read (1 <<20 ),b""):
            h .update (b )
    return h .hexdigest ()


def main ():
    print (f"disk: {bos_gb ():.1f} GB bos (threshold {ESIK_GB })")
    guard ("baslangic")

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    cp =cfg ["current_product"]

    artefakt ={}
    for p in ["results/wire_gate.pkl","results/pose_head.pkl","results/aci_secici.pkl",
    "results/uye_secici.pkl","results/highcp_router.pkl"]:
        if os .path .exists (p ):
            artefakt [p ]={"md5":md5 (p ),"bayt":os .path .getsize (p )}
    for c in (cp .get ("checkpoints")or cfg .get ("robot_vote2_checkpoints")or []):
        if os .path .exists (c ):
            artefakt [c ]={"md5":md5 (c ),"bayt":os .path .getsize (c )}

    headline ={}
    if os .path .exists ("results/headline.json"):
        with io .open ("results/headline.json",encoding ="utf-8")as f :
            m =json .load (f )
        for k ,v in m .get ("bolmeler",{}).items ():
            headline [k .strip ()]={"tespit":v .get ("tespit_F1"),"robot":v .get ("robot_hazir_F1"),
            "n":v .get ("n_parca")}

            # --- CELISKI TARAMASI: belgelerde bayat LOCKED count
    celiski =[]
    for md_ in glob .glob ("*.md"):
        try :
            with io .open (md_ ,encoding ="utf-8",errors ="ignore")as f :
                s =f .read ()
        except Exception :
            continue 
        for m_ in re .finditer (r"(9[0-9])\s*(?:grup-temiz\s*)?LOCKED|LOCKED[^\n]{0,30}?\b(9[0-9])\b",s ):
            n =m_ .group (1 )or m_ .group (2 )
            if n and n !="95":
                celiski .append ({"dosya":md_ ,"bulunan":n ,"baglam":s [max (0 ,m_ .start ()-60 ):m_ .end ()+40 ].replace ("\n"," ")})
    print (f"\nartefakt: {len (artefakt )} dosya damgalandi")
    print (f"headline bolmesi: {len (headline )}")
    print (f"LOCKED celiskisi: {len (celiski )} yer")
    for c_ in celiski [:6 ]:
        print (f"  {c_ ['dosya']}: '{c_ ['bulunan']}' -> {c_ ['baglam'][:90 ]}")

    kilit ={
    "tarih":"2026-08-02 gece",
    "disk_gb":round (bos_gb (),2 ),"disk_esik_gb":ESIK_GB ,
    "artefakt":artefakt ,
    "headline":headline ,
    "gate":cp .get ("wire_gate",{}).get ("gate_kimlik"),
    "egitim_verisi":cp .get ("wire_gate",{}).get ("egitim_verisi"),
    "locked_temiz":95 ,
    "locked_celiskisi":celiski ,
    "dogrulanan_denetim_bulgulari":{
    "s7_brep_graf_eksik":("g_agiz_cev ve g_yuz_alan HIC hesaplanmiyor (graf_ozellik "
    "10 slot acip only 0-7'yi dolduruyor) ve gercek yuz-kenar "
    "komsulugu YOK -> BOSLUK GRAFI DENENMEMIS"),
    "yon_sozlesmesi_bozuk":("robot_cp docstring 'the way the wire goes in' diyor but "
    "measured: candidate yonleri DISARI bakiyor ((p-merkez).d=+3.99), "
    "manufacturer InsertDirection ICERI (-4.8). sina_cluster abs() "
    "kullandigi for metrik bunu GOREMIYOR."),
    },
    "prospektif_109_durumu":("EGITIME GIRMEDI but W2'yi dagitma KARARINDA kullanildi -> "
    "model secimi acisindan HARCANMISTIR. Bakir exam only 95 LOCKED."),
    }
    with io .open (OUT ,"w",encoding ="utf-8")as f :
        json .dump (kilit ,f ,indent =1 ,ensure_ascii =False )
    print (f"\nmakbuz -> {OUT }")


if __name__ =="__main__":
    main ()
