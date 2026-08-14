# -*- coding: utf-8 -*-
"""HAVUZ TAVANI: a secenek onbelleginin YONLU recall'u and F1 tavani.

KAPI A'nin olcusu. `P6_DIZIN` with hangi onbellegin olculecegi secilir; so
"old pool vs three kaldirac open pool" single degiskenli kiyaslanir.

Olcut: mukemmel selector varsayimiyla ulasilabilecek most high F1.
  yonlu recall r  ->  F1 tavani = 2r / (1+r)
Kabul kutusu urun metrigiyle BIREBIR: lateral<=2mm, |axial|<=40mm,
ISARETLI angle<=10 derece.
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402
import direction_bank as YB # noqa: E402

DIZ =os .environ .get ("P6_DIZIN","results/_p6_oz_u25")
ONLER =os .environ .get ("HT_ONLER","d6").split (",")


def _makbuz_yolu ():
    """RECEIPT ADI KUMEYI DE TASIR. Onceden only dizine according to adlandiriliyordu;
    same directory on `d6` and `full` kosulunca ikincisi birincinin USTUNE
    yaziyordu and two different measurement single dosyada karisiyordu."""
    import os as _o 
    return (f"results/havuz_tavani_{_o .path .basename (DIZ )}"
    f"_{'-'.join (ONLER )}.json")
YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 


def rec (P ,D ,G ,Gd ,direction =True ):
    if not len (P )or not len (G ):
        return 0 
    Gn =YB .birim (Gd )
    df =np .asarray (P ,float )[:,None ,:]-np .asarray (G ,float )[None ,:,:]
    al =(df *Gn [None ,:,:]).sum (-1 )
    yan =np .linalg .norm (df -al [...,None ]*Gn [None ,:,:],axis =-1 )
    ok =(yan <=YANAL )&(np .abs (al )<=EKSENEL )
    if direction :
        an =np .degrees (np .arccos (np .clip (YB .birim (D )@Gn .T ,-1.0 ,1.0 )))
        ok =ok &(an <=ACI )
    return int (ok .any (0 ).sum ())


def main ():
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    agg =collections .defaultdict (collections .Counter )
    top =collections .Counter ()
    for on in ONLER :
        fs =sorted (f for f in os .listdir (DIZ )if f .startswith (on +"_"))
        for f in fs :
            pid =f [len (on )+1 :-4 ]
            r =kay .get (pid )
            if r is None or not len (r .get ("G",[])):
                continue 
            z =np .load (f"{DIZ }/{f }")
            idx =np .asarray (z ["idx"],int )
            P =np .asarray (z ["P"],float )
            YD =np .asarray (z ["YD"],float )
            G =np .asarray (r ["G"],float )
            Gd =np .asarray (r ["Gd"],float )
            a =agg [r ["mfg"]]
            a ["part"]+=1 
            a ["gt"]+=len (G )
            a ["konum"]+=rec (P [idx ],YD ,G ,Gd ,direction =False )
            a ["yonlu"]+=rec (P [idx ],YD ,G ,Gd ,direction =True )
            a ["candidate"]+=len (P )
            a ["secenek"]+=len (idx )
            for k_ in ("part","gt","konum","yonlu","candidate","secenek"):
                top [k_ ]+=a [k_ ]-top .get ("_",0 )*0 # (total below)
                # toplami markalardan topla (yukaridaki loop inside birikim wrong olurdu)
    top =collections .Counter ()
    for a in agg .values ():
        for k_ ,v in a .items ():
            top [k_ ]+=v 

    print (f"DIZIN {DIZ } | cluster(ler) {ONLER }")
    print (f"{'brand':<7}{'part':>6}{'GT':>7}{'KONUM':>9}{'YONLU':>9}"
    f"{'F1 tavani':>11}{'candidate/p':>8}{'sec/p':>8}")
    out ={}
    for m ,a in sorted (agg .items (),key =lambda x :-x [1 ]["gt"]):
        g =max (a ["gt"],1 )
        p =max (a ["part"],1 )
        ry =a ["yonlu"]/g 
        out [m ]={"part":a ["part"],"gt":a ["gt"],
        "konum_recall":a ["konum"]/g ,"yonlu_recall":ry ,
        "f1_tavani":2 *ry /(1 +ry ),
        "aday_parca":a ["candidate"]/p ,"secenek_parca":a ["secenek"]/p }
        print (f"{m :<7}{a ['part']:>6}{a ['gt']:>7}{a ['konum']/g :>9.4f}"
        f"{ry :>9.4f}{2 *ry /(1 +ry ):>11.4f}"
        f"{a ['candidate']/p :>8.0f}{a ['secenek']/p :>8.0f}")
    g =max (top ["gt"],1 )
    p =max (top ["part"],1 )
    ry =top ["yonlu"]/g 
    T ={"part":top ["part"],"gt":top ["gt"],
    "konum_recall":top ["konum"]/g ,"yonlu_recall":ry ,
    "f1_tavani":2 *ry /(1 +ry ),
    "aday_parca":top ["candidate"]/p ,"secenek_parca":top ["secenek"]/p }
    print (f"{'TOPLAM':<7}{top ['part']:>6}{top ['gt']:>7}"
    f"{T ['konum_recall']:>9.4f}{ry :>9.4f}{T ['f1_tavani']:>11.4f}"
    f"{T ['aday_parca']:>8.0f}{T ['secenek_parca']:>8.0f}")
    # TAVAN DOYGUNLUGU UYARISI (2026-08-12). Aday basina secenek count
    # `direction_bank.MAX_SEC` tavanina dayanmissa, direction KAYNAKLARINI zenginlestirmek
    # (for example yelpaze cozunurlugunu artirmak) recall'u ARTIRAMAZ: new yonler
    # tavana takilip mevcutlarin instead of geciyordur. Bu, olcumu yorumlarken
    # kolayca gozden kacan a kisittir -- before ceiling buyutulmelidir.
    # KORPUSUN KURULDUGU TAVAN, BU SURECIN TAVANI DEGILDIR. Onbellek hangi
    # `YB_MAX_SEC` with cikarildiysa doygunluk ona according to olculur; surecin own
    # varsayilanina (12) bakmak ceiling-24 korpusunda YANLIS ALARM produces
    # (17.2 secenek/candidate "doygun" sanilir, oysa 24'un %72'si). Korpus tavani
    # npz'de yazili olmadigi for cevreden verilir.
    try :
        _sp =T ["secenek_parca"]/max (T ["aday_parca"],1e-9 )
        _cap =os .environ .get ("HT_KORPUS_MAXSEC")
        if _cap is None :
            import direction_bank as _YB 
            _cap =_YB .MAX_SEC 
            _kaynak =f"surec varsayilani {_cap } (HT_KORPUS_MAXSEC verilmedi)"
        else :
            _cap =int (_cap )
            _kaynak =f"corpus tavani {_cap }"
        if _sp >=0.9 *_cap :
            print (f"\n!! TAVAN DOYGUN: candidate basina {_sp :.1f} secenek, "
            f"{_kaynak }. Yon kaynagi eklemek recall'u ARTIRMAZ; "
            f"once ceiling buyutulmeli (probe_max_sec.py).")
        else :
            print (f"\n   ceiling doygun DEGIL: candidate basina {_sp :.1f} secenek, "
            f"{_kaynak }")
    except Exception :
        pass 
    print (f"\nKAPI A: yonlu recall >= 0.85 mi -> "
    f"{'GECTI'if ry >=0.85 else 'GECMEDI'} ({ry :.4f})")
    json .dump ({"damga":makbuz_hash .damga (),"dizin":DIZ ,"cluster":ONLER ,
    "toplam":T ,"brand":out ,
    "kapi_a_gecti":bool (ry >=0.85 ),
    "not":"Havuz TAVANI (mukemmel selector). Kabul kutusu urun "
    "metrigiyle birebir. D7'ye BAKILMADI."},
    open (_makbuz_yolu (),"w"),indent =1 )
    print (f"receipt -> {_makbuz_yolu ()}")


if __name__ =="__main__":
    main ()
