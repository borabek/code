# -*- coding: utf-8 -*-
"""v6 FARKI, ADIM B: ETIKET TANIMINI DUZELT and genisletilmis havuzu YENIDEN sina.

ADIM A SONUCU: benim recetem v6'nin KENDI korpusunda full **0.1970 / 0.4292**
verdi -- v6'nin kendisiyle birebir. Yani recete DOGRUYDU, difference KORPUSTAYDI.
(Yan kazanc: v6'nin loss kunyesi geri kazanildi -> v6 = refit_gate_v7 recetesi
+ `zengin_parite_v6.npz`.)

FARKIN KAYNAGI BULUNDU (`build_rich_parity.py:142-152`). Projenin etiketi:
  * distance EKSENE DIK (lateral), Oklid DEGIL
  * axial offset <= 40mm AYRI gate
  * ACGOZLU BIRE-BIR eslesme (each candidate a times, each GT a times)
Benimki `min Oklid <= tol` idi: ne lateral/axial ayrimi ne bire-birlik. Sonuc:
axis along uzaktaki DOGRU candidates negatif, a GT cevresindeki TUM candidates
pozitif yazilmis -- gate'e "GT'ye yakin each sey iyi" ogretilmis.

Bu betik onbellekteki OZNITELIKLERE DOKUNMADAN etiketi projenin tanimiyla
yeniden produces, two havuzu same recetede egitir and D7'de olcer:
  A) TEZ-SAF pool   -> hedef ~0.1970 (v6 seviyesi). Cikarsa teshis DOGRULANIR.
  B) GENISLETILMIS   -> havuzun olculmus +0.0386'si here gerceklesmeli.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 
from sklearn .ensemble import RandomForestClassifier 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402
import brep_pool # noqa: E402
import wire_gate # noqa: E402
from p1c_threshold import maske # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

TUM ={}
OZ ="results/_brep_oz"
DONUSUM ="zskor"
EKSENEL =40.0 


def project_label (P ,G ,Gd ,diag ):
    """`build_rich_parity.py` with BIREBIR same tanim.

    Yanal distance + axial 40mm kapisi + acgozlu BIRE-BIR eslesme.
    """
    y =np .zeros (len (P ),int )
    if not len (P )or not len (G ):
        return y 
    tol =max (3.0 ,0.06 *float (diag ))
    Gn =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
    diff =P [:,None ,:]-G [None ,:,:]
    al =(diff *Gn [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (diff -al [...,None ]*Gn [None ,:,:],axis =-1 )
    pe =np .where (np .abs (al )<=EKSENEL ,pe ,np .inf )
    up ,ug =set (),set ()
    for dd ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
    for a in range (len (P ))for b in range (len (G ))):
        if dd >tol or a_ in up or b_ in ug :
            continue 
        up .add (a_ );ug .add (b_ );y [a_ ]=1 
    return y 


CY ={}
AC ={}
for _f in ("results/_brepegit_silindirler.pkl","results/_d6_silindirler.pkl"):
    CY .update (pickle .load (open (_f ,"rb")))
for _f in ("results/_brepegit_acikliklar.pkl","results/_d6_acikliklar.pkl"):
    AC .update (pickle .load (open (_f ,"rb")))


def training (segtek ):
    atlanan_uyumsuz =[]
    R ={str (r ["pid"]):r for r in pickle .load (open (K .KAYIT ,"rb"))}
    R6 =d6_record .yukle (set (d6_record .exam ()["pidler"]))
    M ,Y =[],[]
    eski_poz =yeni_poz =n =0 
    for f in sorted (os .listdir (OZ )):
        if not (f .endswith (".npz")and (f .startswith ("tam_")or f .startswith ("d6_"))):
            continue 
        on ="tam"if f .startswith ("tam_")else "d6"
        pid =f [len (on )+1 :-4 ]
        r =R .get (pid )or R6 .get (pid )
        if r is None :
            continue 
        z =np .load (f"{OZ }/{f }")
        X =np .asarray (z ["X"],float )
        nseg =len (np .asarray (r ["P"],float ))
        if nseg >len (X ):
            continue 
            # `source` tam_* dosyalarinda saklanmadi; seg adaylari HER ZAMAN basta
        kay =np .zeros (len (X ),int )
        kay [nseg :]=1 
        m =(kay ==0 )if segtek else np .ones (len (X ),bool )
        Xm =X [m ]
        if len (Xm )<2 :
            continue 
            # KONUM havuzunu same maskeyle kur.
            # `tam_*` dosyalarinda P SAKLANMADI (only X,y). Onu atlamak korpusun
            # 2583 parcasini SESSIZCE dusuruyordu and training 468 parcaya iniyordu --
            # full da "training buyuklugu karistiricisi" tuzagi. P yeniden HESAPLANIR:
            # `merged_pool` deterministiktir and features same cagriyla
            # uretilmisti, i.e. SIRA birebir ortusur.
        if "P"in z :
            Pb =np .asarray (z ["P"],float )
        else :
            Pb ,_Db ,_kk =brep_pool .merged_pool (
            np .asarray (r ["P"],float ),np .asarray (r ["Pd"],float ),
            CY .get (pid ),AC .get (pid ))
            if len (Pb )!=len (X ):
                atlanan_uyumsuz .append (pid )
                continue 
        Pm =Pb [m ]
        y =project_label (Pm ,np .asarray (r ["G"],float ),
        np .asarray (r ["Gd"],float ),float (r ["diag"]))
        eski_poz +=int (z ["y"][m ].sum ())
        yeni_poz +=int (y .sum ())
        n +=len (y )
        M .append (wire_gate .within_part (Xm ,DONUSUM ))
        Y .append (y )
    M =np .vstack (M )
    Y =np .concatenate (Y )
    if atlanan_uyumsuz :
        print (f"  UYUMSUZ (P yeniden hesabi tutmadi): {len (atlanan_uyumsuz )} part",
        flush =True )
    print (f"  training {M .shape } | ESKI etiket pozitif {eski_poz /max (n ,1 ):.4f} -> "
    f"PROJE etiketi {yeni_poz /max (n ,1 ):.4f}",flush =True )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M ,Y )
    return {"clf":clf ,"cols":None ,"n_feat":M .shape [1 ],"donusum":DONUSUM }


def exam ():
    te =[]
    for f in sorted (os .listdir (OZ )):
        if not (f .startswith ("d7_")and f .endswith (".npz")):
            continue 
        z =np .load (f"{OZ }/{f }")
        te .append ({"pid":f [3 :-4 ],"X":np .asarray (z ["X"],float ),
        "P":z ["P"],"D":z ["D"],"source":z ["source"]})
    kay =K .yukle ([d ["pid"]for d in te ])
    for d in te :
        r =kay [d ["pid"]]
        d .update ({"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
        "Gd":np .asarray (r ["Gd"],float ),"diag":r ["diag"]})
    return te 


def olc (model ,te ,segtek ):
    en =None 
    for tip ,e in ([("mutlak",x )for x in (0.20 ,0.30 ,0.40 )]+
    [("goreli",x )for x in ((0.5 ,0.20 ),(0.5 ,0.30 ))]):
        rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
        tes =[]
        for d in te :
            m =(d ["source"]==0 )if segtek else np .ones (len (d ["source"]),bool )
            X =d ["X"][m ]
            if len (X )<2 :
                continue 
            s =np .asarray (wire_gate .decision_score (model ,X ),float )
            k =(s >=e )if tip =="mutlak"else maske (s ,e [0 ],e [1 ])
            P ,D =d ["P"][m ],d ["D"][m ]
            P ,D =(P [k ],D [k ])if k .any ()else (P [:0 ],D [:0 ])
            if len (P )>1 :
                nm =wire_gate .crowd_mask (P ,s [k ])
                P ,D =P [nm ],D [nm ]
            tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,
            K .ACI ,False ,signed =True )[:3 ]
            a =rob [d ["mfg"]]
            a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
            tes .append ((len (d ["G"]),)+match_hungarian (
            P ,D ,d ["G"],d ["Gd"],d ["diag"],
            max (3.0 ,0.06 *d ["diag"]),180.0 ,True )[:3 ])
        pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
        mi =float (2 *sum (a [0 ]for a in rob .values ())/
        max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
        r ={"rule":f"{tip } {e }","robot":mi ,"tespit":K .mikro (tes ),
        "makro":float (np .mean (list (pm .values ()))),
        "en_kotu":float (min (pm .values ())),"brand":pm }
        TUM .setdefault (id (model ),[]).append (r )
        if en is None or r ["robot"]>en ["robot"]:
            en =r 
    return en 


def main ():
    te =exam ()
    print (f"SINAV {len (te )} part\n",flush =True )
    out ={}
    for ad ,segtek in (("A) TEZ-SAF + proje etiketi",True ),
    ("B) GENISLETILMIS + proje etiketi",False )):
        print (ad ,flush =True )
        m =training (segtek )
        out [ad ]=olc (m ,te ,segtek )
        for rr in TUM .get (id (m ),[]):
            print (f"     {rr ['rule']:<16} robot {rr ['robot']:.4f} | tespit "
            f"{rr ['tespit']:.4f} | makro {rr ['makro']:.4f} | en kotu "
            f"{rr ['en_kotu']:.4f}",flush =True )
        c =out [ad ]
        print (f"  -> robot {c ['robot']:.4f} | tespit {c ['tespit']:.4f} | makro "
        f"{c ['makro']:.4f} | en kotu {c ['en_kotu']:.4f} | {c ['rule']}\n",
        flush =True )
    a =out ["A) TEZ-SAF + proje etiketi"]["robot"]
    b =out ["B) GENISLETILMIS + proje etiketi"]["robot"]
    print (f"ESKI ETIKETLE tez-saf:  0.1159")
    print (f"PROJE ETIKETIYLE:       {a :.4f}   (v6 seviyesi 0.1970)")
    print (f"GENISLETILMIS pool:    {b :.4f}   (fark {b -a :+.4f})")
    print (f"DAGITILAN URUN (NMS'li): 0.2029")
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,
    "eski_etiketle_tezsaf":0.1159 ,"v6":0.1970 ,"urun":0.2029 ,
    "not":"Etiket projenin tanimiyla (lateral + axial 40mm + acgozlu "
    "bire-a) yeniden uretildi; OZNITELIKLER onbellekten, "
    "DEGISMEDI. D7 brand-disi, MIKRO."},
    open ("results/v6_farki_D.json","w"),indent =1 )
    print ("receipt -> results/v6_farki_B.json")


if __name__ =="__main__":
    main ()
