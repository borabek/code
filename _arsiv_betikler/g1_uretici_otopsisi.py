# -*- coding: utf-8 -*-
"""G1: URETICI OTOPSISI -- neden ELMEX 0.615 but most kotusu 0.054?

Gorulmemis manufacturer sinavinda spread 0.5613 (most iyi 0.6154 / most kotu 0.0541). Ortalamayi
kovalamadan ONCE this yayilimin SEBEBINI bilmek is required: kotu ureticilerde kaybin baskin
kovasi hangisi?

  ADAY_YOK       havuzda never candidate absent            -> TEMSIL (G5)
  GATE_REDDI     candidate vardi, gate reddetti       -> GATE (G2/G3/G4)
  ADAY_KALABALIK candidate present but BASKA GT'ye gitti  -> COZUNURLUK
  ATAMA          kabul edildi, baska GT'ye gitti -> METRIK

BU DECISION VERICIDIR: kotu ureticilerde GATE_REDDI baskinsa Gun 1'in gate kollari (G2/G3/G4)
gercekten whereas yarar; ADAY_YOK baskinsa that kollar TAVANA CARPACAK and dogrudan G5'e (temsil)
gecmek is required. Kol acmadan before this tablo cikar.

YAN OLCUMLER (ureticiye ozgu a SISTEMATIK present mi):
  * part olcegi (diag) and CP yogunlugu -- tolerans max(3, 0.06*diag) olcege bagli
  * cerceve residual -- GT with mesh'in median lateral mesafesi (D5-4 layer 4'un olcusu)
  * candidate/GT orani -- gate very mu uretiyor few mi

TEZ DEGISMEZ: no sey egitilmez, no threshold does not change. Bu a MUHASEBE betigi.
"""
import collections 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import wire_gate 
from f0_4_hata_bankasi import esle 
from f2_12_veri_kolu import egit 

SINAV ="results/_der_sinav_yeni.pkl"
MAKBUZ ="results/g1_uretici_otopsisi.json"


def main ():
    import protocol 
    protocol .tez_dogrula ()
    with open (SINAV ,"rb")as f :
        DER =pickle .load (f )
    model ,bilgi =egit ("results/zengin_parite_v3.npz")
    print (f"exam {len (DER )} part | gate: v3 ({bilgi ['part']} part / "
    f"{bilgi ['manufacturer']} manufacturer)\n")

    U =collections .defaultdict (lambda :{"tp":0 ,"fp":0 ,"fn":collections .Counter (),
    "diag":[],"ngt":[],"nad":[],"res":[]})
    for r in DER :
        if r ["X"]is None or r .get ("XR")is None :
            continue 
        M =np .hstack ([r ["X"],r ["XR"]]).astype (float )
        if M .shape [1 ]*2 !=model ["n_feat"]:
            continue 
        k =wire_gate .decision_mask (wire_gate .decision_score (model ,M ))
        Ptum =np .asarray (r ["P"],float )
        P =Ptum [k ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        tol =max (3.0 ,0.06 *float (r ["diag"]))
        u =U [r ["mfg"]]
        cift ,fn_idx =esle (P ,G ,Gd ,tol )
        u ["tp"]+=len (cift );u ["fp"]+=len (P )-len (cift )
        u ["diag"].append (float (r ["diag"]));u ["ngt"].append (len (G ));u ["nad"].append (len (P ))

        cift_tum ,_ =esle (Ptum ,G ,Gd ,tol )
        gt_tum ={v :a for a ,v in cift_tum .items ()}
        if len (Ptum ):
            dd =Ptum [:,None ,:]-G [None ,:,:]
            aa =(dd *Gd [None ,:,:]).sum (-1 )
            pp =np .linalg .norm (dd -aa [...,None ]*Gd [None ,:,:],axis =-1 )
            u ["res"].append (float (np .median (np .where (np .abs (aa )>40 ,np .inf ,pp ).min (0 ))))
            yakin =(np .where (np .abs (aa )>40 ,np .inf ,pp ).min (0 )<=tol )
        else :
            yakin =np .zeros (len (G ),bool )
        for b in fn_idx :
            if b not in gt_tum :
                u ["fn"]["ADAY_YOK"if not yakin [b ]else "ADAY_KALABALIK"]+=1 
            elif not k [gt_tum [b ]]:
                u ["fn"]["GATE_REDDI"]+=1 
            else :
                u ["fn"]["ATAMA"]+=1 

    sat =[]
    for m ,u in U .items ():
        fn =sum (u ["fn"].values ())
        f1 =2 *u ["tp"]/max (2 *u ["tp"]+u ["fp"]+fn ,1 )
        sat .append ((f1 ,m ,u ,fn ))
    sat .sort ()

    print (f"{'manufacturer':<8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}   "
    f"{'ADAY_YOK':>9}{'GATE_RED':>9}{'KALABALIK':>10}{'ATAMA':>7}")
    for f1 ,m ,u ,fn in sat :
        c =u ["fn"]
        print (f"{m :<8}{f1 :>8.4f}{u ['tp']:>6}{u ['fp']:>6}{fn :>6}   "
        f"{c ['ADAY_YOK']:>9}{c ['GATE_REDDI']:>9}{c ['ADAY_KALABALIK']:>10}"
        f"{c ['ATAMA']:>7}")

    print (f"\n{'manufacturer':<8}{'F1':>8}{'part':>7}{'ort diag':>10}{'ort GT':>8}"
    f"{'candidate/GT':>9}{'cerceve':>10}")
    for f1 ,m ,u ,fn in sat :
        ag =np .mean (u ["nad"])/max (np .mean (u ["ngt"]),1e-9 )
        print (f"{m :<8}{f1 :>8.4f}{len (u ['diag']):>7}{np .mean (u ['diag']):>10.1f}"
        f"{np .mean (u ['ngt']):>8.1f}{ag :>9.2f}"
        f"{np .median (u ['res'])if u ['res']else float ('nan'):>9.2f}mm")

        # DECISION: kotu yaridaki ureticilerde baskin kova
    n =len (sat )//2 
    kotu =collections .Counter ();iyi =collections .Counter ()
    for i ,(f1 ,m ,u ,fn )in enumerate (sat ):
        (kotu if i <n else iyi ).update (u ["fn"])
    tk ,ti =sum (kotu .values ()),sum (iyi .values ())
    print (f"\nKOTU YARI ({n } manufacturer, {tk } FN) vs IYI YARI ({ti } FN):")
    for kv in ("ADAY_YOK","GATE_REDDI","ADAY_KALABALIK","ATAMA"):
        print (f"  {kv :<16} kotu %{100 *kotu [kv ]/max (tk ,1 ):>5.1f}   "
        f"iyi %{100 *iyi [kv ]/max (ti ,1 ):>5.1f}")
    bask =kotu .most_common (1 )[0 ][0 ]if tk else "-"
    print (f"\nKOTU YARIDA BASKIN KOVA: {bask }")
    if bask =="GATE_REDDI":
        print ("  -> G2/G3/G4 (gate kollari) GERCEKTEN ise yarayabilir; Gun 1 plani GECERLI")
    elif bask =="ADAY_YOK":
        print ("  -> gate kollari TAVANA CARPAR; dogrudan G5'e (TEMSIL) gecilmeli")
    else :
        print ("  -> cozunurluk/metrik kovasi baskin; G2-G4'ten once o incelenmeli")

    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({m :{"F1":f1 ,"tp":u ["tp"],"fp":u ["fp"],"fn":dict (u ["fn"]),
        "part":len (u ["diag"]),"ort_diag":float (np .mean (u ["diag"])),
        "ort_gt":float (np .mean (u ["ngt"])),
        "cerceve_ortanca":float (np .median (u ["res"]))if u ["res"]else None }
        for f1 ,m ,u ,fn in sat },
        f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
