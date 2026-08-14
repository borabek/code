# -*- coding: utf-8 -*-
"""SEG-1 — URETICI GT'sinden KISMI SEGMENTASYON ETIKETI URET

WHY. Canli segmentasyon kontrol noktalari **27 Temmuz** tarihli and
~200 parcalik a veriyle egitilmis. Kampanya along SECICI optimize
edildi but altindaki segmentasyon DONDURULMUS kaldi. Olculen sonuc:
NIT'te GT'de olasilik 0.5244 / rastgele yuzeyde 0.4394 (1.19x), SUPU'da
83x. Yani zincirin EN BASI dense parcada bilgi uretmiyor
(`results/otopsi_segmentasyon.json`).

Elimizde `full` kumesinde 2583 parcanin URETICI GT'si present. Egitici already
`--partial-dir` with "kablo girisi signed, gerisi MASKELI" formatini
kabul ediyor (simdiye up to only elle isaretlenmis birkac face part).
Bu betik same formati GT'den corpus olceginde produces.

DONGUSEL DEGILDIR: labels model ciktisindan not URETICI JSON'undan
gelir.

SIZINTI KISITI -- EN IMPORTANT RULE. Yalniz `full` parcalari boyanir.
  * D7 = SINAV; GT'si segmentasyona girerse exam yanar.
  * d6 = benim gelistirme kumem; girerse d6'daki each measurement gecersiz becomes.
Bekci: D7/d6 kimlikleri KUMEDEN cikarilir and sayilir; sifir olmalidir.

YARICAP. GT a NOKTADIR, kablo girisi a BOLGEDIR. `PAINT_R` (varsayilan
2.0 mm) lateral kabul toleransiyla same secildi; GT'lerin %98.1'i already
2mm'de a tepeye komsu (`mesh-ceiling-not-lateral-model-hatasi`).
"""
import os 
import sys 
import time 

import numpy as np 
from scipy .spatial import cKDTree 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import connector3d # noqa: E402
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402

CIK =os .environ .get ("GK_CIK","_label_targets_gt2")
MESH =os .environ .get ("GK_MESH","results/_p1_olasilik_brepegit")
PAINT_R =float (os .environ .get ("GK_R","2.0"))
DERIN =float (os .environ .get ("GK_DERIN","6.0"))# axis along ICERI
AGIZ_PAY =float (os .environ .get ("GK_AGIZ","1.0"))# mouth disina small pay
CE =int (connector3d .CABLE_ENTRY )


def main ():
    t0 =time .time ()
    os .makedirs (CIK ,exist_ok =True )
    kay =K .yukle ()
    # --- SIZINTI BEKCISI. Ilk yazimda `d6_record.yukle()` kullanmistim and
    # 6074 kimlik donduruyordu -- that a GENEL KAYIT DEPOSU, d6 bolmesi
    # DEGIL. Bekci correct calisti but wrong listeyle: 3418 parcanin hepsini
    # eledi and HIC label uretilmedi. Gercek split, feature dosyalarinin
    # ONEKINDEN okunur (`_tam_oz/{cluster}_{pid}.npz`) -- kollarin kullandigi
    # bolmenin ta kendisi.
    OZ ="results/_tam_oz"
    def _kume_kimlikleri (on ):
        return {f [len (on )+1 :-4 ]for f in os .listdir (OZ )
        if f .startswith (on +"_")and f .endswith (".npz")}
    tam =_kume_kimlikleri ("tam")
    d6 =_kume_kimlikleri ("d6")
    d7 =_kume_kimlikleri ("d7")
    yasak =d6 |d7 
    print (f"split: tam {len (tam )} | d6 {len (d6 )} | d7 {len (d7 )}",flush =True )
    print (f"leakage bekcisi: {len (yasak )} kimlik (d6+d7) DISARIDA",
    flush =True )
    assert tam and not (tam &yasak ),"tam kumesi d6/d7 with KESISIYOR"

    yazilan =atlanan =empty_ =leakage =0 
    poz_top =tepe_top =0 
    for pid ,r in sorted (kay .items ()):
        pid =str (pid )
        if pid in yasak or pid not in tam :
            leakage +=1 
            continue 
        hedef =f"{CIK }/{pid }"
        if os .path .exists (f"{hedef }/{pid }.labels.txt"):
            atlanan +=1 
            continue 
        G =np .asarray (r .get ("G",[]),float )
        mf =f"{MESH }/{pid }.npz"
        if not len (G )or not os .path .exists (mf ):
            empty_ +=1 
            continue 
        z =np .load (mf )
        V =np .asarray (z ["V"],float )
        F =np .asarray (z ["F"],int )
        if not len (V )or not len (F ):
            empty_ +=1 
            continue 
        L =np .zeros (len (V ),np .int64 )
        agac =cKDTree (V )
        # SEKIL DUZELTMESI (2026-08-13). Ilk version GT noktasinin cevresine
        # 2mm KURE boyuyordu. Kablo girisi a kure DEGIL, axis along
        # uzanan a KANAL YUZEYIDIR; kure ne that sekli ne that yonu carries --
        # model wrong shape ogreniyordu (B kolu 0.5540 vs A 0.6232).
        # Dogrusu: GT EKSENI etrafinda SILINDIRIK KABUK.
        #   * eksene YANAL uzaklik <= PAINT_R
        #   * axis along DISARI not ICERI: [-DERIN, +AGIZ_PAY]
        # GT yonu DISARI baktigi for (sozlesme 1.000 disari) ic taraf
        # negatif axial yondur.
        Gd =np .asarray (r .get ("Gd",[]),float )
        Gn =(Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
        if len (Gd )==len (G )else None )
        for j in range (len (G )):
            kom =agac .query_ball_point (G [j ],PAINT_R +DERIN )
            if not kom :
                continue 
            kom =np .asarray (kom ,int )
            v =V [kom ]-G [j ][None ,:]
            if Gn is None :
                m_ =np .linalg .norm (v ,axis =1 )<=PAINT_R 
            else :
                al =v @Gn [j ]
                yan =np .linalg .norm (v -al [:,None ]*Gn [j ][None ,:],axis =1 )
                m_ =(yan <=PAINT_R )&(al <=AGIZ_PAY )&(al >=-DERIN )
            L [kom [m_ ]]=CE 
        if not (L >0 ).any ():
            empty_ +=1 
            continue 
        os .makedirs (hedef ,exist_ok =True )
        with open (f"{hedef }/{pid }.obj","w")as f :
            for v in V :
                f .write (f"v {v [0 ]:.6f} {v [1 ]:.6f} {v [2 ]:.6f}\n")
            for t in F :
                f .write (f"f {t [0 ]+1 } {t [1 ]+1 } {t [2 ]+1 }\n")
        with open (f"{hedef }/{pid }.labels.txt","w")as f :
            f .write ("\n".join (str (int (x ))for x in L ))
        poz_top +=int ((L >0 ).sum ())
        tepe_top +=len (L )
        yazilan +=1 
        if yazilan %100 ==0 :
            print (f"  {yazilan } yazildi ({time .time ()-t0 :.0f} s)",
            flush =True )

    print (f"\nBITTI: yazilan {yazilan } | cache {atlanan } | bos {empty_ }")
    print (f"SIZINTI BEKCISI: {leakage } part (d6/d7) DISARIDA BIRAKILDI")
    if tepe_top :
        print (f"signed tepe orani: {poz_top /tepe_top :.4f} "
        f"({poz_top } / {tepe_top })")
        print ("  (elle etiketli korpusta kablo girisi ~%1.5; benzer must be)")
    print (f"-> {CIK }")


if __name__ =="__main__":
    main ()
