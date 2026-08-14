# -*- coding: utf-8 -*-
"""F2-11b: STEP'SIZ 2759 PARCANIN JSON AGI KULLANILABILIR MI? (STEP GEREKMEZ)

F2-11 SONUCU: JSON agi genel as parite SAGLAMADI (candidate recall 0.8579 -> 0.5374).
AMA manufacturer kirilimi kapiyi kapatmadi:

    PXC   STEP recall 0.8450 -> JSON 0.2999   (-0.5451)  COKUS
    WEI   STEP recall 0.8764 -> JSON 0.8858   (+0.0094)  KAYIP YOK

Ve this COZUNURLUK DEGIL: two ureticinin JSON agi neredeyse same yogunlukta (medyan
1934 vs 1926 vertex). Yani sorun URETICIYE OZGU TESSELASYON BICIMI -- WEI'nin agi
acikliklari koruyor, PXC'ninki kapatiyor.

BU BETIGIN FIKRI: candidate recall'i olcmek for STEP GEREKMEZ -- only candidates and GT yeter.
Dolayisiyla STEP'i HIC OLMAYAN ureticiler (A-B 822, CWT 785, ABB 324, SIE 311, KLM 101)
for de recall DOGRUDAN olculebilir. Sonuc WEI'ye benziyorsa that data KULLANILABILIR;
PXC'ye benziyorsa kullanilamaz.

TEZ CIZGISI: same network, same uniform ~6000 remesh, same candidate ureticisi. Degisen single sey
mesh kaynagi. STEP olmadigi for B-rep ozellikleri uretilmez (gate'e girmez) but RECALL
gate'ten ONCEKI asamadir -- this measurement ondan etkilenmez.

UC SISME KAYNAGI VE NASIL KAPATILDIGI (this measurement kolayca yaniltir):

  1. ADAY SAYISI SISMESI. Recall, candidate uretirsen BEDAVAYA yukselir. Bu yuzden each
     satirda ADAY/GT ORANI da raporlanir. (Referans: STEP kolu 2.43 candidate/GT, JSON kolu
     1.67 -- JSON more AZ candidate uretip more DUSUK recall veriyor, i.e. tutarli.)

  2. HIZALAMA AVANTAJI. JSON yolunda GT already AYNI CERCEVEDE, `align_frames` YOK.
     WEI'nin +0.0094'u TAMAMEN this may be -- real ustunluk not, EKSIK HATA KAYNAGI.
     Bu yuzden JSON recall'i STEP'ten YUKSEK cikan no sonuc "iyilesme" sayilmaz.

  3. EN TEHLIKELISI -- URETICI TANIDIKLIGI. Ag agirlikli as WEI and PXC on
     egitildi (1926 uygun parcanin 1011'i WEI, 896'si PXC). SIE/A-B/CWT'de low recall
     "JSON agi kotu" mu demek, "network that ureticiyi tanimiyor" mu? YALNIZ JSON ILE AYRISTIRILAMAZ.

     COZUM: measurement IKIYE ayrilir.
       ESLESTIRILMIS  (hem STEP hem JSON which is: PXC, WEI, DIN, WAGO) -> difference YALNIZ mesh
                      kaynagini izole eder, tanidiklik sadelesir. ASIL KANIT BUDUR.
       ESLESTIRILMEMIS (A-B, CWT, ABB, SIE, KLM...) -> mutlak recall; TANIDIKLIKLA
                      KARISIK, single basina evidence DEGIL, only gosterge.

KILL (only ESLESTIRILMIS kolda gecerli): STEP->JSON recall kaybi <= 0.05 whereas that
ureticinin JSON agi kullanilabilir. (WEI loss +0.0094 = kayipsiz; PXC -0.5451 = cokus.)
"""
import argparse 
import collections 
import io 
import json 
import os 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

from f2_11_parite import json_mesh 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--per",type =int ,default =12 ,help ="manufacturer basina part")
    a =ap .parse_args ()

    import protocol 
    protocol .tez_dogrula ()
    import torch 
    import diffusionnet as D_ 
    import robot_cp as RC 
    import thesis_remesh 
    from infer_step_cp import load_any 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    man =json .load (io .open ("results/manifest_korpus.json",encoding ="utf-8"))
    # STEP'i OLMAYAN, CP'si and mesh'i which is parts
    candidate =[m for m in man if not m ["step"]and m ["cp"]>0 and m ["g3d_tepe"]>0 ]
    grup =collections .defaultdict (list )
    for m in candidate :
        grup [m ["manufacturer"]].append (m )
    hedef =[]
    rng =np .random .RandomState (0 )
    for u ,v in sorted (grup .items (),key =lambda x :-len (x [1 ])):
        if len (v )<5 :
            continue 
        idx =rng .permutation (len (v ))[:a .per ]
        hedef +=[v [i ]for i in idx ]
    print (f"STEP'siz manufacturer sayisi {len (grup )} | orneklenen {len (hedef )} part")
    print ("  "+", ".join (f"{u }:{min (len (v ),a .per )}"for u ,v in 
    sorted (grup .items (),key =lambda x :-len (x [1 ]))if len (v )>=5 ))

    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]

    SON =collections .defaultdict (lambda :[0 ,0 ,0 ,[]])
    t0 =time .time ();error =0 
    for k ,m in enumerate (hedef ,1 ):
        if k %15 ==0 :
            print (f"  {k }/{len (hedef )}  {time .time ()-t0 :.0f}s  error={error }",flush =True )
        jf =os .path .join ("_ds1/DataSet",m ["dosya"])
        try :
            Vj ,Fj ,j =json_mesh (jf )
            if Vj is None or len (Fj )<4 :
                error +=1 ;continue 
            V ,F =thesis_remesh .remesh_uniform (Vj ,Fj ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            pbs =[]
            for model ,meta in models :
                _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_json_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pbs .append (np .asarray (pb ,float ))
            cps ,probs ,_ ,_ =RC .derive_candidates (V ,F ,pbs ,None ,cfg =cfg )
            P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            g =j .get ("ConnectionPoints")or []
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in g ],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in g ],float )
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))
            hit =0 
            if len (P )and len (G ):
                d =P [:,None ,:]-G [None ,:,:]
                al =(d *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )>40 ,np .inf ,pe )
                hit =int ((pe .min (0 )<=max (3.0 ,0.06 *diag )).sum ())
            s =SON [m ["manufacturer"]]
            s [0 ]+=hit ;s [1 ]+=len (G );s [2 ]+=len (P );s [3 ].append (len (Vj ))
        except Exception as e :
            error +=1 
            if error <=3 :
                print (f"    {m ['part']}: {type (e ).__name__ }: {str (e )[:70 ]}")

    print (f"\n{'='*84 }\nESLESTIRILMEMIS KOL -- STEP'SIZ URETICILERDE JSON-AGI ADAY RECALL\n"
    f"{'='*84 }")
    print ("DIKKAT: bu satirlar TANIDIKLIKLA KARISIK. Ag agirlikli olarak WEI+PXC uzerinde")
    print ("egitildi; gorulmemis bir ureticide dusuk recall 'JSON agi kotu' DEGIL 'ag bu")
    print ("ureticiyi tanimiyor' da olabilir. TEK BASINA KANIT DEGIL -- gosterge.\n")
    print (f"{'manufacturer':<10}{'part':>6}{'GT':>6}{'candidate':>7}{'candidate/GT':>9}{'RECALL':>9}"
    f"{'JSON tepe':>11}   gosterge")
    R ={}
    for u in sorted (SON ,key =lambda x :-SON [x ][1 ]):
        h ,n ,na ,jt =SON [u ]
        rc =h /max (n ,1 )
        ratio =na /max (n ,1 )
        # ADAY SISMESI KONTROLU: candidate/GT orani referansin (STEP 2.43) COK USTUNDEYSE
        # high recall bedavaya gelmis may be; isaretle.
        sis =" [ADAY SISMESI?]"if ratio >3.5 else ""
        hk =("WEI-benzeri"if rc >=0.70 else 
        "SINIRDA"if rc >=0.55 else "PXC-benzeri (cokus)")
        R [u ]={"recall":rc ,"gt":n ,"candidate":na ,"aday_per_gt":ratio ,"part":len (jt ),
        "json_tepe_medyan":float (np .median (jt ))if jt else 0 }
        print (f"{u :<10}{len (jt ):>6}{n :>6}{na :>7}{ratio :>9.2f}{rc :>9.4f}"
        f"{np .median (jt )if jt else 0 :>11.0f}   {hk }{sis }")
    print (f"\n  REFERANS (ESLESTIRILMIS, tanidiklik sadelesmis):")
    print (f"    WEI  STEP 0.8764 -> JSON 0.8858  (+0.0094)  KAYIPSIZ")
    print (f"    PXC  STEP 0.8450 -> JSON 0.2999  (-0.5451)  COKUS")
    print (f"    candidate/GT referansi: STEP kolu 2.43 | JSON kolu 1.67")
    ok =[u for u ,v in R .items ()if v ["recall"]>=0.70 and v ["aday_per_gt"]<=3.5 ]
    print (f"\n  WEI-benzeri davranan (gosterge): {ok if ok else 'YOK'}")
    print ("  KESIN HUKUM ICIN o ureticiden STEP'li birkac part gerekir (eslestirilmis measurement).")
    with io .open ("results/f2_11b_stepsiz_recall.json","w",encoding ="utf-8")as f :
        json .dump ({"referans_eslestirilmis":{"WEI":{"step":0.8764 ,"json":0.8858 },
        "PXC":{"step":0.8450 ,"json":0.2999 }},
        "eslestirilmemis_gosterge":R ,"wei_benzeri":ok ,
        "uyari":"tanidiklikla karisik; tek basina evidence degil"},
        f ,indent =1 ,ensure_ascii =False )
    print ("receipt -> results/f2_11b_stepsiz_recall.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
