# -*- coding: utf-8 -*-
"""UCUZ TEST (insan-etiketi YOK): STEP B-rep'inin, 6000-remesh'te ATTIGIMIZ huni/koni geometrisi
wire-vs-tool ayrimina YENI sinyal katiyor mu? Hipotez: kablo-girisleri konik-huni (Einführtrichter)
with, alet-delikleri not. extract_cylinders already koni cikariyor.
Test: each STEP silindir-acikligi -> wire(manufacturer GT'ye yakin) vs non-wire etiketle -> koni-varligi
this etiketi ayiriyor mu (AUC). AUC>>0.5 -> real lever. ~0.5 -> huni hipotezi olu.
DURUST: this only 'sinyal VAR mi' testi; entegrasyon/kazanc ayri."""
import os ,sys ,json ,random 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from step_openings import extract_cylinders 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh 
from big_arbiter import eligible 

# Argv SAGLAM okunur: this file pytest adlandirmasina uyuyor and toplanirsa argv pytest'in
# own bayraklarini icerir ('-q' like). Eskiden int('-q') COLLECTION'i patlatiyordu.
def _argv_n (vars_ =80 ):
    for a in sys .argv [1 :]:
        try :
            return int (a )
        except ValueError :
            continue 
    return vars_ 

N =_argv_n ()
oos =set (open ("pxc_out_of_scope.txt").read ().split ())if os .path .exists ("pxc_out_of_scope.txt")else set ()
allp =[(m ,p ,jf ,s )for m ,p ,jf ,s in eligible ()if not (m =="PXC"and p in oos )]
random .seed (0 );random .shuffle (allp )
parts =allp [:N ]

rows =[]# (radius, has_cone, cone_dist, min_cone_taper, y_wire, mfg)
gt_hit =gt_tot =0 
for mfg ,pid ,jf ,stp in parts :
    try :
        j =json .load (open (jf ,encoding ="utf-8-sig"))
        Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
        G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j ["ConnectionPoints"]],float )
        Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j ["ConnectionPoints"]],float )
        if not len (G ):continue 
        Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
        cyls ,bbox =extract_cylinders (stp ,include_cones =True )
        if not cyls :continue 
        Vr ,Fr =step_to_mesh (stp )
        R ,t ,_ =align_frames (Vr ,Vj )
        Gs =(G -t )@R ;Gds =Gd @R # GT -> STEP/gmsh frame
        cyl =[c for c in cyls if c ["kind"]=="cyl"]
        cone =[c for c in cyls if c ["kind"]=="cone"]
        conePos =np .array ([c ["center"]for c in cone ])if cone else np .zeros ((0 ,3 ))
        diag =float (np .linalg .norm (bbox [3 :]-bbox [:3 ]))
        tol =max (4.0 ,0.06 *diag )
        for c in cyl :
            ctr =np .asarray (c ["center"],float )
            # wire etiketi: this silindir manufacturer GT'ye axis-aware yakin mi
            ywire =0 
            if len (Gs ):
                d =Gs -ctr ;al =(d *Gds ).sum (1 )
                perp =np .linalg .norm (d -al [:,None ]*Gds ,axis =1 )
                perp =np .where (np .abs (al )<=40.0 ,perp ,np .inf )
                if perp .min ()<=tol :ywire =1 
                # KONI feature: most yakin koni mesafesi + present mi (<8mm)
            if len (conePos ):
                cd =float (np .linalg .norm (conePos -ctr ,axis =1 ).min ())
            else :
                cd =999.0 
            has_cone =1.0 if cd <=8.0 else 0.0 
            rows .append ((float (c ["radius"]),has_cone ,min (cd ,50.0 ),ywire ,mfg ))
            # sanity: GT'lerin kacini silindir yakaliyor (frame kontrol)
        if len (cyl ):
            cc =np .array ([c ["center"]for c in cyl ])
            for g ,gd in zip (Gs ,Gds ):
                dd =cc -g ;aa =(dd *gd ).sum (1 );pp =np .linalg .norm (dd -aa [:,None ]*gd ,axis =1 )
                pp =np .where (np .abs (aa )<=40.0 ,pp ,np .inf )
                gt_tot +=1 ;gt_hit +=int (pp .min ()<=tol )
    except Exception as e :
        continue 

R_ =np .array ([r [0 ]for r in rows ]);HC =np .array ([r [1 ]for r in rows ])
CD =np .array ([r [2 ]for r in rows ]);Y =np .array ([r [3 ]for r in rows ]);MF =np .array ([r [4 ]for r in rows ])
print (f"{len (parts )} part -> {len (rows )} STEP silindir-opening | wire {int (Y .sum ())} / non-wire {int ((Y ==0 ).sum ())}")
print (f"FRAME sanity: GT'lerin {gt_hit }/{gt_tot } = {gt_hit /max (gt_tot ,1 ):.2f}'i bir silindire denk (dusukse frame bozuk)")
if Y .sum ()<5 or (Y ==0 ).sum ()<5 :
    print ("!!! yetersiz ornek, cikiyorum");sys .exit ()

from sklearn .metrics import roc_auc_score 
print (f"\n=== wire-vs-nonwire AYIRICI GUCU (AUC, tek feature) ===")
print (f"  radius       : AUC {roc_auc_score (Y ,R_ ):.3f}")
print (f"  has_cone(<8mm): AUC {roc_auc_score (Y ,HC ):.3f}   <-- HUNI hipotezi (>>0.5 ise sinyal VAR)")
print (f"  -cone_dist   : AUC {roc_auc_score (Y ,-CD ):.3f}")
print (f"  wire'da koni-var ratio: {HC [Y ==1 ].mean ():.2f} | non-wire'da: {HC [Y ==0 ].mean ():.2f}")
# birlesik (radius+cone) RF OOF
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import cross_val_predict 
X1 =R_ .reshape (-1 ,1 );Xc =np .column_stack ([R_ ,HC ,CD ])
p1 =cross_val_predict (RandomForestClassifier (200 ,min_samples_leaf =3 ,random_state =0 ),X1 ,Y ,cv =5 ,method ="predict_proba")[:,1 ]
pc =cross_val_predict (RandomForestClassifier (200 ,min_samples_leaf =3 ,random_state =0 ),Xc ,Y ,cv =5 ,method ="predict_proba")[:,1 ]
print (f"\n=== BIRLESIK (5-fold OOF) ===")
print (f"  radius-only        : AUC {roc_auc_score (Y ,p1 ):.3f}")
print (f"  radius+cone+conedist: AUC {roc_auc_score (Y ,pc ):.3f}   <-- koni EKLEMEK yardim ediyor mu")
for m in ("WEI","PXC"):
    mk =MF ==m 
    if mk .sum ()>10 and Y [mk ].sum ()>3 and (Y [mk ]==0 ).sum ()>3 :
        print (f"    {m }: has_cone AUC {roc_auc_score (Y [mk ],HC [mk ]):.3f} (wire koni-ratio {HC [mk &(Y ==1 )].mean ():.2f} vs non {HC [mk &(Y ==0 )].mean ():.2f})")
print ("\nKARAR: has_cone AUC >~0.65 -> HUNI GERCEK SINYAL, mevcut 13 feature'a ekle+olc degerli.")
print ("       ~0.5 -> huni yok/ayirt etmiyor, bu lever de olu (durustce kapat).")
