# -*- coding: utf-8 -*-
"""KAPI 1: tez-native metrik. Scheffler tezi mean Jaccard 0.514 raporluyor (SEGMENTASYON). Bizim seg
checkpoint'in val 20-part IoU/Dice/acc'i nedir? >=0.51 -> tezi geciyoruz; ~0.80 -> tez-native 0.80.
val = .obj + .labels.txt (vertex 5-sinif). Etiketsiz kacis (val insan-etiketi already present, GT-eksigi absent)."""
import os ,sys ,glob 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D 
from infer_step_cp import load_any 

dev ="cuda"if torch .cuda .is_available ()else "cpu"
NC =5 


def load_obj (path ):
    V ,F =[],[]
    for ln in open (path ):
        if ln .startswith ("v "):V .append ([float (x )for x in ln .split ()[1 :4 ]])
        elif ln .startswith ("f "):F .append ([int (p .split ("/")[0 ])-1 for p in ln .split ()[1 :4 ]])
    return np .array (V ,np .float64 ),np .array (F ,np .int64 )


CLS =["Housing","Contact","SnapPoint","CableEntry","LabelSurface"]

def metrics (pred ,gt ):
    ious ,dices =[],[];per ={}
    for c in range (NC ):
        p =pred ==c ;g =gt ==c ;inter =(p &g ).sum ();uni =(p |g ).sum ()
        if g .sum ()==0 :per [c ]=None ;continue 
        io =inter /max (uni ,1 );ious .append (io );dices .append (2 *inter /max (p .sum ()+g .sum (),1 ));per [c ]=io 
    return np .mean (ious ),np .mean (dices ),(pred ==gt ).mean (),per 


def main ():
    ckpts =sys .argv [1 :]or ["results/seg_extra/best_full.pt","results/seg_extra/best.pt",
    "results/seg_extra/adj_s0.pt","results/seg_extra/eec_all132.pt",
    "results/seg_extra/human103c_s0.pt","results/seg_extra/recall_hard_s2.pt"]
    parts =sorted (glob .glob ("wscad_corpus_scheffler_exact/val/*/"))
    best =(0 ,None );rows =[]
    for ck in ckpts :
        try :model ,meta =load_any (ck ,dev =dev )[:2 ]
        except Exception as e :print (f"{ck } yuklenemedi: {str (e )[:40 ]}");continue 
        mi ,md ,ac =[],[],[];pc ={c :[]for c in range (NC )}
        for pd in parts :
            pid =os .path .basename (os .path .normpath (pd ))
            obj =os .path .join (pd ,f"{pid }.obj");lab =os .path .join (pd ,f"{pid }.labels.txt")
            if not (os .path .exists (obj )and os .path .exists (lab )):continue 
            V ,F =load_obj (obj );gt =np .loadtxt (lab ,dtype =int )
            if len (gt )!=len (V ):continue 
            try :
                pred ,_ =D .predict (model ,meta ,np .ascontiguousarray (V ),np .ascontiguousarray (F ),
                device =dev ,op_cache_dir ="results/val_ops",return_probs =True )
                pred =np .asarray (pred ).argmax (-1 )if np .asarray (pred ).ndim >1 else np .asarray (pred )
                iou ,dice ,acc ,per =metrics (pred ,gt );mi .append (iou );md .append (dice );ac .append (acc )
                for c in range (NC ):
                    if per [c ]is not None :pc [c ].append (per [c ])
            except Exception :
                continue 
        if not mi :continue 
        name =os .path .basename (ck )
        perstr =" ".join (f"{CLS [c ][:4 ]}:{np .mean (pc [c ]):.2f}"for c in range (NC )if pc [c ])
        print (f"{name :22} IoU {np .mean (mi ):.3f} Dice {np .mean (md ):.3f} acc {np .mean (ac ):.3f} | {perstr }",flush =True )
        rows .append ((name ,float (np .mean (mi )),float (np .mean (md )),float (np .mean (ac )),{CLS [c ]:round (float (np .mean (pc [c ])),3 )for c in range (NC )if pc [c ]}))
        if np .mean (mi )>best [0 ]:best =(np .mean (mi ),name )
    print (f"\nEN IYI tez-native: {best [1 ]} (mean-IoU {best [0 ]:.3f} vs tez 0.514 = +{best [0 ]-0.514 :.3f}, +{(best [0 ]-0.514 )/0.514 *100 :.0f}%)")
    import json 
    json .dump ({"thesis_mean_jaccard":0.514 ,"val_parts":len (parts ),"checkpoints":rows ,"best":best [1 ]},
    open ("results/seg_thesis_val.json","w"),indent =2 )
    print ("-> results/seg_thesis_val.json")


if __name__ =="__main__":
    main ()
