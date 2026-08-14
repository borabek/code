# -*- coding: utf-8 -*-
"""VAL diagnostic: for each PREDICTED CP, is it a TP (matches a GT CP within 5mm) or FP?
Print confidence / area / n_verts distributions for TP vs FP so a precision gate can be chosen
from data (not guessed). Develop ten VAL only; test_locked stays held out."""
import numpy as np 
import scheffler_dataset as dataset 
import diffusionnet ,cp_openings 

CKPT ="results/scheffler_semantic/refit91.pt";OPCACHE ="results/scheffler_semantic/operators"
model ,meta ,_ =diffusionnet .load_checkpoint (CKPT ,device ="cuda")
samples =dataset .load_split ("wscad_corpus_scheffler_exact","val",verify_hashes =False )
tp_c ,fp_c ,tp_a ,fp_a ,tp_n ,fp_n =([]for _ in range (6 ))
for s in samples :
    V ,F =np .asarray (s ["verts"],float ),np .asarray (s ["faces"],int )
    gt =np .array ([c ["point"]for c in cp_openings .connection_points (V ,F ,np .asarray (s ["labels"]),min_v =20 )]or []).reshape (-1 ,3 )
    plab ,probs =diffusionnet .predict (model ,meta ,V ,F ,device ="cuda",op_cache_dir =OPCACHE ,return_probs =True )
    pcps =cp_openings .connection_points (V ,F ,np .asarray (plab ),min_v =20 ,probs =probs )
    for c in pcps :
        near =gt .size and float (np .linalg .norm (gt -c ["point"],axis =1 ).min ())<=5.0 
        (tp_c if near else fp_c ).append (c ["confidence"])
        (tp_a if near else fp_a ).append (c ["area"])
        (tp_n if near else fp_n ).append (c ["n_verts"])

def st (x ):
    x =np .array (x );return f"n={len (x ):3d}  min={x .min ():.3f}  p25={np .percentile (x ,25 ):.3f}  med={np .median (x ):.3f}  mean={x .mean ():.3f}"if len (x )else "n=0"
print ("CONFIDENCE  TP:",st (tp_c ));print ("CONFIDENCE  FP:",st (fp_c ))
print ("AREA(mm2)   TP:",st (tp_a ));print ("AREA(mm2)   FP:",st (fp_a ))
print ("N_VERTS     TP:",st (tp_n ));print ("N_VERTS     FP:",st (fp_n ))
# how many FP does each threshold remove vs TP kept?
tp_c ,fp_c =np .array (tp_c ),np .array (fp_c )
print ("\nthr  ->  TP_kept  FP_kept  (of TP={} FP={})".format (len (tp_c ),len (fp_c )))
for thr in [0.5 ,0.6 ,0.7 ,0.8 ,0.85 ,0.9 ,0.95 ]:
    print (f"conf>={thr :.2f}  TP {int ((tp_c >=thr ).sum ()):3d}  FP {int ((fp_c >=thr ).sum ()):3d}")
print ("DONE")
