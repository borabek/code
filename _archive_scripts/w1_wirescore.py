# -*- coding: utf-8 -*-
"""!!! SUPERSEDED / LEAKY -- KULLANMA. Tek-model (recall_hard_s2) + deploy-gate (held parcalarda egitilmis)
    => wire_score iyimser. Bu scriptin 'gate WEI'yi %29 kesiyor, +0.069 kazanc' bulgusu ARTEFAKT idi.
    DURUST version = w1_gate_oof.py (4-model union + leakage-free OOF gate + nested-CV): 0.35 already optimal.
    Sadece tarihsel referans for tutuluyor. See memory/085-decomposition-leakage-free.md
W1 KANTITATIF: WEI held-out tahminlerini TP/FP ayir, each birinin wire_score'unu kaydet.
Karar:
 - FP'lerin wire_score'u TP'den BELIRGIN DUSUK -> gate FP'leri already eliyor; ham 0.58 = gate-ONCESI measurement,
   GT-eksikligi DEGIL. Urun 0.69 this farki already yakaliyor. -> W1 kapan, lever = wire/tool (W4) or recall (W2).
 - FP wire_score ~ TP (high) -> ya GT-missing (real listelenmemis kablo girisi) ya gate kor.
Axis-aware eslesme (mouth-vs-seat), align_frames with mesh frame."""
import os ,sys ,json 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,wire_gate 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 


def matched_pred_idx (P ,G ,tol ,Gdir ,axis_tol ):
    """big_arbiter.greedy with same eslesme; matched pred-index kumesini dondur."""
    if not len (P )or not len (G ):return set ()
    diff =P [:,None ,:]-G [None ,:,:]
    along =(diff *Gdir [None ,:,:]).sum (-1 )
    dm =np .linalg .norm (diff -along [...,None ]*Gdir [None ,:,:],axis =-1 )
    dm =np .where (np .abs (along )<=axis_tol ,dm ,np .inf )
    order =sorted ((dm [i ,j ],i ,j )for i in range (len (P ))for j in range (len (G )))
    up ,ug =set (),set ()
    for d ,i ,j in order :
        if d >tol :break 
        if i in up or j in ug :continue 
        up .add (i );ug .add (j )
    return up 

dev ="cuda"
N =int (sys .argv [1 ])if len (sys .argv )>1 else 20 
held =set (open ("_hw_r3.txt").read ().split ()[:N ])
parts =[(m ,p ,jf ,s )for m ,p ,jf ,s in eligible ()if m =="WEI"and p in held ]
model ,meta =load_any ("results/seg_extra/recall_hard_s2.pt",dev =dev )[:2 ]
AXIS_TOL =40.0 

tp_ws ,fp_ws =[],[]# TP and FP wire_score'lari
fp_hi =0 # high-skorlu FP (gate'i passing, >=0.35) count
total_G =0 # total manufacturer CP (recall paydasi)
for m ,pid ,jf ,stp in parts :
    j =json .load (open (jf ,encoding ="utf-8-sig"))
    Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
    G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j ["ConnectionPoints"]],float )
    Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]
    for c in j ["ConnectionPoints"]],float )
    if not len (G ):continue 
    total_G +=len (G )
    Vr ,Fr =step_to_mesh (stp );V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
    V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
    _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True );pb =np .asarray (pb ,float )
    cps =cp_openings .connection_points (V ,F ,pb .argmax (-1 ),min_v =30 ,classes =(CE ,CT ),dedupe_mm =10.0 ,
    probs =pb ,vertex_conf =0.5 ,ct_depth_min_mm =1.0 ,cluster_mm =5.0 )
    if not cps :continue 
    wire_gate .apply (V ,F ,pb ,cps ,CE ,CT ,threshold =0.0 )# only wire_score yaz, filtreleme
    R ,t ,_ =align_frames (Vr ,Vj )
    Gm =(G -t )@R 
    Gdm =(Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 ))@R 
    P =np .array ([np .asarray (c ["point"])for c in cps ])
    tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
    matched =matched_pred_idx (P ,Gm ,tol ,Gdm ,AXIS_TOL )
    for k ,c in enumerate (cps ):
        ws =c ["wire_score"]
        if k in matched :tp_ws .append (ws )
        else :
            fp_ws .append (ws )
            if ws >=0.35 :fp_hi +=1 

tp_ws ,fp_ws =np .array (tp_ws ),np .array (fp_ws )
print (f"\n{len (parts )} WEI held-out -- wire_score TP vs FP")
print (f"  TP (n={len (tp_ws )}): ort {tp_ws .mean ():.3f} med {np .median (tp_ws ):.3f}")
print (f"  FP (n={len (fp_ws )}): ort {fp_ws .mean ():.3f} med {np .median (fp_ws ):.3f}")
print (f"  gate esigi 0.35 passing FP: {fp_hi }/{len (fp_ws )} ({100 *fp_hi /max (len (fp_ws ),1 ):.0f}%)")
THR =0.35 


def prf (tp ,fp ,fn ):
    p =tp /max (tp +fp ,1 );r =tp /max (tp +fn ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )


    # pre-gate: butun tahminler
pre_tp ,pre_fp =len (tp_ws ),len (fp_ws );pre_fn =total_G -pre_tp 
# post-gate (0.35): only wire_score>=THR
post_tp =int ((tp_ws >=THR ).sum ());post_fp =int ((fp_ws >=THR ).sum ());post_fn =total_G -post_tp 
tp_lost =pre_tp -post_tp 
pp ,pr ,pf =prf (pre_tp ,pre_fp ,pre_fn )
qp ,qr ,qf =prf (post_tp ,post_fp ,post_fn )
print (f"\n--- GATE ETKISI (threshold {THR }) ---")
print (f"  PRE-gate : P={pp :.3f} R={pr :.3f} F1={pf :.3f}  (TP{pre_tp } FP{pre_fp } FN{pre_fn })")
print (f"  POST-gate: P={qp :.3f} R={qr :.3f} F1={qf :.3f}  (TP{post_tp } FP{post_fp } FN{post_fn })")
print (f"  gate'e KURBAN giden gercek TP: {tp_lost }/{pre_tp } ({100 *tp_lost /max (pre_tp ,1 ):.0f}%)  "
+("<- gate recall'u kesiyor, tuning lever'i"if tp_lost >pre_tp *0.15 else "<- gate TP'yi koruyor, recall lever'i = more very bul (W2)"))

np .savez ("results/w1_wei_scores.npz",tp =tp_ws ,fp =fp_ws ,total_G =total_G )# sweep for cache
print (f"\n--- WEI GATE ESIK SWEEP (F1-optimal ara) ---")
best =(0 ,0 ,0 ,0 ,0 )
for thr in np .arange (0.0 ,0.61 ,0.025 ):
    t =int ((tp_ws >=thr ).sum ());f =int ((fp_ws >=thr ).sum ());n =total_G -t 
    p ,r ,f1 =prf (t ,f ,n )
    if f1 >best [0 ]:best =(f1 ,thr ,p ,r ,t )
    if abs (thr -round (thr ,1 ))<1e-6 :
        print (f"  thr {thr :.2f}: P={p :.3f} R={r :.3f} F1={f1 :.3f}")
print (f"  >>> WEI-OPTIMAL thr={best [1 ]:.3f}: F1={best [0 ]:.3f} (P={best [2 ]:.3f} R={best [3 ]:.3f}) "
f"vs mevcut 0.35 F1={qf :.3f}  ->  KAZANC {best [0 ]-qf :+.3f}")

gap =tp_ws .mean ()-fp_ws .mean ()
print (f"\nKARAR (TP-FP wire_score farki = {gap :+.3f}):")
if gap >0.10 :
    print ("  -> FP'ler BELIRGIN low score: gate onlari already eliyor. Ham 0.58 = gate-ONCESI.")
    print ("  -> GT-eksikligi DEGIL; 0.58 pesimist but urun 0.69 bunu yakaliyor. W1 kapan -> W2(recall)/W4(wire).")
else :
    print ("  -> FP skorlari TP'ye yakin: gate ayirt EDEMIYOR. Ya GT-missing ya gate kor.")
    print (f"  -> {fp_hi } yuksek-skorlu FP GORSEL adjudication gerektirir (gercek kablo girisi mi?).")
