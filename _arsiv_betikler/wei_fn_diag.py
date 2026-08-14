# -*- coding: utf-8 -*-
"""WEI kacirma TESHISI: recall neden ~0.55'te takili? Iki mekanizma ayirt et:
  (A) SEGMENTASYON kacisi -- model kacan manufacturer CP'nin yakininda HIC baglanti-verteksi ateslemi_yor
      -> only VERI cozer (label).
  (B) TURETME kacisi -- model ATESLIYOR (baglanti-verteksleri present) but v_o/clustering/tolerans wrong
      -> POST-PROC/derivation BEDAVA cozebilir.
Her WEI held-out parcasi for: prob'lari al, urun CP'lerini turet, manufacturer G'ye axis-aware esle;
each FN (eslesmeyen G) for, G'nin (mesh frame'e tasinmis) 8mm cevresinde kac CE/CT verteks present +
max baglanti olasiligi (p_CE+p_CT). Ozet: FN'lerin yuzde kaci "model gordu" (B) vs "model kor" (A).
"""
import os ,sys ,json ,argparse ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import greedy ,eligible ,CE ,CT ,OP 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ckpt",default ="results/seg_extra/recall_hard_s2.pt")
    ap .add_argument ("--only-mfg",default ="WEI")
    ap .add_argument ("--only-parts",nargs ="+",default =[])
    ap .add_argument ("--radius",type =float ,default =8.0 ,help ="G cevresinde baglanti-verteks arama yaricapi (mm)")
    ap .add_argument ("--min-v",type =int ,default =30 )
    ap .add_argument ("--vertex-conf",type =float ,default =0.5 )
    ap .add_argument ("--cluster-mm",type =float ,default =5.0 )
    ap .add_argument ("--device",default ="cuda"if torch .cuda .is_available ()else "cpu")
    a =ap .parse_args ()

    model ,meta ,_ =load_any (a .ckpt ,dev =a .device )
    os .environ ["BA_ALLOW_SEEN"]="1"
    raw =eligible ()
    if a .only_mfg :raw =[p for p in raw if p [0 ]==a .only_mfg ]
    if a .only_parts :
        keep =set (a .only_parts );raw =[p for p in raw if p [1 ]in keep ]
    print (f"{len (raw )} part | {os .path .basename (a .ckpt )} | FN teshisi (yaricap {a .radius }mm)",flush =True )

    # FN kovalari: each FN for (fires, nverts_near, maxp_near)
    seg_miss =0 # model kor: yakinda ~0 baglanti-verteksi (A)
    derive_miss =0 # model gordu: baglanti-verteksleri present but CP olmadi (B)
    n_fn =0 ;n_tp =0 ;near_counts =[];near_maxp =[]
    t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (raw ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            _ ,pb =D .predict (model ,meta ,V ,F ,device =a .device ,op_cache_dir =OP ,return_probs =True )
            probs =np .asarray (pb ,float );lab =probs .argmax (-1 )
            cps =cp_openings .connection_points (V ,F ,lab ,min_v =a .min_v ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =probs ,vertex_conf =a .vertex_conf ,ct_depth_min_mm =1.0 ,cluster_mm =a .cluster_mm )
            R ,t ,_ =align_frames (Vr ,Vj )
            P =(np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t )if cps else np .zeros ((0 ,3 ))
            # G'yi mesh frame'e tasi: P0 = (P - t) @ R
            Gm =(G -t )@R 
            conn_p =probs [:,CE ]+probs [:,CT ]# each verteks baglanti olasiligi
            is_conn =np .isin (lab ,(CE ,CT ))
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            # axis-aware esleme with TP/FN indekslerini bul
            # greedy only sayilari donuyor -> here FN'leri kendimiz belirleyelim
            if len (P ):
                diff =P [:,None ,:]-G [None ,:,:]
                al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                order =sorted ((pe [i ,kk ],i ,kk )for i in range (len (P ))for kk in range (len (G ))if pe [i ,kk ]<=tol )
                up ,ug =set (),set ()
                for d ,i ,kk in order :
                    if i in up or kk in ug :continue 
                    up .add (i );ug .add (kk )
                matched_g =ug 
            else :
                matched_g =set ()
            for gi in range (len (G )):
                if gi in matched_g :
                    n_tp +=1 ;continue 
                n_fn +=1 
                dvec =V -Gm [gi ][None ,:]
                near =np .linalg .norm (dvec ,axis =1 )<=a .radius 
                nconn =int ((near &is_conn ).sum ())
                mp =float (conn_p [near ].max ())if near .any ()else 0.0 
                near_counts .append (nconn );near_maxp .append (mp )
                if nconn >=5 :# model that acikligi ateslemis -> turetme kacisi
                    derive_miss +=1 
                else :
                    seg_miss +=1 
        except Exception :
            continue 
        if k %25 ==0 :
            print (f"  {k }/{len (raw )}  {time .time ()-t0 :.0f}s",flush =True )

    print (f"\n=== WEI FN TESHIS ({os .path .basename (a .ckpt )}) ===")
    print (f"  TP {n_tp } | FN {n_fn }")
    if n_fn :
        print (f"  TURETME kacisi (>=5 baglanti-verteks {a .radius }mm icinde): {derive_miss } ({100 *derive_miss /n_fn :.0f}%) <- POST-PROC/derivation BEDAVA cozer")
        print (f"  SEGMENTASYON kacisi (model kor):                          {seg_miss } ({100 *seg_miss /n_fn :.0f}%) <- yalnizca ETIKET cozer")
        print (f"  FN'lerde yakin baglanti-verteks sayisi: medyan {np .median (near_counts ):.0f}  max-olasilik medyan {np .median (near_maxp ):.2f}")
    json .dump ({"ckpt":a .ckpt ,"tp":n_tp ,"fn":n_fn ,"derive_miss":derive_miss ,"seg_miss":seg_miss ,
    "near_counts":near_counts ,"near_maxp":near_maxp },open ("results/wei_fn_diag.json","w"),indent =1 )
    print ("  -> results/wei_fn_diag.json")


if __name__ =="__main__":
    main ()
