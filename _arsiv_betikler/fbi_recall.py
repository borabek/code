# -*- coding: utf-8 -*-
"""FBI: WEI recall platosunu 2 hipotezle sorgula.
H1 (teshis kusuru): old FN teshisi manufacturer CP'sinin (KONTAKT, ~15mm icerde) 8mm cevresine bakti;
    model acikligin AGZINDA (v_o, ~15mm disarda) ateslerse "kor" sayilmis may be. DUZELT:
    each FN for axis-BOYUNCA (Gd yonunde) [-5,+25]mm, perp<6mm silindirde baglanti-verteksi ara.
H2 (tez kolu): CAD-geometrik opening dedektoru (step_openings, ogrenilmemis -> WEI domain-gap'i YOK)
    model kaciran acikligi yakaliyor mu? -> CAD-UNION plato-kiricisi.
Ayrica: model CP + CAD acikliklarinin BIRLESIMI WEI recall'i ne yapar (leakage-siz 145 WEI).
"""
import os ,sys ,json ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,step_openings 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import greedy ,eligible ,CE ,CT ,OP 

CKPT ="results/seg_extra/recall_hard_s2.pt"
HELD =set (open ("_hw_r3.txt").read ().split ())


def match_fn (P ,G ,Gd ,tol ,axis_tol =40.0 ):
    """axis-aware greedy; return set of matched G indices."""
    if not len (P ):return set ()
    diff =P [:,None ,:]-G [None ,:,:]
    al =(diff *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
    pe =np .where (np .abs (al )<=axis_tol ,pe ,np .inf )
    order =sorted ((pe [i ,k ],i ,k )for i in range (len (P ))for k in range (len (G ))if pe [i ,k ]<=tol )
    up ,ug =set (),set ()
    for d ,i ,k in order :
        if i in up or k in ug :continue 
        up .add (i );ug .add (k )
    return ug 


def main ():
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    model ,meta ,_ =load_any (CKPT ,dev =dev )
    os .environ ["BA_ALLOW_SEEN"]="1"
    raw =[p for p in eligible ()if p [0 ]=="WEI"and p [1 ]in HELD ]
    print (f"{len (raw )} WEI held-out | {os .path .basename (CKPT )} | FBI recall sorgusu",flush =True )

    fn_total =0 
    h1_mouth =0 # model AGZINDA atesledi (duzeltilmis) -> turetme/shape, post-proc adayi
    h2_cad =0 # CAD acikligi kapsiyor -> CAD-union adayi
    both_blind =0 # ne model ne CAD -> gercekten gorunmez
    # union recall
    uT =uFp =uFn =0 ;mT =mFp =mFn =0 
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
            _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
            probs =np .asarray (pb ,float );lab =probs .argmax (-1 )
            cps =cp_openings .connection_points (V ,F ,lab ,min_v =30 ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =probs ,vertex_conf =0.5 ,ct_depth_min_mm =1.0 ,cluster_mm =5.0 )
            R ,t ,_ =align_frames (Vr ,Vj )
            Pm =(np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t )if cps else np .zeros ((0 ,3 ))
            # CAD-geometrik acikliklar (STEP frame -> JSON frame)
            try :
                cad =step_openings .openings_from_step (stp ,auto =True ,slot_pairs =True ,rect_clusters =True )
            except Exception :
                cad =[]
            Pc =(np .array ([o ["entry_point"]for o in cad ],float )@R .T +t )if cad else np .zeros ((0 ,3 ))
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))

            # model-only recall
            tp ,fp ,fn =greedy (Pm ,G ,tol ,Gd ,40.0 );mT +=tp ;mFp +=fp ;mFn +=fn 
            # union recall (model CP + CAD entry, dedup 5mm; model before -> onceligi present)
            allp =np .vstack ([Pm ,Pc ])if (len (Pm )and len (Pc ))else (Pm if len (Pm )else Pc )
            U =[]
            for pnt in np .asarray (allp ,float ).reshape (-1 ,3 ):
                if all (np .linalg .norm (pnt -u )>5.0 for u in U ):U .append (pnt )
            U =np .array (U ,float )if U else np .zeros ((0 ,3 ))
            tp ,fp ,fn =greedy (U ,G ,tol ,Gd ,40.0 );uT +=tp ;uFp +=fp ;uFn +=fn 

            # FN teshisi (model-only FN'ler ten)
            matched_g =match_fn (Pm ,G ,Gd ,tol )
            Gm =(G -t )@R ;Gd_m =Gd @R 
            conn_p =probs [:,CE ]+probs [:,CT ];is_conn =np .isin (lab ,(CE ,CT ))
            cad_g =match_fn (Pc ,G ,Gd ,tol )if len (Pc )else set ()
            for gi in range (len (G )):
                if gi in matched_g :continue 
                fn_total +=1 
                # H1: axis-boyu mouth aramasi
                rel =V -Gm [gi ][None ,:]
                al =rel @Gd_m [gi ]
                perp =np .linalg .norm (rel -al [:,None ]*Gd_m [gi ][None ,:],axis =1 )
                near =(al >=-5.0 )&(al <=25.0 )&(perp <=6.0 )
                fired =int ((near &is_conn ).sum ())>=5 
                covered =gi in cad_g 
                if fired :h1_mouth +=1 
                if covered :h2_cad +=1 
                if not fired and not covered :both_blind +=1 
        except Exception :
            continue 
        if k %25 ==0 :
            print (f"  {k }/{len (raw )}  {time .time ()-t0 :.0f}s",flush =True )

    def f1 (T ,Fp ,Fn ):
        p =T /max (T +Fp ,1 );r =T /max (T +Fn ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )
    print (f"\n=== FBI RECALL SORGU ({fn_total } model-FN) ===")
    if fn_total :
        print (f"  H1 model AGIZDA atesledi (axis-boyu, duzeltilmis): {h1_mouth } ({100 *h1_mouth /fn_total :.0f}%) <- POST-PROC/derivation adayi")
        print (f"  H2 CAD-geometri kapsiyor (ogrenilmemis sinyal):     {h2_cad } ({100 *h2_cad /fn_total :.0f}%) <- CAD-UNION adayi")
        print (f"  ikisi de gormuyor (gercekten gorunmez):             {both_blind } ({100 *both_blind /fn_total :.0f}%) <- sadece ETIKET")
    pm =f1 (mT ,mFp ,mFn );pu =f1 (uT ,uFp ,uFn )
    print (f"\n  MODEL-only   WEI: P{pm [0 ]:.3f} R{pm [1 ]:.3f} F1{pm [2 ]:.3f}")
    print (f"  MODEL+CAD union: P{pu [0 ]:.3f} R{pu [1 ]:.3f} F1{pu [2 ]:.3f}   <- CAD-union plato-kirici sonucu")
    json .dump ({"fn_total":fn_total ,"h1_mouth":h1_mouth ,"h2_cad":h2_cad ,"both_blind":both_blind ,
    "model":pm ,"union":pu },open ("results/fbi_recall.json","w"),indent =1 )


if __name__ =="__main__":
    main ()
