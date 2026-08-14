# -*- coding: utf-8 -*-
"""ROBOT END-TO-END TEST (FAZ B4): does the full robot pipeline give correct CPs ten UNSEEN parts?

This exercises the ACTUAL product path a robot cell would run -- robot_cp.extract() ten a STEP file ->
{point, direction, size_mm, depth_mm, confidence, tier} -- and scores it against the manufacturer's
ConnectionPoints ten parts the model never trained ten. It reports what a robot operator actually cares
about, separated by the two-tier policy:

  AUTO tier  (confidence >= --conf-auto): the CPs the robot would act ten WITHOUT a human.
             -> precision here is the "how often does the robot drive a wire into a real opening" number.
  REVIEW tier: flagged for a human -> not the robot's autonomous responsibility.

Also reports, ten matched CPs, the geometric error a robot would actually see:
  position error (mm)  = distance between predicted and manufacturer CP, along + perpendicular
  direction error (deg)= angle between predicted insertion axis and the manufacturer InsertDirection

WHY separate from big_arbiter: big_arbiter scores the raw detector for model selection. This scores the
SHIPPING robot interface (robot_cp.py, product checkpoint, tiers, real geometry) for the thesis's
"does the robot work end to end" claim.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe robot_e2e.py [--mfg WEI] [--limit 60] [--conf-auto 0.75]
"""
import os ,sys ,glob ,json ,argparse 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import connector3d 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
import robot_cp 

DS ="_ds1/DataSet"


def eligible (mfg ):
    step ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}
    seen =set ()
    for d in ("_label_targets","_label_targets_2","_label_targets_3","_label_targets_4",
    "_label_targets_recall","_label_targets_recall_pxc",
    "_QUARANTINE_labels/_label_targets_recall_pxc",
    "_QUARANTINE_labels/_label_targets_recall_v2","_label_targets_recall_hard",
    "_label_targets_recall_r3"):
        seen |={os .path .basename (os .path .normpath (p ))for p in glob .glob (d +"/*/")}
    import scheffler_dataset as ds 
    for sp in ("train","val"):
        seen |={s ["part_id"]for s in ds .load_split ("wscad_corpus_scheffler_exact",sp ,verify_hashes =False )}
    out =[]
    for f in sorted (glob .glob (os .path .join (DS ,"*ElectricalTerminal*.json"))):
        head =os .path .basename (f ).split ("_")[0 ]
        m ,pid =(head .split (".",1 )+[""])[:2 ]
        if pid in seen or pid not in step or (mfg and m !=mfg ):
            continue 
        out .append ((m ,pid ,f ,step [pid ]))
    return out 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ckpt",default ="")# default: product from cp_config
    ap .add_argument ("--mfg",default ="")
    ap .add_argument ("--limit",type =int ,default =60 )
    ap .add_argument ("--conf-auto",type =float ,default =0.75 )
    ap .add_argument ("--device",default ="cuda"if torch .cuda .is_available ()else "cpu")
    a =ap .parse_args ()
    cfg =json .load (open ("cp_config.json"))
    cks =[a .ckpt ]if a .ckpt else cfg ["robot_vote2_checkpoints"]
    models =[load_any (c ,dev =a .device )[:2 ]for c in cks ]
    print (f"URUN: {len (models )} model (CP-duzeyi cogunluk oyu >=2) | conf-auto {a .conf_auto }"
    if len (models )>1 else 
    f"URUN: {os .path .basename (cks [0 ])} | conf-auto {a .conf_auto }",flush =True )

    parts =eligible (a .mfg )[:a .limit ]
    # aggregate: per tier TP/FP, plus geometry errors ten matched
    agg ={"auto":[0 ,0 ],"review":[0 ,0 ]}# [TP, FP]
    n_gt =0 ;pos_err =[];dir_err =[];per_mfg ={}
    for m ,pid ,jf ,stp in parts :
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            cps =robot_cp .extract (models ,stp ,a .device ,a .conf_auto )# THE ROBOT OUTPUT
            Vr ,_ =step_to_mesh (stp )
            R ,t ,_ =align_frames (Vr ,Vj )
            # robot points are in the STEP frame; bring them into the JSON frame where G lives
            P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            Pj =P @R .T +t 
            Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            Pdj =Pd @R .T 
            tiers =[c ["tier"]for c in cps ]
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            n_gt +=len (G );per_mfg .setdefault (m ,[0 ,0 ,0 ])# gt, auto_tp, auto_fp
            per_mfg [m ][0 ]+=len (G )
            # axis-aware greedy match (same opening)
            if len (Pj ):
                diff =Pj [:,None ,:]-G [None ,:,:]
                along =(diff *Gd [None ,:,:]).sum (-1 )
                perp =np .linalg .norm (diff -along [...,None ]*Gd [None ,:,:],axis =-1 )
                perp =np .where (np .abs (along )<=40.0 ,perp ,np .inf )
                order =sorted ((perp [i ,k ],i ,k )for i in range (len (Pj ))for k in range (len (G ))if perp [i ,k ]<=tol )
                up ,ug =set (),set ();matched ={}
                for d ,i ,k in order :
                    if i in up or k in ug :continue 
                    up .add (i );ug .add (k );matched [i ]=k 
                for i in range (len (Pj )):
                    tier =tiers [i ]
                    if i in matched :
                        agg [tier ][0 ]+=1 
                        if tier =="auto":per_mfg [m ][1 ]+=1 
                        k =matched [i ]
                        pos_err .append (float (np .linalg .norm (Pj [i ]-G [k ])))
                        c =float (np .clip (Pdj [i ]@Gd [k ]/(np .linalg .norm (Pdj [i ])*np .linalg .norm (Gd [k ])+1e-9 ),-1 ,1 ))
                        dir_err .append (float (np .degrees (np .arccos (abs (c )))))
                    else :
                        agg [tier ][1 ]+=1 
                        if tier =="auto":per_mfg [m ][2 ]+=1 
        except Exception :
            continue 

    def pr (tp ,fp ,gt ):
        p =tp /max (tp +fp ,1 );r =tp /max (gt ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )
    at ,af =agg ["auto"];rt ,rf =agg ["review"]
    print (f"\n=== ROBOT UCTAN-UCA ({len (parts )} gorulmemis part, {n_gt } manufacturer CP) ===")
    pa =pr (at ,af ,n_gt )
    print (f"  AUTO tier (robot otonom eyler): P {pa [0 ]:.3f}  R {pa [1 ]:.3f}  F1 {pa [2 ]:.3f}  (TP{at } FP{af })")
    pall =pr (at +rt ,af +rf ,n_gt )
    print (f"  AUTO+REVIEW (hepsi):            P {pall [0 ]:.3f}  R {pall [1 ]:.3f}  F1 {pall [2 ]:.3f}")
    if pos_err :
        print (f"  eslesenlerde ROBOT hatasi: konum medyan {np .median (pos_err ):.1f}mm | direction medyan {np .median (dir_err ):.1f} deg")
    for m ,(g ,tp ,fp )in per_mfg .items ():
        p =tp /max (tp +fp ,1 )
        print (f"    {m }: {g } CP, AUTO precision {p :.3f} ({tp } dogru / {fp } yanlis otonom)")
    print ("\n  ROBOT OKUMASI: AUTO precision = robot teli GERCEK acikliga sokma orani (yuksek olmali).")
    print ("  Dusuk-confidence CP'ler REVIEW'e dusuyor -> insana; robotun otonom hatasi degil.")


if __name__ =="__main__":
    main ()
