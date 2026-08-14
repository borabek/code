# -*- coding: utf-8 -*-
"""F1-ARTIRMA: wire-gate UNION'i diriltir mi + ortak (vote-seviyesi x gate-esigi) sweep.

Union (vote>=1) recall'i ~0.74'e cikariyordu but precision cokuyordu (0.40) -> OLU idi. Simdi
wire-gate precision'i geri getiriyor. HIPOTEZ: vote>=1 (max recall) + gate, vote>=2 + gate'i gecebilir.
Sizinti-siz: gate GroupKFold-OOF (a parcaya, that part outside egitilmis gate uygulanir).

Faz 1 (GPU): each part 4 model -> TUM union CP'leri (vote sayisiyla) + feature + tp-label + part + mfg.
Faz 2 (offline): OOF gate skoru; (min_votes x gate_thr) izgarasinda P/R/F1.
"""
import os ,sys ,json ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,connector3d ,wire_gate 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 

HELD =set (open ("_hw_r3.txt").read ().split ())
NPZ ="results/f1_sweep_data.npz"


def union_all (cp_lists ,cluster_mm =5.0 ):
    """TUM CP'ler dedup + vote count (min filtre YOK -- vote>=1)."""
    allc =[c for lst in cp_lists for c in lst ]
    allc .sort (key =lambda c :-float (c .get ("confidence",0.0 )))
    kept =[]
    for c in allc :
        p =np .asarray (c ["point"],float )
        hit =next ((k for k in kept if np .linalg .norm (p -np .asarray (k ["point"],float ))<=cluster_mm ),None )
        if hit is None :
            c =dict (c );c ["_votes"]=1 ;kept .append (c )
        else :
            hit ["_votes"]+=1 
    return kept 


def extract ():
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cks =(os .environ ["F1_CKPTS"].split (",")if os .environ .get ("F1_CKPTS")
    else json .load (open ("cp_config.json"))["robot_vote2_checkpoints"])
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    print (f"UYELER ({len (cks )}): {[c .split ('/')[-1 ]for c in cks ]}",flush =True )
    os .environ ["BA_ALLOW_SEEN"]="1"
    cap =int (os .environ .get ("F1_PXC_CAP","0"))# 0 = tum PXC (full hakem)
    parts =[p for p in eligible ()if (p [0 ]=="WEI"and p [1 ]in HELD )]
    pxc =[p for p in eligible ()if p [0 ]=="PXC"]
    parts +=(pxc [:cap ]if cap else pxc )
    print (f"{len (parts )} part ({len (parts )-len (pxc if not cap else pxc [:cap ])} WEI + {len (pxc if not cap else pxc [:cap ])} PXC) | faz 1 (GPU)",flush =True )
    Xs ,votes ,tp ,grp ,mfgs ,ngt =[],[],[],[],[],{}
    t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (parts ,1 ):
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
            per =[];acc =None 
            for model ,meta in models :
            # per-k_eig cache dir: mixed-k_eig ensembles (64/96/128) share ONE mesh-keyed cache and
            # thrash ("not enough eigenvalues -> overwriting" every part). Separate dirs = no thrash.
                _opd =f"{OP }_k{int (meta .get ('k_eig',64 ))}"
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =_opd ,return_probs =True )
                pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
                _pp =json .load (open ("cp_config.json")).get ("prediction_postproc",{})
                per .append (cp_openings .connection_points (V ,F ,pb .argmax (-1 ),
                min_v =int (_pp .get ("min_vertices",30 )),classes =(CE ,CT ),dedupe_mm =10.0 ,
                probs =pb ,vertex_conf =float (_pp .get ("vertex_confidence_mask",0.5 )),
                ct_depth_min_mm =1.0 ,cluster_mm =float (_pp .get ("cluster_mm",5.0 ))))
            probs =acc /len (per )
            cps =union_all (per )
            if not cps :continue 
            X =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT )
            R ,t ,_ =align_frames (Vr ,Vj )
            P =np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t 
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            yy =np .zeros (len (P ),int );order =sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))if pe [a ,b ]<=tol )
            up ,ug =set (),set ()
            for dd ,a ,b in order :
                if a in up or b in ug :continue 
                up .add (a );ug .add (b );yy [a ]=1 
            Xs .append (X );votes .append (np .array ([c ["_votes"]for c in cps ]));tp .append (yy )
            grp +=[k ]*len (cps );mfgs +=[1 if mfg =="WEI"else 0 ]*len (cps );ngt [k ]=len (G )
        except Exception :
            continue 
        if k %25 ==0 :print (f"  {k }/{len (parts )}  {time .time ()-t0 :.0f}s",flush =True )
    X =np .vstack (Xs );v =np .concatenate (votes );y =np .concatenate (tp )
    grp =np .array (grp );mfgs =np .array (mfgs )
    gid =np .array (sorted (ngt ));ng =np .array ([ngt [g ]for g in gid ])
    np .savez (NPZ ,X =X ,votes =v ,y =y ,groups =grp ,mfg =mfgs ,grp_ids =gid ,ngt =ng )
    print (f"  -> {NPZ } ({len (y )} union CP)")


def sweep ():
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    d =np .load (NPZ ,allow_pickle =True )
    X ,v ,y ,grp ,mfg =d ["X"],d ["votes"],d ["y"],d ["groups"],d ["mfg"]
    ngt_of =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    # OOF gate skoru (leakage absent)
    oof =np .zeros (len (y ));gkf =GroupKFold (n_splits =5 )
    for tr ,te in gkf .split (X ,y ,grp ):
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 )
        clf .fit (X [tr ],y [tr ]);oof [te ]=clf .predict_proba (X [te ])[:,1 ]
    n_gt =sum (ngt_of .values ())
    print (f"\n=== ORTAK SWEEP ({len (y )} union CP, manufacturer {n_gt }) | urun vote>=1 union + RF wire-gate 0.35 (base F1 0.750; GB-era 0.693 RETIRED) ===")
    print (f"{'min_vote':>8s} {'gate':>5s} | {'P':>6s} {'R':>6s} {'F1':>6s}")
    best =None 
    for mv in (1 ,2 ,3 ):
        for th in (0.0 ,0.2 ,0.3 ,0.4 ,0.5 ):
            keep =(v >=mv )&(oof >=th )
            tp =int ((y [keep ]==1 ).sum ());nk =int (keep .sum ())
            p =tp /max (nk ,1 );r =tp /max (n_gt ,1 );f =2 *p *r /max (p +r ,1e-9 )
            tag =""
            if best is None or f >best [0 ]:best =(f ,mv ,th ,p ,r );tag =" <--"
            print (f"{mv :8d} {th :5.2f} | {p :6.3f} {r :6.3f} {f :6.3f}{tag }")
    print (f"\n  EN IYI: vote>={best [1 ]} + gate {best [2 ]:.2f} -> F1 {best [0 ]:.3f} (P{best [3 ]:.3f} R{best [4 ]:.3f})")
    # URUN config (vote>=1, gate 0.35) per-mfg + birlesik, OOF (leakage absent)
    print ("\n  === URUN CONFIG (vote>=1 + gate 0.35) OOF ===")
    for name ,m in (("HEPSI",np .ones (len (y ),bool )),("WEI",mfg ==1 ),("PXC",mfg ==0 )):
        k =m &(v >=1 )&(oof >=0.35 );tp =int ((y [k ]==1 ).sum ())
        gt =sum (ngt_of [g ]for g in np .unique (grp [m ]))
        p =tp /max (k .sum (),1 );r =tp /max (gt ,1 )
        print (f"    {name :6s}: P {p :.3f} R {r :.3f} F1 {2 *p *r /max (p +r ,1e-9 ):.3f}")
        # DAGITILAN gate'i UNION dagiliminda yeniden egit (train/apply uyusmazligini duzelt)
    import wire_gate as WG 
    clf_full =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (X ,y )
    import pickle ;pickle .dump ({"clf":clf_full ,"feat_names":WG .FEAT_NAMES },open (WG .MODEL_PATH ,"wb"))
    print (f"\n  -> dagitilan gate UNION-dagiliminda yeniden egitildi ({len (y )} CP) -> {WG .MODEL_PATH }")


if __name__ =="__main__":
    if "--sweep-only"in sys .argv :
        sweep ()
    else :
        extract ();sweep ()
