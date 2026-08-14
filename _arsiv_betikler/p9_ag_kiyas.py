# -*- coding: utf-8 -*-
"""g9 (sig 2mm boyama) vs g7 (derin 6mm) -- D6'da ADAY KAHINI + real tespit/robot.

KILL: g9 candidate kahininde g7'yi GECMEZSE sig boyama GERI ALINIR
(`_label_auto_derin6` duruyor, `g5_mouth_label._ESKI_DERINLIK` = (1.0, 6.0)).

Aday kahini gate'ten BAGIMSIZ oldugu for agin TEMSIL gucunu dogrudan olcer;
real tespit/robot whereas dagitilan gate with uctan uca sonucu gives.
"""
import argparse ,collections ,glob ,io ,json ,os ,pickle ,sys ,time 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import d6_record 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ckpt",nargs ="+",required =True )
    ap .add_argument ("--name",nargs ="+",required =True )
    ap .add_argument ("--boundary",type =int ,default =0 )
    a =ap .parse_args ()
    import protocol ;protocol .tez_dogrula ()
    import torch ,thesis_remesh ,robot_cp as RC ,diffusionnet as D_ 
    from infer_step_cp import load_any ,step_to_mesh 
    from corpus_identity import step_kimlik as SK 
    from sina_cluster import match_hungarian ,f1w 

    sv =d6_record .exam ();rec_ =d6_record .yukle (set (sv ["pidler"]))
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    with io .open ("cp_config.json",encoding ="utf-8")as f :cfg =json .load (f )
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    pid_l =[p for p in sorted (rec_ )if p in S ]
    if a .bound_ :pid_l =pid_l [:a .bound_ ]
    print (f"D6: {len (pid_l )} part | cihaz {dev }\n")
    res_ ={}
    for ck ,ad in zip (a .ckpt ,a .ad ):
        models =[load_any (ck ,dev =dev )[:2 ]]
        K ,T ,R =[],[],[]
        t0 =time .time ();error =0 
        for i ,pid in enumerate (pid_l ,1 ):
            if i %100 ==0 :
                print (f"  {ad } {i }/{len (pid_l )} {(time .time ()-t0 )/i :.1f}s/part",flush =True )
            try :
                r =rec_ [pid ]
                Vr ,Fr =step_to_mesh (S [pid ])
                V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
                pbs =[]
                for m ,meta in models :
                    _ ,pb =D_ .predict (m ,meta ,V ,F ,device =dev ,op_cache_dir =None ,
                    return_probs =True )
                    pbs .append (np .asarray (pb ,float ))
                cps ,_pr ,_hi ,_u =RC .derive_candidates (V ,F ,pbs ,S [pid ],cfg =cfg )
                P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
                Dd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
                G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
                rj ="very"if r ["n"]>=8 else "low"
                tt =max (3.0 ,0.06 *r ["diag"])
                if len (P )and len (G ):
                    d =P [:,None ,:]-G [None ,:,:]
                    al =(d *Gd [None ,:,:]).sum (-1 )
                    pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
                    kap =int ((np .where (np .abs (al )<=40.0 ,pe ,np .inf ).min (0 )<=tt ).sum ())
                else :kap =0 
                K .append ((rj ,kap ,0 ,len (G )-kap ))
                T .append ((rj ,)+match_hungarian (P ,Dd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ])
                R .append ((rj ,)+match_hungarian (P ,Dd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,
                signed =True )[:3 ])
            except Exception :
                error +=1 
        kh =f1w (K );tf =f1w (T );rf =f1w (R )
        kc =f1w ([s for s in K if s [0 ]=="very"]);rc =f1w ([s for s in R if s [0 ]=="very"])
        print (f"\n{ad :<8} KAHIN {kh :.4f} (cok {kc :.4f}) | TESPIT {tf :.4f} | "
        f"ROBOT {rf :.4f} (cok {rc :.4f}) | error {error }")
        res_ [ad ]={"kahin":kh ,"kahin_cok":kc ,"tespit":tf ,"robot":rf ,
        "robot_cok":rc ,"error":error }
    if len (res_ )==2 :
        a1 ,a2 =a .ad 
        print (f"\nFARK ({a2 } - {a1 }): kahin {res_ [a2 ]['kahin']-res_ [a1 ]['kahin']:+.4f} | "
        f"tespit {res_ [a2 ]['tespit']-res_ [a1 ]['tespit']:+.4f} | "
        f"robot {res_ [a2 ]['robot']-res_ [a1 ]['robot']:+.4f}")
        ok =res_ [a2 ]["kahin"]>res_ [a1 ]["kahin"]
        print (f"KILL OLCUTU (kahin gecmeli): {'GECTI'if ok else 'KALDI -> SIG BOYAMA GERI AL'}")
    with io .open ("results/p9_ag_kiyas.json","w",encoding ="utf-8")as f :
        json .dump (res_ ,f ,indent =1 )
    print ("receipt -> results/p9_ag_kiyas.json")


if __name__ =="__main__":
    main ()
