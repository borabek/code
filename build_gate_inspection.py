# -*- coding: utf-8 -*-
"""P1.9-11: wire-gate HATA inceleme paketi. Gate'in two error tipini part baglaminda renkli goster ki
insan real_wire / tool / ambiguous diye yargilayabilsin:
  YESIL  = dropped_real_wire  (gate ATTI but manufacturer-CP'ye eslesiyor -> gate wrong attiysa real tel)
  TURUNCU= kept_FP            (gate TUTTU but manufacturer-CP'ye eslesmez -> tool VEYA listelenmemis tel)
  GRI    = correct (kept-TP + dropped-tool)
Her CP'de wire_score yazili. NOT: gate already dogrulandi (vote-override %95 tool output); this paket
calibration-ince-setting for, low oncelik.
"""
import os ,sys ,glob ,json 
import numpy as np ,torch 
import matplotlib ;matplotlib .use ("Agg");import matplotlib .pyplot as plt 
from mpl_toolkits .mplot3d .art3d import Poly3DCollection 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,connector3d ,wire_gate 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 
from robot_cp import _vote2 

HELD =set (open ("_hw_r3.txt").read ().split ())
dev ="cuda"if torch .cuda .is_available ()else "cpu"


def main ():
    cks =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    os .environ ["BA_ALLOW_SEEN"]="1"
    parts =[p for p in eligible ()if (p [0 ]=="WEI"and p [1 ]in HELD )][:40 ]+[p for p in eligible ()if p [0 ]=="PXC"][:40 ]
    picked =[]# parts with >=1 gate error, up to 16
    for mfg ,pid ,jf ,stp in parts :
        if len (picked )>=16 :break 
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
                _opd =f"{OP }_k{int (meta .get ('k_eig',64 ))}"
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =_opd ,return_probs =True )
                pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
                per .append (cp_openings .connection_points (V ,F ,pb .argmax (-1 ),min_v =30 ,classes =(CE ,CT ),
                dedupe_mm =10.0 ,probs =pb ,vertex_conf =0.5 ,ct_depth_min_mm =1.0 ,cluster_mm =5.0 ))
            probs =acc /len (per );cps =_vote2 (per )
            if len (cps )<2 :continue 
            wire_gate .apply (V ,F ,probs ,cps ,CE ,CT ,threshold =0.35 )# sets wire_score (no filter: top_n irrelevant)
            R ,t ,_ =align_frames (Vr ,Vj )
            P =np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t 
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .where (np .abs (al )<=40.0 ,np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 ),np .inf )
            order =sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))if pe [a ,b ]<=tol )
            up ,ug =set (),set ();matched =set ()
            for dd ,a ,b in order :
                if a in up or b in ug :continue 
                up .add (a );ug .add (b );matched .add (a )
            cats =[]# 0=grey ok, 1=green dropped_real_wire, 2=orange kept_FP
            for i ,c in enumerate (cps ):
                ws =c .get ("wire_score",1.0 );m =i in matched 
                if ws <0.35 and m :cats .append (1 )
                elif ws >=0.35 and not m :cats .append (2 )
                else :cats .append (0 )
            if 1 in cats or 2 in cats :
                picked .append ((pid ,mfg ,V ,F ,cps ,cats ))
        except Exception :
            continue 

    n =len (picked );cols =4 ;rows =(n +cols -1 )//cols 
    fig =plt .figure (figsize =(4.5 *cols ,4.6 *rows ))
    for k ,(pid ,mfg ,V ,F ,cps ,cats )in enumerate (picked ):
        tri =V [F ];ext =V .max (0 )-V .min (0 );c0 =V .mean (0 );rr =ext .max ()*0.55 ;L =ext .max ()*0.16 
        ax =fig .add_subplot (rows ,cols ,k +1 ,projection ="3d")
        ax .add_collection3d (Poly3DCollection (tri ,facecolors =[0.82 ,0.82 ,0.82 ],edgecolors ="none",alpha =0.9 ))
        col ={0 :"grey",1 :"lime",2 :"orange"}
        for c ,cat in zip (cps ,cats ):
            p =np .asarray (c ["point"],float );dd =np .asarray (c ["direction"],float )
            dd =dd /(np .linalg .norm (dd )+1e-9 )*L 
            ax .quiver (*p ,*dd ,color =col [cat ],linewidth =2.2 ,arrow_length_ratio =0.35 ,zorder =10 )
            ax .scatter (*p ,s =40 ,c =col [cat ],edgecolors ="black",linewidths =0.5 ,depthshade =False ,zorder =11 )
            if cat :ax .text (p [0 ],p [1 ],p [2 ],f"{c .get ('wire_score',0 ):.2f}",fontsize =7 ,zorder =12 )
        ax .set_xlim (c0 [0 ]-rr ,c0 [0 ]+rr );ax .set_ylim (c0 [1 ]-rr ,c0 [1 ]+rr );ax .set_zlim (c0 [2 ]-rr ,c0 [2 ]+rr )
        try :ax .set_box_aspect (ext )
        except Exception :pass 
        thin =int (np .argmin (ext ));ev ,az ={0 :(0 ,0 ),1 :(0 ,-90 ),2 :(90 ,-90 )}[thin ]
        ax .view_init (elev =ev ,azim =az );ax .set_axis_off ()
        ng =cats .count (1 );no =cats .count (2 )
        ax .set_title (f"{pid } ({mfg })  yesil-dropped:{ng } turuncu-keptFP:{no }",fontsize =9 )
    fig .suptitle ("WIRE-GATE INCELEME: YESIL=gate-atti-but-mfg-matched (real tel mi?) TURUNCU=gate-tuttu-but-unmatched (tool mu, listelenmemis tel mi?)",
    fontsize =12 )
    plt .tight_layout (rect =[0 ,0 ,1 ,0.96 ])
    plt .savefig ("results/_gate_inspection.png",dpi =95 ,bbox_inches ="tight")
    print (f"-> results/_gate_inspection.png ({n } part, gate hatalari renkli)")


if __name__ =="__main__":
    main ()
