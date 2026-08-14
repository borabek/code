# -*- coding: utf-8 -*-
"""GECE ISI: ZENGIN TEMSIL cikarimi (tum corpus, urun-sadik path).
RATIONALE (this oturumda measured): 13 el-yapimi feature TEMSIL DARBOGAZI. Sadece 'parcada nerede'
eklemek WEI havuzunda top-N F1 +0.041 (part-out) / +0.031 (AILE-out, genellenir) getirdi.
Bu script same urun yolunu (4-model union + avg probs) runs but each candidate for ZENGIN feature dumper:
  A konum (mesh-bbox normalize, 3) + bbox-face uzakliklari (6)
  B very-radius sinif-olasilik profili (r=3,6,10,15 x 5 sinif = 20) + yogunluk (4)
  C TAPER profili (depth 0..8mm'de mouth yaricapi, 5) -- mesh-tabanli huni testi (B-rep koni olmustu)
  D normal-degisim / egrilik proxy (3 radius)
Cikti: results/rich_feats.npz (X13, XR, y, groups, mfg, ngt, pos) -> CPU analizinde AUC/F1 kazanci olculur."""
import os ,sys ,json ,time 
import numpy as np ,torch ,trimesh 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,wire_gate 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 
from robot_cp import _vote2 

dev ="cuda"if torch .cuda .is_available ()else "cpu"
cfg =json .load (open ("cp_config.json"))
CK =cfg ["current_product"]["checkpoints"]
pp_ =cfg .get ("prediction_postproc",{})
MV ,VC ,CL =int (pp_ .get ("min_vertices",30 )),float (pp_ .get ("vertex_confidence_mask",0.5 )),float (pp_ .get ("cluster_mm",5.0 ))
DD =10.0 
# post-isleme override (ortam degiskeni) -- ceiling/F1 deneyleri for
MV =int (os .environ .get ("CP_MV",MV ));VC =float (os .environ .get ("CP_VC",VC ))
CL =float (os .environ .get ("CP_CL",CL ));DD =float (os .environ .get ("CP_DD",DD ))
PROMOTE =float (os .environ .get ("CP_PROMOTE",0.0 ))# CC-B: argmax kapisini ac (opt-in)
RAD =(3.0 ,6.0 ,10.0 ,15.0 )
DEPTHS =(0.0 ,2.0 ,4.0 ,6.0 ,8.0 )
RICH_N =3 +6 +20 +4 +5 +3 
OUT_NPZ ="results/rich_feats.npz"


def rich_feats (V ,F ,probs ,cps ,Nrm ):
    lo ,hi =V .min (0 ),V .max (0 );diag =float (np .linalg .norm (hi -lo ))+1e-9 
    out =[]
    for c in cps :
        p =np .asarray (c ["point"],float );d =np .asarray (c ["direction"],float )
        d =d /(np .linalg .norm (d )+1e-9 )
        f =list ((p -lo )/np .maximum (hi -lo ,1e-6 ))# A1 konum (3)
        f +=list ((p -lo )/diag )+list ((hi -p )/diag )# A2 face uzakliklari (6)
        rel =V -p ;dist =np .linalg .norm (rel ,axis =1 )
        for r in RAD :# B very-radius (20+4)
            m =dist <=r 
            if m .any ():
                f +=list (probs [m ].mean (0 ));f .append (float (m .sum ())/len (V ))
            else :
                f +=[0.0 ]*probs .shape [1 ];f .append (0.0 )
        al =rel @d ;perp =np .linalg .norm (rel -al [:,None ]*d [None ,:],axis =1 )
        for t in DEPTHS :# C taper profili (5)
            ring =(np .abs (al -t )<=1.0 )&(perp <=12.0 )
            f .append (float (np .median (perp [ring ]))if ring .any ()else 0.0 )
        for r in (3.0 ,6.0 ,10.0 ):# D normal-degisim (3)
            m =dist <=r 
            f .append (float (np .std (Nrm [m ]@d ))if m .sum ()>=4 else 0.0 )
        out .append (f )
    return np .array (out ,float )


def main ():
    models =[load_any (c ,dev =dev )[:2 ]for c in CK ]
    oos =set (open ("pxc_out_of_scope.txt").read ().split ())if os .path .exists ("pxc_out_of_scope.txt")else set ()
    held =set (open ("_hw_r3.txt").read ().split ())
    # FAZ2/P0: TEMIZ set = leakage-muhafizi OPEN (segmentasyon egitiminde gorulmemis parts).
    # Once temizi al, after BA_ALLOW_SEEN with tamamini al and FARKI 'seen' as isaretle ->
    # single kosuda hem temiz hem canonical-kiyaslanabilir number uretilebilir.
    def _filt (lst ):return [(m ,p ,jf ,s )for m ,p ,jf ,s in lst 
    if (m =="WEI"and p in held )or (m =="PXC"and p not in oos )]
    clean =_filt (eligible ());clean_ids ={p for _ ,p ,_ ,_ in clean }
    os .environ ["BA_ALLOW_SEEN"]="1"
    if "--list"in sys .argv :# open part listesi (ek gate-training havuzu for)
        want =set (open (sys .argv [sys .argv .index ("--list")+1 ]).read ().split ())
        parts =[(m ,p ,jf ,s )for m ,p ,jf ,s in eligible ()if p in want ]
        global OUT_NPZ ;OUT_NPZ =sys .argv [sys .argv .index ("--out")+1 ]if "--out"in sys .argv else OUT_NPZ 
    else :
        parts =_filt (eligible ())
    print (f"{len (parts )} part ({sum (1 for m ,*_ in parts if m =='WEI')} WEI + {sum (1 for m ,*_ in parts if m =='PXC')} PXC)"
    f" | TEMIZ (sizintisiz) {len (clean )} | seen-flagli {len (parts )-len (clean )}",flush =True )
    X13 ,XR ,YY ,GG ,MM ,POS ,NGT =[],[],[],[],[],[],{}
    PIDS ,SEEN =[],[]
    # DEVAM-EDEBILIRLIK: onceki kismi kayittan yukle, islenmisleri atla (hang sonrasi is kaybi olmasin)
    _done =set ()
    if os .path .exists (OUT_NPZ ):
        try :
            _p =np .load (OUT_NPZ ,allow_pickle =True )
            _done ={str (x )for x in _p ["part_ids"]}
            X13 =[_p ["X13"]];XR =[_p ["XR"]];YY =[_p ["y"]];POS =[_p ["pos"]]
            GG =list (_p ["groups"]);MM =list (_p ["mfg"])
            PIDS =[str (x )for x in _p ["part_ids"]];SEEN =list (_p ["seen"])
            NGT ={int (g ):int (n )for g ,n in zip (_p ["grp_ids"],_p ["ngt"])}
            print (f"  [devam] {len (_done )} part zaten islenmis, atlanacak",flush =True )
        except Exception :
            _done =set ()
    _skip =set (open ("_skip_parts.txt").read ().split ())if os .path .exists ("_skip_parts.txt")else set ()
    t0 =time .time ();k =0 
    for mfg ,pid ,jf ,stp in parts :
        k +=1 
        if pid in _done or pid in _skip :continue 
        # HANGI PARCADA OLDUGUMUZU DISK'E YAZ: is takilirsa hangi parcanin sucu oldugu ANINDA bilinsin.
        # (2026-07-28: a inference 3 saat sessizce durdu and suclu parcayi bulmak ayri a sorusturma
        # became; log only each 25 parcada a yaziyordu and part ADINI never yazmiyordu.)
        try :
            with open ("results/_extract_current.txt","w")as _cf :
                _cf .write (f"{mfg }.{pid }	{k }/{len (parts )}	{time .strftime ('%H:%M:%S')}	{stp }")
        except OSError :
            pass 
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j ["ConnectionPoints"]],float )
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
                per .append (cp_openings .connection_points (V ,F ,pb .argmax (-1 ),min_v =MV ,classes =(CE ,CT ),
                dedupe_mm =DD ,probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,conn_promote =PROMOTE ))
            probs =acc /len (per )
            cps =_vote2 (per ,min_votes =1 )
            if not cps :continue 
            Nrm =trimesh .Trimesh (V ,F ,process =False ).vertex_normals .view (np .ndarray )
            x13 =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT )
            xr =rich_feats (V ,F ,probs ,cps ,Nrm )
            R ,t ,_ =align_frames (Vr ,Vj )
            P =np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t 
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .where (np .abs (al )<=40.0 ,np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 ),np .inf )
            yy =np .zeros (len (P ),int )
            order =sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))if pe [a ,b ]<=tol )
            up ,ug =set (),set ()
            for dd ,a ,b in order :
                if a in up or b in ug :continue 
                up .add (a );ug .add (b );yy [a ]=1 
            gi =len (NGT );NGT [gi ]=len (G )
            X13 .append (x13 );XR .append (xr );YY .append (yy );POS .append (P )
            GG +=[gi ]*len (cps );MM +=[1 if mfg =="WEI"else 0 ]*len (cps )
            PIDS .append (pid );SEEN .append (0 if pid in clean_ids else 1 )
        except Exception as e :
            continue 
        if k %25 ==0 :
            print (f"  {k }/{len (parts )}  {len (NGT )} ok  {time .time ()-t0 :.0f}s (kaydediliyor)",flush =True )
            _gid =np .array (sorted (NGT ));_ng =np .array ([NGT [g ]for g in _gid ])
            np .savez (OUT_NPZ ,X13 =np .vstack (X13 ),XR =np .vstack (XR ),y =np .concatenate (YY ),
            groups =np .array (GG ),mfg =np .array (MM ),pos =np .vstack (POS ),grp_ids =_gid ,ngt =_ng ,
            part_ids =np .array (PIDS ),seen =np .array (SEEN ))
    gid =np .array (sorted (NGT ));ng =np .array ([NGT [g ]for g in gid ])
    np .savez (OUT_NPZ ,X13 =np .vstack (X13 ),XR =np .vstack (XR ),y =np .concatenate (YY ),
    groups =np .array (GG ),mfg =np .array (MM ),pos =np .vstack (POS ),grp_ids =gid ,ngt =ng ,
    part_ids =np .array (PIDS ),seen =np .array (SEEN ))
    print (f"-> {OUT_NPZ }  {len (np .concatenate (YY ))} candidate, {len (NGT )} part "
    f"({int (np .sum (SEEN ))} seen-flagli), rich-dim {np .vstack (XR ).shape [1 ]}  {time .time ()-t0 :.0f}s")


if __name__ =="__main__":
    main ()
