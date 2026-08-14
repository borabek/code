# -*- coding: utf-8 -*-
"""FB-3: UC KALDIRACI AYNI KOSUDA SINA -- tespiti artirma denemesi.

TESPIT KAYBININ AYRISTIRILMASI (measured):
    A) 188 GT'ye HIC candidate absent        -> network orada CE/CT olasiligi uretmiyor
    B) 248 correct adayi gate reddetti -> ayirt edici feature DUSUK OY (1.33 vs 1.98)
    C) 313 wrong pozitif            -> real opening but TEL GIRISI DEGIL

UC KALDIRAC:
  1. 8 UYELI TOPLULUK (4 old + 4 new). Kayitta yazili: gate'in 13 ozelliginden only
     `votes` durust geometri bolmesinde transfer ediyor (AUC dususu 0.018, digerleri
     0.12-0.18). 4 uyeyle oy {1,2,3,4}; 8 uyeyle {1..8} -> GENELLESEN TEK SINYALIN
     COZUNURLUGU IKIYE KATLANIR. Ayrica B'yi dogrudan hedefler.
  2. TEL/ALET olasiligi (yardimci kafa) gate'e YENI feature -> C'yi hedefler.
     Gate'in bugune up to HIC sahip olmadigi bilgi.
  3. EKSEN-FARKINDALIKLI oy havuzlamasi: lateral<=3mm, depth SERBEST. Olculdu:
     birlesme %67.7 -> %70.7 (marjinal but more DOGRU rule -- urunun puanlayicisi
     `big_arbiter.greedy` already axis-farkindalikli, `_vote2` whereas duz Oklid kullaniyordu).
     Ayrica lateral 3mm siniri komsu kutuplari (step 3.5-6mm) KORUR.

TEZ CIZGISI: network mimarisi, remesh, tezin v_o turetmesi and 5-sinif kaybi DEGISMEZ.
Degisen: kac uye oy veriyor, oylar nasil havuzlaniyor, gate hangi ozniteligi goruyor.

KILL (onceden): tespit +0.01 VE grup bootstrap GA'si sifiri disliyor. Uretici-disi DUSMEMELI.
"""
import io 
import json 
import os 
import sys 
import time 

import numpy as np 
import torch 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def eksen_havuz (cp_lists ,yanal_mm =3.0 ,min_votes =1 ):
    """EKSEN-FARKINDALIKLI oy havuzlamasi.

    Iki candidate AYNI acikliktan sayilir: birinin digerine according to DIK mesafesi <= yanal_mm and
    eksenleri hizali (<=20 derece). Derinlik farki SERBEST -- because tez CP'yi agizda,
    manufacturer kontakta tanimlar and uyeler different derinliklerde durabilir.
    `big_arbiter.greedy` puanlarken already boyle does; oy havuzlamasi duz Oklid
    kullandigi for urun own inside tutarsizdi.
    """
    allc =[dict (c ,_mid =i )for i ,lst in enumerate (cp_lists )for c in lst ]
    allc .sort (key =lambda c :-float (c .get ("confidence",0.0 )))
    kept =[]
    for c in allc :
        p =np .asarray (c ["point"],float )
        d =np .asarray (c ["direction"],float )
        d =d /(np .linalg .norm (d )+1e-9 )
        hit =None 
        for k in kept :
            q =np .asarray (k ["point"],float )
            e =np .asarray (k ["direction"],float );e =e /(np .linalg .norm (e )+1e-9 )
            if abs (float (d @e ))<np .cos (np .radians (20.0 )):
                continue 
            v =p -q 
            if float (np .linalg .norm (v -(v @e )*e ))<=yanal_mm :
                hit =k ;break 
        if hit is None :
            c =dict (c );c ["_mids"]={c ["_mid"]}
            c ["_pts"]=[p ];c ["_ws"]=[float (c .get ("confidence",1.0 ))]
            kept .append (c )
        else :
            if c ["_mid"]in hit ["_mids"]:
                continue 
            hit ["_mids"].add (c ["_mid"])
            hit ["_pts"].append (p );hit ["_ws"].append (float (c .get ("confidence",1.0 )))
    for c in kept :
        c ["_votes"]=len (c ["_mids"])
        w =np .asarray (c ["_ws"],float );P =np .asarray (c ["_pts"],float )
        c ["point"]=list ((P *w [:,None ]).sum (0 )/max (w .sum (),1e-9 ))
    return [c for c in kept if c ["_votes"]>=min_votes ]


def main ():
    import cp_openings 
    import diffusionnet as D_ 
    import measure_set 
    import robot_cp as RC 
    import thesis_remesh 
    import wire_gate 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from infer_step_cp import load_any ,step_to_mesh 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    pp =cfg .get ("prediction_postproc",{})
    mv =int (pp .get ("min_vertices",4 ));vc =float (pp .get ("vertex_confidence_mask",0.3 ))
    cl =float (pp .get ("cluster_mm",3.0 ))
    ESKI =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    YENI =sorted (f"results/seg_extra/fb2_aux_s{i }.pt"for i in range (4 )
    if os .path .exists (f"results/seg_extra/fb2_aux_s{i }.pt"))
    print (f"eski uye {len (ESKI )} | yeni (tel-farkindalikli) uye {len (YENI )}")
    if not YENI :
        print ("YENI UYE YOK -- training bitmemis, cikiliyor");return 

    stp ={p :s for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    M_ESKI =[load_any (c ,dev =dev )[:2 ]for c in ESKI ]
    M_YENI =[load_any (c ,dev =dev )[:2 ]for c in YENI ]
    AUX =[]
    for c in YENI :
        d =torch .load (c ,map_location =dev ,weights_only =False )
        if d .get ("aux_state"):
            lin =torch .nn .Linear (list (d ["aux_state"].values ())[0 ].shape [1 ],1 )
            lin .load_state_dict (d ["aux_state"]);lin .to (dev ).eval ();AUX .append (lin )
        else :
            AUX .append (None )
    print (f"yardimci kafa yuklendi: {sum (1 for a in AUX if a is not None )}/{len (YENI )}")

    def uye_adaylari (V ,F ,model ,meta ,sp ):
        _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
        op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
        return_probs =True )
        q =np .asarray (pb ,float )
        lst =cp_openings .connection_points (
        V ,F ,q .argmax (-1 ),min_v =mv ,classes =(CE ,CT ),dedupe_mm =10.0 ,probs =q ,
        vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =cl ,step_path =sp )
        return lst ,q 

    KOL ={}
    for ad in ("A_ESKI4_duz","B_YENI4_eksen","C_HEPSI8_eksen"):
        KOL [ad ]={"RX":[],"RY":[],"RG":[],"RP":[],"RJ":[],"ALET":[]}
    t0 =time .time ()
    for k ,r in enumerate (DER ,1 ):
        if k %20 ==0 :
            print (f"  {k }/{len (DER )}  {time .time ()-t0 :.0f}s",flush =True )
        try :
            Vr ,Fr =step_to_mesh (stp [r ["pid"]])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            LE ,QE =[],[]
            for m_ ,mt in M_ESKI :
                l_ ,q_ =uye_adaylari (V ,F ,m_ ,mt ,stp [r ["pid"]]);LE .append (l_ );QE .append (q_ )
            LY ,QY ,ZY =[],[],[]
            for (m_ ,mt ),ax in zip (M_YENI ,AUX ):
                tut ={}
                h =m_ .last_lin .register_forward_hook (lambda mo ,gi ,go :tut .__setitem__ ("z",gi [0 ]))
                try :
                    l_ ,q_ =uye_adaylari (V ,F ,m_ ,mt ,stp [r ["pid"]])
                finally :
                    h .remove ()
                LY .append (l_ );QY .append (q_ )
                if ax is not None and "z"in tut :
                    with torch .no_grad ():
                        ZY .append (torch .sigmoid (ax (tut ["z"]).reshape (-1 )).cpu ().numpy ())
        except Exception as e :
            print (f"    {r ['pid']}: {type (e ).__name__ }");continue 
        alet =np .mean (ZY ,axis =0 )if ZY else None 
        for ad ,lists ,q_all ,pool in (
        ("A_ESKI4_duz",LE ,QE ,"duz"),
        ("B_YENI4_eksen",LY ,QY ,"axis"),
        ("C_HEPSI8_eksen",LE +LY ,QE +QY ,"axis")):
            cps =(RC ._vote2 (lists ,cluster_mm =5.0 ,min_votes =1 )if pool =="duz"
            else eksen_havuz (lists ,yanal_mm =3.0 ,min_votes =1 ))
            if not cps :
                continue 
            probs =sum (q_all )/len (q_all )
            P =np .array ([c ["point"]for c in cps ],float )
            X =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ,step_path =stp [r ["pid"]])
            try :
                from build_zengin_parite import _normaller ,zengin 
                XR =zengin (V ,F ,probs ,cps ,_normaller (V ,F ))
                X =np .hstack ([X ,XR ])
            except Exception :
                continue 
            al =np .zeros (len (P ))
            if alet is not None :
                for i in range (len (P )):
                    m2 =np .linalg .norm (V -P [i ],axis =1 )<=5.0 
                    al [i ]=float (alet [m2 ].mean ())if m2 .any ()else 0.5 
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            yy =np .zeros (len (P ),int )
            if len (G ):
                diff =P [:,None ,:]-G [None ,:,:]
                a_ =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -a_ [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (a_ )>40 ,np .inf ,pe )
                tt =max (3.0 ,0.06 *float (r ["diag"]))
                up ,ug =set (),set ()
                for d_ ,x_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
                for b in range (len (G ))):
                    if not np .isfinite (d_ )or d_ >tt or x_ in up or b_ in ug :
                        continue 
                    up .add (x_ );ug .add (b_ );yy [x_ ]=1 
            K =KOL [ad ]
            for i in range (len (P )):
                K ["RX"].append (X [i ]);K ["RY"].append (yy [i ]);K ["RG"].append (r ["geo"])
                K ["RP"].append (r ["pid"]);K ["ALET"].append (al [i ])
            K ["RJ"].append ({"n":r ["n"],"G":G ,"Gd":Gd ,"diag":r ["diag"],
            "geo":r ["geo"],"P":P ,"Pd":np .array (
            [c ["direction"]for c in cps ],float ),"nn":len (P )})

    print (f"\n{'arm':<20}{'candidate':>7}{'candidate-recall':>13}{'TESPIT':>9}{'+alet oz.':>11}")
    SON ,ROWS ={},{}
    for ad ,K in KOL .items ():
        if not K ["RY"]:
            continue 
        RX =np .array (K ["RX"]);RY =np .array (K ["RY"])
        RG =np .array (K ["RG"]);RP =np .array (K ["RP"]);AL =np .array (K ["ALET"])
        Z =np .zeros ((len (RX ),RX .shape [1 ]*2 ))
        for u in np .unique (RP ):
            i =np .where (RP ==u )[0 ]
            Z [i ]=wire_gate .within_part (RX [i ],"zskor")

        def olc (M ):
            o =np .zeros (len (RY ))
            for tr ,te in GroupKFold (n_splits =5 ).split (M ,RY ,RG ):
                o [te ]=RandomForestClassifier (n_estimators =300 ,min_samples_leaf =3 ,
                n_jobs =-1 ,random_state =0 ).fit (
                M [tr ],RY [tr ]).predict_proba (M [te ])[:,1 ]
            det ,gg =[],[];off =0 
            for r in K ["RJ"]:
                n =r ["nn"];sk =o [off :off +n ];off +=n 
                m =((sk >=0.5 *max (sk .max (),1e-9 ))&(sk >=0.25 ))if n else np .zeros (0 ,bool )
                rj ="cok"if r ["n"]>=8 else "dusuk"
                det .append ((rj ,)+esle (r ["P"][m ],r ["Pd"][m ],r ["G"],r ["Gd"],
                r ["diag"],0.0 ,180.0 ,True ))
                gg .append (r ["geo"])
            return det ,gg 
        d0 ,gg =olc (Z )
        d1 ,_ =olc (np .hstack ([Z ,AL [:,None ]]))
        ul =0 ;ngt =0 
        for r in K ["RJ"]:
            G =r ["G"];ngt +=len (G )
            if len (r ["P"])and len (G ):
                diff =r ["P"][:,None ,:]-G [None ,:,:]
                a_ =(diff *r ["Gd"][None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -a_ [...,None ]*r ["Gd"][None ,:,:],axis =-1 )
                pe =np .where (np .abs (a_ )>40 ,np .inf ,pe )
                ul +=int ((pe .min (0 )<=max (3.0 ,0.06 *float (r ["diag"]))).sum ())
        SON [ad ]={"candidate":int (len (RY )),"aday_recall":ul /max (ngt ,1 ),
        "tespit":f1w (d0 ),"tespit_alet":f1w (d1 )}
        ROWS [ad ]=(d0 ,d1 ,gg )
        s =SON [ad ]
        print (f"{ad :<20}{s ['candidate']:>7}{s ['aday_recall']:>13.4f}{s ['tespit']:>9.4f}"
        f"{s ['tespit_alet']:>11.4f}")

    if "A_ESKI4_duz"in ROWS :
        baseline =SON ["A_ESKI4_duz"]["tespit"]
        fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
        print ()
        for ad in SON :
            for j ,et in ((0 ,"tespit"),(1 ,"tespit+alet")):
                if ad =="A_ESKI4_duz"and j ==0 :
                    continue 
                _ ,lo ,hi =measure_set .grup_bootstrap (
                list (zip (ROWS ["A_ESKI4_duz"][0 ],ROWS [ad ][j ])),ROWS [ad ][2 ],fn ,n =2000 )
                v =SON [ad ]["tespit"if j ==0 else "tespit_alet"]
                print (f"  {ad } [{et }]: {v :.4f} ({v -baseline :+.4f}) GA[{lo :+.4f},{hi :+.4f}] "
                f"{'GERCEK'if (lo >0 or hi <0 )else 'noise'}")
    with io .open ("results/fb3_sinav.json","w",encoding ="utf-8")as f :
        json .dump (SON ,f ,indent =1 )
    print ("receipt -> results/fb3_sinav.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
