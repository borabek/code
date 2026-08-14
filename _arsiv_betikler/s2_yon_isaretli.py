# -*- coding: utf-8 -*-
"""D1: YON SECICISINI DAGITILABILIR YAP -- training korpusunda egit, measurement kumesinde sina.

C4 SONUCU: ayrik direction selector, measurement kumesinde GRUP-CAPRAZ as +0.0238 (GA
[+0.0027,+0.0464]) verdi. Bu SIZINTISIZ a TAHMINDIR but DAGITILABILIR DEGILDIR:
dagitilacak selector, measurement kumesini HIC gormemis veriyle egitilmelidir. Aksi halde
headline own training verisiyle olculur.

BU BETIK:
  1. Olcum kumesinin GEOMETRI GRUPLARI and LOCKED disindaki parcalari selects
  2. Her biri for URUNUN yolundan candidate produces, direction SOZLUGUNU kurar, GT with labels
  3. Secici orada egitilir -> results/yon_secici.pkl
  4. Olcum kumesine uygulanir (sozlugu c4_sozluk.pkl'den) and UCTAN UCA olculur
  5. Kill gecerse cp_config'e yazilir

KILL (C4 with AYNI, degistirilmedi): robot +0.01 VE grup bootstrap GA'si sifiri disliyor.
Tespit yapisal as degismemeli; manufacturer-disi dusmemeli.
"""
import io 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 
import torch 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

EGT ="results/yon_egitim_isaretli.pkl"
MODEL ="results/yon_secici_isaretli.pkl"
KAYNAK =["mevcut","ham","uye","obb","dik","uzlasi","yuz_uzlasi","yuzn"]
MARJ =0.05 # C4'te two yari da 0.02-0.05 secti; ortasi sabitlenir


def _birim (v ):
    v =np .asarray (v ,float );n =np .linalg .norm (v )
    return v /n if n >1e-9 else None 


def sozluk_kur (V ,P ,Pd ,Dham ,UYE ,YUZN ):
    from c2_part_level_direction import obb_eksenleri ,uzlasi_yonu 
    OBB =obb_eksenleri (V )if V is not None else []
    UZ =uzlasi_yonu (Pd )
    out =[]
    for i in range (len (P )):
        d0 =_birim (Pd [i ])
        S =[("mevcut",d0 ),("ham",_birim (Dham [i ]))]
        if UYE :
            n_u =0 
            for lst in UYE :
                for cc in (lst if isinstance (lst ,(list ,tuple ))else []):
                    if not isinstance (cc ,dict ):
                        continue 
                    pt =np .asarray (cc .get ("point",P [i ]),float )
                    dd =cc .get ("direction")
                    if dd is None or np .linalg .norm (pt -P [i ])>5.0 :
                        continue 
                    b =_birim (dd )
                    if b is not None :
                        S .append (("uye",b ));n_u +=1 
                    if n_u >=4 :
                        break 
        for a in OBB :
            if a is not None :
                S +=[("obb",a ),("obb",-a )]
        if UZ is not None :
            S .append (("uzlasi",UZ ))
        if OBB and len (P )>2 :
            t_ =P @OBB [0 ]
            ay =np .abs (t_ -t_ [i ])<=3.0 
            if ay .sum ()>=2 :
                fu =uzlasi_yonu (Pd [ay ])
                if fu is not None :
                    S .append (("yuz_uzlasi",fu ))
        if d0 is not None :
            for a in OBB :
                if a is None :
                    continue 
                pr =_birim (a -float (a @d0 )*d0 )
                if pr is not None :
                    S +=[("dik",pr ),("dik",-pr )]
        for a in YUZN :
            S +=[("yuzn",a ),("yuzn",-a )]
        out .append ([(t2 ,v )for t2 ,v in S if v is not None ])
    return out ,(UZ if UZ is not None else np .zeros (3 ))


def satirla (Xk ,SOZ ,UZ ,Pd ,hedef_yon ):
    """(candidate, giris) satirlari + label (hedef_yon verilmisse)."""
    RX ,RY ,RJ =[],[],[]
    for i in range (len (SOZ )):
        d0 =Pd [i ]
        for gi ,(tip ,v )in enumerate (SOZ [i ]):
            oz =[1.0 if tip ==t2 else 0.0 for t2 in KAYNAK ]
            oz .append (float (np .degrees (np .arccos (np .clip (abs (float (v @d0 )),0 ,1 )))))
            oz .append (float (np .degrees (np .arccos (np .clip (abs (float (v @UZ )),0 ,1 ))))
            if np .linalg .norm (UZ )>0.5 else 90.0 )
            oz .append (float (gi ));oz .append (float (len (SOZ [i ])))
            RX .append (np .concatenate ([Xk [i ],oz ]));RJ .append ((i ,gi ))
            if hedef_yon is not None :
                g =hedef_yon [i ]
                # ISARETLI ETIKET: direction, ureticinin InsertDirection'iyla AYNI tarafa bakmali.
                # Onceki version abs() kullaniyordu -> 180 derece ters a direction "correct" sayiliyor,
                # selector de onu secebiliyordu. Sonuc: EKSEN metrigi +0.0319 but FIZIKSEL
                # (signed) metrik 0.5818 -> 0.5813, i.e. kazanc SAHTEYDI. Bu row that hatayi
                # kapatir: sign KORUNUR.
                RY .append (int (g is not None and 
                np .degrees (np .arccos (np .clip (float (v @g ),-1 ,1 )))<=10.0 ))
    return RX ,RY ,RJ 


def main ():
    import cad_eval 
    import diffusionnet as D_ 
    import gate_bench as T 
    import measure_set 
    import robot_cp as RC 
    import thesis_remesh 
    import wire_gate 
    from big_arbiter import eligible 
    from infer_step_cp import load_any ,step_to_mesh 
    from sina_cluster import esle 
    from sklearn .ensemble import RandomForestClassifier 
    from gece_kilit import guard 

    guard ("d1")
    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    cfg =D ["cfg"]
    gk =D ["gk"];tg ={r ["geo"]for r in D ["DER"]}
    s3 =json .load (io .open ("results/split3.json",encoding ="utf-8"))
    lock_geo ={gk .get (str (p ),"absent:"+str (p ))for p in s3 ["locked"]["parts"]}
    E =[(m ,p ,jf ,s )for m ,p ,jf ,s in eligible ()
    if gk .get (p ,"absent:"+p )not in tg and gk .get (p ,"absent:"+p )not in lock_geo ]
    print (f"\nEGITIM havuzu: {len (E )} part (measurement + LOCKED gruplari CIKARILDI)")

    if os .path .exists (EGT ):
        with open (EGT ,"rb")as f :
            RX ,RY =pickle .load (f )
        print (f"training onbellekten: {len (RY )} satir",flush =True )
    else :
        cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
        dev ="cuda"if torch .cuda .is_available ()else "cpu"
        models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
        from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
        RX ,RY =[],[]
        t0 =time .time ();ok =0 
        for k ,(mfg ,p ,jf ,stp_ )in enumerate (E ,1 ):
            if k %25 ==0 :
                print (f"  {k }/{len (E )} ok={ok }  {time .time ()-t0 :.0f}s",flush =True )
                guard (f"d1 {k }")
                with open (EGT ,"wb")as f :
                    pickle .dump ((RX ,RY ),f )
            try :
                j =json .load (io .open (jf ,encoding ="utf-8-sig"))
                cps_gt =j .get ("ConnectionPoints")or []
                if not cps_gt :
                    continue 
                G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in cps_gt ],float )
                Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in cps_gt ],float )
                Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
                Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
                Vr ,Fr =step_to_mesh (stp_ )
                V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
                R ,t_ ,_ =cad_eval .align_frames (Vr ,Vj )
                Gm =(G -t_ )@R ;Gdm =Gd @R 
                pbs =[]
                for model ,meta in models :
                    _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                    op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                    return_probs =True )
                    pbs .append (np .asarray (pb ,float ))
                cps ,probs ,_ ,_ =RC .derive_candidates (V ,F ,pbs ,stp_ ,cfg =cfg )
                if not cps :
                    continue 
                Xf =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ,step_path =stp_ )
                Pham =np .array ([c ["point"]for c in cps ],float )
                Dham =np .array ([c ["direction"]for c in cps ],float )
                cc =[{"point":Pham [i ],"direction":Dham [i ]}for i in range (len (cps ))]
                cc =wire_gate .pose_correct (Xf ,cc )
                if cfg .get ("robot_aci_secici"):
                    cc =wire_gate .angle_correct (Xf ,cc )
                P =np .array ([x ["point"]for x in cc ],float )
                Pd =np .array ([x ["direction"]for x in cc ],float )
                YUZN =[]
                try :
                    import brep_axes as _ba 
                    pl =_ba .planes (stp_ )
                    if pl is not None and len (pl ):
                        _n =np .asarray (pl [1 ],float );_rr =np .asarray (pl [2 ],float )
                        for jj in np .argsort (-_rr )[:6 ]:
                            b =_birim (_n [jj ])
                            if b is not None :
                                YUZN .append (b )
                except Exception :
                    pass 
                SOZ ,UZ =sozluk_kur (V ,P ,Pd ,Dham ,None ,YUZN )
                # ETIKET: each adayi most yakin GT'ye bagla (tespit toleransi)
                diff =P [:,None ,:]-Gm [None ,:,:]
                al =(diff *Gdm [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gdm [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )>40 ,np .inf ,pe )
                tt =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
                hedef =[]
                for i in range (len (P )):
                    if np .isfinite (pe [i ]).any ()and pe [i ].min ()<=tt :
                        hedef .append (Gdm [int (np .argmin (pe [i ]))])
                    else :
                        hedef .append (None )
                ax ,ay ,rj_ =satirla (Xf ,SOZ ,UZ ,Pd ,hedef )
                # YALNIZ eslesen adaylarin satirlari: label however GT'ye baglanabilen
                # adaylarda tanimlidir. (satirla() RJ'de (candidate, giris) ciftini returns.)
                for n ,(i ,_gi )in enumerate (rj_ ):
                    if hedef [i ]is not None :
                        RX .append (ax [n ]);RY .append (ay [n ])
                ok +=1 
            except Exception as e :
                if k <=5 :
                    print (f"    {p }: {type (e ).__name__ }: {e }")
        with open (EGT ,"wb")as f :
            pickle .dump ((RX ,RY ),f )
        print (f"-> {EGT } | {len (RY )} satir / {ok } part",flush =True )

    RX =np .array (RX ,float );RY =np .array (RY )
    print (f"\negitim: {len (RY )} satir | dogru giris {RY .mean ():.1%} | sutun {RX .shape [1 ]}")
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (RX ,RY )
    with open (MODEL ,"wb")as f :
        pickle .dump ({"clf":clf ,"source":KAYNAK ,"marj":MARJ ,
        "not":"D1: measurement+LOCKED gruplari DISINDA egitildi"},f )
    print (f"-> {MODEL }")

    # --- OLCUM KUMESINDE SINA (dictionary c4'ten)
    with open ("results/c4_sozluk.pkl","rb")as f :
        PARCA =pickle .load (f )
    rob0 ,rob1 ,det0 ,det1 ,gg =[],[],[],[],[]
    for r in D ["DER"]:
        d_ =PARCA .get (r ["pid"])
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="very"if r ["n"]>=8 else "low"
        if d_ is None :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ));Pn =Pd 
        else :
            P =d_ ["P"];Pd =d_ ["Pd"];Pn =Pd .copy ()
            ax ,_ ,RJ =satirla (d_ ["X"],d_ ["SOZ"],d_ ["UZ"],Pd ,None )
            if ax :
                pr =clf .predict_proba (np .array (ax ,float ))[:,1 ]
                SK ={}
                for n_ ,(i ,gi )in enumerate (RJ ):
                    SK .setdefault (i ,{})[gi ]=pr [n_ ]
                for i ,s in SK .items ():
                    gi =max (s ,key =s .get );s0 =s .get (0 ,0.0 )
                    if gi !=0 and s [gi ]-s0 >=MARJ :
                        Pn [i ]=d_ ["SOZ"][i ][gi ][1 ]
        rob0 .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        rob1 .append ((rj ,)+esle (P ,Pn ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        det0 .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
        det1 .append ((rj ,)+esle (P ,Pn ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
        gg .append (r ["geo"])
    fn =lambda rows :T .f1w ([q for _ ,q in rows ])-T .f1w ([p for p ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (rob0 ,rob1 )),gg ,fn ,n =3000 )
    d =T .f1w (rob1 )-T .f1w (rob0 )
    print (f"\nDAGITILABILIR SECICI (training korpusunda egitildi):")
    print (f"  robot  {T .f1w (rob0 ):.4f} -> {T .f1w (rob1 ):.4f}  ({d :+.4f})  GA[{lo :+.4f},{hi :+.4f}]")
    print (f"  tespit {T .f1w (det0 ):.4f} -> {T .f1w (det1 ):.4f}")
    gecti =d >=0.01 and lo >0 
    print (f"\nKILL: robot +0.01 VE GA>0 -> {'GECTI -> DAGITILIR'if gecti else 'GECMEDI'}")
    with io .open ("results/s2_yon_isaretli.json","w",encoding ="utf-8")as f :
        json .dump ({"robot_once":T .f1w (rob0 ),"robot_sonra":T .f1w (rob1 ),"difference":d ,
        "ga":[lo ,hi ],"tespit_once":T .f1w (det0 ),"tespit_sonra":T .f1w (det1 ),
        "marj":MARJ ,"egitim_satir":int (len (RY )),"gecti":bool (gecti )},
        f ,indent =1 )
    print ("receipt -> results/s2_yon_isaretli.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
