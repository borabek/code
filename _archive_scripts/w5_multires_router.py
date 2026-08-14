# -*- coding: utf-8 -*-
"""W5: MULTIRES KOLUNU URUNUN GERCEK YONLENDIRMESIYLE OLC (denetimin P1.1 + P1.2).

s9 NE YAPMISTI: hedef parcalari `r["n"] >= 8` with, i.e. GT'DEKI CP SAYISIYLA secti and
only that 60 parcayi puanladi. Iki kusur:
  1. Urun calisma aninda GT rejimini BILMEZ -- `_highcp_router` geometriden prediction eder.
     (w4: 58 correct / 3 YANLIS-POZITIF / 2 kacan. Router'in hedefi 61 part.)
  2. Yalniz very-CP lower kumesinde puanlandi; wrong yonlendirilen DUSUK-CP parcalarin
     BEDELI never olculmedi and genel etki 0.105 agirlikla TAHMIN edildi.

W5 DUZELTIR: arm, router'in "+" dedigi parcalara uygulanir; puanlama TUM 194 parcada is done
(yonlendirilmeyen parts two kolda ozdestir -> katkilari dogal as sifirdir). Boylece
produced number dogrudan MANSETLE karsilastirilabilir, agirlikla prediction edilmez.

Gate: DAGITILAN path (zengin_parite_w2.npz), measurement gruplari egitimden CIKARILIR.
KILL (onceden): agirlikli detection kazanci >= +0.02 VE grup bootstrap GA'si sifiri disliyor.
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
OUT ="results/w5_multires_router.json"
ONBELLEK ="results/_w5_9k.pkl"


def main ():
    import diffusionnet as D 
    import measure_set 
    import robot_cp as RC 
    import thesis_remesh 
    import wire_gate 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from infer_step_cp import load_any ,step_to_mesh 
    from sina_cluster import esle ,f1_rejim ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    NPZ =cfg ["current_product"]["wire_gate"]["egitim_verisi"]
    stp_of ={p :s for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }

    with io .open ("results/w4_router_regime.json",encoding ="utf-8")as f :
        W4 =json .load (f )
    yonlendirilen ={s ["pid"]for s in W4 ["satirlar"]if s ["router"]}
    print (f"\nrouter '+' part: {len (yonlendirilen )} (GT very-CP: "
    f"{sum (1 for s in W4 ['satirlar']if s ['gt_cok'])})",flush =True )

    # --- GATE: dagitilan veriyle, measurement gruplari disarida
    zen =np .load (NPZ ,allow_pickle =True )
    Xt =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    ytr =np .asarray (zen ["y"]);tpid =np .array ([str (x )for x in zen ["pids"]])
    keep =~np .isin (np .array ([gk .get (p ,"absent:"+p )for p in tpid ]),list (tg ))
    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")
    Z =np .zeros ((len (Xt ),Xt .shape [1 ]*2 ))
    for u in np .unique (tpid ):
        i =np .where (tpid ==u )[0 ]
        Z [i ]=wire_gate .within_part (Xt [i ],DON )
    gate ={"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],ytr [keep ]),
    "n_feat":Z .shape [1 ],"donusum":DON }
    print (f"gate: {NPZ } | {int (keep .sum ())} candidate egitimde",flush =True )

    def son_islem (X ,P ,Pd ):
        """URUNUN last islem zinciri: pose + secili angle duzeltmesi."""
        cps =[{"point":P [j ],"direction":Pd [j ]}for j in range (len (P ))]
        cps =wire_gate .pose_correct (X ,cps )
        if cfg .get ("robot_aci_secici"):
            cps =wire_gate .angle_correct (X ,cps )
        return (np .array ([c ["point"]for c in cps ],float ),
        np .array ([c ["direction"]for c in cps ],float ))

    K9 ={}
    if os .path .exists (ONBELLEK ):
        with open (ONBELLEK ,"rb")as f :
            K9 =pickle .load (f )
        print (f"9k onbellegi: {len (K9 )} part",flush =True )

    cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =None 
    det ={"A":[],"B":[]};rob ={"A":[],"B":[]};grup =[]
    t0 =time .time ()
    for i_ ,r in enumerate (DER ,1 ):
        if i_ %25 ==0 :
            print (f"  {i_ }/{len (DER )}  {time .time ()-t0 :.0f}s",flush =True )
            with open (ONBELLEK ,"wb")as f :
                pickle .dump (K9 ,f )
                # --- A: dagitilan urun (6k), measurement onbelleginden
        PA =np .zeros ((0 ,3 ));PdA =np .zeros ((0 ,3 ))
        if r ["X"]is not None and r .get ("XR")is not None :
            X58 =np .hstack ([r ["X"],r ["XR"]])
            k =wire_gate .decision_mask (wire_gate .decision_score (gate ,X58 ))
            if k .any ():
                PA ,PdA =son_islem (X58 [k ],r ["P"][k ],r ["Pd"][k ])
        PB ,PdB =PA .copy (),PdA .copy ()

        # --- B: only ROUTER '+' dediyse 9k birlesimi ekle
        if r ["pid"]in yonlendirilen :
            if r ["pid"]not in K9 :
                try :
                    if models is None :
                        models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
                    stp =stp_of .get (r ["pid"])
                    Vr ,Fr =step_to_mesh (stp )
                    V9 ,F9 =thesis_remesh .remesh_uniform (Vr ,Fr ,target =9000 )
                    V9 =np .ascontiguousarray (V9 ,np .float64 )
                    F9 =np .ascontiguousarray (F9 ,np .int64 )
                    pbs9 =[]
                    for model ,meta in models :
                        _ ,pb =D .predict (model ,meta ,V9 ,F9 ,device =dev ,
                        op_cache_dir =f"results/step_infer/ops9_k{int (meta .get ('k_eig',64 ))}",
                        return_probs =True )
                        pbs9 .append (np .asarray (pb ,float ))
                    cps9 ,probs9 ,_ ,_ =RC .derive_candidates (V9 ,F9 ,pbs9 ,stp ,cfg =cfg )
                    if cps9 :
                        X9 =wire_gate .feats_for (V9 ,F9 ,probs9 ,cps9 ,CE ,CT ,step_path =stp )
                        K9 [r ["pid"]]=(X9 ,
                        np .array ([c ["point"]for c in cps9 ],float ),
                        np .array ([c ["direction"]for c in cps9 ],float ))
                    else :
                        K9 [r ["pid"]]=None 
                except Exception as e :
                    print (f"    {r ['pid']}: 9k atlandi ({type (e ).__name__ }: {e })",flush =True )
                    K9 [r ["pid"]]=None 
            data_ =K9 .get (r ["pid"])
            if data_ is not None :
                X9 ,P9a ,Pd9a =data_ 
                k9 =wire_gate .decision_mask (wire_gate .decision_score (gate ,X9 ))
                if k9 .any ():
                    P9 ,Pd9 =son_islem (X9 [k9 ],P9a [k9 ],Pd9a [k9 ])
                    ek_p ,ek_d =[],[]
                    for j in range (len (P9 )):
                        if not len (PB )or np .min (np .linalg .norm (PB -P9 [j ],axis =1 ))>5.0 :
                            ek_p .append (P9 [j ]);ek_d .append (Pd9 [j ])
                    if ek_p :
                        PB =np .vstack ([PB ,np .array (ek_p )])if len (PB )else np .array (ek_p )
                        PdB =np .vstack ([PdB ,np .array (ek_d )])if len (PdB )else np .array (ek_d )
        rj ="very"if r ["n"]>=8 else "low"
        for ad ,(P ,Pd )in (("A",(PA ,PdA )),("B",(PB ,PdB ))):
            det [ad ].append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob [ad ].append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        grup .append (r ["geo"])
    with open (ONBELLEK ,"wb")as f :
        pickle .dump (K9 ,f )

    ra ,rb =f1_rejim (det ["A"]),f1_rejim (det ["B"])
    qa ,qb =f1_rejim (rob ["A"]),f1_rejim (rob ["B"])
    print (f"\n{'olcu':<22}{'A (urun)':>11}{'B (+9k)':>11}{'difference':>10}")
    for ad ,x ,y in (("detection AGIRLIKLI",f1w (det ["A"]),f1w (det ["B"])),
    ("  detection low-CP",ra ["F1"]["low"],rb ["F1"]["low"]),
    ("  detection very-CP",ra ["F1"]["very"],rb ["F1"]["very"]),
    ("robot AGIRLIKLI",f1w (rob ["A"]),f1w (rob ["B"])),
    ("  robot low-CP",qa ["F1"]["low"],qb ["F1"]["low"]),
    ("  robot very-CP",qa ["F1"]["very"],qb ["F1"]["very"])):
        print (f"{ad :<22}{x :>11.4f}{y :>11.4f}{y -x :>+10.4f}")

    fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (det ["A"],det ["B"])),grup ,fn ,n =4000 )
    d =f1w (det ["B"])-f1w (det ["A"])
    gecti =(d >=0.02 )and lo >0 
    print (f"\ntespit AGIRLIKLI diff {d :+.4f}  GA[{lo :+.4f},{hi :+.4f}]"
    f"  -> {'GECTI'if gecti else 'GECMEDI'} (bar +0.02 VE GA>0)")
    # 3 YANLIS YONLENDIRILEN parcanin bedeli
    yy =set (W4 ["yanlis_yonlendirilen"])
    idx =[i for i ,r in enumerate (DER )if r ["pid"]in yy ]
    if idx :
        ba =f1w ([det ["A"][i ]for i in idx ]);bb =f1w ([det ["B"][i ]for i in idx ])
        print (f"wrong yonlendirilen {len (idx )} dusuk-CP part: {ba :.4f} -> {bb :.4f} "
        f"({bb -ba :+.4f})  <- s9 bunu HIC olcmemisti")
    with io .open (OUT ,"w",encoding ="utf-8")as f :
        json .dump ({"tespit_A":f1w (det ["A"]),"tespit_B":f1w (det ["B"]),"difference":d ,
        "ga":[lo ,hi ],"gecti":bool (gecti ),
        "rejim_A":ra ["F1"],"rejim_B":rb ["F1"],
        "robot_A":f1w (rob ["A"]),"robot_B":f1w (rob ["B"]),
        "yonlendirilen":len (yonlendirilen ),
        "not":("Kol ROUTER karariyla uygulandi (GT rejimi DEGIL) and puanlama TUM "
        "194 parcada yapildi; genel etki agirlikla prediction edilmedi.")},f ,indent =1 )
    print (f"receipt -> {OUT }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
