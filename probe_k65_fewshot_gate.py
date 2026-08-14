"""K6.5-a: FEW-SHOT egrisi, GATE katmani (CPU; segmentasyon fine-tune AYRI).

Senaryo: gorulmemis a markadan k part ELLE etiketlenir (CP noktalari verilir).
Bu k parcanin adaylari gate egitimine eklenir, AYNI markanin KALAN parcalarinda
olculur. k=0 = bugunku sifir-atis durumu.

Bu a ALT SINIRDIR: only gate adapte oluyor, segmentasyon not. Gorulmemis
markada FN'lerin %76.2'si ADAY_YOK (temsil) oldugu for asil kaldirac seg
fine-tune'dur; this probe gate'in TEK BASINA ne up to tasidigini olcer.

Protokol notlari:
  - Ayni part hem adaptasyona hem olcume GIRMEZ.
  - Her k for R different cekilis; mean +- std raporlanir (single cekilis gurultulu).
  - k=0 single times (cekilis absent).
  - Olcum urunun TAM zincirini runs (product_zinciri.tam_poz).
"""
import os 
import sys 
import glob 
import json 
import pickle 
import collections 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")

import d6_record # noqa: E402
import wire_gate # noqa: E402
import product_zinciri # noqa: E402
from p1c_threshold import maske # noqa: E402
from sina_cluster import match_hungarian ,f1w # noqa: E402
from korpus_kimlik import step_kimlik as SK # noqa: E402
from sklearn .ensemble import RandomForestClassifier # noqa: E402

MARKALAR =["SUPU","UPUN","NIT"]
KLAR =[0 ,1 ,3 ,5 ]
CEKILIS =3 
ROBOT_YANAL ,ROBOT_ACI =2.0 ,10.0 
OB ="results/_p1_olasilik"
CIKTI ="results/k65_fewshot_gate.json"


def candidate_label (P ,G ,diag ):
    """Aday TESPIT toleransi inside a GT'ye dusuyor mu."""
    if not len (P )or not len (G ):
        return np .zeros (len (P ),int )
    tol =max (3.0 ,0.06 *diag )
    d =np .linalg .norm (P [:,None ,:]-G [None ,:,:],axis =2 )
    return (d .min (axis =1 )<=tol ).astype (int )


def donustur (X ,pids ,donusum ):
    """URUNUN training donusumu: within_part PARCA PARCA uygulanir.

    parca_ici_dagit.py with AYNI loop. Toptan uygulanirsa z-skor corpus geneli
    becomes and calisma anindaki (part-ici) anlamini KAYBEDER -- olculen sey urunun
    yaptigi sey olmaz.
    """
    X =np .asarray (X ,float )
    M =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pids ):
        i =np .where (pids ==u )[0 ]
        M [i ]=wire_gate .within_part (X [i ],donusum )
    return M 


def main ():
# Urunun gate'inden donusumu and genisligi AL (elle sabitlemek deviation kaynagi)
    with open ("results/wire_gate_v5.pkl","rb")as f :
        urun_gate =pickle .load (f )
    DONUSUM =urun_gate ["donusum"]
    N_FEAT =urun_gate ["n_feat"]
    print (f"urun gate: donusum={DONUSUM } n_feat={N_FEAT }")

    baseline =np .load ("results/zengin_parite_v3.npz",allow_pickle =True )
    Xb_ham =np .hstack ([baseline ["X22"],baseline ["XR"]])
    yb =baseline ["y"].astype (int )
    pb_ids =np .array ([str (p )for p in baseline ["pids"]])
    Xb =donustur (Xb_ham ,pb_ids ,DONUSUM )
    assert Xb .shape [1 ]==N_FEAT ,(Xb .shape [1 ],N_FEAT )
    print (f"baseline gate korpusu: {Xb .shape } | pozitif %{100 *yb .mean ():.1f} | "
    f"{len (np .unique (pb_ids ))} part")

    sv =d6_record .exam ()
    kayit =d6_record .yukle (set (sv ["pidler"]))
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}

    sonuc ={}
    for M in MARKALAR :
        pidler =sorted (p for p ,r in kayit .items ()
        if r ["mfg"]==M and os .path .exists (f"{OB }/{p }.npz"))
        if len (pidler )<max (KLAR )+10 :
            print (f"{M }: {len (pidler )} part, ATLANDI")
            continue 
        print (f"\n=== {M }: {len (pidler )} part ===",flush =True )

        # --- mesh + kayit onbellegi (this brand for a times)
        ob ={}
        for p in pidler :
            d =np .load (f"{OB }/{p }.npz")
            ob [p ]=(np .ascontiguousarray (d ["V"],np .float64 ),
            np .ascontiguousarray (d ["F"],np .int64 ),
            np .asarray (d ["pbs"],float ).mean (0 ))

        rng =np .random .RandomState (0 )
        sonuc [M ]={}
        for k in KLAR :
            cekilisler =1 if k ==0 else CEKILIS 
            tf ,rf =[],[]
            for c in range (cekilisler ):
                if k ==0 :
                    adapt ,olc =[],pidler 
                else :
                    adapt =list (rng .choice (pidler ,k ,replace =False ))
                    olc =[p for p in pidler if p not in adapt ]

                    # --- gate egitimi: baseline + adaptasyon parcalarinin adaylari
                X ,y =Xb ,yb 
                if adapt :
                    Xa ,ya =[],[]
                    for p in adapt :
                        r =kayit [p ]
                        Mx =d6_record .x58 (r )
                        if Mx is None or r .get ("P")is None or not len (r ["P"]):
                            continue 
                            # adaptasyon parcasi da PARCA PARCA donusur
                        Xa .append (wire_gate .within_part (np .asarray (Mx ,float ),
                        DONUSUM ))
                        ya .append (candidate_label (np .asarray (r ["P"],float ),
                        np .asarray (r ["G"],float ),r ["diag"]))
                    if Xa :
                        X =np .vstack ([Xb ]+Xa )
                        y =np .concatenate ([yb ]+ya )
                _clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,
                n_jobs =-1 ,random_state =c )
                _clf .fit (X ,y )
                g ={"clf":_clf ,"n_feat":N_FEAT ,"donusum":DONUSUM ,
                "cols":None ,"feat_names":None ,
                "topo_r":urun_gate .get ("topo_r")}

                # --- measurement: TAM zincir
                T ,R =[],[]
                for p in olc :
                    r =kayit [p ]
                    Mx =d6_record .x58 (r )
                    if Mx is None or r .get ("P")is None or not len (r ["P"]):
                        continue 
                    sk =np .asarray (wire_gate .decision_score (g ,Mx ),float )
                    kk =maske (sk ,0.40 ,0.30 )
                    if not kk .any ():
                        continue 
                    P =np .asarray (r ["P"],float )[kk ]
                    D =np .asarray (r ["Pd"],float )[kk ]
                    V ,F ,pb =ob [p ]
                    P ,D =product_zinciri .tam_poz (V ,F ,pb ,P ,D ,step_path =S .get (p ))
                    G =np .asarray (r ["G"],float )
                    Gd =np .asarray (r ["Gd"],float )
                    if not len (G ):
                        continue 
                    T .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],
                    0.0 ,180.0 ,True )[:3 ])
                    R .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],
                    ROBOT_YANAL ,ROBOT_ACI ,
                    False ,signed =True )[:3 ])
                if T :
                    tf .append (f1w (T ))
                    rf .append (f1w (R ))
                del g 
            if tf :
                sonuc [M ][k ]={"tespit":float (np .mean (tf )),
                "tespit_std":float (np .std (tf )),
                "robot":float (np .mean (rf )),
                "robot_std":float (np .std (rf )),
                "n_olc":len (olc ),"cekilis":cekilisler }
                s =sonuc [M ][k ]
                print (f"  k={k }: tespit {s ['tespit']:.4f}+-{s ['tespit_std']:.4f} | "
                f"robot {s ['robot']:.4f}+-{s ['robot_std']:.4f} "
                f"(measurement {s ['n_olc']} part)",flush =True )
        del ob 

    print ("\n=== OZET: k part etiketlemenin robot F1 kazanci ===")
    for M ,d in sonuc .items ():
        if 0 in d :
            t0 =d [0 ]["robot"]
            art ="  ".join (f"k={k }:{d [k ]['robot']:+.4f}"
            for k in KLAR if k in d and k >0 )
            print (f"  {M }: k=0 baseline {t0 :.4f}  ->  "+
            "  ".join (f"k={k }:{d [k ]['robot']:.4f}"
            for k in KLAR if k in d and k >0 ))
            print (f"       fark: "+"  ".join (
            f"k={k }:{d [k ]['robot']-t0 :+.4f}"for k in KLAR if k in d and k >0 ))
    with open (CIKTI ,"w")as f :
        json .dump ({"sonuc":sonuc ,"not":"GATE-only few-shot; seg fine-tune AYRI. "
        "Alt sinirdir.","markalar":MARKALAR ,"klar":KLAR ,
        "cekilis":CEKILIS },f ,indent =1 )
    print (f"\nmakbuz -> {CIKTI }")


if __name__ =="__main__":
    main ()
