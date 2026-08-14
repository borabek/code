# -*- coding: utf-8 -*-
"""K: gate hangi ozellikle OGRENIYOR, hangisiyle EZBERLIYOR?

FINDING (2026-07-31): geometrik ikizler bolmeden cikarilinca gate precision'i 0.917 -> 0.630
dustu. Bu, gate'in "tel mi alet mi" sorusunu ogrenmek instead of large olcude PARCA GEOMETRISINI
ezberledigini gosterir: ikiz egitimde oldugu surece test parcasinin cevabini already biliyor.

OLCUM: each feature for, that feature TEK BASINA ne up to tasiyor -- and ezber bolmesi with durust
split arasindaki farki. Fark buyukse that feature EZBERLETIYOR demektir.
Ayrica: modeli KUCULTMEK (more sig agac) durust bolmede yardim eder mi?
"""
import os ,sys ,json 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def main ():
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .linear_model import LogisticRegression 
    from sklearn .model_selection import GroupKFold ,cross_val_score 
    from sklearn .preprocessing import StandardScaler 
    from sklearn .pipeline import make_pipeline 
    import wire_gate 

    d =np .load ("results/gate_regrow_data_rt2.npz",allow_pickle =True )
    X =d ["X"];y =d ["y"];fams =d ["fams"].astype (str )
    pids =np .array ([str (x )for x in d ["pids"]])
    gk =json .load (open ("results/_geometry_keys.json"))
    Gg =np .array ([gk .get (p ,"yok:"+p )for p in pids ])
    names =wire_gate .FEAT_NAMES_13 
    print (f"{len (y )} candidate, {X .shape [1 ]} ozellik, TP orani %{100 *y .mean ():.1f}\n",flush =True )

    def auc (o ):
    # Mann-Whitney (bag duzeltmeli) -- kisa-path AUC dengesiz veride SISIYOR (auc-formula-bug)
        from scipy .stats import rankdata 
        r =rankdata (o );n1 =int (y .sum ());n0 =len (y )-n1 
        if n1 ==0 or n0 ==0 :return 0.5 
        return (r [y ==1 ].sum ()-n1 *(n1 +1 )/2 )/(n1 *n0 )

    def oof (M ,groups ,model =None ):
        o =np .zeros (len (y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (M ,y ,groups ):
            m =model ()if model else RandomForestClassifier (
            n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 )
            o [te ]=m .fit (M [tr ],y [tr ]).predict_proba (M [te ])[:,1 ]
        return o 

    print (f"{'ozellik':<12}{'AUC ezber':>11}{'AUC durust':>12}{'DUSUS':>9}")
    rows =[]
    for i ,nm in enumerate (names ):
        a_f =auc (oof (X [:,[i ]],fams ))
        a_g =auc (oof (X [:,[i ]],Gg ))
        rows .append ((nm ,a_f ,a_g ,a_f -a_g ))
    for nm ,a ,b ,dd in sorted (rows ,key =lambda r :-r [3 ]):
        flag ="  <- EZBERLETIYOR"if dd >0.03 else ("  genellesiyor"if dd <0.01 else "")
        print (f"{nm :<12}{a :>11.3f}{b :>12.3f}{dd :>9.3f}{flag }")

    print (f"\n{'model':<34}{'AUC ezber':>11}{'AUC durust':>12}{'DUSUS':>9}")
    cfgs =[("RF 400/leaf3 (mevcut)",lambda :RandomForestClassifier (
    n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 )),
    ("RF 400/leaf30 (sig)",lambda :RandomForestClassifier (
    n_estimators =400 ,min_samples_leaf =30 ,n_jobs =-1 ,random_state =0 )),
    ("RF 400/depth6",lambda :RandomForestClassifier (
    n_estimators =400 ,max_depth =6 ,n_jobs =-1 ,random_state =0 )),
    ("Lojistik regresyon",lambda :make_pipeline (
    StandardScaler (),LogisticRegression (max_iter =2000 )))]
    for lab ,mk in cfgs :
        a_f =auc (oof (X ,fams ,mk ));a_g =auc (oof (X ,Gg ,mk ))
        print (f"{lab :<34}{a_f :>11.3f}{a_g :>12.3f}{a_f -a_g :>9.3f}",flush =True )
    print ("\nYORUM: DUSUS kucuk olan model/ozellik GENELLESIYOR; buyuk olan EZBERLIYOR.")


if __name__ =="__main__":
    main ()
