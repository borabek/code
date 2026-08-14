# -*- coding: utf-8 -*-
"""CC-D: min_v10 post-islemesinin etkisi REJIM AYRIMLI (low-CP vs very-CP).

Onceki measurement (+0.021) agirlikli as DUSUK-CP parcalarindan geliyordu; CC-A and CC-B, asil
engelin very-CP rejiminde oldugunu showed. Bu measurement ikisini AYIRIR.

DURUSTLUK: aile-disi GroupKFold(5) + IC-ICE threshold secimi (threshold test katindan ogrenilmez),
`seen` bayrakli parts skorlanmaz, kilitli holdout'a DOKUNULMAZ, two arm AYNI part kumesinde.
"""
import json ,sys 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 

ARMS ={
"min_v30 (mevcut urun)":["results/rich_feats.npz","results/rich_ds3.npz","results/rich_new863.npz"],
"min_v10 (gevsek)":["results/rich_mv10_all.npz"],
"min_v10 + promote0.25":["results/rich_promote.npz"],
}
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )
HIGH_CP =8 


def load (paths ):
    X ,Y ,PID ,SEEN ,NGT =[],[],[],{},{}
    got =set ()
    for p in paths :
        r =np .load (p ,allow_pickle =True )
        rp =[str (v )for v in r ["part_ids"]]
        gp =np .array ([rp [int (g )]for g in r ["groups"]])
        keep =np .array ([q not in got for q in gp ])
        got |=set (gp [keep ])
        X .append (np .hstack ([r ["X13"],r ["XR"][:,0 :33 ]])[keep ])
        Y .append (r ["y"][keep ]);PID .append (gp [keep ])
        for i ,q in enumerate (rp ):
            SEEN .setdefault (q ,int (r ["seen"][i ]))
        ngt ={int (g ):int (n )for g ,n in zip (r ["grp_ids"],r ["ngt"])}
        for g in np .unique (r ["groups"][keep ]):
            NGT [rp [int (g )]]=ngt [int (g )]
    return np .vstack (X ),np .concatenate (Y ),np .concatenate (PID ),SEEN ,NGT 


def f1_of (tp ,nk ,gt ):
    p =tp /max (nk ,1 );r =tp /max (gt ,1 )
    return 2 *p *r /max (p +r ,1e-9 )


def evaluate (X ,Y ,PID ,NGT ,G ,score_mask ,seeds =(0 ,1 ,2 )):
    """Aile-disi OOF + ic-ice threshold. Doner: (F1 ortalamasi, sd)."""
    outs =[]
    for s in seeds :
        oof =np .zeros (len (Y ))
        for tr ,te in GroupKFold (5 ).split (X ,Y ,G ):
            oof [te ]=RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =s ).fit (X [tr ],Y [tr ]).predict_proba (X [te ])[:,1 ]
        uf =np .unique (G [score_mask ])
        rs =np .random .RandomState (0 );uf =uf [rs .permutation (len (uf ))]
        tp =nk =gt =0 
        for fo in np .array_split (uf ,5 ):
            te =score_mask &np .isin (G ,fo )
            tr =score_mask &~np .isin (G ,fo )
            gt_tr =sum (NGT .get (q ,0 )for q in set (PID [tr ]))
            bt ,bf =THRS [0 ],-1.0 
            for t in THRS :
                k =tr &(oof >=t )
                f =f1_of (int (Y [k ].sum ()),int (k .sum ()),gt_tr )
                if f >bf :bf ,bt =f ,t 
            k =te &(oof >=bt )
            tp +=int (Y [k ].sum ());nk +=int (k .sum ())
            gt +=sum (NGT .get (q ,0 )for q in set (PID [te ]))
        outs .append (f1_of (tp ,nk ,gt ))
    return float (np .mean (outs )),float (np .std (outs ))


def main ():
    lock =json .load (open ("results/split_lock.json"));LOCK =set (lock ["locked_parts"])
    fams ={k :v .get ("family",f"nr:{k }")for k ,v in lock ["parts"].items ()}

    data ={k :load (v )for k ,v in ARMS .items ()}
    # ORTAK, TEMIZ, kilitli-olmayan parts -- two arm AYNI kumede olculur
    common =None 
    for k ,(X ,Y ,PID ,SEEN ,NGT )in data .items ():
        s ={q for q in set (PID )if SEEN .get (q ,1 )==0 and q not in LOCK }
        common =s if common is None else (common &s )
    ngt_ref =data [list (ARMS )[0 ]][4 ]
    hi ={q for q in common if ngt_ref .get (q ,0 )>=HIGH_CP }
    lo =common -hi 
    print (f"ortak skor kumesi {len (common )} temiz part  ->  dusuk-CP {len (lo )} | cok-CP {len (hi )}\n")

    print (f"{'arm':<24}{'DUSUK-CP':>12}{'COK-CP':>12}{'HEPSI':>12}")
    res ={}
    for label ,paths in ARMS .items ():
        X ,Y ,PID ,SEEN ,NGT =data [label ]
        m =np .array ([q not in LOCK for q in PID ])
        X ,Y ,PID =X [m ],Y [m ],PID [m ]
        fam =np .array ([fams .get (q ,f"nr:{q }")for q in PID ])
        fid ={v :i for i ,v in enumerate (sorted (set (fam )))}
        G =np .array ([fid [v ]for v in fam ])
        row ={}
        for tag ,ps in [("dusuk",lo ),("cok",hi ),("hepsi",common )]:
            mask =np .array ([q in ps for q in PID ])
            if mask .sum ()<30 :
                row [tag ]=(float ("nan"),0.0 );continue 
            row [tag ]=evaluate (X ,Y ,PID ,NGT ,G ,mask )
        res [label ]=row 
        print (f"{label :<24}{row ['dusuk'][0 ]:>12.4f}{row ['cok'][0 ]:>12.4f}{row ['hepsi'][0 ]:>12.4f}")

    a ,b =list (ARMS )
    print (f"\n{'FARK (min_v10 - min_v30)':<24}"
    f"{res [b ]['dusuk'][0 ]-res [a ]['dusuk'][0 ]:>+12.4f}"
    f"{res [b ]['cok'][0 ]-res [a ]['cok'][0 ]:>+12.4f}"
    f"{res [b ]['hepsi'][0 ]-res [a ]['hepsi'][0 ]:>+12.4f}")
    print (f"{'(seed sd)':<24}{res [b ]['dusuk'][1 ]:>12.4f}{res [b ]['cok'][1 ]:>12.4f}{res [b ]['hepsi'][1 ]:>12.4f}")

    w =len (hi )/max (len (common ),1 )
    print (f"\ncok-CP orani skor kumesinde %{100 *w :.1f}")
    json .dump ({k :{t :list (v )for t ,v in r .items ()}for k ,r in res .items ()},
    open ("results/cc_d_minv_rejim.json","w"),indent =1 )
    print ("receipt -> results/cc_d_minv_rejim.json")


if __name__ =="__main__":
    main ()
