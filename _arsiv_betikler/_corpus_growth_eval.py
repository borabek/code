# -*- coding: utf-8 -*-
"""VERI KALDIRACI OLCUMU: buyutulmus corpus CP F1'i yukseltiyor mu?

DURUSTLUK KURALLARI (see cp_config.locked_holdout_policy_2026_07_28):
  * Kilitli holdout parcalari TAMAMEN disarida -- this measurement ona DOKUNMAZ.
  * `seen` bayrakli parts (etiketleri segmentasyon egitiminde gorulmus) EGITIMDE kullanilabilir
    but SKORLANMAZ; otherwise leakage F1'i sisirir.
  * Aile-disi (GroupKFold family) + ic-ice CV threshold secimi: threshold test katindan OGRENILMEZ.
  * Iki arm AYNI protokolde, AYNI skorlanan part kumesinde olculur -- single degisen EGITIM havuzu.
    (Aksi halde 'more very data' with 'more easy test' karisir.)
"""
import json ,sys 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 

BASE =["results/rich_feats.npz","results/rich_ds3.npz"]
NEW =["results/rich_new863.npz"]
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )


def load (paths ):
    """Birden very npz'yi birlestir; part kimligi with TEKILLESTIR (same part two dosyada may be)."""
    X ,Y ,PID ,MFG ,SEEN ,NGT =[],[],[],[],{},{}
    seen_pid =set ()
    for p in paths :
        try :
            r =np .load (p ,allow_pickle =True )
        except FileNotFoundError :
            print (f"  [yok] {p }");continue 
        rp =[str (v )for v in r ["part_ids"]]
        rs ={rp [i ]:int (r ["seen"][i ])for i in range (len (rp ))}
        ngt ={int (g ):int (n )for g ,n in zip (r ["grp_ids"],r ["ngt"])}
        feats =np .hstack ([r ["X13"],r ["XR"][:,0 :33 ]])
        gp =np .array ([rp [int (g )]for g in r ["groups"]])
        keep =np .array ([q not in seen_pid for q in gp ])
        seen_pid |=set (gp [keep ])
        X .append (feats [keep ]);Y .append (r ["y"][keep ]);PID .append (gp [keep ])
        MFG .append (r ["mfg"][keep ])
        for q in set (gp [keep ]):
            SEEN [q ]=rs [q ]
        for g in np .unique (r ["groups"][keep ]):
            NGT [rp [int (g )]]=ngt [int (g )]
    return (np .vstack (X ),np .concatenate (Y ),np .concatenate (PID ),
    np .concatenate (MFG ),SEEN ,NGT )


def f1_of (tp ,n_keep ,n_gt ):
    p =tp /max (n_keep ,1 );r =tp /max (n_gt ,1 )
    return 2 *p *r /max (p +r ,1e-9 )


def run (paths ,label ,score_pids ,lock ,fams ):
    X ,Y ,PID ,MFG ,SEEN ,NGT =load (paths )
    LOCK =set (lock ["locked_parts"])
    m =np .array ([q not in LOCK for q in PID ])
    X ,Y ,PID ,MFG =X [m ],Y [m ],PID [m ],MFG [m ]
    fam =np .array ([fams .get (q ,f"nr:{q }")for q in PID ])
    fid ={v :i for i ,v in enumerate (sorted (set (fam )))}
    FG =np .array ([fid [v ]for v in fam ])

    oof =np .full (len (Y ),np .nan )
    for tr ,te in GroupKFold (5 ).split (X ,Y ,FG ):
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 )
        c .fit (X [tr ],Y [tr ])
        oof [te ]=c .predict_proba (X [te ])[:,1 ]

        # SKORLAMA: only ortak, temiz (seen==0), kilitli-olmayan parts
    s =np .array ([q in score_pids for q in PID ])

    # IC-ICE ESIK SECIMI: threshold, test katindaki ailelerden OGRENILMEZ. (Skorlanan kumede most iyi esigi
    # secmek each two kolu da sisirir -- more before this hataya dusuldu, see p2 notlari.)
    fam_s =FG [s ]
    uf =np .unique (fam_s )
    rs =np .random .RandomState (0 );uf =uf [rs .permutation (len (uf ))]
    folds =np .array_split (uf ,5 )
    tp_t =nk_t =gt_t =0 
    for f_out in folds :
        te =s &np .isin (FG ,f_out )
        tr =s &~np .isin (FG ,f_out )
        best_t ,best_f =THRS [0 ],-1.0 
        gt_tr =int (sum (NGT .get (q ,0 )for q in set (PID [tr ])))
        for t in THRS :# threshold SADECE ic (train) ailelerde secilir
            k =tr &(oof >=t )
            f =f1_of (int (Y [k ].sum ()),int (k .sum ()),gt_tr )
            if f >best_f :best_f ,best_t =f ,t 
        k =te &(oof >=best_t )# secilen threshold DIS katta uygulanir
        tp_t +=int (Y [k ].sum ());nk_t +=int (k .sum ())
        gt_t +=int (sum (NGT .get (q ,0 )for q in set (PID [te ])))
    f1 =f1_of (tp_t ,nk_t ,gt_t )
    n_train_parts =len (set (PID ))
    print (f"{label :<22} training-part {n_train_parts :>4} | candidate {len (Y ):>5} | "
    f"skorlanan part {len (set (PID [s ])):>3} | F1 {f1 :.4f}  (ic-ice threshold)")
    return f1 ,n_train_parts 


if __name__ =="__main__":
    lock =json .load (open ("results/split_lock.json"))
    fams ={k :v .get ("family",f"nr:{k }")for k ,v in lock ["parts"].items ()}
    LOCK =set (lock ["locked_parts"])

    # ORTAK SKOR KUMESI: two kolda da bulunan, TEMIZ, kilitli-olmayan parts
    _ ,_ ,P0 ,_ ,S0 ,_ =load (BASE )
    _ ,_ ,P1 ,_ ,S1 ,_ =load (BASE +NEW )
    common ={q for q in set (P0 )if S0 .get (q ,1 )==0 and q not in LOCK }
    print (f"ortak skor kumesi: {len (common )} temiz part (iki kolda da ayni)\n")

    a ,na =run (BASE ,"TABAN corpus",common ,lock ,fams )
    b ,nb =run (BASE +NEW ,"BUYUTULMUS corpus",common ,lock ,fams )
    print (f"\nEgitim havuzu {na } -> {nb } part (+{nb -na }, x{nb /max (na ,1 ):.2f})")
    print (f"CP F1 {a :.4f} -> {b :.4f}  ({b -a :+.4f})")
    json .dump ({"base_f1":a ,"grown_f1":b ,"delta":b -a ,
    "base_parts":na ,"grown_parts":nb ,"scored_parts":len (common ),
    "protocol":"family-out GroupKFold(5), seen-excluded from scoring, locked holdout untouched"},
    open ("results/corpus_growth_eval.json","w"),indent =1 )
