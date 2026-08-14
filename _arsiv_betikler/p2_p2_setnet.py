# -*- coding: utf-8 -*-
"""FAZ 2 / P2 -- PART-LEVEL SET SECICI + LISTWISE RANKING LOSS.
Bugune kadarki each gate BAGIMSIZ ikili siniflandirmaydi: each candidate single basina puanlandi.
Oysa karar PARCA ICINDE goreceli: "this parcadaki 17 adayin hangileri tel girisi?" Set-transformer
candidates arasi self-attention kurar (order/pitch/ayna/komsuluk iliskisi ORTUK ogrenilir) and
LISTWISE loss dogrudan part-ici siralamayi optimize eder (top-N modunun full hedefi).
Modlar: count-BILINMEYEN (threshold) and count-BILINEN (top-N) AYRI egitilir/olculur.
GO (kullanici): family-out'ta P1 sonucunun uzerine >= +0.010 and 3 seed'de kararli.
Kullanim: python p2_p2_setnet.py [--emb]"""
import sys ,json ,math 
import numpy as np ,torch ,torch .nn as nn 
from sklearn .model_selection import GroupKFold 
from sklearn .ensemble import RandomForestClassifier 

USE_EMB ="--emb"in sys .argv 
dev ="cuda"if torch .cuda .is_available ()else "cpu"
r =np .load ("results/rich_feats.npz",allow_pickle =True )
lock =json .load (open ("results/split_lock.json"))
LOCKED =set (lock ["locked_parts"]);META =lock ["parts"]
X13 ,XR ,Y ,G ,MF =r ["X13"],r ["XR"],r ["y"],r ["groups"],r ["mfg"]
RICH =np .hstack ([X13 ,XR [:,0 :33 ]])
rp =[str (x )for x in r ["part_ids"]]
gpid =np .array ([rp [int (g )]for g in G ]);gseen =np .array ([int (r ["seen"][int (g )])for g in G ])
ngt_of =dict (zip (r ["grp_ids"].tolist (),r ["ngt"].tolist ()))
if USE_EMB :
    e =np .load ("results/embed_feats.npz",allow_pickle =True )
    ep =[str (x )for x in e ["part_ids"]];emap ={p :i for i ,p in enumerate (ep )}
    EM =np .zeros ((len (Y ),e ["EMB"].shape [1 ]))
    ok =np .zeros (len (Y ),bool )
    for gi_r ,pid in enumerate (rp ):
        if pid not in emap :continue 
        ir =np .where (G ==gi_r )[0 ];ie =np .where (e ["groups"]==emap [pid ])[0 ]
        if len (ir )!=len (ie ):continue 
        EM [ir ]=e ["EMB"][ie ];ok [ir ]=True 
    RICH =np .hstack ([RICH ,EM ])
    print (f"embedding dahil ({EM .shape [1 ]}d), eslesen candidate {int (ok .sum ())}/{len (Y )}")

WORK =(gseen ==0 )&np .array ([p not in LOCKED for p in gpid ])
fam =np .array ([META .get (p ,{}).get ("family",f"nr:{p }")for p in gpid ])
FG =np .array ([{v :i for i ,v in enumerate (sorted (set (fam )))}[v ]for v in fam ])
THRS =np .round (np .arange (0.10 ,0.71 ,0.02 ),3 )
masks ={"ALL":WORK ,"WEI":WORK &(MF ==1 ),"PXC":WORK &(MF ==0 )}
mu ,sd =RICH [WORK ].mean (0 ),RICH [WORK ].std (0 )+1e-6 
XN =(RICH -mu )/sd 
print (f"WORK {int (WORK .sum ())} candidate / {len (set (gpid [WORK ]))} part | feat {XN .shape [1 ]}d | dev {dev }")


class SetNet (nn .Module ):
    """DeepSets + self-attention: candidate-ici temsil + part-ici baglam -> skor."""
    def __init__ (self ,d ,h =128 ,nl =2 ):
        super ().__init__ ()
        self .inp =nn .Sequential (nn .Linear (d ,h ),nn .ReLU (),nn .Linear (h ,h ))
        layer =nn .TransformerEncoderLayer (h ,4 ,h *2 ,dropout =0.1 ,batch_first =True ,norm_first =True )
        self .enc =nn .TransformerEncoder (layer ,nl )
        self .out =nn .Sequential (nn .ReLU (),nn .Linear (h ,1 ))

    def forward (self ,x ):# x: (1, n, d) single part
        z =self .inp (x );z =self .enc (z );return self .out (z ).squeeze (-1 )


def train_setnet (idx_tr ,seed ,listwise ,epochs =60 ):
    torch .manual_seed (seed );np .random .seed (seed )
    net =SetNet (XN .shape [1 ]).to (dev )
    opt =torch .optim .AdamW (net .parameters (),lr =1e-3 ,weight_decay =1e-4 )
    groups =[np .where ((G ==g )&WORK )[0 ]for g in np .unique (G [idx_tr ])]
    groups =[g for g in groups if len (g )>=2 and Y [g ].sum ()>0 ]
    bce =nn .BCEWithLogitsLoss ()
    for ep in range (epochs ):
        np .random .shuffle (groups );net .train ()
        for gi in groups :
            x =torch .tensor (XN [gi ],dtype =torch .float32 ,device =dev ).unsqueeze (0 )
            yy =torch .tensor (Y [gi ],dtype =torch .float32 ,device =dev )
            s =net (x ).squeeze (0 )
            if listwise :# LISTWISE: part-ici softmax capraz-entropi
                tgt =yy /yy .sum ()
                loss =-(tgt *torch .log_softmax (s ,0 )).sum ()
            else :
                loss =bce (s ,yy )
            opt .zero_grad ();loss .backward ();opt .step ()
    return net 


@torch .no_grad ()
def predict (net ,idx_te ):
    net .eval ();out =np .zeros (len (Y ))
    for g in np .unique (G [idx_te ]):
        gi =np .where ((G ==g )&WORK )[0 ]
        if not len (gi ):continue 
        x =torch .tensor (XN [gi ],dtype =torch .float32 ,device =dev ).unsqueeze (0 )
        out [gi ]=torch .sigmoid (net (x ).squeeze (0 )).cpu ().numpy ()
    return out 


def prf (tp ,nk ,gt ):
    p =tp /max (nk ,1 );rr =tp /max (gt ,1 );return 2 *p *rr /max (p +rr ,1e-9 )
def gt_of (m ):return int (sum (ngt_of [int (g )]for g in np .unique (G [m ])))
def sc_at (sc ,m ,t ):
    k =m &(sc >=t );return prf (int (Y [k ].sum ()),int (k .sum ()),gt_of (m ))


def nested (sc ,m ,groups ,K =5 ):
    gp =np .unique (groups [np .where (m )[0 ]]);rs =np .random .RandomState (0 )
    gp =gp [rs .permutation (len (gp ))];TP =NK =GT =0 
    for f in np .array_split (gp ,K ):
        tg =set (f .tolist ())
        trm =m &np .array ([g not in tg for g in groups ]);tem =m &np .array ([g in tg for g in groups ])
        if not trm .any ()or not tem .any ():continue 
        bt ,bf =0.35 ,-1 
        for t in THRS :
            ff =sc_at (sc ,trm ,t )
            if ff >bf :bf ,bt =ff ,t 
        k =tem &(sc >=bt );TP +=int (Y [k ].sum ());NK +=int (k .sum ());GT +=gt_of (tem )
    return prf (TP ,NK ,GT )


def topn (sc ,m ):
    TP =NK =GT =0 
    for g in np .unique (G [m ]):
        i =np .where (m &(G ==g ))[0 ];n =ngt_of [int (g )]
        j =i [np .argsort (-sc [i ])[:n ]];TP +=int (Y [j ].sum ());NK +=len (j );GT +=n 
    return prf (TP ,NK ,GT )


def oof_rf (groups ):
    o =np .zeros (len (Y ));idx =np .where (WORK )[0 ]
    for tr ,te in GroupKFold (5 ).split (RICH [idx ],Y [idx ],groups [idx ]):
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 )
        c .fit (RICH [idx ][tr ],Y [idx ][tr ]);o [idx [te ]]=c .predict_proba (RICH [idx ][te ])[:,1 ]
    return o 


def oof_setnet (groups ,seed ,listwise ):
    o =np .zeros (len (Y ));idx =np .where (WORK )[0 ]
    for tr ,te in GroupKFold (5 ).split (RICH [idx ],Y [idx ],groups [idx ]):
        net =train_setnet (idx [tr ],seed ,listwise )
        o +=predict (net ,idx [te ])*np .isin (np .arange (len (Y )),idx [te ])
    return o 


print ("\n================ family-out (KARAR SPLIT'I) ================")
base =oof_rf (FG )
print (f"{'model':22s} {'ALL':>7s} {'WEI':>7s} {'PXC':>7s} {'topN':>7s}")
print (f"{'RF (baseline)':22s} {nested (base ,masks ['ALL'],FG ):7.4f} {nested (base ,masks ['WEI'],FG ):7.4f} "
f"{nested (base ,masks ['PXC'],FG ):7.4f} {topn (base ,WORK ):7.4f}",flush =True )
res ={"rf_base":{"ALL":nested (base ,masks ["ALL"],FG ),"topN":topn (base ,WORK )}}
for lw in (False ,True ):
    tag ="listwise"if lw else "BCE"
    alls ,tns =[],[]
    for seed in (0 ,1 ,2 ):
        sc =oof_setnet (FG ,seed ,lw )
        a ,tn =nested (sc ,masks ["ALL"],FG ),topn (sc ,WORK )
        alls .append (a );tns .append (tn )
        print (f"{'SetNet-'+tag +f' s{seed }':22s} {a :7.4f} {nested (sc ,masks ['WEI'],FG ):7.4f} "
        f"{nested (sc ,masks ['PXC'],FG ):7.4f} {tn :7.4f}",flush =True )
    res [f"setnet_{tag }"]={"ALL_mean":float (np .mean (alls )),"ALL_std":float (np .std (alls )),
    "topN_mean":float (np .mean (tns )),"topN_std":float (np .std (tns ))}
    print (f"  -> {tag }: ALL {np .mean (alls ):.4f}+-{np .std (alls ):.4f} | topN {np .mean (tns ):.4f}+-{np .std (tns ):.4f}")

b =res ["rf_base"]["ALL"]
print (f"\n=== GO/NO-GO (family-out'ta RF tabanina gore >= +0.010 VE 3 seed kararli) ===")
for k in ("setnet_BCE","setnet_listwise"):
    dA =res [k ]["ALL_mean"]-b ;dT =res [k ]["topN_mean"]-res ["rf_base"]["topN"]
    stable =res [k ]["ALL_std"]<0.010 
    print (f"  {k :16s}: dALL {dA :+.4f} (std {res [k ]['ALL_std']:.4f}) | dtopN {dT :+.4f} -> "
    f"{'GO'if (dA >=0.010 and stable )else 'no'}")
json .dump (res ,open ("results/p2_setnet_eval.json","w"),indent =1 )
print ("-> results/p2_setnet_eval.json")
