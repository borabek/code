# -*- coding: utf-8 -*-
"""FAZ 2 / P7 -- FINAL VERIFICATION. KILITLI HOLDOUT TEK KEZ ACILIR.
Kural zinciri (kullanici listesi):
 * tum model and esikler holdout ACILMADAN kilitlendi -> results/final_config.json
 * holdout SADECE here, TEK KEZ calistirilir (no secim/tuning yapilmaz)
 * ALL / WEI / PXC / high-CP / family-out AYRI raporlanir
 * 3 seed ortalamasi + bootstrap confidence araligi
 * AUTO precision with AUTO+REVIEW F1 AYRI gosterilir
 * BASARI KAPISI: locked holdout ALL CP-F1 >= 0.85
Egitim: SADECE WORK (541 part). Tahmin: 129 kilitli part."""
import json 
import numpy as np ,torch ,torch .nn as nn 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 

dev ="cuda"if torch .cuda .is_available ()else "cpu"
CFG =json .load (open ("results/final_config.json"))
THR =float (CFG ["locked_threshold"])
print (f"KILITLI KONFIGURASYON: {CFG ['winner']} | threshold {THR :.2f} | WORK ALL {CFG ['WORK_ALL']:.4f}")

r =np .load ("results/rich_feats.npz",allow_pickle =True )
e =np .load ("results/embed_feats.npz",allow_pickle =True )
lock =json .load (open ("results/split_lock.json"))
LOCKED =set (lock ["locked_parts"]);META =lock ["parts"]
X13 ,XR ,Y ,G ,MF =r ["X13"],r ["XR"],r ["y"],r ["groups"],r ["mfg"]
rp =[str (x )for x in r ["part_ids"]];ep =[str (x )for x in e ["part_ids"]]
emap ={p :i for i ,p in enumerate (ep )}
EM =np .zeros ((len (Y ),e ["EMB"].shape [1 ]))
for gi ,pid in enumerate (rp ):
    if pid not in emap :continue 
    ir =np .where (G ==gi )[0 ];ie =np .where (e ["groups"]==emap [pid ])[0 ]
    if len (ir )==len (ie ):EM [ir ]=e ["EMB"][ie ]
RICH =np .hstack ([X13 ,XR [:,0 :33 ]]);FULL =np .hstack ([RICH ,EM ])
gpid =np .array ([rp [int (g )]for g in G ]);gseen =np .array ([int (r ["seen"][int (g )])for g in G ])
ngt_of =dict (zip (r ["grp_ids"].tolist (),r ["ngt"].tolist ()))
WORK =(gseen ==0 )&np .array ([p not in LOCKED for p in gpid ])
HOLD =np .array ([p in LOCKED for p in gpid ])# KILITLI HOLDOUT
CONF =X13 [:,12 ]# segmentasyon guveni (AUTO tier for)
mu ,sd =FULL [WORK ].mean (0 ),FULL [WORK ].std (0 )+1e-6 
XN =(FULL -mu )/sd 
print (f"EGITIM (WORK): {int (WORK .sum ())} candidate / {len (set (gpid [WORK ]))} part")
print (f"HOLDOUT (KILITLI): {int (HOLD .sum ())} candidate / {len (set (gpid [HOLD ]))} part "
f"({len (set (gpid [HOLD &(MF ==1 )]))} WEI + {len (set (gpid [HOLD &(MF ==0 )]))} PXC)")


class SetNet (nn .Module ):
    def __init__ (self ,d ,h =128 ,nl =2 ):
        super ().__init__ ()
        self .inp =nn .Sequential (nn .Linear (d ,h ),nn .ReLU (),nn .Linear (h ,h ))
        self .enc =nn .TransformerEncoder (nn .TransformerEncoderLayer (h ,4 ,h *2 ,dropout =0.1 ,
        batch_first =True ,norm_first =True ),nl )
        self .out =nn .Sequential (nn .ReLU (),nn .Linear (h ,1 ))
    def forward (self ,x ):return self .out (self .enc (self .inp (x ))).squeeze (-1 )


def train_setnet (seed ,epochs =60 ):
    torch .manual_seed (seed );np .random .seed (seed )
    net =SetNet (XN .shape [1 ]).to (dev )
    opt =torch .optim .AdamW (net .parameters (),lr =1e-3 ,weight_decay =1e-4 )
    bce =nn .BCEWithLogitsLoss ()
    groups =[np .where ((G ==g )&WORK )[0 ]for g in np .unique (G [WORK ])]
    groups =[g for g in groups if len (g )>=2 and Y [g ].sum ()>0 ]
    for _ in range (epochs ):
        np .random .shuffle (groups );net .train ()
        for gi in groups :
            x =torch .tensor (XN [gi ],dtype =torch .float32 ,device =dev ).unsqueeze (0 )
            yy =torch .tensor (Y [gi ],dtype =torch .float32 ,device =dev )
            loss =bce (net (x ).squeeze (0 ),yy )
            opt .zero_grad ();loss .backward ();opt .step ()
    return net 


@torch .no_grad ()
def pred_setnet (net ,mask ):
    net .eval ();o =np .zeros (len (Y ))
    for g in np .unique (G [mask ]):
        gi =np .where ((G ==g )&mask )[0 ]
        if not len (gi ):continue 
        x =torch .tensor (XN [gi ],dtype =torch .float32 ,device =dev ).unsqueeze (0 )
        o [gi ]=torch .sigmoid (net (x ).squeeze (0 )).cpu ().numpy ()
    return o 


def prf (tp ,nk ,gt ):
    p =tp /max (nk ,1 );rr =tp /max (gt ,1 );return p ,rr ,2 *p *rr /max (p +rr ,1e-9 )
def gt_of (m ):return int (sum (ngt_of [int (g )]for g in np .unique (G [m ])))


# ---- 3 seed: each seed for RF + SetNet egit, probability ortalamasi ----
print ("\nEGITILIYOR (3 seed, SADECE WORK)...",flush =True )
seed_scores =[]
for seed in (0 ,1 ,2 ):
    rf =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =seed )
    rf .fit (RICH [WORK ],Y [WORK ])
    s_rf =np .zeros (len (Y ));s_rf [HOLD ]=rf .predict_proba (RICH [HOLD ])[:,1 ]
    net =train_setnet (seed )
    s_set =pred_setnet (net ,HOLD )
    seed_scores .append ((s_rf +s_set )/2 )
    print (f"  seed {seed } tamam",flush =True )

hi_cp =np .array ([ngt_of [int (g )]>=11 for g in G ])
subsets ={"ALL":HOLD ,"WEI":HOLD &(MF ==1 ),"PXC":HOLD &(MF ==0 ),"high-CP(>=11)":HOLD &hi_cp }

print (f"\n================ KILITLI HOLDOUT SONUCLARI (threshold {THR :.2f}, TEK KEZ) ================")
print (f"{'altkume':16s} {'part':>6s} {'GT':>5s} {'P':>7s} {'R':>7s} {'F1 (3 seed ort)':>18s}")
final ={}
for nm ,m in subsets .items ():
    if not m .any ():print (f"{nm :16s} (none)");continue 
    f1s ,ps ,rs =[],[],[]
    for sc in seed_scores :
        k =m &(sc >=THR )
        p ,rr ,f =prf (int (Y [k ].sum ()),int (k .sum ()),gt_of (m ))
        f1s .append (f );ps .append (p );rs .append (rr )
    final [nm ]={"F1_mean":float (np .mean (f1s )),"F1_std":float (np .std (f1s )),
    "P_mean":float (np .mean (ps )),"R_mean":float (np .mean (rs )),
    "n_parts":len (set (gpid [m ])),"n_gt":gt_of (m )}
    print (f"{nm :16s} {len (set (gpid [m ])):6d} {gt_of (m ):5d} {np .mean (ps ):7.4f} {np .mean (rs ):7.4f} "
    f"{np .mean (f1s ):11.4f} +-{np .std (f1s ):.4f}")

    # ---- bootstrap CI (part bazli yeniden ornekleme) ----
sc =np .mean (seed_scores ,0 )
pids =np .array (sorted (set (gpid [HOLD ])));rs_ =np .random .RandomState (0 )
boot =[]
for _ in range (2000 ):
    samp =rs_ .choice (pids ,len (pids ),replace =True )
    tp =nk =gt =0 
    for p in samp :
        m =HOLD &(gpid ==p );k =m &(sc >=THR )
        tp +=int (Y [k ].sum ());nk +=int (k .sum ());gt +=gt_of (m )
    boot .append (prf (tp ,nk ,gt )[2 ])
lo ,hi =np .percentile (boot ,[2.5 ,97.5 ])
print (f"\nBOOTSTRAP %95 GA (ALL, 2000 tekrar, part-bazli): [{lo :.4f}, {hi :.4f}]  (nokta {np .mean (boot ):.4f})")

# ---- AUTO / REVIEW ayrimi (urun konvansiyonu: conf >= 0.5 -> AUTO) ----
kept =HOLD &(sc >=THR )
auto =kept &(CONF >=0.5 );rev =kept &(CONF <0.5 )
gt_all =gt_of (HOLD )
p_auto =int (Y [auto ].sum ())/max (int (auto .sum ()),1 )
p_all ,r_all ,f_all =prf (int (Y [kept ].sum ()),int (kept .sum ()),gt_all )
print (f"\n=== AUTO / REVIEW (urun konvansiyonu conf>=0.5) ===")
print (f"  AUTO       : n={int (auto .sum ()):4d} | PRECISION {p_auto :.4f}  (guvenlik metrigi)")
print (f"  REVIEW     : n={int (rev .sum ()):4d} | precision {int (Y [rev ].sum ())/max (int (rev .sum ()),1 ):.4f}")
print (f"  AUTO+REVIEW: P {p_all :.4f} R {r_all :.4f} F1 {f_all :.4f}  (birincil metrik)")

gate =final ["ALL"]["F1_mean"]>=0.85 
print (f"\n================ BASARI KAPISI ================")
print (f"  locked holdout ALL CP-F1 = {final ['ALL']['F1_mean']:.4f}  (hedef >= 0.85)  -> {'GECTI'if gate else 'GECMEDI'}")
final ["bootstrap_CI95_ALL"]=[float (lo ),float (hi )]
final ["AUTO_precision"]=float (p_auto );final ["AUTO_n"]=int (auto .sum ())
final ["AUTO_REVIEW"]={"P":float (p_all ),"R":float (r_all ),"F1":float (f_all )}
final ["config"]=CFG ;final ["gate_passed"]=bool (gate )
final ["note"]="KILITLI HOLDOUT TEK KEZ acildi; no secim/tuning this veride yapilmadi."
json .dump (final ,open ("results/p7_final_holdout.json","w"),indent =1 )
print ("-> results/p7_final_holdout.json")
