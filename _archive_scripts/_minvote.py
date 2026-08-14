# -*- coding: utf-8 -*-
"""8 uyeli havuzda min_votes taramasi -- DURUST secimle.

RATIONALE: min_votes=1 (birlesim) karari 4 UYE for verilmisti; gerekcesi "birlesimin high
recall'i hayatta kalir because gate kesinligi geri getirir". 8 uyede candidate count GT basina
2.4'ten 4.6'ya output and gate yuku kaldiramadi (detection -0.031). Artik "3 uye anlasti" like
anlamli a sinyal present; birlesim zorunlu not.

DURUST SECIM: threshold BIR YARIDA secilir, DIGER yaride olculur. Ayni kumede tarayip most iyiyi
almak sisirmedir (this night G5'te same tuzagi olcmustum).
"""
import io ,json ,os ,sys 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import measure_set ,wire_gate 
from sina_cluster import esle ,f1w 
from sklearn .ensemble import RandomForestClassifier 

d =np .load ("results/gate_8uye.npz",allow_pickle =True )
Xtr =np .hstack ([np .asarray (d ["X22"],float ),np .asarray (d ["XR"],float )])
ytr =np .asarray (d ["y"]);ptr =np .array ([str (x )for x in d ["pids"]]);vtr =np .asarray (d ["votes"])
DER ,rap =measure_set .cluster ("results/_der_8uye.pkl");measure_set .rapor_bas (rap )
AD =None 
try :
    import gate_bench as T ;AD =T .yukle ()["name"]
except Exception :pass 
i_vot =AD .index ("votes")if AD and "votes"in AD else 11 

Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
for u in np .unique (ptr ):
    i =np .where (ptr ==u )[0 ];Ztr [i ]=wire_gate .within_part (Xtr [i ],"zskor")

def gate_ile (mv ):
    m =vtr >=mv 
    if m .sum ()<500 or len (np .unique (ytr [m ]))<2 :return None 
    return {"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Ztr [m ],ytr [m ]),"n_feat":Ztr .shape [1 ],"donusum":"zskor"}

def puanla (mv ,alt ):
    g =gate_ile (mv )
    if g is None :return None ,None ,None 
    det ,rob ,gg =[],[],[]
    for r in alt :
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
        if r ["X"]is not None and r .get ("XR")is not None :
            X =np .hstack ([r ["X"],r ["XR"]])
            vv =X [:,i_vot ]
            sec =vv >=mv 
            if sec .any ():
                Xs =X [sec ]
                k =wire_gate .decision_mask (wire_gate .decision_score (g ,Xs ))
                if k .any ():
                    idx =np .where (sec )[0 ][k ]
                    P =np .asarray (r ["P"],float )[idx ].copy ();Pd =np .asarray (r ["Pd"],float )[idx ].copy ()
                    c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                    c =wire_gate .pose_correct (X [idx ],c );c =wire_gate .angle_correct (X [idx ],c )
                    if r .get ("UYE"):c =wire_gate .pick_member_direction (X [idx ],c ,r ["UYE"])
                    P =np .array ([x ["point"]for x in c ],float );Pd =np .array ([x ["direction"]for x in c ],float )
        rj ="very"if r ["n"]>=8 else "low"
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        det .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
        rob .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        gg .append (r ["geo"])
    return det ,rob ,gg 

MV =(1 ,2 ,3 ,4 )
print (f"\n{'min_votes':<11}{'training candidate':>13}{'detection':>9}{'robot':>9}")
TAM ={}
for mv in MV :
    det ,rob ,gg =puanla (mv ,DER )
    if det is None :continue 
    TAM [mv ]=(f1w (det ),f1w (rob ))
    print (f"{mv :<11}{int ((vtr >=mv ).sum ()):>13}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}",flush =True )

    # --- DURUST CAPRAZ SECIM
gruplar =sorted ({r ["geo"]for r in DER });rng =np .random .default_rng (0 )
kar =list (gruplar );rng .shuffle (kar );A =set (kar [:len (kar )//2 ])
altA =[r for r in DER if r ["geo"]in A ];altB =[r for r in DER if r ["geo"]not in A ]
dc ,gg2 =[],[]
for sec ,olc in ((altA ,altB ),(altB ,altA )):
    en ,ea =-1 ,1 
    for mv in MV :
        dd ,_ ,_ =puanla (mv ,sec )
        if dd and f1w (dd )>en :en ,ea =f1w (dd ),mv 
    d1 ,_ ,g1 =puanla (ea ,olc )
    dc +=d1 ;gg2 +=g1 
    print (f"  yarida selected min_votes {ea } -> diger yaride measured ({len (olc )} part)")
print (f"\nDURUST (capraz secim) TESPIT: {f1w (dc ):.4f}   [baseline 0.7584]")
json .dump ({"full":{str (k ):{"detection":v [0 ],"robot":v [1 ]}for k ,v in TAM .items ()},
"durust_tespit":f1w (dc )},io .open ("results/g4_minvotes.json","w"),indent =1 )
print ("receipt -> results/g4_minvotes.json")
