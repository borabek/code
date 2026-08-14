# -*- coding: utf-8 -*-
"""P1 ONCULU: gate ONCE mi SONRA mi? Tavan farki real mi?

IDDIA: ham candidate + (silindir + planar) poz secenekleriyle bire-a kahin 0.6097;
ONCE gate uygulanirsa same kahin 0.4837. Yani mevcut `gate -> pose` order
tavani BASTAN kirpiyor. Bu probe that iddiayi KENDI verimizde produces.

KAHIN = each candidate for seceneklerin EN IYISI secilebilseydi (bire-a Macar,
robot toleransi 2mm/10 derece ISARETLI). Ulasilabilir not, TAVAN.

TEZE SADIK: `v_o` (mevcut poz) HER ZAMAN seceneklerden biri.
"""
import glob ,json ,os ,pickle ,sys 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record ,robot_cp ,wire_gate ,brep_snap 
from p1c_threshold import maske 
from sina_cluster import match_hungarian ,f1w 
from korpus_kimlik import step_kimlik as SK 

OB ="results/_p1_olasilik_g10";GATE ="results/wire_gate_v7.pkl"
YANAL ,ACI =2.0 ,10.0 
sv =d6_record .exam ();rec_ =d6_record .yukle (set (sv ["pidler"]))
gate =pickle .load (open (GATE ,"rb"))
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
cyl =pickle .load (open ("results/_d6_silindirler.pkl","rb"))
pidler =sorted ({f [:-4 ]for f in os .listdir (OB )if f .endswith (".npz")}&set (rec_ ))
print (f"part {len (pidler )}\n",flush =True )


def secenekler (P ,D ,cyls ,mm =8.0 ):
    """Her candidate for (konum, direction) secenekleri. ILK oge HER ZAMAN MEVCUT (tez)."""
    uy =[c for c in (cyls or [])if brep_snap .R_MIN <=c ["radius"]<=brep_snap .R_MAX ]
    out =[]
    for i in range (len (P )):
        o =[(P [i ],D [i ])]
        for c in uy :
            a =np .asarray (c ["axis"],float )
            for m in (c ["mouth_a"],c ["mouth_b"]):
                m =np .asarray (m ,float )
                if np .linalg .norm (m -P [i ])<=mm :
                    o .append ((m ,a ));o .append ((m ,-a ))
        out .append (o )
    return out 


def oracle_ (P ,D ,cyls ,G ,Gd ,diag ):
    """ORTAK (bipartite) kahin -- candidate BASINA acgozlu DEGIL.

    ILK SURUMUM YANLISTI: each candidate KENDI most iyi secenegini bagimsiz seciyordu.
    Boylece bircok candidate AYNI GT'ye yigiliyor, Macar bire-a atayinca most bosa
    gidiyordu and kahin uctan uca sonucun however biraz ustunde cikiyordu (0.151) --
    hatta gate'li hali (0.199) DAHA YUKSEK gorunuyordu, ki a upper kumede
    imkansizdir. Isaret: kahin ORTAK must be.

    Dogrusu: (candidate i, GT j) for i'nin O GT'ye according to EN IYI secenegini bul,
    kabul edilebilirse 1 score; after bipartite EN BUYUK ESLESME (Hungarian).
    """
    if not len (P )or not len (G ):
        return (len (G ),0.0 ,0.0 ,0.0 )
    S_ =secenekler (P ,D ,cyls )
    n ,m =len (P ),len (G )
    C =np .ones ((n ,m ))# maliyet: 0 = kabul edilebilir, 1 = not
    for i ,o in enumerate (S_ ):
        for j in range (m ):
            g =Gd [j ]/(np .linalg .norm (Gd [j ])+1e-9 )
            for (p ,d )in o :
                u =d /(np .linalg .norm (d )+1e-9 )
                v =G [j ]-p 
                lat =np .linalg .norm (v -(v @u )*u )
                ax =abs (float (v @u ))
                ac =np .degrees (np .arccos (np .clip (float (u @g ),-1 ,1 )))
                if lat <=YANAL and ac <=ACI and ax <=40.0 :
                    C [i ,j ]=0.0 
                    break 
    from scipy .optimize import linear_sum_assignment 
    ri ,ci =linear_sum_assignment (C )
    tp =int (sum (1 for a ,b_ in zip (ri ,ci )if C [a ,b_ ]==0.0 ))
    # NULL SECENEGI: p5-v2 tasariminda a candidate ATILABILIR. Kahin de atabilmeli,
    # otherwise ham havuzun extra adaylari otomatik FP becomes and GENIS pool DAHA KOTU
    # gorunur -- ki this, olculmek istenen seyin (pool zenginligi) TERSINI olcer.
    # Ikinci surumumde this yoktu and gate'li arm more high cikiyordu.
    fp =0 
    fn =m -tp 
    return (m ,float (tp ),float (fp ),float (fn ))


R_ham ,R_gate =[],[]
for pid in pidler :
    r =rec_ [pid ]
    G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
    if not len (G ):continue 
    d =np .load (f"{OB }/{pid }.npz")
    V =np .ascontiguousarray (d ["V"],np .float64 );F =np .ascontiguousarray (d ["F"],np .int64 )
    cps ,_o ,_c ,_p =robot_cp .derive_candidates (V ,F ,[np .asarray (q ,float )for q in d ["pbs"]],
    S .get (pid ))
    if not cps :
        R_ham .append ((len (G ),0. ,0. ,0. ));R_gate .append ((len (G ),0. ,0. ,0. ));continue 
    P =np .asarray ([c ["point"]for c in cps ],float )
    D =np .asarray ([c ["direction"]for c in cps ],float )
    cy =cyl .get (pid )
    R_ham .append (oracle_ (P ,D ,cy ,G ,Gd ,r ["diag"]))
    avg =np .asarray (d ["pbs"],float ).mean (0 )
    Xp =np .asarray (wire_gate .feats_for (V ,F ,avg ,cps ,robot_cp .CE ,robot_cp .CT ,
    step_path =S .get (pid )),float )
    k =maske (np .asarray (wire_gate .decision_score (gate ,Xp ),float ),0.40 ,0.30 )
    if k .any ():
        R_gate .append (oracle_ (P [k ],D [k ],cy ,G ,Gd ,r ["diag"]))
    else :
        R_gate .append ((len (G ),0. ,0. ,0. ))
a ,b =f1w (R_ham ),f1w (R_gate )
print (f"HAM candidate + poz secenekleri (gate YOK) : {a :.4f}")
print (f"ONCE gate, sonra poz secenekleri      : {b :.4f}")
print (f"GATE'IN KIRPTIGI TAVAN                : {a -b :+.4f}")
json .dump ({"ham_kahin":a ,"gate_once_kahin":b ,"fark":a -b ,"n_parca":len (pidler ),
"not":"D6 (DEV). Kahin = ulasilabilir not TAVAN."},
open ("results/p1_sira_tavani.json","w"),indent =1 )
print ("receipt -> results/p1_sira_tavani.json")
