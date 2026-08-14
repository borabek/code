# -*- coding: utf-8 -*-
"""Gate v7 REFIT: g10 with yeniden turetilmis corpus ten.

`parca_ici_dagit.py` with AYNI recete (within_part PARCA PARCA + RF 400/leaf3), but
input/output PARAMETRIK -- dagitilan gate'in uzerine YAZMAZ, A/B yapilabilsin.

WHY ZORUNLU (gate-refit-minv4 dersi): "two gate AYNI dagilimda egitilmeli".
g10 different candidates uretiyor; old gate onlari gormedi.
"""
import argparse ,pickle ,numpy as np 
from sklearn .ensemble import RandomForestClassifier 
import wire_gate 

ap =argparse .ArgumentParser ()
ap .add_argument ("--corpus",default ="results/zengin_parite_v4_g10.npz")
ap .add_argument ("--output",default ="results/wire_gate_v7.pkl")
ap .add_argument ("--donusum",default ="zskor")
a =ap .parse_args ()

d =np .load (a .corpus ,allow_pickle =True )
X =np .hstack ([d ["X22"],d ["XR"]]).astype (float )
y =np .asarray (d ["y"]).astype (int )
pids =np .array ([str (p )for p in d ["pids"]])
M =np .zeros ((len (X ),X .shape [1 ]*2 ))
for u in np .unique (pids ):
    i =np .where (pids ==u )[0 ]
    M [i ]=wire_gate .within_part (X [i ],a .donusum )
print (f"training: {M .shape [0 ]} candidate x {M .shape [1 ]} sutun | "
f"{len (np .unique (pids ))} part | pozitif {y .mean ():.4f}",flush =True )
clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
random_state =0 ).fit (M ,y )
with open (a .out_ ,"wb")as f :
    pickle .dump ({"clf":clf ,
    "feat_names":list (wire_gate .FEAT_NAMES )+
    [n +"_z"for n in wire_gate .FEAT_NAMES ],
    "cols":None ,"n_feat":M .shape [1 ],"donusum":a .donusum ,
    "topo_r":float (wire_gate .TOPO_R ),
    "corpus":a .corpus ,
    "note":"gate v7: g10 with yeniden turetilmis corpus (44675 candidate / "
    "2596 part). parca_ici_dagit.py with same recete."},f )
print (f"-> {a .out_ }")
