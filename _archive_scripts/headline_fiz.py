# -*- coding: utf-8 -*-
"""Yeni headline: 18-sutunlu gate, URUN birlestiricisi, DEV+VAL havuzlanmis."""
import os ,sys ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1");os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["MERGE"]="urun"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from sina_cluster import f1w ,pr 

# q6 kosulari part-basina sayimlari yazmiyordu; makbuzlardan HAVUZLANMIS value
# full as hesaplanamaz -> two kumenin agirlikli ortalamasi (equal n) alinir and
# GA'lar for q6 makbuzlarindaki cluster-ici degerler raporlanir.
d ={k :json .load (open (f"results/q6_fiz_uctan_uca_{k }.json"))for k in ("dev","val")}
print (f"{'arm':<22}{'DEV detection':>12}{'VAL detection':>12}{'ort':>9}"
f"{'DEV robot':>11}{'VAL robot':>11}{'ort':>9}")
for tag ,lab in (("rt2 13","onceki (13 column)"),("fiz18","YENI (18 column)")):
    dt ,vt =d ["dev"]["detection"][tag ],d ["val"]["detection"][tag ]
    dr ,vr =d ["dev"]["robot"][tag ],d ["val"]["robot"][tag ]
    print (f"{lab :<22}{dt :>12.4f}{vt :>12.4f}{(dt +vt )/2 :>9.4f}"
    f"{dr :>11.4f}{vr :>11.4f}{(dr +vr )/2 :>9.4f}")
o ={}
for m in ("detection","robot"):
    a =(d ["dev"][m ]["rt2 13"]+d ["val"][m ]["rt2 13"])/2 
    b =(d ["dev"][m ]["fiz18"]+d ["val"][m ]["fiz18"])/2 
    o [m ]={"onceki":a ,"new":b ,"difference":b -a }
    print (f"\n{m }: {a :.4f} -> {b :.4f}  ({b -a :+.4f})")
o ["bootstrap_dev"]=d ["dev"]["bootstrap"];o ["bootstrap_val"]=d ["val"]["bootstrap"]
o ["kesinlik_yeni"]={k :d [k ]["precision"]["fiz18"]for k in ("dev","val")}
o ["recall_yeni"]={k :d [k ]["recall"]["fiz18"]for k in ("dev","val")}
json .dump (o ,open ("results/headline_fiz.json","w"),indent =1 )
print ("\nmakbuz -> results/headline_fiz.json")
