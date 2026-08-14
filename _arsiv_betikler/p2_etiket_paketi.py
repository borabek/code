# -*- coding: utf-8 -*-
"""P2 PARTI-1 label paketi: `.obj` + MODELIN TAHMINIYLE DOLDURULMUS label.

WHY TAHMIN TOHUMLU: is SIFIRDAN BOYAMA not DUZELTME. Olculdu
(`results/p2_getirisi.json`, D7 brand-disi, 3087 GT): mouth cevresi segmentasyon
kalitesinin most iyi ceyreginde robot 0.4870 / lateral<=2mm %65.4 / medyan lateral
1.38mm; most kotu ceyrekte 0.0557 / %10.4 / 8.64mm. Herkes most iyi ceyrek like
davransa robot recall 0.1995 -> 0.4870. Duzeltilecek sey AGZIN KENARLARI.

TEZE SADIK: ~6000 tepeye uniform izotropik remesh, 5 sinif, `v_o` turetmesi
DEGISMEZ. Degisen single sey label kalitesi.

Cikti: `_p2_etiket/<pid>/<pid>.obj` + `<pid>.labels.txt` (seed) +
`<pid>.labels.template.txt` (empty) + `README.md`.
Etiketleyici `label_tool.html`'i acar, `.obj`'i selects, seed etiketi yukler,
agizlari fixes, indirir.
"""
import argparse 
import glob 
import os 
import sys 
import time 

import numpy as np 
import torch 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
os .environ .setdefault ("BA_ALLOW_SEEN","1")
import diffusionnet as D_ # noqa: E402
import thesis_remesh # noqa: E402
from infer_step_cp import load_any ,step_to_mesh # noqa: E402
from corpus_identity import step_kimlik as SK # noqa: E402

CIK ="_p2_etiket"
CKPT ="results/seg_g7/g7_s0.pt"
SINIF ="0=Housing 1=Contact 2=SnapPoint 3=CableEntry 4=LabelSurface"

KILAVUZ ="""# P2 parti 1 -- mouth segmentasyonu correction

**Is SIFIRDAN boyama DEGIL.** Her parcanin etiketi modelin own tahminiyle
doldurulmus geliyor; senin yapacagin **agizlarin kenarlarini duzeltmek**.

## Sinif sozlesmesi
`{sinif}`

* **CableEntry (3)** = telin girdigi acikligin YUZEYI. Yuvarlak hole, KARE
  giris, push-in yarik -- all of them CableEntry. Sadece yuvarlak hole not.
* **Contact (1)** = metal kontak yuzeyi / alet yuvasi.
* Emin olamadigin yeri **DEGISTIRME** -- wrong label bosluktan kotudur.

## Neden this is
Olculdu (D7 gorulmemis brand, 3087 CP): mouth cevresindeki segmentasyon
kalitesinin most iyi ceyreginde robot **0.4870**, most kotu ceyrekte **0.0557**;
medyan lateral error 1.38mm'ye karsi 8.64mm. Herkes most iyi ceyrek like davransa
robot recall **0.1995 -> 0.4870**. Bugune up to denenen tum last-islem kollari
+-0.005 bandindaydi; this arm two buyukluk mertebesi more large.

## Nasil
1. `label_tool.html`'i tarayicida ac
2. `<pid>.obj` dosyasini sec
3. `<pid>.labels.txt` (seed) dosyasini yukle
4. Agiz kenarlarini duzelt, indir, same dizine `<pid>.labels.txt` as kaydet

## Parcalar
{list}
"""


def obj_yaz (yol ,V ,F ):
    with open (yol ,"w")as f :
        for v in V :
            f .write (f"v {v [0 ]:.6f} {v [1 ]:.6f} {v [2 ]:.6f}\n")
        for t in F :
            f .write (f"f {t [0 ]+1 } {t [1 ]+1 } {t [2 ]+1 }\n")


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--list",default ="results/p2_parti1_pidler.txt")
    ap .add_argument ("--output",default =CIK )
    ap .add_argument ("--ckpt",default =CKPT )
    a =ap .parse_args ()

    pids =[x .strip ()for x in open (a .lst_ )if x .strip ()]
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    eksik =[p for p in pids if p not in S ]
    if eksik :
        raise SystemExit (f"STEP bulunamayan {len (eksik )} part: {eksik [:5 ]} "
        "-- silent atlamak instead of DURUYORUM")
    os .makedirs (a .out_ ,exist_ok =True )

    cihaz ="cuda"if torch .cuda .is_available ()else "cpu"
    # URUNUN own yukleyicisi -- k_eig/meta ckpt'ten gelir. Elle `n_eig`
    # varsaymak this projede a times agi YANLIS ozvektor tabaninda kosturmustu.
    model ,meta =load_any (a .ckpt ,dev =cihaz )[:2 ]
    print (f"ckpt {a .ckpt } | cihaz {cihaz } | part {len (pids )}",flush =True )

    t0 =time .time ()
    basarili =[]
    for i ,pid in enumerate (pids ,1 ):
        d =os .path .join (a .out_ ,pid )
        os .makedirs (d ,exist_ok =True )
        obj =os .path .join (d ,f"{pid }.obj")
        seed =os .path .join (d ,f"{pid }.labels.txt")
        if os .path .exists (obj )and os .path .exists (seed ):
            basarili .append (pid )
            continue 
        Vr ,Fr =step_to_mesh (S [pid ])
        V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )# TEZ
        V =np .ascontiguousarray (V ,np .float64 )
        F =np .ascontiguousarray (F ,np .int64 )
        obj_yaz (obj ,V ,F )
        _lb ,pb =D_ .predict (model ,meta ,V ,F ,device =cihaz ,
        op_cache_dir =None ,return_probs =True )
        pred =np .asarray (np .argmax (np .asarray (pb ,float ),axis =1 ),int )
        np .savetxt (seed ,pred ,fmt ="%d")
        np .savetxt (os .path .join (d ,f"{pid }.labels.template.txt"),
        np .zeros (len (V ),int ),fmt ="%d")
        basarili .append (pid )
        if i %5 ==0 :
            print (f"  {i }/{len (pids )}  {(time .time ()-t0 )/i :.1f}s/part",flush =True )

    with open (os .path .join (a .out_ ,"README.md"),"w",encoding ="utf-8")as f :
        f .write (KILAVUZ .format (sinif =SINIF ,
        lst_ ="\n".join (f"* {p }"for p in basarili )))
    print (f"\nBITTI: {len (basarili )} part -> {a .out_ }")
    print (f"Etiket araci: label_tool.html")


if __name__ =="__main__":
    main ()
