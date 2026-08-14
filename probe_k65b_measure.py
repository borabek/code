"""K6.5-b OLCUM: few-shot adapte edilmis aglari UCTAN UCA olcer.

Egitimden AYRI betik: training a times runs (saatler), measurement tekrar tekrar
kosulabilir (dakikalar). `run_k65b_fewshot_seg.py` makbuzunu reads.

PROTOKOL:
  * k=0 TABANI: degistirilmemis urun agi (g10) same parcalarda
  * adaptasyon parcalari OLCUME GIRMEZ (makbuzdaki `adapt` listesi dislanir)
  * TAM zincir: gate -> product_chain.tam_poz -> Macar eslestirme
  * tespit VE robot AYRI raporlanir; k=0'a according to difference verilir
  * cekilisler arasi mean +- std (single cekilis gurultulu)

HATA YUTULMAZ: a kosum olculemezse betik PATLAR (see. g10 kill kapisi dersi).
"""
import argparse 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")

import d6_record # noqa: E402
import robot_cp # noqa: E402
import wire_gate # noqa: E402
import product_chain # noqa: E402
from p1c_threshold import maske # noqa: E402
from sina_cluster import match_hungarian ,f1w # noqa: E402
from corpus_identity import step_kimlik as SK # noqa: E402

ROBOT_YANAL ,ROBOT_ACI =2.0 ,10.0 
TABAN_OB ="results/_p1_olasilik_g10"# k=0: degistirilmemis urun agi


def olc (pidler ,ob_dir ,rec_ ,gate ,S ):
    """Bir cache dizini for (tespit_F1, robot_F1). Hata YUTULMAZ."""
    T ,R =[],[]
    for pid in pidler :
        f =f"{ob_dir }/{pid }.npz"
        if not os .path .exists (f ):
            raise RuntimeError (f"onbellekte YOK: {f } -- measurement eksik kalirdi")
        r =rec_ [pid ]
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        d =np .load (f )
        V =np .ascontiguousarray (d ["V"],np .float64 )
        F =np .ascontiguousarray (d ["F"],np .int64 )
        pbs =[np .asarray (q ,float )for q in d ["pbs"]]
        cps ,_op ,_cok ,_per =robot_cp .derive_candidates (V ,F ,pbs ,S .get (pid ))
        if not cps :
            T .append ((len (G ),0.0 ,0.0 ,0.0 ))
            R .append ((len (G ),0.0 ,0.0 ,0.0 ))
            continue 
            # ANAHTAR "direction" -- "dir" DEGIL. Ilk surumumde `c.get("dir",[0,0,1])`
            # yazmistim: TUM yonler [0,0,1] oluyordu and signed angle metrigi each places
            # dusuyordu -> robot TAM 0.0000. Varsayilan vermek instead of PATLIYORUZ.
        P =np .asarray ([c ["point"]for c in cps ],float )
        D =np .asarray ([c ["direction"]for c in cps ],float )

        # GATE OZNITELIKLERI YENI ADAYLAR ICIN HESAPLANIR.
        # Ilk surumumde `d6_record.x58(r)` kullaniyordum: that features KAYITTAKI
        # (old agdan gelen) candidates for uretilmisti. Uzunluklar tutmadigi for
        # (45 vs 23) gate SESSIZCE ATLANIYORDU -- i.e. "uctan uca" dedigim measurement
        # gate'siz kosuyordu. Simdi urunun own fonksiyonu is used.
        avg =np .asarray (d ["pbs"],float ).mean (0 )
        Xp =wire_gate .feats_for (V ,F ,avg ,cps ,robot_cp .CE ,robot_cp .CT ,
        step_path =S .get (pid ))
        Xp =np .asarray (Xp ,float )
        if len (Xp )!=len (P ):
            raise RuntimeError (f"{pid }: gate oznitelik {len (Xp )} != candidate {len (P )}")
        k =maske (np .asarray (wire_gate .decision_score (gate ,Xp ),float ),0.40 ,0.30 )
        if k .any ():
            P ,D =P [k ],D [k ]
        P ,D =product_chain .tam_poz (V ,F ,np .asarray (d ["pbs"],float ).mean (0 ),
        P ,D ,step_path =S .get (pid ))
        T .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ])
        R .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],ROBOT_YANAL ,
        ROBOT_ACI ,False ,signed =True )[:3 ])
    return f1w (T ),f1w (R )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--receipt",default ="results/k65b_fewshot_seg.json")
    ap .add_argument ("--cikti",default ="results/k65b_fewshot_olcum.json")
    a =ap .parse_args ()

    with open (a .receipt )as f :
        mk =json .load (f )
    brand =mk ["brand"]
    sv =d6_record .exam ()
    rec_ =d6_record .yukle (set (sv ["pidler"]))
    gate =pickle .load (open ("results/wire_gate_v5.pkl","rb"))
    import glob 
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    hepsi =sorted (p for p ,r in rec_ .items ()if r ["mfg"]==brand )
    print (f"=== {brand }: {len (hepsi )} part ===\n")

    res_ ={}
    for k ,kosumlar in sorted (mk ["kosumlar"].items (),key =lambda t :int (t [0 ])):
        tl ,rl =[],[]
        for ko in kosumlar :
            adapt =set (map (str ,ko ["adapt"]))
            olcp =[p for p in hepsi if p not in adapt ]
            # k=0 TABANI AYNI PARCALARDA: adaptasyon parcalari here da disarida,
            # otherwise two arm FARKLI kumede olculur and difference anlamsizlasir.
            t0 ,r0 =olc (olcp ,TABAN_OB ,rec_ ,gate ,S )
            t1 ,r1 =olc (olcp ,ko ["cache"],rec_ ,gate ,S )
            tl .append ((t0 ,t1 ))
            rl .append ((r0 ,r1 ))
            print (f"  k={k } cekilis {ko ['cekilis']}: tespit {t0 :.4f} -> {t1 :.4f} "
            f"({t1 -t0 :+.4f}) | robot {r0 :.4f} -> {r1 :.4f} ({r1 -r0 :+.4f})",
            flush =True )
        t0m =float (np .mean ([x [0 ]for x in tl ]));t1m =float (np .mean ([x [1 ]for x in tl ]))
        r0m =float (np .mean ([x [0 ]for x in rl ]));r1m =float (np .mean ([x [1 ]for x in rl ]))
        res_ [k ]={"tespit_k0":t0m ,"tespit_k":t1m ,"tespit_fark":t1m -t0m ,
        "robot_k0":r0m ,"robot_k":r1m ,"robot_fark":r1m -r0m ,
        "robot_std":float (np .std ([x [1 ]for x in rl ])),
        "n_cekilis":len (kosumlar )}
        print (f"  --> k={k } ORTALAMA: tespit {t1m -t0m :+.4f} | robot {r1m -r0m :+.4f} "
        f"(std {res_ [k ]['robot_std']:.4f})\n",flush =True )

    with open (a .out_ ,"w")as f :
        json .dump ({"brand":brand ,"sonuc":res_ ,"baseline":TABAN_OB ,
        "not":"k=0 tabani AYNI parcalarda measured; adaptasyon parcalari "
        "each iki kolda da DISARIDA. Son-epoch ckpt kullanildi "
        "(genel val'e per secim adaptasyonu cezalandirir)."},
        f ,indent =1 )
    print (f"receipt -> {a .out_ }")


if __name__ =="__main__":
    main ()
