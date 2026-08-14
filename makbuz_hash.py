# -*- coding: utf-8 -*-
"""P0-c: HER MAKBUZA kod + config + model HASH'leri.

SORUN (plan, 2026-08-09): canli sistem / D7 / p3c / p5 FARKLI gate modelleri and
esikleri kullaniyordu and makbuzlarda bunu gosteren HICBIR SEY yoktu. Iki receipt
same sayiyi verse bile same yigindan gelip gelmedigi ANLASILAMIYORDU.

Kullanim:  import makbuz_hash;  makbuz_hash.damga()  -> dict
Her measurement betigi ciktisina `damga()` eklemeli.
"""
import hashlib ,io ,json ,os ,sys 

KOD =["robot_cp.py","wire_gate.py","cp_openings.py","product_chain.py",
"p3c_axis_selector.py","p5v2_secenek.py","p5v2_egit.py",
"sina_cluster.py","p1c_threshold.py"]
MODEL =["results/wire_gate_v5.pkl","results/wire_gate_v7.pkl",
"results/p3c_axis_selector.pkl","results/seg_g10/g10_s0.pt",
"results/seg_g7/g7_s0.pt"]
CONFIG =["cp_config.json","results/metrik_dondurulmus.json"]


def _h (yol ,n =16 ):
    if not os .path .exists (yol ):
        return None 
    h =hashlib .sha256 ()
    with open (yol ,"rb")as f :
        for blok in iter (lambda :f .read (1 <<20 ),b""):
            h .update (blok )
    return h .hexdigest ()[:n ]


def damga ():
    """Olcumun HANGI yigindan geldigini single sozlukte returns."""
    return {
    "kod":{k :_h (k )for k in KOD if os .path .exists (k )},
    "model":{k :_h (k )for k in MODEL if os .path .exists (k )},
    "config":{k :_h (k )for k in CONFIG if os .path .exists (k )},
    "python":sys .version .split ()[0 ],
    }


if __name__ =="__main__":
    d =damga ()
    json .dump (d ,io .open ("results/damga.json","w",encoding ="utf-8"),indent =1 )
    print (json .dumps (d ,indent =1 )[:900 ])
