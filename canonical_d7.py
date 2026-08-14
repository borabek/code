# -*- coding: utf-8 -*-
"""KANONIK D7 ZINCIRI -- 0.2344'u URETEN girdilerle.

DUN WHY URETEMEDIM (2026-08-09): two girdiyi wrong kullanmisim.
  * kayitlar `_der_yeni_G7BIRLESIK.pkl` (ben d6_record.yukle with baska shard'lar)
  * gate `wire_gate_v6.pkl` (ben v5)
Makbuzlardan (`audit_d7_pose_dictionary_oracle.json`) okundu.

Bu modul TEK giris noktasidir. Yeni a yigin olculecekse BURADAN gecer.
"""
import glob ,os ,pickle ,sys 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import wire_gate ,product_zinciri 
from korpus_kimlik import step_kimlik as SK 

KAYIT ="results/_der_yeni_G7BIRLESIK.pkl"
GATE ="results/wire_gate_v6.pkl"
OB ="results/_p1_olasilik_d7"

# YIGIN SECIMI (2026-08-10). Kanonik yigin YUKARIDAKI ucludur and VARSAYILANDIR;
# headline that uclu with uretilir. Ama a baska segmentasyon agini AYNI measurement yoluyla
# sinamak for ayri a betik yazmak, measurement yolunun urunden ayrismasi demekti
# (this projede two times became). Bunun instead of same harness cevre degiskeniyle
# yeniden hedeflenir; damga (`makbuz_hash`) hangi ucluyle olculdugunu carries.
#   KD7_KAYIT / KD7_GATE / KD7_OB
_ov =lambda ad ,var :os .environ .get (ad )or var # noqa: E731
KAYIT =_ov ("KD7_KAYIT",KAYIT )
GATE =_ov ("KD7_GATE",GATE )
OB =_ov ("KD7_OB",OB )
YANAL ,ACI =2.0 ,10.0 


def yukle (pidler =None ):
    R =pickle .load (open (KAYIT ,"rb"))
    if pidler is not None :
        pidler =set (map (str ,pidler ))
        R =[r for r in R if str (r ["pid"])in pidler ]
    return {str (r ["pid"]):r for r in R }


def gate_yukle ():
    return pickle .load (open (GATE ,"rb"))


def x58 (r ):
    X =np .asarray (r ["X"],float );XR =np .asarray (r ["XR"],float )
    return np .hstack ([X ,XR ])if X .shape [1 ]==22 else X 


def mikro (rows ):
    """MIKRO F1 (havuzlanmis TP/FP/FN) -- headline this olcekte.

    DUN'KU HATA: `f1w` (regime-agirlikli, low 0.895 / very 0.105) kullaniyordum;
    same veride 0.0838 veriyor, mikro whereas 0.1956. Manset 0.2344 MIKRO olcekte.
    """
    TP =sum (x [1 ]for x in rows );FP =sum (x [2 ]for x in rows )
    FN =sum (x [3 ]for x in rows )
    return 2 *TP /max (2 *TP +FP +FN ,1 )


def makro (per ):
    """Uretici bazinda F1'lerin ortalamasi. per: mfg -> [tp, fp, fn]."""
    import numpy as _np 
    return float (_np .mean ([2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )
    for a in per .values ()]))if per else 0.0 


def urun_poz (r ,gate ,S ,threshold =(0.40 ,0.30 ),tam_zincir =True ,p3c =None ,
p3c_esik =0.0 ,cyl =None ):
    """URUNUN output pozlari. Doner: (P, D).

    `p3c` verilirse EKSEN SECICISI de runs -- headline yolunda kosuyordu and
    bende yoktu (kalan farkin most guclu adayi).
    """
    from p1c_threshold import maske 
    M =x58 (r )
    P =np .asarray (r ["P"],float );D =np .asarray (r ["Pd"],float )
    if not len (P ):
        return P ,D 
    s =np .asarray (wire_gate .decision_score (gate ,M ),float )
    k =maske (s ,threshold [0 ],threshold [1 ])
    P ,D =(P [k ],D [k ])if k .any ()else (P [:0 ],D [:0 ])
    # KALABALIK BASTIRMA -- urunun `wire_gate.apply` yolundakiyle AYNI fonksiyon.
    # Buraya konmazsa measurement yolu urunden ayrisir (2026-08-10'da full bunu yakaladim).
    if len (P )>1 :
        _n =wire_gate .crowd_mask (P ,s [k ])
        P ,D =P [_n ],D [_n ]
    if p3c is not None and len (P )and cyl is not None :
        import p3c_axis_selector as P3 
        cy =cyl .get (r ["pid"])
        komsu =None 
        if len (D )>1 :
            B =D *np .sign (D @D [0 ])[:,None ]
            komsu =B .mean (0 );komsu /=(np .linalg .norm (komsu )+1e-12 )
        sk =s [k ]if k .any ()else s [:0 ]
        P2 ,D2 =P .copy (),D .copy ()
        for i in range (len (P )):
            opt =P3 .secenekler (cy ,P [i ],D [i ],r ["diag"],float (sk [i ]),
            komsu ,len (P ))
            if len (opt )==1 :
                continue 
            X2 =np .asarray ([o [2 ]for o in opt ],float )
            pr =p3c ["clf"].predict_proba (P3 ._uyumlu (p3c ["clf"],X2 ))[:,1 ]
            j =int (np .argmax (pr ))
            if j !=0 and pr [j ]>=pr [0 ]+p3c_esik :
                P2 [i ],D2 [i ]=opt [j ][0 ],opt [j ][1 ]
        P ,D =P2 ,D2 
    if tam_zincir and len (P ):
        f =f"{OB }/{r ['pid']}.npz"
        if os .path .exists (f ):
            z =np .load (f )
            P ,D =product_zinciri .tam_poz (
            np .ascontiguousarray (z ["V"],np .float64 ),
            np .ascontiguousarray (z ["F"],np .int64 ),
            np .asarray (z ["pbs"],float ).mean (0 ),P ,D ,
            step_path =S .get (r ["pid"]))
    return P ,D 


def step_map ():
    return {SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
