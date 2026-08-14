# -*- coding: utf-8 -*-
"""A4: FIZIKSEL GECERLILIK ORANI -- KORPUS CAPINDA (194 part), metrikten BAGIMSIZ column.

OTOPSIDE (3 part) uretilen CP'lerin %40/%50/%96'si fiziksel as kusurluydu. O a
ORNEKLEMDI; this betik 194 parcanin TAMAMINDA olcer.

ALETLER A0'DA DOGRULANDI: `is_inside` known geometrilerde 26/26 (yanilma %0.0),
`mouth_width` deviation %1, isin count 12/24/48'de birebir same. Yani asagidaki bayraklar
guvenilir and B1-B4'te RED KRITERI olabilirler.

DORT BAYRAK (all of them tez-notr; network/remesh/v_o'ya dokunmaz):
    govde_ici      point malzemenin ICINDE  -> robot oraya tel sokamaz
    onu_kapali     takma yonunde <5mm'de malzeme -> direction kanala girmiyor
    duvara_yapisik mouth ic capi small -> point agzin ORTASINDA not, kenarda
    axial        GT agzina according to |axial| large (only ESLESENLERDE tanimli)

ASIL SORU (B kolunun kaderi): kusurlu CP'ler FP mi TP mi?
  * cogunlukla FP whereas -> B1-B3 kesinligi yukseltir, arm yasar
  * cogunlukla TP whereas -> reddetmek recall'u cokertir, arm becomes
Bu ayrim OLCULMEDEN B'ye gecilmez.
"""
import io 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
from sina_cluster import match_greedy ,f1w 
from a1a3_metric_cerrahi import urun_ciktisi 

ONB ="results/_a4_bayraklar.pkl"
ILERI_MIN =5.0 # takma yonunde at least this up to bosluk must be
IC_CAP_MIN =0.8 # mouth ic capi (merkezlenme) -- B2'de DEV'de ayarlanacak
EKSEN_MAX =8.0 


def bayraklari_uret (DER ,CIKTI ):
    if os .path .exists (ONB ):
        d =pickle .load (open (ONB ,"rb"))
        print (f"bayraklar onbellekten: {len (d )} part",flush =True )
        return d 
    import trimesh 
    import thesis_remesh 
    from big_arbiter import eligible 
    from cp_geometry import is_inside ,mouth_width ,ray_hits 
    from infer_step_cp import step_to_mesh 
    stp ={p :s for m ,p ,jf ,s in eligible ()}
    out ={}
    t0 =time .time ()
    for k ,r in enumerate (DER ,1 ):
        if k %20 ==0 :
            print (f"  {k }/{len (DER )}  {time .time ()-t0 :.0f}s",flush =True )
            pickle .dump (out ,open (ONB ,"wb"))
        P ,Pd =CIKTI [r ["pid"]]
        if not len (P )or r ["pid"]not in stp :
            out [r ["pid"]]=[]
            continue 
        try :
            Vr ,Fr =step_to_mesh (stp [r ["pid"]])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            mesh =trimesh .Trimesh (vertices =np .ascontiguousarray (V ,float ),
            faces =np .ascontiguousarray (F ,np .int64 ),process =False )
        except Exception :
            out [r ["pid"]]=[]
            continue 
        sat =[]
        for i in range (len (P )):
            p ,d =P [i ],Pd [i ]
            try :
                ic =bool (is_inside (mesh ,p ))
            except Exception :
                ic =None 
            try :
                h =ray_hits (mesh ,p +1e-3 *d ,d ,60.0 )
                ileri =float (min (h ))if len (h )else float ("inf")
            except Exception :
                ileri =float ("nan")
            try :
                cap =float (mouth_width (mesh ,p ,d )[0 ])
            except Exception :
                cap =float ("nan")
            sat .append ({"govde_ici":ic ,"ileri":ileri ,"ic_cap":cap })
        out [r ["pid"]]=sat 
    pickle .dump (out ,open (ONB ,"wb"))
    print (f"-> {ONB }",flush =True )
    return out 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    DER ,gate ,ek =T2 .yukle ()
    CIKTI ={r ["pid"]:urun_ciktisi (r ,gate )for r in DER }
    B =bayraklari_uret (DER ,CIKTI )
    AGIZ =pickle .load (open ("results/_gt_agiz.pkl","rb"))

    line_ =[]
    for r in DER :
        P ,Pd =CIKTI [r ["pid"]]
        bl =B .get (r ["pid"])or []
        if not len (P )or len (bl )!=len (P ):
            continue 
        G =AGIZ [r ["pid"]][0 ];Gd =np .asarray (r ["Gd"],float )
        tp ,fp ,fn ,bi =match_greedy (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )
        matched ={e [0 ]:(e [3 ],e [4 ])for e in bi ["eslesme"]}
        rj ="very"if r ["n"]>=8 else "low"
        for i in range (len (P )):
            b =bl [i ]
            ax ,ac =matched .get (i ,(None ,None ))
            line_ .append ({"pid":r ["pid"],"mfg":r .get ("mfg","?"),"regime":rj ,
            "tp":i in matched ,
            "govde_ici":bool (b ["govde_ici"]),
            "onu_kapali":bool (np .isfinite (b ["ileri"])and b ["ileri"]<ILERI_MIN ),
            "duvara_yapisik":bool (np .isfinite (b ["ic_cap"])and 0 <b ["ic_cap"]<IC_CAP_MIN ),
            "eksenel_buyuk":bool (ax is not None and abs (ax )>EKSEN_MAX ),
            "ic_cap":float (b ["ic_cap"]),"ileri":float (b ["ileri"])})
    n =len (line_ )
    BAY =("govde_ici","onu_kapali","duvara_yapisik","eksenel_buyuk")
    kusur =lambda s :any (s [k ]for k in BAY )
    print (f"\n{'='*78 }\nA4 -- FIZIKSEL GECERLILIK (194 part, {n } uretilen CP)\n{'='*78 }")
    print (f"{'bayrak':<18}{'number':>7}{'ratio':>8}   {'TP inside':>10}{'FP inside':>11}")
    ntp =sum (1 for s in line_ if s ["tp"]);nfp =n -ntp 
    for k in BAY :
        c =sum (1 for s in line_ if s [k ])
        t =sum (1 for s in line_ if s [k ]and s ["tp"])
        print (f"{k :<18}{c :>7}{100 *c /max (n ,1 ):>7.1f}%   {t :>10}{c -t :>11}")
    ck =sum (1 for s in line_ if kusur (s ))
    ckt =sum (1 for s in line_ if kusur (s )and s ["tp"])
    print (f"{'HERHANGI BIRI':<18}{ck :>7}{100 *ck /max (n ,1 ):>7.1f}%   {ckt :>10}{ck -ckt :>11}")
    print (f"\nTP {ntp } | FP {nfp }")
    print (f"  TP'lerin kusurlu orani: %{100 *ckt /max (ntp ,1 ):.1f}")
    print (f"  FP'lerin kusurlu orani: %{100 *(ck -ckt )/max (nfp ,1 ):.1f}")

    print (f"\n{'regime':<10}{'CP':>7}{'kusurlu':>9}{'ratio':>8}{'TP kusur':>10}{'FP kusur':>10}")
    for rj in ("low","very"):
        alt =[s for s in line_ if s ["regime"]==rj ]
        c =sum (1 for s in alt if kusur (s ))
        t =sum (1 for s in alt if kusur (s )and s ["tp"])
        print (f"{rj :<10}{len (alt ):>7}{c :>9}{100 *c /max (len (alt ),1 ):>7.1f}%{t :>10}{c -t :>10}")

    print (f"\n{'manufacturer':<10}{'CP':>7}{'kusurlu':>9}{'ratio':>8}")
    for m in sorted ({s ["mfg"]for s in line_ }):
        alt =[s for s in line_ if s ["mfg"]==m ]
        c =sum (1 for s in alt if kusur (s ))
        print (f"{m :<10}{len (alt ):>7}{c :>9}{100 *c /max (len (alt ),1 ):>7.1f}%")

        # --- B kolunun kaderi
        # DUZELTME (rule 9): first version HAM SAYI oranini kullaniyordu (FP 73 / TP 61 = 1.20x)
        # and "TP agirlikli, reddetme" diyordu. YANLIS: korpusta TP count FP'nin 2.6 KATI,
        # therefore ham ratio each bayragi TP'ye kaydirir. Dogru olcu ZENGINLESME ORANI:
        # (bayragin FP'ler icindeki ORANI) / (TP'ler icindeki ORANI). Duzeltilince three bayrak da
        # FP-ZENGIN cikiyor (2.0x - 3.2x) and B kolu YASIYOR.
    print (f"\n{'='*78 }\nB KOLU ICIN HUKUM (ZENGINLESME ORANI)\n{'='*78 }")
    print (f"  {'bayrak':<16}{'FP ratio':>9}{'TP ratio':>9}{'zenginlesme':>13}   verdict")
    for k in BAY :
        t =sum (1 for s in line_ if s [k ]and s ["tp"])
        f =sum (1 for s in line_ if s [k ]and not s ["tp"])
        rt =t /max (ntp ,1 );rf =f /max (nfp ,1 )
        z =rf /max (rt ,1e-9 )
        hkm =("YAPISAL (only TP'de tanimli)"if k =="eksenel_buyuk"else 
        "AYIRT EDICI -- arm yasar"if z >=1.5 else 
        "ZAYIF"if z >=1.0 else "TERS (TP'de more sik)")
        print (f"  {k :<16}{100 *rf :>8.1f}%{100 *rt :>8.1f}%{z :>12.2f}x   {hkm }")
        # BIRLESIK satiri EKSENEL HARIC is computed: that bayrak only matched CP'lerde tanimli,
        # birlige katilinca ratio yapisal as TP'ye kayar and kolu haksiz yere olu gosterir.
    FIZ =("govde_ici","onu_kapali","duvara_yapisik")
    fk =lambda s :any (s [k ]for k in FIZ )
    ck2 =sum (1 for s in line_ if fk (s ));ckt2 =sum (1 for s in line_ if fk (s )and s ["tp"])
    rk_t =ckt2 /max (ntp ,1 );rk_f =(ck2 -ckt2 )/max (nfp ,1 )
    print (f"  {'BIRLESIK(fiz)':<16}{100 *rk_f :>8.1f}%{100 *rk_t :>8.1f}%{rk_f /max (rk_t ,1e-9 ):>12.2f}x"
    f"   {ck2 } CP (%{100 *ck2 /max (n ,1 ):.1f})")
    print ("\n  RULE: zenginlesme >= 1.5x whereas bayrak FP'yi ayirt eder -> FIX/RED kolu mesru")

    with io .open ("results/a4_fiziksel_gecerlilik.json","w",encoding ="utf-8")as f :
        json .dump ({"n_cp":n ,"tp":ntp ,"fp":nfp ,
        "kusurlu":ck ,"kusurlu_tp":ckt ,
        "bayraklar":{k :{"total":sum (1 for s in line_ if s [k ]),
        "tp":sum (1 for s in line_ if s [k ]and s ["tp"]),
        "fp":sum (1 for s in line_ if s [k ]and not s ["tp"])}
        for k in BAY },
        "esikler":{"ileri_min":ILERI_MIN ,"ic_cap_min":IC_CAP_MIN ,
        "eksen_max":EKSEN_MAX }},f ,indent =1 )
    print ("\nmakbuz -> results/a4_fiziksel_gecerlilik.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
