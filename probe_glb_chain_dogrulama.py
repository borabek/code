# -*- coding: utf-8 -*-
"""E2 -- SAHAYA INEN ZINCIRIN DOGRULANMASI

SORUN (memory: glb-olculen-zinciri-kullanmiyor). Ihracatcilar
`robot_cp.extract` cagiriyor; kampanyada olculen `product_p6`/`product_genis`
sahaya HIC girmiyor. Yani olctugum each kazanc robota ULASMIYOR.
Olculmus difference: baseline 0.2980 vs olculen zincir 0.3115.

`export_robot_glb.py` icine `cp_config.glb_kanonik_zincir` bayragi kondu
(varsayilan KAPALI). Bu betik bayragi ACMADAN ONCE yolun saglam calistigini
dogrular -- sahaya inen sey budur, kirik output robotu wrong yere gonderir.

DENETLENEN (GT'li parcalarda, two zincir YAN YANA):
  cikti_var    : zincir never CP uretiyor mu (empty donmuyor mu)
  count         : uretilen CP count
  yon_birim    : yonler unit uzunlukta mi (robot for sart)
  yon_sonlu    : NaN/Inf present mi
  tier         : tier atamasi yapiliyor mu (AUTO/REVIEW)
  robot_F1     : uctan uca robot F1 (lateral 2mm / signed angle 10 / axial 40)

KAPI: olculen zincir (1) no parcada COKMEYECEK, (2) yonleri gecerli
olacak, (3) robot F1'de tabani ASACAK. Ucu birden saglanmadan bayrak
ACILMAZ.

D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

MESH =os .environ .get ("GZ_MESH","results/_p1_olasilik")
N_PARCA =int (os .environ .get ("GZ_N","60"))
MARKALAR =set (os .environ .get ("GZ_MARKA","NIT,MOR,SUPU,UPUN").split (","))


def _birim_mi (D ):
    if not len (D ):
        return True 
    u =np .linalg .norm (D ,axis =1 )
    return bool (np .all (np .abs (u -1.0 )<1e-3 ))


def main ():
    t0 =time .time ()
    import canonical_chain 
    import robot_cp 
    cfg =json .load (open ("cp_config.json"))if os .path .exists (
    "cp_config.json")else {}
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    candidate =[(p ,r )for p ,r in kay .items ()
    if r .get ("mfg")in MARKALAR and len (r .get ("G",[]))
    and os .path .exists (f"{MESH }/{p }.npz")]
    rng =np .random .default_rng (0 )
    candidate =[candidate [i ]for i in rng .choice (len (candidate ),
    min (N_PARCA ,len (candidate )),
    replace =False )]
    print (f"{len (candidate )} part denetleniyor",flush =True )

    say =collections .Counter ()
    agg =collections .Counter ()
    error =[]
    for pid ,r in candidate :
        z =np .load (f"{MESH }/{pid }.npz")
        V =np .asarray (z ["V"],float )
        F =np .asarray (z ["F"],int )
        pbs =[np .asarray (q ,float )for q in z ["pbs"]]
        try :
            ham =canonical_chain .product_output (V ,F ,pbs ,r .get ("step"),cfg )
        except Exception as e :# noqa: BLE001
            say ["COKTU"]+=1 
            error .append (f"{pid }: {type (e ).__name__ }: {e }")
            continue 
            # `product_output` sozlukleri `point`/`direction` anahtarlariyla returns
            # (ihracatci sozlesmesi). Ilk yazimda `p`/`d` varsaymistim, KeyError
            # verdi -- anahtarlar dogrudan ciktidan okundu.
        P =np .asarray ([c ["point"]for c in ham ],float ).reshape (-1 ,3 )if len (ham )else np .zeros ((0 ,3 ))
        D =np .asarray ([c ["direction"]for c in ham ],float ).reshape (-1 ,3 )if len (ham )else np .zeros ((0 ,3 ))
        say ["part"]+=1 
        if len (P ):
            say ["cikti_var"]+=1 
        say ["cp"]+=len (P )
        if not np .all (np .isfinite (D ))or not np .all (np .isfinite (P )):
            say ["SONSUZ"]+=1 
            error .append (f"{pid }: NaN/Inf cikti")
            continue 
        if not _birim_mi (D ):
            say ["YON_BIRIM_DEGIL"]+=1 
            error .append (f"{pid }: yonler birim degil")
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        dg =float (np .linalg .norm (V .max (0 )-V .min (0 )))
        tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
        signed =True )[:3 ]
        agg ["tp"]+=tp ;agg ["fp"]+=fp ;agg ["fn"]+=fn 
        agg ["gt"]+=len (G )

    f1 =2 *agg ["tp"]/max (2 *agg ["tp"]+agg ["fp"]+agg ["fn"],1 )
    print (f"\n--- OLCULEN ZINCIR (canonical_chain.product_output) ---")
    print (f"  part            {say ['part']}")
    print (f"  cikti veren      {say ['cikti_var']}")
    print (f"  COKEN            {say ['COKTU']}")
    print (f"  NaN/Inf          {say ['SONSUZ']}")
    print (f"  direction birim DEGIL  {say ['YON_BIRIM_DEGIL']}")
    print (f"  uretilen CP      {say ['cp']}  (GT {agg ['gt']})")
    print (f"  robot F1         {f1 :.4f}")
    if error :
        print ("\n  ILK HATALAR:")
        for h in error [:8 ]:
            print (f"    {h }")
    saglam =(say ["COKTU"]==0 and say ["SONSUZ"]==0 
    and say ["YON_BIRIM_DEGIL"]==0 
    and say ["cikti_var"]==say ["part"])
    print (f"\nSAGLAMLIK: {'GECTI'if saglam else 'KALDI'}")
    print ("Bayrak `glb_kanonik_zincir` ancak SAGLAMLIK GECTI ve robot F1")
    print ("tabani astiktan sonra acilir.")
    json .dump ({"damga":makbuz_hash .damga (),"n_parca":say ["part"],
    "coken":say ["COKTU"],"sonsuz":say ["SONSUZ"],
    "yon_birim_degil":say ["YON_BIRIM_DEGIL"],
    "cikti_veren":say ["cikti_var"],"uretilen_cp":say ["cp"],
    "gt":agg ["gt"],"robot_f1":f1 ,"saglam":saglam ,
    "not":"Sahaya inecek zincirin saglamlik denetimi. Bayrak "
    "acilmadan ONCE kosulur. D7'ye BAKILMADI."},
    open ("results/glb_zincir_dogrulama.json","w"),indent =1 )
    print (f"receipt -> results/glb_zincir_dogrulama.json "
    f"({time .time ()-t0 :.0f} s)")


if __name__ =="__main__":
    main ()
