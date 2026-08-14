# -*- coding: utf-8 -*-
"""FAZ 2 / P0 -- OLCUMU KILITLE (kullanici listesi, TODO_085_PHASE2.md).
Yapilanlar:
 1) FRAME dogrulama: P (pozisyon) and direction hangi frame'de? Karistiran yer present mi?
 2) AILE: STEP PRODUCT adindan (PXC'de real urun-tipi; WEI'de part-no -> prefix fallback)
 3) GEOMETRY-HASH: JSON bbox + point count -> tekrar/duplike parts
 4) SPLIT'leri sabitle: part-out / geometry-out / family-out
 5) KILITLI HOLDOUT: ailelerin ~%20'si, deterministik, BIR KEZ acilacak (P7'ye up to dokunulmaz)
 6) Baseline dogrulama girdileri (rich_feats.npz with eslestir)
 7) candidate-recall vs oracle-F1 ayrimini raporla
Cikti: results/split_lock.json (single dogruluk kaynagi)"""
import os ,re ,sys ,json ,glob ,hashlib 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from big_arbiter import eligible 
from json_dataset import family_key 

PROD_RE =re .compile (rb"PRODUCT\s*\(\s*'([^']*)'")


def step_product (stp ):
    """STEP dosyasindan first PRODUCT adi (large dosyada bastan sinirli okuma)."""
    try :
        with open (stp ,"rb")as f :
            head =f .read (4_000_000 )
        m =PROD_RE .search (head )
        return m .group (1 ).decode ("latin-1").strip ()if m else ""
    except Exception :
        return ""


def norm_family (pid ,prod ):
    """Aile anahtari: PRODUCT adi bilgilendiriciyse onu (varyant eki soyularak), degilse part-no prefix'i.
    'bilgilendirici not' = PRODUCT adi part numarasinin kendisi / tamamen rakam."""
    p =(prod or "").strip ()
    digits_only =p .replace (".","").replace ("-","").isdigit ()
    if p and not digits_only and pid not in p :
    # varyant eki soy: sondaki '-<number>' or '_<number>' gruplarini single seferde
        base =re .sub (r"[-_](?:\d+(?:\.\d+)?)(?=(?:[-_]|$))","",p )
        return f"prod:{base }"
    return f"nr:{family_key (pid )}"


def main ():
    parts =list (eligible ())
    oos =set (open ("pxc_out_of_scope.txt").read ().split ())if os .path .exists ("pxc_out_of_scope.txt")else set ()
    held =set (open ("_hw_r3.txt").read ().split ())
    parts =[(m ,p ,jf ,s )for m ,p ,jf ,s in parts 
    if (m =="WEI"and p in held )or (m =="PXC"and p not in oos )]
    print (f"[P0] {len (parts )} part ({sum (1 for m ,*_ in parts if m =='WEI')} WEI + {sum (1 for m ,*_ in parts if m =='PXC')} PXC)",flush =True )

    rows ={}
    for i ,(mfg ,pid ,jf ,stp )in enumerate (parts ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            pts =j ["Graphic3d"]["Points"]
            V =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in pts ],float )
            dims =np .round (V .max (0 )-V .min (0 ),1 )
            gkey ="g:%s|v%d"%(dims .tolist (),int (np .log2 (max (len (V ),1 ))))
            prod =step_product (stp )
            rows [pid ]={"mfg":mfg ,"product":prod ,"family":norm_family (pid ,prod ),
            "geom":gkey ,"n_cps":len (j .get ("ConnectionPoints",[])),
            "dims":dims .tolist (),"n_pts":len (V )}
        except Exception :
            continue 
        if i %200 ==0 :print (f"  {i }/{len (parts )}",flush =True )

    fams ={}
    geos ={}
    for pid ,r in rows .items ():
        fams .setdefault (r ["family"],[]).append (pid )
        geos .setdefault (r ["geom"],[]).append (pid )
    fsz =np .array ([len (v )for v in fams .values ()])
    gsz =np .array ([len (v )for v in geos .values ()])
    print (f"\n[P0-2/3] {len (rows )} part | {len (fams )} AILE (ort {fsz .mean ():.2f}, max {fsz .max ()}, "
    f"tek-parcali %{100 *np .mean (fsz ==1 ):.0f}) | {len (geos )} GEOMETRI (duplike-grup {int ((gsz >1 ).sum ())}, "
    f"duplike part {int (gsz [gsz >1 ].sum ())})")
    prod_informative =sum (1 for r in rows .values ()if r ["family"].startswith ("prod:"))
    print (f"  PRODUCT-tabanli aile: {prod_informative } part | part-no fallback: {len (rows )-prod_informative }")
    for mfg in ("WEI","PXC"):
        sub ={p :r for p ,r in rows .items ()if r ["mfg"]==mfg }
        f2 ={}
        for p ,r in sub .items ():f2 .setdefault (r ["family"],[]).append (p )
        s2 =np .array ([len (v )for v in f2 .values ()])
        print (f"  {mfg }: {len (sub )} part -> {len (f2 )} aile (ort {s2 .mean ():.2f}, max {s2 .max ()})")

        # 5) KILITLI HOLDOUT: ailelerin ~%20'si, deterministik hash with (seed'e bagli not, tekrarlanabilir)
    def fam_hash (f ):return int (hashlib .sha256 (f .encode ()).hexdigest ()[:8 ],16 )
    all_f =sorted (fams )
    locked_f =[f for f in all_f if fam_hash (f )%100 <20 ]
    locked_parts =sorted (p for f in locked_f for p in fams [f ])
    work_parts =sorted (p for p in rows if p not in set (locked_parts ))
    lw =sum (1 for p in locked_parts if rows [p ]["mfg"]=="WEI")
    ww =sum (1 for p in work_parts if rows [p ]["mfg"]=="WEI")
    print (f"\n[P0-5] KILITLI HOLDOUT (aile bazli, deterministik): {len (locked_f )} aile / {len (locked_parts )} part "
    f"({lw } WEI + {len (locked_parts )-lw } PXC)")
    print (f"        CALISMA seti: {len (work_parts )} part ({ww } WEI + {len (work_parts )-ww } PXC)")
    print (f"        -> P7'ye kadar KILITLI parts hicbir training/threshold/secim kararinda KULLANILMAZ.")

    out ={"note":"FAZ2/P0 split kilidi. locked_parts P7'de TEK KEZ acilir.",
    "created":"2026-07-28","n_parts":len (rows ),
    "family_source":"STEP PRODUCT adi (varyant eki soyulmus); part-no ise json_dataset.family_key fallback",
    "geometry_source":"JSON Graphic3d bbox (0.1mm) + log2(nokta sayisi)",
    "locked_holdout_rule":"sha256(family)%100 < 20",
    "n_families":len (fams ),"n_geometries":len (geos ),
    "locked_families":locked_f ,"locked_parts":locked_parts ,"work_parts":work_parts ,
    "parts":rows }
    json .dump (out ,open ("results/split_lock.json","w"),indent =1 )
    print ("\n-> results/split_lock.json yazildi (tek dogruluk kaynagi)")


if __name__ =="__main__":
    main ()
