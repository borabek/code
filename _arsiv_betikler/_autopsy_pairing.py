# -*- coding: utf-8 -*-
"""OTOPSI KANITI: Contact bolgeleri tel-girisiyle (CableEntry) EŞLEŞIYOR mu, otherwise YALNIZ mi?

Tez: Contact sinifi = 'Kontaktierung bzw. Werkzeugeinschub' = kontakt YA DA alet-sokma agzi (same sinif).
Hipotez: GERCEK tel-CP'de path = CableEntry (surface agzi) -> Contact (icerde). Bir TOOL agzi (Werkzeug-
einschub) whereas Contact'tir but yakininda CableEntry YOKTUR. Yani:
  Contact bileseni yakininda CableEntry VARSA -> real tel-baglantisi
  Contact bileseni YALNIZ (yakinda CableEntry absent) -> muhtemelen TOOL/actuator agzi -> CP DEGIL
Bu dogruysa: cp-v3'un extra-saymasi = only-Contact'lari CP sayması. Fix = Contact'i CableEntry with ESLE.
Sadece label geometrisi (GPU absent). Korpus + insan etiketleri ten olcer.
"""
import os ,sys ,numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import connector3d ,train_seg_extra as T 
import scheffler_dataset as ds 
from derive_3d_boxes import components 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )


def comp_centroids (V ,F ,L ,cls ,min_v =8 ):
    return [V [c ].mean (0 )for c in components (V ,F ,np .isin (L ,(cls ,)))if len (c )>=min_v ]


def main ():
    samples =[]
    for sp in ("train","val"):
        for s in ds .load_split ("wscad_corpus_scheffler_exact",sp ,verify_hashes =False ):
            samples .append (s )
            # insan kismi etiketleri connection-kanali (CE/CT ayrimi absent) -> otopsi for KORPUS'u kullan (full 5-sinif)
    print (f"{len (samples )} tam-etiketli corpus parcasi (5-sinif)",flush =True )

    tot_ce =tot_ct =paired =lonely =0 
    per_part =[]
    dists =[]
    for s in samples :
        try :
            V =np .asarray (s ["verts"],float );F =np .asarray (s ["faces"],np .int64 );L =np .asarray (s ["labels"],np .int64 )
            ce =comp_centroids (V ,F ,L ,CE );ct =comp_centroids (V ,F ,L ,CT )
            tot_ce +=len (ce );tot_ct +=len (ct )
            diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))
            thr =max (6.0 ,0.10 *diag )# 'yakin' esigi
            p =l =0 
            for c in ct :
                if ce :
                    dmin =min (np .linalg .norm (c -e )for e in ce )
                    dists .append (dmin )
                    if dmin <=thr :p +=1 
                    else :l +=1 
                else :
                    l +=1 
            paired +=p ;lonely +=l 
            per_part .append ((s .get ("part_id","?"),len (ce ),len (ct ),p ,l ))
        except Exception :
            continue 

    print (f"\n=== OTOPSI: Contact <-> CableEntry eslesmesi ===")
    print (f"  toplam CableEntry bileseni: {tot_ce }")
    print (f"  toplam Contact bileseni:    {tot_ct }")
    print (f"  Contact YAKININDA CableEntry VAR (gercek baglanti):  {paired } ({100 *paired /max (tot_ct ,1 ):.0f}%)")
    print (f"  Contact YALNIZ (muhtemel TOOL/actuator agzi):        {lonely } ({100 *lonely /max (tot_ct ,1 ):.0f}%)")
    if dists :
        d =np .array (dists )
        print (f"  Contact->en yakin CableEntry mesafesi: medyan {np .median (d ):.1f}mm  (yakinlar gercek, uzaklar tool)")
    print (f"\n  Contact/CableEntry orani: {tot_ct /max (tot_ce ,1 ):.2f}x  (>1.5 ise Contact'lar tel-girislerinden FAZLA = tool aglari)")
    # at most only-Contact'li parts
    per_part .sort (key =lambda r :-r [4 ])
    print ("\n  at most YALNIZ-Contact'li 8 part (pid, #CE, #CT, esli, only):")
    for pid ,nce ,nct ,p ,l in per_part [:8 ]:
        print (f"    {pid }: CE {nce }  CT {nct }  esli {p }  yalniz {l }")


if __name__ =="__main__":
    main ()
