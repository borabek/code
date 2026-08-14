"""Pre-flight gate for v31: the run is a one-shot, so every check that CAN be made
before the GPU starts IS made here.

Checks, in order of how badly they would burn us:

 1. LEAK: the same catalog number downloaded twice (208 of them) must not end up
    ten both sides of the split -- that inflates val F1 and would fake a pass.
 2. LABEL QUALITY ON NEW VENDORS: the CAD labeller was tuned ten Phoenix Contact.
    If Weidmueller / the other vendors are drawn differently, their labels are
    junk and the model both learns AND is scored ten junk. Measured as the
    GT-decode ceiling per vendor + the zero-CP rate per vendor.
 3. ACCESSORY RESIDUE: an accessory (cover, jumper) has no wire entry, so it
    shows up as zero-CP or as a wild CP count. Flags families to inspect.
 4. DISK: does the run have room for prep cache + checkpoints?
"""
import argparse 
import glob 
import hashlib 
import json 
import os 
import re 
import shutil 
from collections import Counter ,defaultdict 

import numpy as np 

import json_dataset as jd 
import cp_targets as ct 
import metrics as mcp 


def vendor_of (part_nr ):
    m =re .search (r"wscaduniverse_([0-9A-Za-z-]+)_",str (part_nr ))
    if not m :
        return "human-GT"
    art =m .group (1 )
    if len (art )>=9 :
        return "Weidmueller"
    if art .startswith (("3","0"))and len (art )==7 :
        return "PhoenixContact"
    return "other"


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--corpus",default ="wscad_corpus_v6")
    ap .add_argument ("--pool",default ="all_wscad_stp")
    args =ap .parse_args ()

    print ("="*62 )
    print ("V31 ON-UCUS KONTROLU")
    print ("="*62 )

    # ---------- 1) duplicate-catalog leak ----------
    cats =defaultdict (list )
    for f in glob .glob (os .path .join (args .corpus ,"*.json")):
        m =re .search (r"wscaduniverse_([0-9A-Za-z-]+)_",os .path .basename (f ))
        if m :
            cats [m .group (1 )].append (f )
    dup_cats ={c :fs for c ,fs in cats .items ()if len (fs )>1 }
    print (f"\n[1] KOPYA SIZINTISI")
    print (f"    corpus'ta ayni katalogdan >1 part: {len (dup_cats )}")
    if dup_cats :
        print (f"    !! bunlar train/val'e AYRI dusebilir -> sahte val skoru")
        print (f"    ornek: {list (dup_cats )[:5 ]}")
    else :
        print (f"    OK -- geometrik dedup hepsini elemis")

        # ---------- 2/3) label quality + zero-CP, per vendor ----------
    print (f"\n[2] ETIKET KALITESI (GT-decode tavani) + SIFIR-CP, URETICI BAZINDA")
    by_vendor =defaultdict (lambda :{"n":0 ,"zero":0 ,"cps":0 ,
    "tp":0 ,"fp":0 ,"fn":0 })
    fam_zero =Counter ()
    for part in jd .iter_parts (args .corpus ):
        v =vendor_of (part .part_nr )
        d =by_vendor [v ]
        d ["n"]+=1 
        _ ,pts ,dirs =jd .dedup_connection_points (part )
        if not len (pts ):
            d ["zero"]+=1 
            fam_zero [v ]+=1 
            continue 
        d ["cps"]+=len (pts )
        # ceiling: decode the GROUND TRUTH target -- anything lost here is lost
        # to every model that will ever be trained ten this corpus
        tgt ,_ ,_ =ct .encode_targets (part .vertices ,pts ,dirs )
        preds =ct .decode_predictions (part .vertices ,tgt ,heatmap_thresh =0.3 ,
        nms_radius_mm =5.0 ,min_votes =1 )
        P =np .array ([p ["point"]for p in preds ])if preds else np .zeros ((0 ,3 ))
        matches ,un_p ,un_g =mcp .match_predictions (P ,np .asarray (pts ),5.0 )
        d ["tp"]+=len (matches );d ["fp"]+=len (un_p );d ["fn"]+=len (un_g )

    print (f"    {'manufacturer':<16}{'part':>7}{'sifir-CP':>10}{'CP/part':>10}{'TAVAN F1':>10}")
    for v ,d in sorted (by_vendor .items (),key =lambda kv :-kv [1 ]["n"]):
        tp ,fp ,fn =d ["tp"],d ["fp"],d ["fn"]
        f1 =2 *tp /(2 *tp +fp +fn )if (tp +fp +fn )else 0.0 
        pos =d ["n"]-d ["zero"]
        cpp =d ["cps"]/pos if pos else 0 
        flag ="  <-- SUPHELI"if (f1 <0.97 and pos >20 )or (d ["zero"]/max (d ["n"],1 )>0.15 )else ""
        print (f"    {v :<16}{d ['n']:>7}{d ['zero']:>10}{cpp :>10.1f}{100 *f1 :>9.1f}%{flag }")

        # ---------- 4) disk ----------
    free =shutil .disk_usage (".").free /(1 <<30 )
    print (f"\n[3] DISK: {free :.1f} GB bos")
    if free <6 :
        print ("    !! RISKLI -- prep-cache + checkpoint for at least 6GB oneriliyor")
    else :
        print ("    OK")

    n_parts =sum (d ["n"]for d in by_vendor .values ())
    print (f"\n[4] CORPUS: {n_parts } part")
    print ("="*62 )


if __name__ =="__main__":
    main ()
