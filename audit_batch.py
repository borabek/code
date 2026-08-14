"""Audit a fresh batch of vendor STEP files before it is allowed into the corpus.

Three gates, learned from the 2026-07-12 batch (1000 files: 79 were distribution
blocks -- not terminal blocks at all -- and would have taught the model to find
connection points ten parts the robot never wires):

  1. SCOPE      -- product families that are out of scope (distribution blocks,
                   manifolds, bridge/jumper accessories) are deleted and appended
                   to the re-download guard list.
  2. GUARD      -- anything already ten the guard list (a previously-rejected
                   catalog) is deleted ten sight.
  3. GEOMETRY   -- files with no usable 3D body (WSCAD sometimes serves an empty
                   record) are deleted.

Nothing here touches parts already in the corpus; it only filters the NEW files.
"""
import argparse 
import os 
import re 
import sys 
from collections import Counter 

# Families that are NOT terminal blocks (the robot never inserts a wire into
# these). Established by inspecting the vendor product names of the 4th batch.
OUT_OF_SCOPE ={
# distribution blocks (not terminal blocks -- no wire entry the robot uses)
"RBO",
"PTRV","PTRVB","FTRV","FTRVB",# potential distributors
"PTFIX",# fixed-bridge distribution blocks
# ACCESSORIES. These ride along in vendor catalogs next to the blocks they
# belong to, and 612 of them were downloaded before this list existed. A cover
# plate or a jumper has no wire entry at all, so labelling one teaches the
# model to hunt for connection points ten a part that has none.
"D","D-ST","D-PT","D-STS","D-RSC","D-ST25","D-ST4","D-STTB",
"D-UK","D-UT","D-STTB-4","D-ST-4-QUATTRO",# end covers
"FBS","FBSR","FBST","FBI","FBRN",# jumpers / plug-in bridges
"ATP","ATP-ST-TWIN","ATP-ST-QUATTRO",# partition plates
"PAI","PS","PSBJ","DP",# bridge / separator sets
"SK","SKS",# shield connection clamps
"AB","AB-PTI","AB-STI",# end brackets
"RB","AGK","AKG","C-ME","CLIPFIX",# reducing bridge, covers, clips
"EB","ZB","UBE","KLM","BST",# marker/label carriers
"ZFM","ZQV","WAP","WQV","ZEW","ZAD","ZTH",# Weidmueller accessories
}

PRODUCT =re .compile (r"PRODUCT\s*\(\s*'([^']*)'")
POINT =re .compile (r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)",re .S )
NUM =re .compile (r"-?[\d.]+(?:[eE][+-]?\d+)?")


def product_family (path ):
    try :
        with open (path ,errors ="ignore")as fh :
            head =fh .read (300000 )
    except OSError :
        return None ,None 
    m =PRODUCT .search (head )
    if not m :
        return None ,None 
    name =re .sub (r"[-_]select$","",m .group (1 ).strip (),flags =re .I )
    return name ,re .split (r"[_ ]",name )[0 ].upper ()


def has_body (path ,min_points =20 ):
    """True if the STEP carries a real 3D body (>= min_points 3D vertices).

    finditer, not findall: the old version materialised EVERY CARTESIAN_POINT in
    the file (a 500KB STEP has thousands) before checking the first one, so the
    early-exit below never actually saved anything. Same decision, same result --
    it just stops reading before the body is proven."""
    try :
        txt =open (path ,errors ="ignore").read ()
    except OSError :
        return False 
    pts =0 
    for m in POINT .finditer (txt ):
        if len (NUM .findall (m .group (1 )))==3 :
            pts +=1 
            if pts >=min_points :
                return True 
    return False 


def _classify (path ):
    """Per-file work (header read + geometry probe) -- pure, so it parallelises."""
    cat_m =re .search (r"wscaduniverse_([0-9A-Za-z-]+)_",os .path .basename (path ))
    cat =cat_m .group (1 )if cat_m else None 
    name ,fam =product_family (path )
    body =has_body (path )
    return path ,cat ,name ,fam ,body 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--files",required =True ,help ="text file: one STEP path per line")
    ap .add_argument ("--guard",default ="pxc_out_of_scope.txt")
    ap .add_argument ("--delete-out-of-scope",action ="store_true")
    args =ap .parse_args ()

    paths =[l .strip ()for l in open (args .files ,encoding ="utf-8")if l .strip ()]
    guard_ids =set ()
    if os .path .exists (args .guard ):
        for line in open (args .guard ,encoding ="utf-8"):
            m =re .match (r"\s*PXC\.([0-9A-Za-z-]+)",line )
            if m :
                guard_ids .add (m .group (1 ))

    drop_scope ,drop_guard ,drop_empty ,keep =[],[],[],[]
    fams =Counter ()
    # every gate below is a pure function of one file, so the SAME checks run in a
    # thread pool (the work is file-I/O + regex, both GIL-releasing enough to scale)
    # -- identical verdicts, a fraction of the wall clock ten a 1000-file batch.
    from concurrent .futures import ThreadPoolExecutor 
    workers =min (16 ,(os .cpu_count ()or 4 )*2 )
    with ThreadPoolExecutor (max_workers =workers )as pool :
        results =list (pool .map (_classify ,paths ))

    for p ,cat ,name ,fam ,body in results :
        fams [fam or "?"]+=1 
        if cat and cat in guard_ids :
            drop_guard .append ((p ,cat ,name ))
        elif fam in OUT_OF_SCOPE :
            drop_scope .append ((p ,cat ,name ))
        elif not body :
            drop_empty .append ((p ,cat ,name ))
        else :
            keep .append (p )

    print (f"yeni file: {len (paths )}")
    print (f"  KAPSAM DISI (dagitim blogu vb): {len (drop_scope )}")
    print (f"  YASAKLI LISTEDE (more before elenmis): {len (drop_guard )}")
    print (f"  BOS GEOMETRI: {len (drop_empty )}")
    print (f"  TEMIZ -> etiketlenecek: {len (keep )}")
    print ("\n  aile dagilimi (ilk 12):",
    ", ".join (f"{f }:{c }"for f ,c in fams .most_common (12 )))

    if args .delete_out_of_scope :
        with open (args .guard ,"a",encoding ="utf-8")as gh :
            if drop_scope :
                gh .write (f"\n# --- yeni parti denetimi ({len (drop_scope )} kapsam-disi) ---\n")
            for p ,cat ,name in drop_scope :
                if cat :
                    gh .write (f"PXC.{cat }   # {name }\n")
        for p ,_ ,_ in drop_scope +drop_guard +drop_empty :
            try :
                os .remove (p )
            except OSError :
                pass 
        print (f"\n  {len (drop_scope )+len (drop_guard )+len (drop_empty )} file SILINDI "
        f"(kapsam-disi olanlar koruma listesine yazildi)")

    if not keep :
        sys .exit ("HATA: temiz file kalmadi")


if __name__ =="__main__":
    main ()
