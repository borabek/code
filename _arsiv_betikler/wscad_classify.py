"""Ask WSCAD what each downloaded part ACTUALLY is, and delete anything that is
not a terminal block.

Why: the catalog PDFs I mined for article numbers list the whole product family --
terminal blocks AND their accessories (end covers 'D', jumpers 'FBS', partition
plates 'ATP', shield clamps 'SK', plug-in bridges 'PAI/PS/DP'). Those have no wire
entry at all, so labelling them would teach the model to hunt for connection points
ten a cover plate. The STEP header is unreliable (653 files carry no product name at
all -- a bare Creo export), so the authoritative source is WSCAD's own part page,
which states  Category / Subcategory / Part Type.

Writes a report and (with --delete) removes the non-terminal parts from the pool.
"""
import argparse 
import os 
import re 
import sys 
from collections import Counter 

from playwright .sync_api import sync_playwright 

PART =("https://www.wscaduniverse.com/part?manufacturerId={mid}&partNumber={art}"
"&format=WSCAD&norm=IEC")

# WSCAD category text that means "this is a terminal block we can train ten"
KEEP_CAT =re .compile (r"terminal",re .I )
# subcategories that are accessories even when the category says Terminals
DROP_SUB =re .compile (r"accessor|bridge|cover|marker|separat|clamp|plate|jumper|"
r"shield|end\s*stop|partition",re .I )


def classify (page ,art ,mid ,timeout =30000 ):
    page .goto (PART .format (mid =mid ,art =art ),wait_until ="domcontentloaded",
    timeout =timeout )
    page .wait_for_timeout (1200 )
    body =page .inner_text ("body")
    def field (label ):
        m =re .search (rf"{label }\s*\n\s*([^\n]+)",body )
        return (m .group (1 ).strip ()if m else "")
    return {"art":art ,"type":field ("Part Type:"),
    "cat":field ("Category:"),"sub":field ("Subcategory:")}


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--files",required =True ,help ="list of STEP paths")
    ap .add_argument ("--profile",default ="_wscad_profile")
    ap .add_argument ("--shard",type =int ,default =0 )
    ap .add_argument ("--nshards",type =int ,default =1 )
    ap .add_argument ("--out",required =True )
    args =ap .parse_args ()

    paths =[l .strip ()for l in open (args .files ,encoding ="utf-8")if l .strip ()]
    paths =[p for i ,p in enumerate (paths )if i %args .nshards ==args .shard ]

    rows =[]
    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (
        os .path .abspath (args .profile ),headless =True )
        page =ctx .pages [0 ]if ctx .pages else ctx .new_page ()
        for i ,path in enumerate (paths ,1 ):
            m =re .search (r"wscaduniverse_([0-9A-Za-z-]+)_",os .path .basename (path ))
            if not m :
                continue 
            art =m .group (1 )
            mid ="88"if len (art )>=9 else "63"# Weidmuller ids are long
            try :
                r =classify (page ,art ,mid )
            except Exception :# noqa: BLE001
                r ={"art":art ,"type":"","cat":"","sub":""}
            r ["path"]=path 
            rows .append (r )
            if i %25 ==0 :
                print (f"  [{args .shard }] {i }/{len (paths )}",flush =True )
        ctx .close ()

    with open (args .out ,"w",encoding ="utf-8")as fh :
        for r in rows :
            fh .write (f"{r ['path']}\t{r ['art']}\t{r ['type']}\t{r ['cat']}\t{r ['sub']}\n")
    print (f"[{args .shard }] bitti: {len (rows )} part -> {args .out }",flush =True )


if __name__ =="__main__":
    sys .exit (main ())
