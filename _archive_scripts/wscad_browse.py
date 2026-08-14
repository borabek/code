"""Harvest REAL terminal-block article numbers from WSCAD Universe's own search.

Each result card carries a product description ("Single Level Feed-through Terminal
Block, Screw Connection"), so a part is accepted ONLY when that description says
terminal block and does NOT say cover / bridge / jumper / plate / marker / clamp.
This is the gate that was missing when 612 accessories got downloaded from the
vendor PDFs -- those catalogs list a family's accessories next to its blocks, and
nothing in a bare article number tells them apart.

Parts already in the pool, and anything ten the out-of-scope guard list, are dropped
here so they are never fetched again.
"""
import argparse 
import os 
import re 

from playwright .sync_api import sync_playwright 

SEARCH =("https://www.wscaduniverse.com/search?searchText={q}&norm=IEC"
"&format=WSCAD&manufacturerIds={mids}&symbolsInTechnologies=&category="
"&limit=96&offset={off}")

IS_TERMINAL =re .compile (r"terminal\s*block",re .I )
IS_ACCESSORY =re .compile (
r"\b(cover|bridge|jumper|plate|marker|label|clamp|separator|partition|"
r"end\s*stop|end\s*bracket|shield|accessor|insert|screwdriver|tool|"
r"carrier|holder|rail|busbar|comb|test\s*plug|adapter)\b",re .I )


def pool_and_guard (pool_dir ,guard_file ):
    have =set ()
    if os .path .isdir (pool_dir ):
        for f in os .listdir (pool_dir ):
            m =re .search (r"wscaduniverse_([0-9A-Za-z-]+)_",f )
            if m :
                have .add (m .group (1 ).upper ())
    if os .path .exists (guard_file ):
        for line in open (guard_file ,encoding ="utf-8"):
            m =re .match (r"\s*PXC\.([0-9A-Za-z-]+)",line )
            if m :
                have .add (m .group (1 ).upper ())
    return have 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--query",required =True )
    ap .add_argument ("--mids",default ="")
    ap .add_argument ("--pages",type =int ,default =14 )
    ap .add_argument ("--profile",default ="_wscad_profile")
    ap .add_argument ("--pool",default ="all_wscad_stp")
    ap .add_argument ("--guard",default ="pxc_out_of_scope.txt")
    ap .add_argument ("--out",required =True )
    args =ap .parse_args ()

    have =pool_and_guard (args .pool ,args .guard )
    keep ,seen =[],set ()
    n_acc =n_have =0 

    with sync_playwright ()as p :
        ctx =p .chromium .launch_persistent_context (
        os .path .abspath (args .profile ),headless =True )
        page =ctx .pages [0 ]if ctx .pages else ctx .new_page ()

        for i in range (args .pages ):
            page .goto (SEARCH .format (q =args .query .replace (" ","%20"),
            mids =args .mids ,off =i *96 ),
            wait_until ="domcontentloaded",timeout =60000 )
            page .wait_for_timeout (3500 )
            cards =page .eval_on_selector_all (
            "a[href*='partNumber']",
            "els => els.map(e => ({h: e.getAttribute('href'), "
            "t: (e.innerText||'').trim()}))")
            # the card renders as two <a>: one empty (image), one with the text --
            # fold them onto the href so every part keeps its description
            desc ={}
            for c in cards :
                m =re .search (r"partNumber=([^&]+)",c ["h"]or "")
                if not m :
                    continue 
                art =m .group (1 )
                if c ["t"]:
                    desc [art ]=c ["t"]
                desc .setdefault (art ,"")

            new =0 
            for c in cards :
                m =re .search (r"partNumber=([^&]+)",c ["h"]or "")
                mm =re .search (r"manufacturerId=(\d+)",c ["h"]or "")
                if not m :
                    continue 
                art =m .group (1 )
                if art in seen :
                    continue 
                seen .add (art )
                d =desc .get (art ,"")
                if art .upper ()in have :
                    n_have +=1 
                    continue 
                if not IS_TERMINAL .search (d )or IS_ACCESSORY .search (d ):
                    n_acc +=1 
                    continue 
                keep .append ((art ,mm .group (1 )if mm else "?",d [:70 ]))
                new +=1 
            print (f"  s{i +1 }: +{new } klemens (total {len (keep )}; "
            f"elenen aksesuar {n_acc }, elimizde {n_have })",flush =True )
            if not cards :
                break 

        ctx .close ()

    with open (args .out ,"w",encoding ="utf-8")as fh :
        for art ,mid ,d in keep :
            fh .write (f"{art }\t{mid }\t{d }\n")
    print (f"SONUC: {len (keep )} yeni KLEMENS -> {args .out }")


if __name__ =="__main__":
    main ()
