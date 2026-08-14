"""Mine vendor catalog PDFs for TERMINAL BLOCK article numbers only.

The first version of this took every 7-digit number ten any line and shipped 612
accessories (end covers 'D', jumpers 'FBS', partition plates 'ATP', shield clamps
'SK', bridge sets 'PAI/PS/DP') into the download list -- parts with no wire entry
at all, which would have taught the model to find connection points ten a cover.

The catalog rows are 'TYPE  ARTICLE  QTY' (sometimes twice per line), so the type
sits immediately BEFORE its article number. We parse that pair and keep the row
only when the type's FIRST TOKEN is a known terminal-block family. Checking the
first token matters: 'AGK 10-UKH 50' is an end cover FOR a UKH block, not a block.
"""
import argparse 
import os 
import re 

import pypdf 

# families that ARE terminal blocks (a wire goes into them)
TERMINAL ={
# Phoenix screw / universal
"UT","UK","UKH","USLKG","UKK","UDK","URTK","UKM","MBK","UTI","UTN",
"UIK","USST","UTTB","UKKB","UTME","UTTB","UKN","USED","UKD",
# Phoenix push-in / spring
"PT","PTI","PTIO","PTTB","PTTBS","PTS","PTU","PTC","PTME","PTMEB",
"ST","STTB","STTBS","STS","STME","STI","STIO",
# fuse / disconnect / special
"FT","FTTB","FTTBS","UKSI","USI","FBSI","UT4HESI","STSI",
# miniature / others in the corpus
"MSB","MSDB","MPT","MUT","MT","QTC","QTCU","RTO","BT","BTO","BTH",
"BTP","PTPOWER","UKH240",
# Weidmueller
"WDU","WPE","ZDU","ZPE","SAK","AKZ","WDK","WTR","ZDK","WDL","WDT",
"ZDL","WFF","WSI","ZSI",
}

# explicit accessory families (belt and braces -- anything not in TERMINAL is
# dropped anyway, but naming them makes the report readable)
ACCESSORY ={
"D","FBS","FBSR","FBST","FBI","AGK","ATP","PAI","PS","PSBJ","DP",
"SK","SKS","AB","RB","CLIPFIX","E","ZB","UBE","KLM","BST","C-ME",
"EB","ESL","ZFM","ZQV","WAP","WQV","ZEW","ZAD","ZTH",
}

ROW =re .compile (r"([A-Z][A-Za-z0-9][A-Za-z0-9,\.\-/ ]{0,24}?)\s+(3[0-9]{6}|1[0-9]{9})\b")


def first_token (t ):
    t =re .sub (r"[-_]select$","",t .strip (),flags =re .I )
    tok =re .split (r"[\s_]",t )[0 ].upper ()
    return re .sub (r"[^A-Z0-9\-]","",tok )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--pdfs",nargs ="+",required =True )
    ap .add_argument ("--have",default ="_mevcut_katalog_numaralari.txt")
    ap .add_argument ("--guard",default ="pxc_out_of_scope.txt")
    ap .add_argument ("--pool",default ="all_wscad_stp")
    ap .add_argument ("--out",required =True )
    args =ap .parse_args ()

    have =set ()
    if os .path .exists (args .have ):
        have |={l .strip ()for l in open (args .have ,encoding ="utf-8")if l .strip ()}
        # whatever is physically in the pool right now (incl. today's downloads)
    if os .path .isdir (args .pool ):
        for f in os .listdir (args .pool ):
            m =re .search (r"wscaduniverse_([0-9A-Za-z-]+)_",f )
            if m :
                have .add (m .group (1 ))
    guard =set ()
    if os .path .exists (args .guard ):
        for line in open (args .guard ,encoding ="utf-8"):
            m =re .match (r"\s*PXC\.([0-9A-Za-z-]+)",line )
            if m :
                guard .add (m .group (1 ))

    from collections import Counter 
    keep ,seen =[],set ()
    stats =Counter ()
    for pdf in args .pdfs :
        try :
            reader =pypdf .PdfReader (pdf )
        except Exception :# noqa: BLE001
            continue 
        for page in reader .pages :
            for line in (page .extract_text ()or "").splitlines ():
                for typ ,art in ROW .findall (line ):
                    fam =first_token (typ )
                    if art in seen :
                        continue 
                    if fam not in TERMINAL :
                        stats [f"aksesuar/unknown:{fam }"]+=1 
                        continue 
                    seen .add (art )
                    if art in have :
                        stats ["already elimizde"]+=1 
                        continue 
                    if art in guard :
                        stats ["kara listede"]+=1 
                        continue 
                    keep .append ((art ,f"{fam } ({typ .strip ()})"))
                    stats [f"KLEMENS:{fam }"]+=1 

    with open (args .out ,"w",encoding ="utf-8")as fh :
        for a ,t in keep :
            fh .write (f"{a }\t{t }\n")

    print (f"YENI KLEMENS: {len (keep )} -> {args .out }")
    print ("  already elimizde:",stats ["already elimizde"],
    "| kara listede:",stats ["kara listede"])
    print ("  aile dagilimi:",", ".join (
    f"{k .split (':')[1 ]}:{v }"for k ,v in stats .most_common (40 )
    if k .startswith ("KLEMENS:")))


if __name__ =="__main__":
    main ()
