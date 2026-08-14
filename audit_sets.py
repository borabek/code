# -*- coding: utf-8 -*-
"""Guard against the 'test set silently changed mid-experiment' bug (bit us twice: dun gece the
fair comparison drifted 182->186 as SIE downloaded; and seed2_wei lost 1 part to a transient remesh
fail). Any two big_arbiter runs compared for a DECISION must be ten identical part sets. Run this
before trusting a comparison.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe audit_sets.py tagA tagB ...
"""
import sys ,json 
def parts (tag ):
    try :return frozenset (p ['part_id']for p in json .load (open (f'results/big_arbiter_{tag }.json'))['per_part'])
    except Exception :return None 
tags =sys .argv [1 :]
sets ={t :parts (t )for t in tags if parts (t )is not None }
if len (sets )<2 :
    print ("at least 2 gecerli tag gerekli");sys .exit (1 )
ref =next (iter (sets .values ()))
allsame =True 
for t ,s in sets .items ():
    ok =s ==ref 
    allsame &=ok 
    print (f"  {t :22s} {len (s ):4d} part  {'AYNI'if ok else 'FARKLI -> KARSILASTIRMA GECERSIZ'}")
common =set .intersection (*[set (s )for s in sets .values ()])
print (f"\n  ortak part: {len (common )}")
print ("  "+("HEPSI AYNI SET -- karsilastirma gecerli"if allsame else 
f"UYARI: setler farkli. Ortak {len (common )} parcaya indirgeyerek karsilastir."))
