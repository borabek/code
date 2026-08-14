# -*- coding: utf-8 -*-
"""g1_training_derive.py'ye PATOLOJIK PARCA atlama listesi adds.

2026-08-03: part 2502740000 (WEI, STEP 1.2 MB) 16+ dakika %100 CPU harcadi and tum
kosuyu rehin aldi. Kilitlenme DEGILDI -- CPU ilerliyordu, i.e. geometride patoloji
(mean part 3.5 sn; this 170 fold aykiri). `big_arbiter` da same sebeple --skip-parts
bayragi tasiyor.
"""
import io 

P ="g1_training_derive.py"
s =io .open (P ,encoding ="utf-8",newline ="").read ()


def patch (old ,new ,s ):
    assert old in s ,"KALIP YOK: "+repr (old [:70 ])
    return s .replace (old ,new ,1 )


s =patch (
'ARA = "results/_g1_ara.pkl"\n',
'ARA = "results/_g1_ara.pkl"\n'
'# PATOLOJIK PARCALAR: single a part tum kosuyu rehin alabiliyor. 2026-08-03\'te\n'
'# 2502740000 (WEI, STEP 1.2 MB) 16+ dakika %100 CPU harcadi -- KILITLENME DEGIL\n'
'# (CPU ilerliyordu), geometride patoloji: mean part 3.5 sn, this 170 fold aykiri.\n'
'# `big_arbiter` da same sebeple --skip-parts bayragi carries. Atlanan part makbuza yazilir.\n'
'ATLA = {"2502740000"}\n',s )

s =patch (
'        try:\n            j = json.load(io.open(jf, encoding="utf-8-sig"))\n',
'        if pid in ATLA:\n'
'            print(f"    {pid}: ATLANDI (patolojik -- see. ATLA)", flush=True)\n'
'            continue\n'
'        try:\n            j = json.load(io.open(jf, encoding="utf-8-sig"))\n',s )

s =patch (
'json.dump({"ckpt": CKS, "n_parca": ok, "n_aday": len(Y),',
'json.dump({"ckpt": CKS, "n_parca": ok, "n_aday": len(Y), "skipped": sorted(ATLA),',s )

io .open (P ,"w",encoding ="utf-8",newline ="").write (s )
print ("atlama listesi eklendi: 2502740000")
