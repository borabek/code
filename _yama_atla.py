# -*- coding: utf-8 -*-
"""g1_egitim_turet.py'ye PATOLOJIK PARCA atlama listesi ekler.

2026-08-03: parca 2502740000 (WEI, STEP 1.2 MB) 16+ dakika %100 CPU harcadi ve tum
kosuyu rehin aldi. Kilitlenme DEGILDI -- CPU ilerliyordu, yani geometride patoloji
(ortalama parca 3.5 sn; bu 170 kat aykiri). `big_arbiter` da ayni sebeple --skip-parts
bayragi tasiyor.
"""
import io

P = "g1_egitim_turet.py"
s = io.open(P, encoding="utf-8", newline="").read()


def yama(old, new, s):
    assert old in s, "KALIP YOK: " + repr(old[:70])
    return s.replace(old, new, 1)


s = yama(
    'ARA = "results/_g1_ara.pkl"\n',
    'ARA = "results/_g1_ara.pkl"\n'
    '# PATOLOJIK PARCALAR: tek bir parca tum kosuyu rehin alabiliyor. 2026-08-03\'te\n'
    '# 2502740000 (WEI, STEP 1.2 MB) 16+ dakika %100 CPU harcadi -- KILITLENME DEGIL\n'
    '# (CPU ilerliyordu), geometride patoloji: ortalama parca 3.5 sn, bu 170 kat aykiri.\n'
    '# `big_arbiter` da ayni sebeple --skip-parts bayragi tasir. Atlanan parca makbuza yazilir.\n'
    'ATLA = {"2502740000"}\n', s)

s = yama(
    '        try:\n            j = json.load(io.open(jf, encoding="utf-8-sig"))\n',
    '        if pid in ATLA:\n'
    '            print(f"    {pid}: ATLANDI (patolojik -- bkz. ATLA)", flush=True)\n'
    '            continue\n'
    '        try:\n            j = json.load(io.open(jf, encoding="utf-8-sig"))\n', s)

s = yama(
    'json.dump({"ckpt": CKS, "n_parca": ok, "n_aday": len(Y),',
    'json.dump({"ckpt": CKS, "n_parca": ok, "n_aday": len(Y), "atlanan": sorted(ATLA),', s)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("atlama listesi eklendi: 2502740000")
