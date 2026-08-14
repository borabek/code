# -*- coding: utf-8 -*-
"""Dosya adlarini cevir + TUM referanslari guncelle."""
import os, re, subprocess, sys, keyword
import importlib.util as _u
_s=_u.spec_from_file_location("p","_dosya_adlari.py"); _m=_u.module_from_spec(_s); _s.loader.exec_module(_m)
P=_m.PARCA

def yeni_ad(tab):
    parts=tab.split("_")
    out=[P.get(p,p) for p in parts]
    # ardisik tekrarlari sadelestir (run_run -> run)
    ded=[]
    for x in out:
        if not ded or ded[-1]!=x: ded.append(x)
    return "_".join(ded)

def main():
    uygula="--uygula" in sys.argv
    fs=[f for f in subprocess.run(["git","ls-files","*.py"],capture_output=True,text=True).stdout.split() if "/" not in f]
    mevcut={os.path.splitext(f)[0] for f in fs}
    stdlib=set(sys.stdlib_module_names) if hasattr(sys,"stdlib_module_names") else set()
    plan={}
    for f in fs:
        tab=os.path.splitext(f)[0]
        y=yeni_ad(tab)
        if y==tab: continue
        if y in mevcut or y in plan.values() or y in stdlib or keyword.iskeyword(y):
            continue                      # CAKISMA -> dokunma
        plan[tab]=y
    print(f"{len(fs)} dosya | {len(plan)} yeniden adlandirilacak")
    if not uygula:
        for k,v in list(plan.items())[:14]: print(f"  {k}.py -> {v}.py")
        print("(kuru kosum)"); return
    # 1) dosyalari tasi
    for k,v in plan.items():
        subprocess.run(["git","mv",f"{k}.py",f"{v}.py"],capture_output=True)
    # 2) TUM metin dosyalarinda modul adini guncelle
    D=re.compile(r"\b("+"|".join(re.escape(k) for k in sorted(plan,key=len,reverse=True))+r")\b")
    n=0
    for f in subprocess.run(["git","ls-files"],capture_output=True,text=True).stdout.split():
        if not f.endswith((".py",".sh",".md",".json",".txt",".yaml",".yml",".ini")): continue
        if not os.path.exists(f): continue
        try: s=open(f,encoding="utf-8",errors="ignore").read()
        except Exception: continue
        y=D.sub(lambda m:plan[m.group(1)],s)
        if y!=s:
            open(f,"w",encoding="utf-8").write(y); n+=1
    print(f"-> {len(plan)} dosya adi, {n} dosyada referans guncellendi")

main()
