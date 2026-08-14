import glob, os, pickle, sys, time, json
sys.path.insert(0, ".")
os.environ.setdefault("BA_ALLOW_SEEN", "1")
import brep_aciklik
from korpus_kimlik import step_kimlik as SK
# VARSAYILAN ARGUMAN TUZAGI: `acikliklar(..., esd_min=ESD_MIN)` varsayilani
# TANIM ANINDA baglar; modul sabitini sonradan degistirmek ETKISIZ. Bu yuzden
# parametreler ACIKCA gecirilir. (Ilk denemede etkisiz kalmisti ve aday sayisi
# BIREBIR ayni cikinca yakalandi.)
S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
pids = [str(p) for p in json.load(open("results/dusuk_marka_kumesi.json"))["pidler"]]
ob, hata = {}, 0
t0 = time.time()
for i, p in enumerate(pids, 1):
    if p not in S:
        continue
    try:
        ob[p] = brep_aciklik.acikliklar(S[p], esd_min=0.3, esd_max=25.0)
    except Exception as e:
        ob[p] = []; hata += 1
    if i % 100 == 0:
        print(f"  {i}/{len(pids)} {(time.time()-t0)/i:.2f}s/parca hata {hata}", flush=True)
pickle.dump(ob, open("results/_genisband_acikliklar.pkl", "wb"))
n = sum(len(v) for v in ob.values())
print(f"BITTI {len(ob)} parca | toplam aciklik {n} ({n/max(len(ob),1):.1f}/parca) | hata {hata}")
