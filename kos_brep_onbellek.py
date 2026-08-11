# -*- coding: utf-8 -*-
"""B-rep onbellekleri (silindir + aciklik) EGITIM korpusu icin -- yeniden baslatilabilir.

`p3c_eksen_secici.silindir_onbellek` ayni isi yapiyor ama basarisiz parcayi
sessizce bos liste yaziyor; burada BASARISIZLIK SAYILIR ve makbuza yazilir
(bos havuz ile cikarilamayan parca AYNI SEY DEGIL).

Egitim ve sinav havuzlari AYNI bicimde kurulmali: D7 tarafinda hem silindir hem
aciklik var, o yuzden egitim tarafinda da IKISI de uretilir.
"""
import glob, json, os, pickle, sys, time
sys.path.insert(0, ".")
os.environ.setdefault("BA_ALLOW_SEEN", "1")
import brep_snap, brep_aciklik
from korpus_kimlik import step_kimlik as SK

KUME = os.environ.get("BREP_KUME", "results/brep_egitim_kumesi.json")
ON = os.environ.get("BREP_ON", "_brepegit")
ISLER = [("silindir", f"results/{ON}_silindirler.pkl",
          lambda p: brep_snap.exact_cylinders(p)),
         ("aciklik", f"results/{ON}_acikliklar.pkl",
          lambda p: brep_aciklik.acikliklar(p))]
S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
pidler = [str(p) for p in json.load(open(KUME))["pidler"]]
print(f"kume {len(pidler)} | STEP eslesen {sum(1 for p in pidler if p in S)}", flush=True)

for ad, yol, fn in ISLER:
    ob, hata = {}, {}
    if os.path.exists(yol):
        with open(yol, "rb") as f:
            ob = pickle.load(f)
    eksik = [p for p in pidler if p not in ob and p in S]
    print(f"\n{ad}: onbellekte {len(ob)} | cikarilacak {len(eksik)}", flush=True)
    t0 = time.time()
    for i, p in enumerate(eksik, 1):
        try:
            ob[p] = fn(S[p])
        except Exception as e:                    # yutulmaz: SAYILIR ve raporlanir
            ob[p] = []
            hata[p] = f"{type(e).__name__}: {e}"[:200]
        if i % 100 == 0 or i == len(eksik):
            with open(yol, "wb") as f:
                pickle.dump(ob, f)
            print(f"  {ad} {i}/{len(eksik)}  {(time.time()-t0)/i:.2f}s/parca  "
                  f"hata {len(hata)}", flush=True)
    with open(yol, "wb") as f:
        pickle.dump(ob, f)
    bos = sum(1 for p in ob if not ob[p])
    print(f"{ad} BITTI: {len(ob)} parca | BOS {bos} | HATA {len(hata)}", flush=True)
    json.dump({"n": len(ob), "bos": bos, "hata_sayisi": len(hata),
               "hatalar": dict(list(hata.items())[:40])},
              open(f"results/brep_onbellek_{ad}.json", "w"), indent=1)
