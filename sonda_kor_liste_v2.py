# -*- coding: utf-8 -*-
"""KOR NOKTA LISTESI v2 -- insan emegine gore siralanmis.

v1 KUSURLUYDU ve kullanici yakaladi: liste `(eslesme, -GT_sayisi)` ile
siralanmisti, yani en cok CP'li parcalar ONE geliyordu (ilk bes parca 60-64 CP'li
VTK cok katli klemens -- biri bile saatler alir). Ayrica geometri anahtari
30/30 "farkli" dese de parcalar AYNI URUN AILESININ varyantlariydi
(VTK-5MD-16P / 5ME-16P / 5MD-15P): olcu farkli, BILGI ayni.

v2 kurallari:
  * GT sayisi [2, 12] -- etiketlemesi HIZLI, ogrenme sinyali yine yeterli
  * AILE basina en fazla 1 parca (ad kokunden: rakam bloklari silinir)
  * marka basina en fazla `MARKA_TAVANI`
  * eslesme orani dusuk olan once (otomatigin en kor oldugu yer)
"""
import collections, json, os, pickle, re, sys
import numpy as np
sys.path.insert(0, ".")
os.environ.setdefault("BA_ALLOW_SEEN", "1")
import p2_brep_agiz_etiket as P2
import d6_kayit, kanonik_d7 as K

GT_ALT, GT_UST = 2, 12
MARKA_TAVANI = 6
HEDEF = 24


def aile(pid):
    """Urun ailesi kokunu cikar: rakam bloklari ve ayraclar silinir."""
    return re.sub(r"[0-9]+", "#", str(pid)).strip("-_. ").upper()


isler = [("korpus", "results/_brepegit_silindirler.pkl",
          "results/_brepegit_acikliklar.pkl",
          K.yukle([str(p) for p in json.load(
              open("results/brep_egitim_kumesi.json"))["pidler"]])),
         ("d6", "results/_d6_silindirler.pkl", "results/_d6_acikliklar.pkl",
          d6_kayit.yukle(set(d6_kayit.sinav()["pidler"])))]
sat = []
for ad, cylf, acf, kay in isler:
    cy = pickle.load(open(cylf, "rb")); ac = pickle.load(open(acf, "rb"))
    for pid, r in kay.items():
        G = np.asarray(r.get("G", []), float)
        if not (GT_ALT <= len(G) <= GT_UST):
            continue
        Gd = np.asarray(r["Gd"], float)
        A = P2.agizlar(cy.get(str(pid)), ac.get(str(pid)))
        if not A:
            sat.append((str(pid), r["mfg"], len(G), 0.0)); continue
        M = np.asarray([a[0] for a in A], float)
        AX = np.asarray([a[1] for a in A], float)
        es = 0
        for j in range(len(G)):
            u = P2._birim(Gd[j])
            if u is None:
                continue
            w = G[j][None] - M
            e = np.einsum("ij,ij->i", w, AX)
            y = np.linalg.norm(w - e[:, None] * AX, axis=1)
            a2 = np.degrees(np.arccos(np.clip(np.abs(AX @ u), -1.0, 1.0)))
            if ((y <= P2.ESLES_YANAL) & (np.abs(e) <= P2.ESLES_EKSENEL) &
                    (a2 <= P2.ESLES_ACI)).any():
                es += 1
        sat.append((str(pid), r["mfg"], len(G), es / len(G)))

kor = [s for s in sat if s[3] < 0.2]
kor.sort(key=lambda x: (x[3], x[2]))          # once en kor, sonra en AZ CP'li
sec, gorulen_aile, marka = [], set(), collections.Counter()
for p, m, n, o in kor:
    a = aile(p)
    if a in gorulen_aile or marka[m] >= MARKA_TAVANI:
        continue
    sec.append((p, m, n, o)); gorulen_aile.add(a); marka[m] += 1
    if len(sec) >= HEDEF:
        break
print(f"kor havuz (GT {GT_ALT}-{GT_UST}, eslesme<0.2): {len(kor)}")
print(f"secilen {len(sec)} | farkli aile {len(gorulen_aile)} | "
      f"marka {dict(marka)}")
print(f"\n{'pid':<20} {'marka':<7} {'GT':>3} {'eslesme':>8}")
for p, m, n, o in sec:
    print(f"{p:<20} {m:<7} {n:>3} {o:>8.2f}")
print(f"\ntoplam etiketlenecek CP: {sum(x[2] for x in sec)} "
      f"(v1'de ilk 5 parca zaten {60+64+60+64+60})")
with open("results/p2_kor_v2.txt", "w") as f:
    for p, _, _, _ in sec:
        f.write(p + "\n")
json.dump({"secilen": [list(x) for x in sec], "kor_havuz": len(kor),
           "kural": {"gt": [GT_ALT, GT_UST], "aile_basina": 1,
                     "marka_tavani": MARKA_TAVANI},
           "not": "v1 en COK CP'liyi one koyuyordu (hata). v2 hizli etiketlenen, "
                  "aile-cesitli, markaya yayilmis parcalar."},
          open("results/p2_kor_v2.json", "w"), indent=1)
print("-> results/p2_kor_v2.txt")
