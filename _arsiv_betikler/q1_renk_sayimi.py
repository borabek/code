# -*- coding: utf-8 -*-
"""Q1: STEP yuz-basina RENK kanali BOS mu?

BAGLAM: kayitli bulgu, AP214 renginin yuz basina ayristirilabildigi (OVER_RIDING_STYLED_ITEM,
108/108 yuz) ama o parcada TUM yuzlerin ayni gumusi rengi tasidigi. Entegrasyon da bloke:
gmsh rengi atiyor, sira-eslemesi null testte cakti, XCAF 0 alt-sekil etiketi veriyor.

SIRALAMA MANTIGI: "yuzleri geometriyle eslestiren bir matcher yaz" pahali bir istir. Ondan ONCE
kanalda BILGI olup olmadigi olculur -- eslestirmeye gerek YOK, cunku bir parcadaki TUM yuzler
ayni renkteyse hangi yuzun hangi renk oldugu zaten onemsizdir.

KILL (onceden yazildi): parcalarin %10'undan azinda >=2 ayrik renk varsa kanal BOS ilan edilir
ve matcher isi ACILMAZ. >=2 renk yayginsa, renk metal kontagi plastik govdeden ayiriyor olabilir
-- o zaman matcher mesru bir yatirimdir.
"""
import os, sys, re, json, glob, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# AP214 renk tanimi: COLOUR_RGB('',r,g,b)
RGB = re.compile(rb"COLOUR_RGB\s*\(\s*'[^']*'\s*,\s*([-0-9.E+]+)\s*,\s*([-0-9.E+]+)\s*,"
                 rb"\s*([-0-9.E+]+)\s*\)", re.I)
STYLED = re.compile(rb"STYLED_ITEM|OVER_RIDING_STYLED_ITEM", re.I)


def renkler(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    out = []
    for m in RGB.finditer(raw):
        try:
            out.append(tuple(round(float(x), 3) for x in m.groups()))
        except ValueError:
            continue
    return out, len(STYLED.findall(raw))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    files = sorted(glob.glob("all_wscad_stp/*.stp"))[:n]
    print(f"{len(files)} STEP taraniyor...", flush=True)
    cok, hic, tek = 0, 0, 0
    hist = collections.Counter()
    styled_yok = 0
    ornek = []
    for f in files:
        try:
            cs, ns = renkler(f)
        except Exception:
            continue
        u = sorted(set(cs))
        hist[len(u)] += 1
        if ns == 0:
            styled_yok += 1
        if len(u) == 0:
            hic += 1
        elif len(u) == 1:
            tek += 1
        else:
            cok += 1
            if len(ornek) < 6:
                ornek.append((os.path.basename(f), u[:4]))
    tot = max(hic + tek + cok, 1)
    print(f"\n{'renk sayisi':<16}{'parca':>8}{'oran':>9}")
    for k in sorted(hist):
        print(f"{k:<16}{hist[k]:>8}{hist[k]/tot:>9.3f}")
    print(f"\nhic renk yok      : {hic:>5} ({hic/tot:.3f})")
    print(f"tek renk          : {tek:>5} ({tek/tot:.3f})")
    print(f">=2 ayrik renk    : {cok:>5} ({cok/tot:.3f})   <- KANALIN BILGISI BURADA")
    print(f"STYLED_ITEM yok   : {styled_yok:>5}")
    if ornek:
        print("\ncok renkli ornekler:")
        for a, b in ornek:
            print(f"  {a}: {b}")
    oran = cok / tot
    print(f"\nKILL (onceden yazili): >=2 renkli parca orani <0.10 ise kanal BOS -> "
          f"{oran:.3f} => {'KANAL CANLI, matcher mesru' if oran >= 0.10 else 'KANAL BOS, ACMA'}")
    json.dump({"n": tot, "hic": hic, "tek": tek, "cok": cok, "oran_cok": oran,
               "styled_yok": styled_yok,
               "karar": "CANLI" if oran >= 0.10 else "BOS"},
              open("results/q1_renk_sayimi.json", "w"), indent=1)
    print("makbuz -> results/q1_renk_sayimi.json")


if __name__ == "__main__":
    main()
