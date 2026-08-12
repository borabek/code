# -*- coding: utf-8 -*-
"""SABAH OZETI: gecenin butun makbuzlarini topla, ONCEDEN ILAN EDILEN kapilari
uygula, tek sayfa rapor yaz.

NEDEN. Gece programi onlarca log ve makbuz uretiyor. Sabah bunlari elle okumak
hem yavas hem de yanlis okumaya acik. Bu betik her makbuzu kapisiyla birlikte
gosterir ve KARARI yazar.

DUMAN MAKBUZU KORUMASI. Duman testleri (kucuk ornek, minik parametre) ayni
dosya adlarina yaziyor ve gercek sonuc sanilabiliyor -- bu gece bir kez oldu.
`n_parca` esigin altindaysa makbuz GECERSIZ isaretlenir, sayisi RAPOR EDILMEZ.

Kullanim:  python sabah_ozeti.py
"""
import glob
import json
import os
import time

KOK = os.path.dirname(os.path.abspath(__file__))
EN_AZ_PARCA = 500          # bunun altindaki makbuz DUMAN sayilir
KAPI_EK = 0.01             # ek oznitelik blogu kapisi
KAPI_HAVUZ = 0.85          # yonlu havuz recall kapisi
KAPI_D7 = 0.10             # D7 OKUMA #2 icin kumulatif kazanc kapisi


def oku(y):
    try:
        with open(y, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def sat(x):
    return "-" if x is None else (f"{x:+.4f}" if isinstance(x, float) else str(x))


def ek_bloklar():
    out = []
    for y in sorted(glob.glob(os.path.join(KOK, "results", "ek_blok_*.json"))):
        d = oku(y)
        if not d:
            continue
        ad = d.get("blok", os.path.basename(y))
        n = int(d.get("n_parca", 0))
        if n < EN_AZ_PARCA:
            out.append((ad, None, None, None, f"GECERSIZ (duman, {n} parca)"))
            continue
        f = float(d.get("fark", 0.0))
        out.append((ad, float(d.get("yok", 0)), float(d.get("var", 0)), f,
                    "GECTI" if f >= KAPI_EK else "gecmedi"))
    return out


def havuz(dizin="_p6_oz_tam3", kume=None):
    """Degerler KOKTE degil `toplam` altinda: konum_recall / yonlu_recall /
    f1_tavani. (Ilk yazimda kokte aranmisti ve rapor bos gosteriyordu.)

    Makbuz adi artik kumeyi de tasiyor; eski (kumesiz) ada da bakilir."""
    adlar = ([f"havuz_tavani_{dizin}_{kume}.json"] if kume else []) + \
            [f"havuz_tavani_{dizin}_d6.json", f"havuz_tavani_{dizin}.json"]
    d = None
    for a in adlar:
        d = oku(os.path.join(KOK, "results", a))
        if d:
            break
    if not d:
        return None
    t = d.get("toplam", {})
    return (t.get("konum_recall"), t.get("yonlu_recall"), t.get("f1_tavani"),
            t.get("aday_parca"), bool(d.get("kapi_a_gecti")))


def kademe2():
    out = []
    for ad, y in (("taban (u25)", "p6_kademe2_tam.json"),
                  ("SIRA kapali", "p6_kademe2_sira0.json"),
                  ("SIRA acik", "p6_kademe2_sira1.json"),
                  ("tam3 korpus", "p6_kademe2_tam3.json")):
        d = oku(os.path.join(KOK, "results", y))
        if not d:
            continue
        t = d.get("toplam", {})
        kollar = {k: v.get("robot") for k, v in t.items()
                  if isinstance(v, dict) and "robot" in v}
        out.append((ad, d.get("dizin", "?"), d.get("secilen"), kollar))
    return out


def gece_fazlari():
    y = os.path.join(KOK, "results", "_gece", "ANA.log")
    if not os.path.exists(y):
        return []
    with open(y, encoding="utf-8", errors="replace") as f:
        return [s.strip() for s in f
                if "BITTI:" in s or "DUSTU:" in s]


def main():
    L = []
    L.append("# SABAH RAPORU -- " + time.strftime("%Y-%m-%d %H:%M"))
    L.append("")
    L.append("Butun sayilar `tam` MARKA KATLARINDA (LOMO). D7 SINAVINA "
             "BAKILMADI. Manset metrik MIKRO robot F1.")
    L.append("")

    L.append("## 1. KAPI A -- tam-acik havuzun yonlu recall'u")
    yeni = havuz("_p6_oz_tam3")
    eski = havuz("_p6_oz_u25")
    if yeni is None:
        L.append("Henuz olculmedi (A2 fazi kosmadi).")
        if eski:
            L.append(f"- ONCEKI havuz (u25): yonlu recall {eski[1]:.4f}, "
                     f"F1 tavani {eski[2]:.4f} -> kapi GECMEMISTI")
    else:
        L.append("| havuz | konum recall | yonlu recall | F1 tavani | aday/parca |")
        L.append("|---|---|---|---|---|")
        for ad, h in (("onceki (u25)", eski), ("**tam-acik (tam3)**", yeni)):
            if h:
                L.append(f"| {ad} | {h[0]:.4f} | {h[1]:.4f} | {h[2]:.4f} | "
                         f"{h[3]:.0f} |")
        L.append("")
        L.append(f"- KAPI A (yonlu recall >= {KAPI_HAVUZ}) -> "
                 f"**{'GECTI' if yeni[4] else 'GECMEDI'}** ({yeni[1]:.4f})")
        if eski:
            L.append(f"- onceki havuza gore yonlu recall farki: "
                     f"**{yeni[1] - eski[1]:+.4f}**, tavan farki: "
                     f"**{yeni[2] - eski[2]:+.4f}**")
        L.append("")
        L.append("Kapi gecmezse havuz genisletme kolu KAPANIR: tavan "
                 "yetmiyorsa secici ne kadar iyilesirse iyilessin hedefe "
                 "ulasilamaz. TAVAN, mukemmel bir secicinin alacagi F1'dir -- "
                 "VAAT DEGIL, UST SINIR.")
    L.append("")

    L.append("## 1b. SECENEK TAVANI (MAX_SEC) BAGLIYOR MU?")
    ms = sorted(glob.glob(os.path.join(KOK, "results",
                                       "max_sec_sondasi*.json")))
    if not ms:
        L.append("- olcum yok (`MS_MARKA=NIT python sonda_max_sec.py`)")
    for y in ms:
        d = oku(y)
        if not d:
            continue
        L.append(f"**orneklem: {d.get('marka', '?')} / {d.get('n_parca')} "
                 f"parca, yelpaze {d.get('yelpaze')}**")
        L.append("")
        L.append("| tavan | yonlu recall | secenek maliyeti |")
        L.append("|---|---|---|")
        for t, v in sorted(d.get("sonuc", {}).items(), key=lambda kv: int(kv[0])):
            L.append(f"| {t} | {v['yonlu_recall']:.4f} | "
                     f"{v['maliyet_kat']:.2f}x |")
        L.append("")
    if ms:
        L.append("> Bugunku tavan **12**. Tavan bagliyorsa yon kaynagi "
                 "eklemek (yelpaze cozunurlugu) recall'u ARTIRMAZ -- yeni "
                 "yonler tavana takilip mevcutlarin yerini alir. "
                 "`YB_MAX_SEC` ile ayarlanir.")
    L.append("")

    L.append("## 2. EK OZNITELIK BLOKLARI (kapi +0.01)")
    eb = ek_bloklar()
    if not eb:
        L.append("Henuz makbuz yok.")
    else:
        L.append("| blok | bloksuz | blokla | fark | karar |")
        L.append("|---|---|---|---|---|")
        for ad, y0, v0, f, k in eb:
            L.append(f"| {ad} | {'-' if y0 is None else f'{y0:.4f}'} | "
                     f"{'-' if v0 is None else f'{v0:.4f}'} | "
                     f"{'-' if f is None else f'{f:+.4f}'} | {k} |")
    L.append("")

    L.append("## 3. KADEME2 KOL KIYASLARI")
    for ad, dz, sec, kollar in kademe2():
        ks = ", ".join(f"{k} {v:.4f}" for k, v in sorted(kollar.items()))
        L.append(f"- **{ad}** ({dz}) secilen={sec} -> {ks}")
    L.append("")

    L.append("## 3b. SECICI VERIMLILIGI -- 0.50 nereden gelebilir?")
    tv = havuz("_p6_oz_u25", "tam") or havuz("_p6_oz_tam3", "tam")
    ger = None
    for ad, _dz, _sec, kollar in kademe2():
        if "P6" in kollar:
            ger = kollar["P6"]
            break
    if tv and ger:
        tavan = tv[2]
        verim = ger / max(tavan, 1e-9)
        L.append(f"- havuz F1 TAVANI (`tam`, mukemmel secici): **{tavan:.4f}**")
        L.append(f"- GERCEKLESEN (P6 kolu): **{ger:.4f}**")
        L.append(f"- **secici verimliligi = {verim:.1%}**")
        L.append("")
        ger_tavan = 0.50 / max(verim, 1e-9)
        ger_verim = 0.50 / max(tavan, 1e-9)
        L.append(f"0.50'ye iki yoldan gidilebilir:")
        L.append(f"1. **Havuzla:** verimlilik sabit kalirsa tavanin "
                 f"**{ger_tavan:.4f}** olmasi gerekir"
                 + ("  -> 1.0'i asiyor, TEK BASINA IMKANSIZ"
                    if ger_tavan > 1.0 else ""))
        L.append(f"2. **Seciciyle:** tavan sabit kalirsa verimliligin "
                 f"**{ger_verim:.1%}** olmasi gerekir "
                 f"({ger_verim / max(verim, 1e-9):.2f}x iyilesme)")
        L.append("")
        L.append("> Havuz kolu tek basina hedefe goturmuyor; SECICI kolu "
                 "zorunlu. Bu, EK bloklarina ve aday-kumesi modeline "
                 "(D2) verilen onceligi belirler.")
    else:
        L.append("- `tam` kumesinde tavan olcumu henuz yok "
                 "(`HT_ONLER=tam python sonda_havuz_tavani.py`)")
    L.append("")

    L.append("## 4. GECE FAZLARI")
    fz = gece_fazlari()
    L.extend(f"- {s}" for s in fz) if fz else L.append("- log yok")
    L.append("")

    L.append("## 4b. SAHA -- AUTO KATMANI (tier cokusu)")
    tc = oku(os.path.join(KOK, "results", "tier_cokusu_d7.json"))
    if not tc:
        L.append("- olcum yok (`python sonda_tier_cokusu.py`)")
    else:
        L.append(f"Dagitilan AUTO esigi = **{tc.get('dagitilan_esik')}**")
        for ad, k in tc.get("kumeler", {}).items():
            if k.get("durum"):
                L.append(f"- `{ad}`: **{k['durum']}** "
                         f"({k['n_isaret']} isaretin hepsi ayni skor)")
                continue
            de = k.get("dagitilan_esikte", {})
            L.append(f"- `{ad}`: AUTO payi **{de.get('auto_pay', 0):.4f}**, "
                     f"kesinlik **{de.get('kesinlik') or 0:.4f}**"
                     + ("  <- REVIEW KATMANI BOS" if de.get("review_bos") else ""))
        L.append("")
        L.append("> Gorulmemis markada robot HER isarete otonom guveniyor. "
                 "Esigi yukseltmek kurtarmiyor (0.95'te bile kesinlik ~0.47). "
                 "Oneri: gorulmemis marka icin AUTO katmani KAPATILSIN.")
    L.append("")

    L.append("## 5. D7 OKUMA #2 KARARI")
    kaz = [f for _, _, _, f, k in eb if f is not None and k == "GECTI"]
    top = sum(kaz)
    L.append(f"- kapiyi gecen blok sayisi: **{len(kaz)}**")
    L.append(f"- bu bloklarin toplam kazanci: **{top:+.4f}** "
             f"(kapi +{KAPI_D7:.2f})")
    L.append("")
    if top >= KAPI_D7:
        L.append("**KARAR: D7 OKUMA #2 HAK EDILDI.** Yine de okuma ancak "
                 "kanonik zincirle (`kanonik_d7.mikro()`) yapilir.")
    else:
        L.append("**KARAR: D7 OKUNMAZ.** Kumulatif kazanc kapinin altinda; "
                 "okuma HARCANMAZ. Butcede kalan okuma sayisi degismez.")
    L.append("")
    L.append("> Kazanclar TOPLANARAK tahmin edilir; gercek birlesik kazanc "
             "genellikle DAHA AZ olur (bloklar ayni hatalari duzeltir). "
             "Toplam yalnizca KAPI kararidir, VAAT DEGILDIR.")

    metin = "\n".join(L)
    y = os.path.join(KOK, "docs", "SABAH_RAPORU.md")
    os.makedirs(os.path.dirname(y), exist_ok=True)
    with open(y, "w", encoding="utf-8") as f:
        f.write(metin + "\n")
    print(metin)
    print(f"\n-> {y}")


if __name__ == "__main__":
    main()
