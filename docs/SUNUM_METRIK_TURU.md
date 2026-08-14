# SUNUM — BEKLEYEN METRIK TURU (2026-08-14)

`ConnectionPointDetector_Bora_Bayrakci.pptx` icerik olarak guncellendi
(tarih, boru hatti, modeller, ayarlar, yol haritasi). **METRIK SAYILARINA
DOKUNULMADI** -- bu dosya o turu mekanik hale getirmek icindir.

Yedek: `ConnectionPointDetector_Bora_Bayrakci_YEDEK_2026-08-14.pptx`

---

## ONCE KARAR VERILMESI GEREKEN SEY (find-replace DEGIL)

Temmuz sayilari **farkli bir degerlendirme cercevesinde** measured
(hold-out, eski split, eski zincir). Bugunku sayilar **VAL 100 part**
uzerinde ve iki ayri populasyon icin ayri ayri var. Yani ayni hucreye
yeni sayi yazmak yetmez; **hangi cerceveyi sunacagina** karar vermek
gerekir.

**Onerilen cerceve** (savunmasi en kolay olan):
> "Bilinen bir ureticiden gelen, daha once GORULMEMIS bir .stp dosyasi"
> = VAL 100 part. Kesisim sifir (training/gelistirme/exam).

---

## BUGUNKU SAYILAR (hazir, makbuzlu)

| criterion | deger | %95 GA |
|---|---|---|
| Tespit F1 | **0.7878** | [0.7464, 0.8263] |
| Robot-hazir, axis (unsigned) | 0.5764 | [0.5067, 0.6438] |
| Robot-hazir, **ISARETLI** | **0.4839** | [0.4003, 0.5671] |
| Kesinlik / Recall | 0.5008 / 0.4682 | — |
| **Alan farki** (tanidik 0.5603 vs gorulmemis 0.3135) | **+0.2468** | — |
| Dusuk-CP / Cok-CP | 0.7701 / 0.6583 | — |

Makbuz: `results/TABAN_VAL_uc_metrik.json`,
`results/guven_araligi_saha.json`

---

## SLAYT SLAYT: DEGISMESI GEREKEN HUCRELER

| slayt | hucre | bugunku deger (Temmuz) | not |
|---|---|---|---|
| 6 | Connection-point F1 | 62.5% | cerceve farkli -- karar gerekli |
| 9 | Mean IoU (urun / en iyi) | 63.3% / 68.1% | **muhtemelen GECERLI** (seg modeli degismedi) |
| 9 | sinif bazli IoU (5 satir) | 77.3 / 68.8 / 64.6 / 64.2 / 45.4 | ayni model -> gecerli |
| 10 | The running system | 62.5% | -> tespit 0.7878 (cerceve notu) |
| 10 | Robot-ready (2mm + 10 deg) | 41.6% | -> **signed 0.4839** (yeni: signed/unsigned AYRIMI) |
| 10 | weaker split (siblings shared) | 77.9% | kalabilir (sisme ornegi) |
| 10 | Ceiling with a perfect filter | 89.4% | **yeni ceiling olcumu var**: 21.60 |
| 11 | Low / High CP-F1 | 61.8% / 68.2% | -> 0.7701 / 0.6583 |
| 12 | ornek part F1'leri | 100% / 0% | yeni ornek secilebilir (10 parcalik ara test var) |
| 13 | "+12.7 points, measured" | — | **GECERLI**, degismedi |

---

## SUNUMA EKLENMESI ONERILEN IKI YENI SAYI

Bunlar Temmuz'da YOKTU ve bu kampanyanin en guclu iki sonucudur:

1. **Alan farki +0.2468** (tanidik manufacturer 0.5603 / gorulmemis 0.3135).
   Ayni sistem, ayni veri, yalniz split tipi farkli. Projenin nereye
   yatirim yapmasi gerektigini soyleyen tek sayi.

2. **Baglayici kisit tablosu** (VAL 100 / 660 CP): yonu TAMAMEN mukemmel
   yapmak kapsamayi +0.170 aciyor, lateral toleransi 2->10 mm yapmak
   **+0.358**. Yani darbogaz direction degil **lateral konum**.

Ayrica bir metodoloji sayisi: ayni recete, yalniz seed farkli ->
0.6528 / 0.6990 / 0.6673 (**yayilim 0.046**). Tek kosumla dagitilan
her kolun neden supheli oldugunu tek satirda anlatir.


---

## 14 AGUSTOS OGLE — SUNUMA GIREBILECEK IKI YENI SAYI

Bunlar bugun measured ve sunumda YOK:

**1. Offset basi: direction ogreniliyor, buyukluk buzuluyor.**
13 epok, 700 part, leakage kapisi kurulu:
* hedef offset buyuklugu 4.299 mm, tahmin 2.871 mm
* **direction uyumu kosinus +0.675** (~48 derece ortanca aci hatasi;
  rastgele 0 olurdu)
Bu, "sonraki adim" satirini tahminden OLCUME cevirir.

**2. Yogun yolda dort bayatlama onarildi.**
tespit 0.3402 -> **0.3673**, unsigned robot 0.1546 -> **0.2143**.
Taban hala onde (0.7907 / 0.4651), o yuzden DAGITILMADI -- ama yolun
bozuk olmadigi, yalnizca secicisinin eski oldugu artik olculu.

**3. Gate esikleri OLU parametre cikti** (slayt 8 duzeltildi).
`GORELI_ESIK` 2026-07-31'de dagitilmis; config'deki 0.40/0.35 ve
sunumdaki "40% / 35%" ARTIK KULLANILMIYOR. Canli kural:
skor >= part maksimumunun %50'si VE skor >= %20.
