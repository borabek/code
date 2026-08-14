# SUNUM — BEKLEYEN METRIK TURU (2026-08-14)

`ConnectionPointDetector_Bora_Bayrakci.pptx` icerik as guncellendi
(tarih, boru hatti, modeller, settings, path haritasi). **METRIK SAYILARINA
DOKUNULMADI** -- this file o turu mekanik hale getirmek icindir.

Yedek: `ConnectionPointDetector_Bora_Bayrakci_YEDEK_2026-08-14.pptx`

---

## ONCE DECISION VERILMESI GEREKEN SEY (find-replace DEGIL)

Temmuz sayilari **different a degerlendirme cercevesinde** measured
(hold-out, eski split, eski zincir). Bugunku sayilar **VAL 100 part**
uzerinde ve iki ayri populasyon for ayri ayri present. Yani same hucreye
yeni number yazmak yetmez; **hangi cerceveyi sunacagina** karar vermek
gerekir.

**Onerilen cerceve** (savunmasi en kolay which):
> "Bilinen a ureticiden gelen, more before GORULMEMIS a .stp dosyasi"
> = VAL 100 part. Kesisim sifir (training/gelistirme/exam).

---

## BUGUNKU SAYILAR (hazir, makbuzlu)

| criterion | value | %95 GA |
|---|---|---|
| Tespit F1 | **0.7878** | [0.7464, 0.8263] |
| Robot-hazir, axis (unsigned) | 0.5764 | [0.5067, 0.6438] |
| Robot-hazir, **ISARETLI** | **0.4839** | [0.4003, 0.5671] |
| Kesinlik / Recall | 0.5008 / 0.4682 | — |
| **Alan farki** (familiar 0.5603 vs unseen 0.3135) | **+0.2468** | — |
| Dusuk-CP / Cok-CP | 0.7701 / 0.6583 | — |

Makbuz: `results/baseline_VAL_uc_metric.json`,
`results/confidence_interval_saha.json`

---

## SLAYT SLAYT: DEGISMESI GEREKEN HUCRELER

| slayt | hucre | bugunku value (Temmuz) | not |
|---|---|---|---|
| 6 | Connection-point F1 | 62.5% | cerceve different -- karar gerekli |
| 9 | Mean IoU (urun / en iyi) | 63.3% / 68.1% | **muhtemelen VALID** (seg modeli degismedi) |
| 9 | sinif bazli IoU (5 satir) | 77.3 / 68.8 / 64.6 / 64.2 / 45.4 | same model -> valid |
| 10 | The running system | 62.5% | -> detection 0.7878 (cerceve notu) |
| 10 | Robot-ready (2mm + 10 deg) | 41.6% | -> **signed 0.4839** (yeni: signed/unsigned AYRIMI) |
| 10 | weaker split (siblings shared) | 77.9% | kalabilir (sisme ornegi) |
| 10 | Ceiling with a perfect filter | 89.4% | **yeni ceiling olcumu present**: 21.60 |
| 11 | Low / High CP-F1 | 61.8% / 68.2% | -> 0.7701 / 0.6583 |
| 12 | ornek part F1'leri | 100% / 0% | yeni ornek secilebilir (10 parcalik ara test present) |
| 13 | "+12.7 points, measured" | — | **VALID**, degismedi |

---

## SUNUMA EKLENMESI ONERILEN IKI YENI SAYI

Bunlar Temmuz'da YOKTU ve this kampanyanin en guclu iki sonucudur:

1. **Alan farki +0.2468** (familiar manufacturer 0.5603 / unseen 0.3135).
   Ayni sistem, same data, only split tipi different. Projenin nereye
   yatirim yapmasi gerektigini soyleyen tek number.

2. **Baglayici kisit tablosu** (VAL 100 / 660 CP): direction TAMAMEN mukemmel
   yapmak kapsamayi +0.170 aciyor, lateral toleransi 2->10 mm yapmak
   **+0.358**. Yani darbogaz direction not **lateral konum**.

Ayrica a metodoloji sayisi: same recete, only seed different ->
0.6528 / 0.6990 / 0.6673 (**spread 0.046**). Tek kosumla dagitilan
each kolun why supheli oldugunu tek satirda anlatir.


---

## 14 AGUSTOS OGLE — SUNUMA GIREBILECEK IKI YENI SAYI

Bunlar bugun measured ve sunumda YOK:

**1. Offset basi: direction ogreniliyor, buyukluk buzuluyor.**
13 epok, 700 part, leakage kapisi kurulu:
* hedef offset buyuklugu 4.299 mm, prediction 2.871 mm
* **direction uyumu kosinus +0.675** (~48 derece median aci hatasi;
  rastgele 0 olurdu)
Bu, "sonraki adim" satirini tahminden OLCUME cevirir.

**2. Yogun yolda dort bayatlama onarildi.**
detection 0.3402 -> **0.3673**, unsigned robot 0.1546 -> **0.2143**.
Taban hala onde (0.7907 / 0.4651), o yuzden NOT DEPLOYED -- but yolun
bozuk olmadigi, yalnizca secicisinin eski oldugu residual olculu.

**3. Gate esikleri OLU parametre cikti** (slayt 8 duzeltildi).
`GORELI_ESIK` 2026-07-31'de dagitilmis; config'deki 0.40/0.35 ve
sunumdaki "40% / 35%" ARTIK KULLANILMIYOR. Canli rule:
score >= part maksimumunun %50'si VE score >= %20.
