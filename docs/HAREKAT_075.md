# HAREKAT PLANI: GORULMEMIS MARKA ROBOT F1 -> 0.75

## 0. CEPHENIN DURUMU (hepsi olculdu, 2026-08-12)

| | deger |
|---|---|
| bugunku F1 (gorulmemis marka) | **0.31** |
| havuz F1 tavani (tavan-24) | **0.9432** |
| secici verimliligi | ~%33 |
| **0.75 icin gereken verimlilik** | **%79.5** |
| bugunku EN IYI markanin verimliligi | **%65.5** (UPUN) |

**Hedefin gercek anlami:** 0.75, bugun EN IYI markada aldigimiz verimin
USTUNU **her markada** istiyor. Bu, artimli kazanclarla ulasilamaz. Sekiz
koldan yalnizca biri gecti (kanonik +0.0151); bu hizla 0.75'e giden yol YOK.

**Demek ki problemin KURGUSU degismeli.** Asagisi onun icin.

---

## 1. COZULMEMIS PARADOKS -- harekatin kilit noktasi

NIT markasinda:

| olcu | deger |
|---|---|
| havuzda GT (tavan) | 0.898 |
| dogru secenegin sira yuzdeligi | **0.005** (7470 secenek icinde ilk ~40) |
| parca basina CP | **24.4** |
| uctan uca F1 | **0.005** |

**Siralama iyi, havuz iyi, sonuc sifir.** Tek tutarli aciklama: parca basina
~24 CP secilmesi gerekiyor ve ESIK TABANLI kural bunu yapamiyor. Skor ayrimi
0.05 iken esik ya yuzlerce sey aliyor (kesinlik cokuyor) ya bir avuc
(recall cokuyor).

**Bu hipotez TEST EDILIYOR** (`sonda_ustk_kahin.py`): ayni skorla ilk `k`
secilir, `k` = parcanin GERCEK CP sayisi.

- **USTK_KAHIN >> KURAL** -> sorun ADET/ESIK. Asagidaki I. KOL acilir.
- **USTK_KAHIN ~ KURAL** -> sorun SIRALAMA. II. ve III. KOL'a gecilir.

Bu tek olcum, harekatin agirlik merkezini belirler. Once o, sonra gerisi.

---

## I. KOL -- YAPISAL URETIM ("hapishane kacisi": 24 karari 2 karara indir)

**Fikir.** Klemens bloklarindaki CP'ler BAGIMSIZ DEGIL: bir KAFES olustururlar
(sabit adim, tek eksen). 24 bagimsiz secim yerine **1 tohum + 1 adim vektoru +
1 adet** tahmin edilir, gerisi URETILIR. Karar sayisi ~7470'ten ~3'e iner.

Elimizde zaten olculmus destek var: `kafes.py` -- D6 GT'sinin **%90.8'i**
parcanin en yaygin oteleme vektoruyle uretilebiliyor.

**I.1 Kafes ile konum uretimi.** En guvenli tohumu bul, kafes adimini olc,
govde uzunlugu boyunca uret.
**I.2 Uretilen her konumda YONU YENIDEN SEC.** Onceki denemenin (K2.1)
cokme sebebi buydu: yon KOPYALANMISTI (tespit +0.0126 ama robot -0.0100).
Yon, o konumdaki yon-bankasi secenekleri arasindan YENIDEN puanlanmali.
**I.3 Adet tahmini.** `n ≈ govde uzunlugu / kafes adimi`. Geometriden gelir,
modelden degil.
**I.4 Kapi.** NIT/MOR gibi COKEN markalarda F1 >= 0.30 (bugun 0.005-0.05).
Calisan markalarda KAYIP OLMAMALI (rejim kapisi ile ayrilir).

**Neden bu kol ozel:** cokus yogunlukla buyuyor, kafes de yogunlukla
GUCLENIYOR. Kolun en iyi calistigi yer, sistemin en cok kanadigi yer.

---

## II. KOL -- KARAR KURALINI OGREN (skoru degil, ESIGI degistir)

Bugun kural KURESEL: tum parcalar icin tek esik. Oysa parcalar cok farkli
(3 CP'lik UPUN vs 24 CP'lik NIT).

**II.1 Parca basina esik.** Kucuk bir model parca ozniteliklerinden (aday
sayisi, skor dagilimi, kafes adimi, tahmini adet) O PARCA ICIN esigi secsin.
**II.2 Adet-kisitli secim.** Esik yerine "ilk k" -- k tahmini adet.
**II.3 Kapi.** LOMO'da +0.03.

**Onemli ayrim:** bu, dusen `ozkalib`/`kume`/S4 kollarindan FARKLI. Onlar
SKORU degistiriyordu (ve goreli kural zaten normalizasyon yaptigi icin
tekrar oluyordu); bu KARARI degistiriyor.

---

## III. KOL -- YONU OGRENME, HESAPLA

Konum recall'u `tam` kumesinde **0.9789**; kayip neredeyse tamamen YON.
Ama B-rep agizlarinda yon ANALITIKTIR (silindir ekseni). Ogrenmeye gerek yok.

**III.1** Kaynak B-rep olan adaylarda yonu analitik eksene SABITLE, secimi
yalnizca konumda yap.
**III.2** Mesh adaylarinda yonu yerel yuzey normalinden turet.
**III.3 Kapi:** yonlu recall sabitken secenek sayisi >= %50 duser (arama
uzayi kuculur, ayrim keskinlesir).

---

## IV. KOL -- EGITIMIN BIRIMI PARCA OLSUN

Bugun egitim satir bazli: 7470 secenekli NIT parcasi, 1000 secenekli UPUN
parcasindan 7 kat fazla agirlik tasiyor. Metrik ise MIKRO ama karar PARCA
basina veriliyor.

**IV.1** Parca basina agirliklandirma (her parca esit katki).
**IV.2** Kayip dogrudan siralama uzerinde (listwise), parca icinde.
**IV.3 Kapi:** +0.02.

---

## V. KOL -- VERI (tek "kaba kuvvet" kolu ve en ongorulebiliri)

**V.1 SENTETIK KLEMENS URETECI.** Klemensler PARAMETRIK: kutup sayisi, adim,
delik capi, govde olculeri. Binlerce etiketli sentetik parca uretmek
mumkun ve bu DOGRUDAN "gorulmemis marka" sorununa calisir -- cunku sentetik
uretec marka kavrami tanimaz.
**V.2** SIE +310 parca (elde, kullanilmamis).
**V.3** Oz-egitim (kapili).
**V.4 Kapi:** +0.03.

**Ogrenme egrisi uyarisi:** hafizadaki olcum "0.80 icin 3.8x, 0.85 icin 17.3x
korpus" diyor. Yani veri TEK BASINA 0.75'e goturmez ama I/II kollarinin
tabanini yukseltir.

---

## VI. KOL -- SIMETRI VE TEKRAR (ucuz, kismen kuyrukta)

Klemensler aynali simetriktir. Bir tarafta bulunan CP, karsi tarafta
BEKLENIR. `simetri` blogu kuyrukta; kafes ile birlestirilirse uretim kolu
guclenir.

---

## HAREKAT SIRASI (agirlik merkezine gore)

| faz | is | kapi | neden bu sirada |
|---|---|---|---|
| **0** | ustk-kahin ayristirmasi (KOSUYOR) | - | agirlik merkezini belirler |
| **1** | I. KOL kafes uretimi | coken markada F1 >= 0.30 | en cok kanayan yer |
| **2** | II. KOL ogrenilen karar | +0.03 | I ile dogrudan birlesir |
| **3** | III. KOL analitik yon | secenek -%50, recall sabit | arama uzayini kucultur |
| **4** | IV. KOL parca birimli egitim | +0.02 | ucuz, bagimsiz |
| **5** | V. KOL sentetik veri | +0.03 | uzun soluklu, tabani yukseltir |

## DURUST BEKLENTI

- **I+II calisirsa** (coken markalar 0.005 -> 0.30-0.45): mikro F1 **0.45-0.55**.
  Cunku coken markalar GT'nin buyuk kismini tutuyor (NIT %45.7, D7'de CWT %37).
- **Ustune III+IV+V**: **0.55-0.65**.
- **0.75**, ancak yukaridakilerin HEPSI calisirsa ve calisan markalarda da
  kayip olmazsa. **Olasiligi dusuk goruyorum ama YOLU VAR** -- bugunku
  yaklasimin ise yolu yoktu.

**Kural:** her kol TEK DEGISKENLI olculur, kapisi ONCEDEN ilan edilir, D7'ye
kumulatif +0.10 birikmeden BAKILMAZ.
