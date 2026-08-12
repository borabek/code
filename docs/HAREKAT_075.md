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

---

# FAZ 0 SONUCU -- AGIRLIK MERKEZI BELIRLENDI (ve hipotezim KISMEN CURUDU)

Makbuz `results/ustk_kahin_d6.json`.

| marka | GT | KURAL | USTK_KAHIN | TAVAN | kazanc |
|---|---|---|---|---|---|
| NIT | 1222 | 0.0089 | 0.0434 | 0.9147 | +0.0344 |
| SUPU | 547 | 0.4494 | 0.4808 | 0.9591 | +0.0314 |
| **UPUN** | 370 | 0.5359 | **0.6892** | 0.9807 | **+0.1533** |
| MOR | 274 | 0.1783 | 0.1861 | 0.9297 | +0.0078 |
| **TOPLAM** | 2413 | **0.2954** | **0.2578** | 0.9372 | **-0.0376** |

## Dort markanin DORDU kazaniyor, TOPLAM kaybediyor

Hata degil, MIKRO metrigin dogasi: NIT'te gercek adet kadar (parca basina
~24) tahmin uretince yuzlerce YANLIS POZITIF ekleniyor. NIT kendi icinde
kazaniyor (cunku FN yigini devasa, recall artisi kesinlik kaybini yeniyor)
ama HAVUZLANMIS toplami batiriyor.

**Bu, projedeki MIKRO/MAKRO tuzaklar ailesinin yeni bir uyesi:** marka basina
iyilesme, havuzlanmis metrikte KOTULESME olarak gorunebilir. Kol kararlari
HER ZAMAN havuzlanmis mikro uzerinden verilmeli -- ama TESHIS marka basina
okunmali, yoksa bu ayrim kacar.

## ASIL BULGU: iki AYRI hastalik

**1. Siralamanin CALISTIGI markada, baglayici kisit ADET/ESIK.**
UPUN: 0.5359 -> **0.6892** yalnizca dogru adedi bilmekle. Bu, tek bir
degisiklikten gelen en buyuk kazanc (+0.1533) -- bugune kadar olctugum
her seyden buyuk.

**2. Siralamanin COKTUGU markada, adet KURTARMIYOR.**
NIT: gercek adet verilse bile 0.0434 (tavan 0.9147). Yani NIT'te siralama
GERCEKTEN bozuk; esik/adet duzeltmesi ise yaramaz.

## PLANIN GUNCELLENMIS AGIRLIK MERKEZI

| kol | hedef kitle | dayanak |
|---|---|---|
| **II. KOL (ogrenilen karar/adet)** | siralamanin CALISTIGI markalar | UPUN +0.1533 OLCULDU |
| **I. KOL (kafes ile yapisal uretim)** | siralamanin COKTUGU markalar | NIT'te adet kurtarmiyor -> yapi sart |

**Ve kritik tasarim kisiti:** adet-kisitli secim KURESEL uygulanamaz --
NIT'te FP patlamasi yaratir. Parca basina, modelin O PARCADA guvenilir olup
olmadigina gore uygulanmali. Yani II. KOL'un ic mekanizmasi:

    guven yuksek -> ilk-k (k = tahmini adet)
    guven dusuk  -> mevcut esik kurali (muhafazakar)

"Guven"in kendisi olculebilir: parca ici skor ayrimi (S7'nin poz-neg olcusu
bunun kahin surumuydu; urun surumu skor dagiliminin bicimi olabilir).

## Sonraki adim

II. KOL'u bu haliyle kur: parca basina rejim (guven) + adet tahmini + ilk-k.
Kapi: havuzlanmis MIKRO'da +0.03, VE coken markalarda kayip olmamasi.

---

# DAHA DERIN: BIRLESTIRICI ILKE -- "PARCA KENDI SABLONUNU TANIMLAR"

## Teshisimdeki kusur (once bunu duzeltmek gerek)

S7'de "dogru secenegin sira yuzdeligi 0.005" diye olctugum sey, parcadaki
**EN IYI siralanmis** dogru secenegin sirasiydi. 24 CP'li bir parcada
**24.'sunun nerede oldugu hakkinda hicbir sey soylemiyor.** "Siralama iyi"
cikarimim fazla iyimserdi.

Dogru soru: **k'inci dogru secenek kacinci sirada?** Ve bu soruyu sorunca
butun basarisizliklar tek cumleyle aciklaniyor:

> **Model ILK CP'yi bulabiliyor, TEKRARLARI bulamiyor.**

Gozlenen deseni birebir aciklar: az CP'li marka (UPUN 3.2) calisiyor, cok
CP'li (NIT 24.4) cokuyor -- cunku bugun HER CP kendi basina markalar-arasi
bir karar gerektiriyor. (`sonda_tekrar.py` bunu olcuyor.)

## Felsefi kirilma

Simdiye kadar hep **MARKALAR ARASI TRANSFER** yapmaya calistik. Dusen uc kol
(ozkalib, kume, S4) da bunu yapiyordu: parca-ici baglami, markalar-arasi bir
siniflandiriciyi iyilestirmek icin kullanmak.

Ama en guclu sinyal transferde degil, **PARCANIN KENDI ICINDE**. Hic
gormedigimiz bir markanin klemensinde de 24 delik BIRBIRININ AYNISIDIR.
Bu bilgi hicbir markadan tasinmaz -- parcanin kendisinden gelir.

**Kendine-benzerlik, TANIMI GEREGI marka-bagimsizdir.**

Bu, "24 zor markalar-arasi karar"i -> "1 markalar-arasi karar + 23 parca-ici
esleştirme"ye indirir. Model o 1 kararda zaten iyi (NIT parcalarinin %76'sinda
dogru bir secenek ilk 50'de).

## Ilke KONUMLA SINIRLI DEGIL

Bir klemenste CP'ler yalnizca dizilişte degil, HER NITELIKTE aynidir:

| nitelik | parca-ici tutarlilik |
|---|---|
| delik yaricapi | hepsi ayni bore capinda |
| derinlik | hepsi ayni |
| yon | hepsi paralel |
| adim | sabit kafes araligi |
| agiz bicimi | ayni huni profili |

**Sonuc 1 -- ADET GEOMETRIDEN OKUNUR.** Bir klemensteki CP sayisi ~ B-rep'teki
KIPSEL YARICAPLI silindir sayisi. Ust-k deneyinde +0.1533 kazandiran "adet",
buyuk olcude OGRENMEYE GEREK OLMADAN elde edilebilir.

**Sonuc 2 -- KIPSEL IMZA SUZGECI.** Parcanin kipsel yaricapina/derinligine
UYMAYAN aday supheli. Bu, markalar-arasi bir siniflandiriciya hic ihtiyac
duymadan yanlis pozitifleri kirpar.

**Sonuc 3 -- KATALOG ONCELI (bedava saglamlik).** Klemens adimlari
STANDARTTIR (3.5 / 5.0 / 5.08 / 7.5 / 10.16 mm). Olculen adimi en yakin
katalog degerine oturtmak, gurultuye karsi bedava bir duzeltmedir.

## VII. KOL (YENI, ONCELIK 1) -- PARCA-ICI SABLON

**VII.1** Parcanin kipsel imzasini cikar (yaricap, derinlik, yon, adim).
**VII.2** Adet = kipsel yaricapli silindir sayisi (geometrik, ogrenmesiz).
**VII.3** Kipsel imzaya uymayan adaylari kirp.
**VII.4** En guvenli tohumdan otelemeyle uret; her uretilen konumda YONU
YENIDEN sec (K2.1'in cokme sebebi yon kopyalamaydi).
**VII.5 Kapi:** coken markada F1 >= 0.30, calisan markada kayip YOK.

**Neden bu kol digerlerinden farkli:** hicbir markadan bilgi tasimiyor.
Gorulmemis marka sinavinda BOZULMASI icin bir sebep yok -- oysa bugune kadar
dusen her kol, tam da o sinavda bozuldugu icin dustu.

## Bu ilke, DUSEN kollari da acikliyor

`ozkalib` / `kume` / S4 parca-ici bilgiyi SKORA katti; ama karar kurali zaten
goreli oldugu icin bu tekrar oldu ve gurultu ekledi. Parca-ici bilginin dogru
kullanimi skoru duzeltmek DEGIL, **URETMEK ve KIRPMAK**.

---

# TEKRAR SONDASI SONUCU -- HIPOTEZ DOGRULANDI

Makbuz `results/tekrar_sondasi_d6.json`.

| marka | CP/p | ILK GT sirasi | ortanca | SON GT sirasi | ilk-k icinde | **tekrar tavani** | adim mm |
|---|---|---|---|---|---|---|---|
| NIT | 24.4 | **18** | 157 | **1065** | 0.053 | **0.557** | 10.50 |
| MOR | 3.4 | 7 | 28 | 89 | 0.152 | 0.673 | 13.00 |
| SUPU | 3.3 | 0 | 3 | 8 | 0.468 | 0.796 | 12.00 |
| UPUN | 3.2 | 0 | 2 | 5 | 0.591 | 0.781 | 21.03 |

## "Model ilkini buluyor, tekrarlari bulamiyor" -- KANITLANDI

NIT'te ILK dogru secenek **18. sirada** (gayet iyi), SONUNCUSU **1065.**
Siralama ilkten sonuncuya ~60 KAT bozuluyor; gercek adet kadar secim yapilsa
GT'nin yalnizca **%5.3'u** yakalanir.

UPUN/SUPU'da ilk 0., son 5-8. sirada -- neredeyse kusursuz. **Calisan ve
coken marka arasindaki fark TAM OLARAK BUDUR.**

Bu ayni zamanda ust-k deneyindeki celiskiyi de acikliyor: UPUN'da adet vermek
+0.1533 kazandiriyor cunku siralama zaten dogru; NIT'te kazandirmiyor cunku
24. dogru secenek 1065. sirada -- ilk 24'e hicbir zaman giremez.

## Yayilim kolunun tavani ve aritmetigi

Tek bir otelemeyle uretilebilen GT orani: NIT **0.557**, MOR 0.673,
SUPU 0.796, UPUN 0.781. **Bu bir ALT SINIR** -- sonda TEK oteleme denedi,
klemenslerde cogu zaman 2 sira/2 kat vardir.

| senaryo | GT agirlikli F1 (D6) |
|---|---|
| bugun | **0.2088** |
| yayilim tavanin %60'ini alirsa | **0.4750** |
| yayilim tavanin tamamini alirsa | ~0.78 (kesinlik mukemmel varsayimiyla) |

NIT tek basina 0.0089 -> 0.7155 tavanina sahip ve D6 GT'sinin %45.7'si.

## Karar

**VII. KOL (parca-ici sablon + yayilim) ONCELIK 1'e alindi.** Gerekcesi
artik varsayim degil olcum:
 * kayip yeri kesin (tekrarlar, ilk degil)
 * tavani olculu (NIT 0.557, alt sinir)
 * mekanizmasi marka-bagimsiz (parcanin kendi otelemesi)
 * bugune kadar dusen her kolun dustugu yerde (gorulmemis marka) bozulmasi
   icin YAPISAL bir sebep yok

**Adim degerleri (10.5 / 13.0 / 12.0 / 21.0 mm) standart klemens
adimlarindan (3.5-7.5 mm) buyuk** -- sonda muhtemelen 2x harmonigi buluyor.
Kol kurulurken adim, katalog degerlerine ve alt harmoniklere karsi
sinanmali; yoksa uretilen izgara her ikinci CP'yi atlar.
