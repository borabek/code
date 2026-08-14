# HAREKAT PLANI: GORULMEMIS MARKA ROBOT F1 -> 0.75

## 0. CEPHENIN DURUMU (hepsi measured, 2026-08-12)

| | deger |
|---|---|
| bugunku F1 (gorulmemis brand) | **0.31** |
| pool F1 tavani (ceiling-24) | **0.9432** |
| selector verimliligi | ~%33 |
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
| havuzda GT (ceiling) | 0.898 |
| dogru secenegin sira yuzdeligi | **0.005** (7470 secenek icinde ilk ~40) |
| part basina CP | **24.4** |
| uctan uca F1 | **0.005** |

**Siralama iyi, pool iyi, sonuc sifir.** Tek tutarli aciklama: part basina
~24 CP secilmesi gerekiyor ve ESIK TABANLI kural bunu yapamiyor. Skor ayrimi
0.05 iken threshold ya yuzlerce sey aliyor (precision cokuyor) ya bir avuc
(recall cokuyor).

**Bu hipotez TEST EDILIYOR** (`probe_ustk_kahin.py`): ayni skorla ilk `k`
secilir, `k` = parcanin GERCEK CP sayisi.

- **USTK_KAHIN >> KURAL** -> sorun ADET/ESIK. Asagidaki I. KOL acilir.
- **USTK_KAHIN ~ KURAL** -> sorun SIRALAMA. II. ve III. KOL'a gecilir.

Bu tek measurement, harekatin agirlik merkezini belirler. Once o, sonra gerisi.

---

## I. KOL -- YAPISAL URETIM ("hapishane kacisi": 24 karari 2 karara indir)

**Fikir.** Klemens bloklarindaki CP'ler BAGIMSIZ DEGIL: bir KAFES olustururlar
(sabit adim, tek axis). 24 bagimsiz secim yerine **1 seed + 1 adim vektoru +
1 adet** tahmin edilir, gerisi URETILIR. Karar sayisi ~7470'ten ~3'e iner.

Elimizde zaten olculmus destek var: `lattice.py` -- D6 GT'sinin **%90.8'i**
parcanin en yaygin oteleme vektoruyle uretilebiliyor.

**I.1 Kafes ile konum uretimi.** En guvenli tohumu bul, lattice adimini olc,
body uzunlugu boyunca uret.
**I.2 Uretilen her konumda YONU YENIDEN SEC.** Onceki denemenin (K2.1)
cokme sebebi buydu: direction KOPYALANMISTI (tespit +0.0126 ama robot -0.0100).
Yon, o konumdaki direction-bankasi secenekleri arasindan YENIDEN puanlanmali.
**I.3 Adet tahmini.** `n ≈ body uzunlugu / lattice adimi`. Geometriden gelir,
modelden degil.
**I.4 Kapi.** NIT/MOR gibi COKEN markalarda F1 >= 0.30 (bugun 0.005-0.05).
Calisan markalarda KAYIP OLMAMALI (regime kapisi ile ayrilir).

**Neden bu arm ozel:** cokus yogunlukla buyuyor, lattice de yogunlukla
GUCLENIYOR. Kolun en iyi calistigi yer, sistemin en cok kanadigi yer.

---

## II. KOL -- KARAR KURALINI OGREN (skoru degil, ESIGI degistir)

Bugun kural KURESEL: tum parts icin tek threshold. Oysa parts cok farkli
(3 CP'lik UPUN vs 24 CP'lik NIT).

**II.1 Parca basina threshold.** Kucuk bir model part ozniteliklerinden (candidate
sayisi, skor dagilimi, lattice adimi, tahmini adet) O PARCA ICIN esigi secsin.
**II.2 Adet-kisitli secim.** Esik yerine "ilk k" -- k tahmini adet.
**II.3 Kapi.** LOMO'da +0.03.

**Onemli ayrim:** bu, dusen `ozkalib`/`cluster`/S4 kollarindan FARKLI. Onlar
SKORU degistiriyordu (ve goreli kural zaten normalizasyon yaptigi icin
tekrar oluyordu); bu KARARI degistiriyor.

---

## III. KOL -- YONU OGRENME, HESAPLA

Konum recall'u `tam` kumesinde **0.9789**; loss neredeyse tamamen YON.
Ama B-rep agizlarinda direction ANALITIKTIR (silindir ekseni). Ogrenmeye gerek yok.

**III.1** Kaynak B-rep olan adaylarda yonu analitik eksene SABITLE, secimi
yalnizca konumda yap.
**III.2** Mesh adaylarinda yonu yerel yuzey normalinden turet.
**III.3 Kapi:** yonlu recall sabitken secenek sayisi >= %50 duser (arama
uzayi kuculur, ayrim keskinlesir).

---

## IV. KOL -- EGITIMIN BIRIMI PARCA OLSUN

Bugun training satir bazli: 7470 secenekli NIT parcasi, 1000 secenekli UPUN
parcasindan 7 fold fazla agirlik tasiyor. Metrik ise MIKRO ama karar PARCA
basina veriliyor.

**IV.1** Parca basina agirliklandirma (her part esit katki).
**IV.2** Kayip dogrudan siralama uzerinde (listwise), part icinde.
**IV.3 Kapi:** +0.02.

---

## V. KOL -- VERI (tek "kaba kuvvet" kolu ve en ongorulebiliri)

**V.1 SENTETIK KLEMENS URETECI.** Klemensler PARAMETRIK: kutup sayisi, adim,
delik capi, body olculeri. Binlerce etiketli sentetik part uretmek
mumkun ve bu DOGRUDAN "gorulmemis brand" sorununa calisir -- cunku sentetik
uretec brand kavrami tanimaz.
**V.2** SIE +310 part (elde, kullanilmamis).
**V.3** Oz-training (kapili).
**V.4 Kapi:** +0.03.

**Ogrenme egrisi uyarisi:** hafizadaki measurement "0.80 icin 3.8x, 0.85 icin 17.3x
corpus" diyor. Yani veri TEK BASINA 0.75'e goturmez ama I/II kollarinin
tabanini yukseltir.

---

## VI. KOL -- SIMETRI VE TEKRAR (ucuz, kismen kuyrukta)

Klemensler aynali simetriktir. Bir tarafta bulunan CP, karsi tarafta
BEKLENIR. `simetri` blogu kuyrukta; lattice ile birlestirilirse uretim kolu
guclenir.

---

## HAREKAT SIRASI (agirlik merkezine gore)

| faz | is | gate | neden bu sirada |
|---|---|---|---|
| **0** | ustk-kahin ayristirmasi (KOSUYOR) | - | agirlik merkezini belirler |
| **1** | I. KOL lattice uretimi | coken markada F1 >= 0.30 | en cok kanayan yer |
| **2** | II. KOL ogrenilen karar | +0.03 | I ile dogrudan birlesir |
| **3** | III. KOL analitik direction | secenek -%50, recall sabit | arama uzayini kucultur |
| **4** | IV. KOL part birimli training | +0.02 | ucuz, bagimsiz |
| **5** | V. KOL sentetik veri | +0.03 | uzun soluklu, tabani yukseltir |

## DURUST BEKLENTI

- **I+II calisirsa** (coken markalar 0.005 -> 0.30-0.45): mikro F1 **0.45-0.55**.
  Cunku coken markalar GT'nin buyuk kismini tutuyor (NIT %45.7, D7'de CWT %37).
- **Ustune III+IV+V**: **0.55-0.65**.
- **0.75**, ancak yukaridakilerin HEPSI calisirsa ve calisan markalarda da
  loss olmazsa. **Olasiligi dusuk goruyorum ama YOLU VAR** -- bugunku
  yaklasimin ise yolu yoktu.

**Kural:** her arm TEK DEGISKENLI olculur, kapisi ONCEDEN ilan edilir, D7'ye
kumulatif +0.10 birikmeden BAKILMAZ.

---

# FAZ 0 SONUCU -- AGIRLIK MERKEZI BELIRLENDI (ve hipotezim KISMEN CURUDU)

Makbuz `results/ustk_kahin_d6.json`.

| brand | GT | KURAL | USTK_KAHIN | TAVAN | kazanc |
|---|---|---|---|---|---|
| NIT | 1222 | 0.0089 | 0.0434 | 0.9147 | +0.0344 |
| SUPU | 547 | 0.4494 | 0.4808 | 0.9591 | +0.0314 |
| **UPUN** | 370 | 0.5359 | **0.6892** | 0.9807 | **+0.1533** |
| MOR | 274 | 0.1783 | 0.1861 | 0.9297 | +0.0078 |
| **TOPLAM** | 2413 | **0.2954** | **0.2578** | 0.9372 | **-0.0376** |

## Dort markanin DORDU kazaniyor, TOPLAM kaybediyor

Hata degil, MIKRO metrigin dogasi: NIT'te gercek adet kadar (part basina
~24) tahmin uretince yuzlerce YANLIS POZITIF ekleniyor. NIT kendi icinde
kazaniyor (cunku FN yigini devasa, recall artisi precision kaybini yeniyor)
ama HAVUZLANMIS toplami batiriyor.

**Bu, projedeki MIKRO/MAKRO tuzaklar ailesinin yeni bir uyesi:** brand basina
iyilesme, havuzlanmis metrikte KOTULESME olarak gorunebilir. Kol kararlari
HER ZAMAN havuzlanmis mikro uzerinden verilmeli -- ama TESHIS brand basina
okunmali, yoksa bu ayrim kacar.

## ASIL BULGU: iki AYRI hastalik

**1. Siralamanin CALISTIGI markada, baglayici kisit ADET/ESIK.**
UPUN: 0.5359 -> **0.6892** yalnizca dogru adedi bilmekle. Bu, tek bir
degisiklikten gelen en buyuk kazanc (+0.1533) -- bugune kadar olctugum
her seyden buyuk.

**2. Siralamanin COKTUGU markada, adet KURTARMIYOR.**
NIT: gercek adet verilse bile 0.0434 (ceiling 0.9147). Yani NIT'te siralama
GERCEKTEN bozuk; threshold/adet duzeltmesi ise yaramaz.

## PLANIN GUNCELLENMIS AGIRLIK MERKEZI

| arm | hedef kitle | dayanak |
|---|---|---|
| **II. KOL (ogrenilen karar/adet)** | siralamanin CALISTIGI markalar | UPUN +0.1533 OLCULDU |
| **I. KOL (lattice ile yapisal uretim)** | siralamanin COKTUGU markalar | NIT'te adet kurtarmiyor -> yapi sart |

**Ve kritik tasarim kisiti:** adet-kisitli secim KURESEL uygulanamaz --
NIT'te FP patlamasi yaratir. Parca basina, modelin O PARCADA guvenilir olup
olmadigina gore uygulanmali. Yani II. KOL'un ic mekanizmasi:

    confidence yuksek -> ilk-k (k = tahmini adet)
    confidence dusuk  -> mevcut threshold kurali (muhafazakar)

"Guven"in kendisi olculebilir: part ici skor ayrimi (S7'nin poz-neg olcusu
bunun kahin surumuydu; urun surumu skor dagiliminin bicimi olabilir).

## Sonraki adim

II. KOL'u bu haliyle kur: part basina regime (confidence) + adet tahmini + ilk-k.
Kapi: havuzlanmis MIKRO'da +0.03, VE coken markalarda loss olmamasi.

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

Gozlenen deseni birebir aciklar: az CP'li brand (UPUN 3.2) calisiyor, cok
CP'li (NIT 24.4) cokuyor -- cunku bugun HER CP kendi basina markalar-arasi
bir karar gerektiriyor. (`probe_tekrar.py` bunu olcuyor.)

## Felsefi kirilma

Simdiye kadar hep **MARKALAR ARASI TRANSFER** yapmaya calistik. Dusen uc arm
(ozkalib, cluster, S4) da bunu yapiyordu: part-ici baglami, markalar-arasi bir
siniflandiriciyi iyilestirmek icin kullanmak.

Ama en guclu sinyal transferde degil, **PARCANIN KENDI ICINDE**. Hic
gormedigimiz bir markanin klemensinde de 24 delik BIRBIRININ AYNISIDIR.
Bu bilgi hicbir markadan tasinmaz -- parcanin kendisinden gelir.

**Kendine-benzerlik, TANIMI GEREGI brand-bagimsizdir.**

Bu, "24 zor markalar-arasi karar"i -> "1 markalar-arasi karar + 23 part-ici
esleştirme"ye indirir. Model o 1 kararda zaten iyi (NIT parcalarinin %76'sinda
dogru bir secenek ilk 50'de).

## Ilke KONUMLA SINIRLI DEGIL

Bir klemenste CP'ler yalnizca dizilişte degil, HER NITELIKTE aynidir:

| nitelik | part-ici tutarlilik |
|---|---|
| delik yaricapi | hepsi ayni bore capinda |
| depth | hepsi ayni |
| direction | hepsi paralel |
| adim | sabit lattice araligi |
| mouth bicimi | ayni huni profili |

**Sonuc 1 -- ADET GEOMETRIDEN OKUNUR.** Bir klemensteki CP sayisi ~ B-rep'teki
KIPSEL YARICAPLI silindir sayisi. Ust-k deneyinde +0.1533 kazandiran "adet",
buyuk olcude OGRENMEYE GEREK OLMADAN elde edilebilir.

**Sonuc 2 -- KIPSEL IMZA SUZGECI.** Parcanin kipsel yaricapina/derinligine
UYMAYAN candidate supheli. Bu, markalar-arasi bir siniflandiriciya hic ihtiyac
duymadan yanlis pozitifleri kirpar.

**Sonuc 3 -- KATALOG ONCELI (bedava saglamlik).** Klemens adimlari
STANDARTTIR (3.5 / 5.0 / 5.08 / 7.5 / 10.16 mm). Olculen adimi en yakin
katalog degerine oturtmak, gurultuye karsi bedava bir duzeltmedir.

## VII. KOL (YENI, ONCELIK 1) -- PARCA-ICI SABLON

**VII.1** Parcanin kipsel imzasini cikar (yaricap, depth, direction, adim).
**VII.2** Adet = kipsel yaricapli silindir sayisi (geometrik, ogrenmesiz).
**VII.3** Kipsel imzaya uymayan adaylari kirp.
**VII.4** En guvenli tohumdan otelemeyle uret; her uretilen konumda YONU
YENIDEN sec (K2.1'in cokme sebebi direction kopyalamaydi).
**VII.5 Kapi:** coken markada F1 >= 0.30, calisan markada loss YOK.

**Neden bu arm digerlerinden farkli:** hicbir markadan bilgi tasimiyor.
Gorulmemis brand sinavinda BOZULMASI icin bir sebep yok -- oysa bugune kadar
dusen her arm, tam da o sinavda bozuldugu icin dustu.

## Bu ilke, DUSEN kollari da acikliyor

`ozkalib` / `cluster` / S4 part-ici bilgiyi SKORA katti; ama karar kurali zaten
goreli oldugu icin bu tekrar oldu ve noise ekledi. Parca-ici bilginin dogru
kullanimi skoru duzeltmek DEGIL, **URETMEK ve KIRPMAK**.

---

# TEKRAR SONDASI SONUCU -- HIPOTEZ DOGRULANDI

Makbuz `results/tekrar_sondasi_d6.json`.

| brand | CP/p | ILK GT sirasi | ortanca | SON GT sirasi | ilk-k icinde | **tekrar tavani** | adim mm |
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
coken brand arasindaki fark TAM OLARAK BUDUR.**

Bu ayni zamanda ust-k deneyindeki celiskiyi de acikliyor: UPUN'da adet vermek
+0.1533 kazandiriyor cunku siralama zaten dogru; NIT'te kazandirmiyor cunku
24. dogru secenek 1065. sirada -- ilk 24'e hicbir zaman giremez.

## Yayilim kolunun tavani ve aritmetigi

Tek bir otelemeyle uretilebilen GT orani: NIT **0.557**, MOR 0.673,
SUPU 0.796, UPUN 0.781. **Bu bir ALT SINIR** -- probe TEK oteleme denedi,
klemenslerde cogu zaman 2 sira/2 fold vardir.

| senaryo | GT agirlikli F1 (D6) |
|---|---|
| bugun | **0.2088** |
| yayilim tavanin %60'ini alirsa | **0.4750** |
| yayilim tavanin tamamini alirsa | ~0.78 (precision mukemmel varsayimiyla) |

NIT tek basina 0.0089 -> 0.7155 tavanina sahip ve D6 GT'sinin %45.7'si.

## Karar

**VII. KOL (part-ici sablon + yayilim) ONCELIK 1'e alindi.** Gerekcesi
artik varsayim degil measurement:
 * loss yeri kesin (tekrarlar, ilk degil)
 * tavani olculu (NIT 0.557, alt sinir)
 * mekanizmasi brand-bagimsiz (parcanin kendi otelemesi)
 * bugune kadar dusen her kolun dustugu yerde (gorulmemis brand) bozulmasi
   icin YAPISAL bir sebep yok

**Adim degerleri (10.5 / 13.0 / 12.0 / 21.0 mm) standart klemens
adimlarindan (3.5-7.5 mm) buyuk** -- probe muhtemelen 2x harmonigi buluyor.
Kol kurulurken adim, katalog degerlerine ve alt harmoniklere karsi
sinanmali; yoksa uretilen izgara her ikinci CP'yi atlar.

---

# VII.0 SONUCU -- 0.70 YAPISAL OLARAK MUMKUN (ceiling degil, ULASIM sorunu)

Makbuz `results/coklu_kafes.json`. **Model yok, yalnizca GT geometrisi.**

| brand | 1 lattice | **2 lattice** | 3 lattice | adim1 | F1 TAVANI |
|---|---|---|---|---|---|
| NIT | 0.500 | **0.983** | 0.983 | 10.50 mm | **0.9914** |
| SUPU | 0.623 | 0.861 | 0.864 | 10.00 mm | 0.9270 |
| MOR | 0.498 | 0.799 | 0.811 | 6.37 mm | 0.8956 |
| UPUN | 0.689 | 0.810 | 0.810 | 15.30 mm | 0.8950 |

**TEK OTELEME OLCUMUM YANILTICIYDI.** NIT'i 0.557 diye olcmustum; gercekte
IKI kafesle **0.983**. NIT parcalari IKI SIRALI yapilar (on/arka ya da iki
fold) ve tek kafesle bakmak yarisini goruyordu.

Adimlar 6.37-15.30 mm, hepsi fiziksel olarak makul -- dejenere kucuk adim yok,
yani bu bir arama artefakti degil.

## Yeni aritmetik

| | GT agirlikli F1 |
|---|---|
| bugun | 0.1985 |
| **YAPISAL TAVAN (2-3 lattice)** | **0.9474** |
| tavanin %50'si yakalanirsa | 0.4852 |
| tavanin %60'i | 0.5685 |
| **tavanin %70'i** | **0.6632** |
| **tavanin %80'i** | **0.7580** |

**0.70, tavanin ~%74'unu yakalamak demek.** Artik "yapisal olarak imkansiz"
degil; **ulasim sorunu**.

## AMA -- kritik durustluk kaydi

Bu bir **KAHIN TAVANIDIR**: "GERCEK CP'ler verildiginde, 2-3 kafesle
tanimlanabilirler mi?" sorusunun cevabi EVET. Urun ise kafesi **GT'yi
bilmeden, gurultulu adaylardan** bulmak zorunda. Asil zorluk orada.

Yani bu measurement sunu kanitlar: **yapi VAR ve GUCLU.** Bu, gerekli bir kosuldu
ve artik saglandi. Yeterli oldugunu gostermez.

## 0.70 icin revize edilmis cevap

- **Yapisal ceiling:** 0.9474 (engel DEGIL)
- **0.70 icin gereken:** tavanin ~%74'u
- **Kiyas:** bugun tavanin %21'ini yakaliyoruz
- **Karar:** 0.70 **konusulabilir** ama ucurumu kapatan sey kafesi VERIDEN
  bulma basarisi olacak. Onu olcmeden sayi vermem.

**Sonraki measurement (VII.0b):** lattice, GT yerine ADAYLARDAN bulunabiliyor mu?
Ayni acgozlu arama, girdi olarak model skorunun en yuksek N adayini alsin.
Kapsama orani duserse, dusus miktari "ulasim acigi"nin dogrudan olcusudur.

---

# VII.2 SONUCU -- ADET GEOMETRIDEN OKUNAMIYOR (bu haliyle)

Makbuz `results/adet_geometri.json` (397 part, model yok).

| brand | gercek adet | n_kipsel | ortanca error | isabet(<=1) |
|---|---|---|---|---|
| NIT | 24.4 | 104.5 | 75 | 0.00 |
| SUPU | 3.3 | 82.9 | 52 | 0.00 |
| UPUN | 3.1 | 51.8 | 25 | 0.00 |
| MOR | 3.8 | 97.3 | 84 | 0.00 |

Uc tahmincinin ucu de AGIR bicimde FAZLA sayiyor; tam-isabete yakin ratio her
yerde **0.00**.

**Sebep:** B-rep'te yuzlerce silindir var (orneklerde 474 / 266 / 34) ve cogu
vida deligi, ic yapi, pah. Kipsel yaricap kovasi bile ~50-100 silindir
tutuyor. "Kipsel yaricap = CP yaricapi" varsayimi YANLIS.

**Hipotez bu haliyle KAPANDI.** Kurtarma yolu var ama olcmeden varsayilmaz:
silindirleri AGZI OLAN (disaridan erisilebilir) olanlarla sinirlamak
(`mouth_a`/`mouth_b` alanlari onbellekte VAR). Ayri madde olarak eklendi.

**Onemli sonuc:** ust-k deneyinin gosterdigi +0.1533'luk adet kazanci
GERCEKTIR, ama adet KOLAY elde edilmiyor. Adet tahmini artik ogrenmeli bir
alt problem (part ozniteliklerinden regresyon) ya da lattice adimindan
turetme (body uzunlugu / adim) olarak ele alinmali.

---

# VII.0m -- KAFES GT OLMADAN BULUNABILIYOR (mesh havuzu + urun kutusu)

Makbuz `results/kafes_v2_d6.json`. Arama olcutu **GT'SIZ** (izgaraya dusen
ADAY sayisi x doluluk); GT yalnizca degerlendirmede.

| brand | 1k | 2k | 3k | 4k | 5k | **6 lattice** | KAHIN | acik |
|---|---|---|---|---|---|---|---|---|
| **NIT** | 0.097 | 0.316 | 0.450 | 0.565 | 0.623 | **0.676** | 0.983 | 0.307 |
| MOR | 0.144 | 0.163 | 0.385 | 0.447 | 0.466 | 0.510 | 0.811 | 0.301 |
| SUPU | 0.165 | 0.244 | 0.326 | 0.369 | 0.440 | 0.494 | 0.864 | 0.370 |
| UPUN | 0.093 | 0.183 | 0.247 | 0.308 | 0.351 | 0.394 | 0.810 | 0.416 |

**NIT'te kapsama 0.001 -> 0.099 -> 0.676** (oklit / B-rep+urun kutusu /
mesh+urun kutusu) ve 6 kafeste HALA TIRMANIYOR (0.623 -> 0.676).

Uc kusurun ucu de duzeltildikten sonra arm CANLI:
1. arama olcutu GT'siz oldu
2. degerlendirme URUN KUTUSUNA cevrildi (oklit degil)
3. cipa B-rep degil MESH havuzu (B-rep NIT'te CP'lerin uzerinde durmuyor)

## Aritmetik -- uretilen nokta sayisina gore

| senaryo | GT agirlikli F1 (d6) |
|---|---|
| bugun | 0.1789 |
| **izgaradan TAM ADET kadar uret** | **0.5876** |
| adedin 1.5 kati | 0.4701 |
| 2 kati | 0.3917 |
| 3 kati (kotu adet kontrolu) | 0.2938 |

**Adet kontrolu berbat olsa bile (3x fazla uretim) bugunku tabanin USTUNDE.**

## KRITIK EKSIK -- direction hesaba KATILMADI

Bu kapsama olcumu yalnizca KONUMU kontrol ediyor (lateral 2mm / axial 40mm).
Robot metrigi ayrica **ISARETLI ACI <= 10 derece** istiyor. Yani yukaridaki
sayilar, uretilen her konumda YONUN DE dogru secilebildigini VARSAYIYOR.

Bu tam olarak VII.4 maddesi ve K2.1'in coktugu yer: onceki deneme yonu
KOPYALAMIS ve robot metrigi -0.0100 dusmustu (tespit +0.0126 iken).

**Sonraki belirleyici measurement:** uretilen konumlarda direction-bankasi secenekleri
arasindan direction YENIDEN secilirse, kapsamanin ne kadari ISARETLI ACI kutusundan
da gecer? O sayi gelmeden yukaridaki F1 tahminleri VAAT DEGILDIR.

---

# VII.4 SONUCU -- YON KURTARILIYOR (arm CANLI, ama SUZGEC sart)

Makbuz `results/kafes_yon_d6.json`. TAM kabul kutusu: lateral <=2mm, axial
<=40mm, **ISARETLI aci <=10 derece**.

| brand | yalniz KONUM | **KONUM+YON** | direction kaybi | uret/part |
|---|---|---|---|---|
| **NIT** | 0.676 | **0.527** | 0.149 | 179 |
| MOR | 0.510 | **0.495** | 0.014 | 108 |
| SUPU | 0.494 | 0.415 | 0.079 | 101 |
| UPUN | 0.394 | 0.333 | 0.061 | 87 |

**Yon kaybi kucuk (0.014-0.149).** K2.1'in coktugu yer buydu ve orada direction
KOPYALANIYORDU; uretilen konumda direction-bankasi seceneklerinden YENIDEN secilince
loss kuculuyor.

## Aritmetik -- SUZGEC olmadan arm ISE YARAMAZ

| senaryo | GT agirlikli F1 (d6) |
|---|---|
| bugun | 0.1789 |
| **suzgecsiz** (179 nokta/part yayinla) | **0.0845** |
| adet kadar uret (n=k) | **0.4769** |
| 2 kati uret (n=2k) | 0.3179 |

**Kritik:** ham haliyle arm bugunku tabandan KOTU (0.0845 < 0.1789), cunku
part basina 179 nokta yayinlamak kesinligi oldururyor. Kolun butun degeri
SUZGECTE:

| brand | suzgecsiz | n=k | bugun |
|---|---|---|---|
| NIT | 0.126 | **0.527** | 0.009 |
| MOR | 0.034 | **0.495** | 0.178 |
| SUPU | 0.026 | 0.415 | 0.449 |
| UPUN | 0.023 | 0.333 | 0.536 |

**Ve regime ayrimi SART:** NIT/MOR'da arm muazzam kazandiriyor (0.009->0.527,
0.178->0.495) ama SUPU/UPUN'da KAYBETTIRIYOR (0.449->0.415, 0.536->0.333).
Kol ancak COKEN markalarda devreye girmeli -- bu zaten VII.5 kapisinin sarti.

## Sonraki adimlar (oncelik sirasi)

1. **SUZGEC**: uretilen 179 noktayi ~k'ya indir. Uc mekanizma: (a) izgara
   noktasinda GERCEKTEN delik var mi (isin/mesh dogrulamasi, VII.0l),
   (b) kipsel imza uyumu (VII.1), (c) mevcut model skoru.
2. **ADET (k)**: n=k senaryosu adedin bilindigini varsayiyor. VII.2 curudu
   (silindir sayimi); kalan yollar lattice adimindan turetme (VII.2c) ve
   ogrenmeli regresyon (VII.2d).
3. **REJIM KAPISI**: arm yalnizca coken markalarda. Ayirt edici olcu S7'de
   var (poz-neg skor ayrimi) ama urun surumu gerekiyor.

---

# VII SONUCU -- KOL CANLI AMA DARBOGAZ YINE AYNI YERDE

## Suzgec olcumu (uretilen noktalari model skoruyla sirala, ilk k)

| brand | bugun | KOL (ilk k) | KAHIN direction | arm/kahin |
|---|---|---|---|---|
| **NIT** | 0.009 | **0.077** | 0.527 | **%15** |
| MOR | 0.178 | 0.120 | 0.495 | %24 |
| SUPU | 0.449 | 0.155 | 0.415 | %37 |
| UPUN | 0.536 | 0.204 | 0.333 | %61 |

| uygulama | GT agirlikli F1 (d6) |
|---|---|
| baseline | 0.1789 |
| arm KURESEL uygulanirsa | **0.1129** (TABANDAN KOTU) |
| **arm yalniz NIT'te (regime kapili)** | **0.2184** (**+0.0394**) |

## Iki asamali darbogaz

**Asama 1 -- konum uretimi CALISIYOR.** Kafes GT'siz bulunuyor, NIT'te
kapsama 0.676; direction KAHIN gibi secilirse 0.527.

**Asama 2 -- direction SECIMI cokuyor.** Gercek selector kullanildiginda 0.527 ->
0.122 (kipsel direction) -> 0.077 (ilk k). Yani **dogru direction MEVCUT ama model onu
SECEMIYOR.**

Bu, S7'nin bulgusunun aynisi: NIT'te poz-neg skor ayrimi 0.05. Yani yayilim
kolu konum sorununu cozuyor, **ama direction secimi ayni zayif skora geri
bagimli** ve orada tikaniyor.

**KIPSEL YON denendi** (parcadaki butun CP'ler paralel; yonu nokta basina
degil part basina oy birligiyle sec): NIT'i 0.054 -> 0.122 ile IKI KATINA
cikardi ama MOR'u 0.221 -> 0.139 dusurdu. Net etki sinirli.

## Durust bilanco

- Kol **kuresel uygulanamaz** (0.1129 < 0.1789).
- **Rejim kapili** haliyle **+0.0394** getiriyor (0.1789 -> 0.2184). Bu
  gercek ama mutevazi bir kazanc.
- Kolun TAVANI (0.527 NIT) ile GERCEKLESENI (0.077) arasindaki 7 fold fark,
  tamamen YON SECIMINDEN geliyor.

## Bundan sonrasi

Yayilim kolunu buyutmenin yolu daha iyi lattice aramasindan DEGIL, **uretilen
konumda direction secmekten** geciyor. Uc candidate:
1. **III. KOL (analitik direction)** -- B-rep agzinda direction hesaplanabilir,
   ogrenilmesi gerekmez. Simdi cok daha degerli: yayilim kolunun darbogazi
   dogrudan bu.
2. **Isin/mesh dogrulamasi** -- uretilen konumda hangi yonde gercekten delik
   var? Bu da yonu GEOMETRIDEN verir.
3. Yon icin ayri, kucuk bir siniflandirici (mevcut skorun zayif oldugu yer).

**0.75 hedefi acisindan:** yapisal ceiling 0.9474 duruyor, ama ona ulasmanin
onunde artik tek somut engel var ve adi konmus durumda: **uretilen konumda
direction secimi.**

---

# III. KOL SONUCU -- YON HESAPLANABILIR, AMA ISARET MARKAYA GORE DEGISIYOR

Makbuz `results/analitik_yon.json`. B-rep agzi + ANALITIK silindir ekseni.

| brand | KONUM | +axis (ISARETSIZ) | merkez->mouth | mouth->merkez |
|---|---|---|---|---|
| SUPU | 0.777 | 0.612 | **0.514** | 0.208 |
| MOR | 0.738 | 0.700 | **0.586** | 0.536 |
| UPUN | 0.950 | 0.793 | 0.130 | **0.743** |
| **NIT** | **0.011** | 0.000 | 0.000 | 0.000 |

## Bulgu 1 -- YON PROBLEMI TEK BIR BITE INIYOR

Analitik axis ZATEN dogru: unsigned ceiling 0.612-0.793. Geriye kalan tek
karar **part basina BIR ISARET BITI**. Bu, secenek basina direction secmekten
(bugun 24 secenek arasindan) fold fold kolay ve model skoruyla oylanabilir.

Kiyas: yayilim kolunda direction secimi 0.527 -> 0.077 dusuruyordu. Burada dogru
isaretle 0.61-0.79 dogrudan elde ediliyor.

## Bulgu 2 -- GT'NIN YON SOZLESMESI MARKALAR ARASINDA TUTARSIZ

SUPU ve MOR'da GT yonu DISARI, UPUN'da ICERI bakiyor. Tek bir kuresel
sozlesme YOK. Bu, direction kullanan HER yaklasimi etkiler ve bugune kadar kayda
gecmemis bir olgudur.

Ihtimaller: (a) manufacturer JSON'larinda tanim farkli, (b) fiziksel olarak farkli
(bazi klemenslerde tel girisi karsi yuzden), (c) GT uretim hatasi. **Hangisi
oldugu arastirilmali** -- eger (c) ise D7 dahil butun direction olcumleri etkilenir.

## Bulgu 3 -- NIT'te B-rep AGIZLARI CP'lerin UZERINDE DEGIL

KONUM 0.011. Yani analitik direction NIT'e HIC yardim edemez; oradaki darbogaz
temsil (B-rep agzi yok), direction degil. NIT icin tek yol mesh tabanli kalir.

Bu, ayni acigin UCUNCU bagimsiz olcumu (daha once: GT'ye oklit ortanca
2.44mm; lattice cipasi 0.099).

## Kolun degeri

SUPU/MOR/UPUN icin analitik direction, dogru isaretle **0.61-0.79 recall** veriyor
-- bugunku F1'leri 0.449/0.178/0.536. NIT icin degeri YOK.

**Sonraki adim:** part basina sign bitini SEC (model skoru oyu / kanonik
cerceve / lattice tutarliligi) ve uctan uca olc.

---

# GT YON SOZLESMESI COZULDU -- sign YEREL bir ozellik

Makbuz `results/gt_yon_sozlesmesi.json`. Olcu: GT yonu, parcanin agirlik
merkezinden DISARI mi bakiyor?

| brand | GT | disari orani | **part ici baskinlik** |
|---|---|---|---|
| NIT | 1222 | 1.000 | **1.000** |
| CWT | 2758 | 0.836 | 0.981 |
| A-B | 1587 | 0.653 | 0.972 |
| MOR | 586 | 0.937 | 0.945 |
| WEI | 3765 | 0.623 | 0.935 |
| SE | 244 | 0.167 | 1.000 |
| EFX | 198 | 0.148 | 0.880 |
| **UPUN** | 433 | 0.589 | **0.597** |
| **DIN** | 155 | 0.515 | **0.540** |

## Uc olgu

1. **Cogu markada sign PARCA ICINDE TUTARLI** (baskinlik 0.88-1.00).
   "Parca basina bir bit" fikri cogunluk icin gecerli.
2. **Yon MARKAYA gore degisiyor:** NIT tamamen DISARI (1.000), SE/EFX
   agirlikla ICERI (0.167/0.148). Kuresel sozlesme gercekten YOK.
3. **UPUN ve DIN part ICINDE BILE karisik** (0.597/0.540). Fiziksel olarak
   anlamli: iki yuzunde de giris olan klemensler.

## Tasarim sonucu

Isaret, **kuresel agirlik merkezinden DEGIL, YEREL DIS NORMALDEN** turetilmeli.
Iki yuzlu bir parcada her agzin kendi dis normali dogru isareti verir; tek bir
part biti onlari ayiramaz.

Bu, III. KOL'un tasarimini belirler:
- direction = analitik silindir ekseni (zaten dogru: unsigned ceiling 0.61-0.79)
- sign = mouth noktasindaki YEREL dis normal (mesh yuzeyinden)
- artik "brand sozlesmesi" diye bir sey aramaya gerek yok

## NIT icin ek evidence

NIT'in direction sozlesmesi KUSURSUZ tutarli (disari 1.000, baskinlik 1.000).
Yani NIT'teki sorun YON DEGIL, tamamen TEMSIL: B-rep agizlari CP'lerin
uzerinde yok (KONUM 0.011). Bu, ayni acigin DORDUNCU bagimsiz olcumu.

---

# YON, MESH GEOMETRISINDEN OKUNUYOR -- 0.70'i menzile sokan bulgu

Makbuz `results/mesh_yon.json`. Model yok, B-rep yok; yalniz mesh.

| brand | KONUM | **tepe normali** | komsu ort. | kipsel | duzlem |
|---|---|---|---|---|---|
| NIT | 1.000 | **0.800** | 0.000 | 0.002 | 0.002 |
| SUPU | 0.996 | **0.936** | 0.035 | 0.035 | 0.035 |
| UPUN | 1.000 | **0.878** | 0.032 | 0.030 | 0.032 |
| MOR | 0.825 | **0.617** | 0.000 | 0.000 | 0.022 |

**Mesh tepesinin KENDI normali GT yonunu ISARETLI olarak NIT'te %80,
SUPU'da %93.6 tutuyor.** Fizik basit: bir deligin agzinda dis yuzeyin normali
zaten delik eksenidir.

**Komsuluk ortalamalari COKUYOR** (0.000-0.035) cunku 2mm'lik komsuluk dis yuz
normaliyle DELIK DUVARININ normallerini karistirip sinyali yok ediyor. Dogru
sinyal TEK TEPENIN kendi normali -- ortalama almak zarar veriyor.

## Neden bu, 0.70 tartismasini degistiriyor

| | bugun | mesh normali |
|---|---|---|
| secenek / part | 4857 | **~350** |
| direction / candidate | 24 | **1** |
| NIT yonlu recall | 0.8429 | 0.800 |

Secicinin isi:
* bugun : 4857 secenek icinden ~24 dogruyu bul -> pozitif yogunlugu **1/202**
* yeni  :  350 candidate icinden ~24 dogruyu bul   -> pozitif yogunlugu **1/15**

**Pozitif yogunlugu 13.5 KAT artiyor, recall bedeli yalnizca 0.04.**

Ve S7'nin teshisi hatirlanirsa: cokusun sebebi POZ-NEG SKOR AYRIMIYDI
(NIT'te 0.050). Ayrim, sinif dengesizligi 13.5 fold azalinca DOGRUDAN
iyilesmeli -- cunku model artik 202 celdiriciyle degil 15 celdiriciyle
yarisiyor.

## DURUSTLUK KAYDI

Olcu "kutuda EN AZ BIR tepe" seklinde (pool recall olcumleriyle AYNI
mantik, dolayisiyla 0.8429 ile kiyaslanabilir). Ama axial tolerans 40mm
oldugu icin kutu UZUN bir silindir; icinde cok tepe var. Bu sayi bir
TAVANDIR -- selector o tepeyi SECEBILMELI.

Yine de kritik fark su: bugun selector hem KONUMU hem YONU secmek zorunda;
mesh normaliyle direction KARAR OLMAKTAN CIKIYOR, geriye yalnizca konum secimi
kaliyor.

## SONRAKI OLCUM (belirleyici)

Yon-bankasini KALDIR, her adaya TEK direction ver (kendi mesh normali), kademe2'yi
tek degiskenli kos. Kapi: **uctan uca robot F1 >= bugunku 0.3124.** Recall
bedeli 0.04 iken pozitif yogunlugu 13.5 fold artiyorsa precision cok daha
fazla artmali.

---

# MESH NORMALI MEKANIZMASI OLDU -- ve 0.800 tahminim SISIKTI

Makbuz `results/aday_normal_tavan.json`.

| brand | KONUM | **adayin KENDI normali** | banka (24 secenek) |
|---|---|---|---|
| NIT | 0.843 | **0.029** | 0.843 |
| SUPU | 0.951 | 0.631 | 0.921 |
| UPUN | 0.965 | 0.546 | 0.962 |
| MOR | 0.869 | 0.635 | 0.869 |
| **TOPLAM** | — | **0.334** | **0.893** |

## Kusur nerede

"Mesh normali yonu NIT'te %80 tutuyor" olcumum, kutudaki **6000 TEPENIN
HERHANGI BIRINI** kabul ediyordu. Mekanizma ise ~490 SECILMIS candidate uzerinde
calisacakti. O candidates cogunlukla deligin DIS DUZ YUZUNDE degil BORU
DUVARINDA oturuyor ve oradaki normal eksene DIKTIR.

Yani dogru normali tasiyan tepeler candidate havuzunda YOK. Sonda olcutu
mekanizmayla UYUSMUYORDU.

Ara adim da ayni yone sign etmisti: banka-yaklasimli suzgec korpusunda
yonlu recall 0.8926 -> 0.3679 dusmustu. Iki bagimsiz measurement ayni sonucu
veriyor.

## Bugunun BESINCI measurement kusuru -- ve tek IYIMSER olani

| # | kusur | direction |
|---|---|---|
| 1 | lattice aramasinda KAHIN hedef | kotumser (kolu olu gosterdi) |
| 2 | OKLIT kutu (urun kutusu yerine) | kotumser |
| 3 | B-rep cipasi (mesh yerine) | kotumser |
| 4 | tek oteleme (coklu lattice yerine) | kotumser |
| **5** | **"herhangi bir tepe" olcutu** | **IYIMSER** |

Ilk dordu kolu haksiz yere OLU gosteriyordu; besincisi haksiz yere CANLI.
Ders ayni: **olcutun MEKANIZMAYLA birebir ayni olmasi gerekir.**

## 0.70 tahminine etkisi

Mesh-normal rotasi KAPANDI. Tahmin ~0.42'de kaliyor; 0.70 icin NIT tipi
markalarda TEMSIL acigini kapatan baska bir sey gerekiyor.

**Kapanmayan tek somut candidate:** deligin DIS YUZ tepelerini candidate havuzuna
sokmak. Bugun pool segmentasyon olasiligina gore seyreltiliyor ve o tepeler
eleniyor olabilir. Bu, "candidate uretimi" kolunun yeni ve olculebilir bir alt
maddesi.
