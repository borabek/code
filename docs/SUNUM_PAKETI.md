# SUNUM PAKETI — klemens kablo girisi (CP) tespiti

Tarih: 2026-08-13 · Tum sayilar OLCULMUS ve makbuzludur.
Kaynak makbuzlar: `results/TABAN_VAL_uc_metrik.json`,
`results/zincir_esli_kiyas.json`, `results/uzman_router_d6.json`,
`results/yeni_konum_d6_TANIDIK.json`

---

## 0. BIR SAYFADA — soylenecek alti cumle

1. **Sistem calisiyor.** Bilinen bir ureticiden gelen, daha once
   GORULMEMIS bir .stp dosyasinda kablo girislerinin yerini
   **%78.8** F1 ile buluyor; robotun dogrudan kullanabilecegi
   (konum + signed direction) cikti orani **%48.4**.

2. **Darbogaz kapasite degil, YENI URETICIYE TRANSFER.** Ayni sistem,
   ayni veri, yalnizca split tipi degistirilerek olculdugunde: tanidik
   manufacturer **0.5603**, gorulmemis manufacturer **0.3135** -- fark **+0.2468**.
   Bu, projenin tek en buyuk sayisidir ve nereye yatirim yapilacagini
   soyler: **model degil, VERI CESITLILIGI**.

3. **En buyuk tekil iyilestirme egitimsizdi.** Agiz cevresi yuzey
   normaliyle direction isaretini duzeltmek, sifir training ve sifir ayarli
   parametreyle **+0.0195** (GA sifiri icermiyor) kazandirdi. Ayni fikir
   YANLIS referansla once **−0.0266** vermisti; "arm olu" denmeyip kok
   neden olculdugu icin kazanca donustu.

4. **Kalan isin ADRESI measured, tahmin edilmedi.** VAL'in tamaminda
   (100 part / 660 CP) kabul kutusunun her ekseni tek tek gevsetildi:
   yonu TAMAMEN mukemmel yapmak kapsamayi yalnizca **+0.170** aciyor,
   ama lateral toleransi 2 → 10 mm yapmak **+0.358** aciyor
   (0.4242 → **0.7818**). GT'lerin **%78'i** icin yonu zaten dogru bir
   candidate 10 mm yakinda duruyor. **Darboğaz lateral konum hassasiyeti** --
   lateral axis, yonun iki katindan fazlasini tasiyor.

5. **Kollarin cogu gurultuydu ve bunu OLCEREK gosterdik.** Ayni recete,
   yalnizca training tohumu degistirilerek 0.6528 / 0.6990 / 0.6673 verdi
   (**yayilim 0.046**). Tek kosumla pozitif gorunen **dort arm** cok
   tohumda curudu; biri sign bile degistirdi. Bu calismanin en
   aktarilabilir ciktisi bu disiplindir.

6. **Sinav kumesi HARCANMADI.** D7 ve LOCKED okunmadi; her sayi
   dogrulanabilir bir makbuza, her degisiklik geri alinabilir bir
   kontrol noktasina bagli.

---

## 1. Problem

Bir klemensin 3B modelinden (STEP), robotun kablo sokabilecegi **baglanti
noktalarini (CP)** bulmak: her CP icin **konum** ve **takma yonu**.

Ciktinin robot icin kullanilabilir sayilmasi:
lateral error ≤ **2 mm** · **signed** aci hatasi ≤ **10°** · axial ≤ 40 mm.

---

## 2. Degerlendirme kumesi — neden guvenilir

`VAL`: **100 part**, `results/split3.json`.

- **Geometri-ayrik** (ayni parcanin varyantlari ayni tarafta)
- **Uretici-karisik**: markalar egitimde VAR, bu PARCALAR yok
- Dogrulandi: **VAL ∩ training = 0, VAL ∩ gelistirme = 0, VAL ∩ exam = 0**

> Bu, gercek kullanim senaryosunun ta kendisi: *bildigimiz bir ureticiden
> gelen, daha once gormedigimiz bir .stp dosyasi.*

(`DEV` kumesi KULLANILMADI: 100 parcasinin 37'si training kumesinde.)

---

## 3. ANA SONUCLAR

| metrik | deger |
|---|---|
| **Tespit F1** (CP'nin yerini buluyor mu) | **0.7878** |
| Robot-hazir F1, **axis** olcutu | 0.5764 |
| Robot-hazir F1, **signed** criterion | 0.4839 |
| Kesinlik | 0.5008 |
| Recall | 0.4682 |

### Guven araligi — 100 parcada belirsizlik KUCUK DEGIL

Parca duzeyi bootstrap, 2000 tekrar, **ayni cluster ve ayni sistem**:

| metrik | F1 | %95 GA |
|---|---|---|
| **tespit** | **0.7878** | **[0.7464, 0.8263]** |
| robot, axis olcutu | 0.5764 | [0.5067, 0.6438] |
| robot, signed | 0.4839 | [0.4003, 0.5671] |

Tespitin araligi dar (±0.04); robot-signed daha genis (±0.08) cunku
100 parcada sign hatalari daha oynak.

**Sunumda her sayi araligiyla verilmelidir.** Ayrica projede **%80
geometrik ikiz** measured; gercek GA burada hesaplanandan GENIS olabilir.
(Makbuz: `results/guven_araligi_saha.json`)

**Tekrarlanabilirlik:** ayni measurement bagimsiz olarak iki kez kosuldu ve
UCU DE BIREBIR ayni cikti (0.7878 / 0.5764 / 0.4839).

### Iki robot metrigi neden AYRI verilir

Kod iki criterion hesapliyor:
- **axis (unsigned)**: 180° ters bir direction DOGRU sayilir
- **signed**: direction ureticinin belirttigi yonle AYNI olmak zorunda

Robot kabloyu fiziksel olarak sokacaksa gecerli criterion **isaretlidir**.
Literaturde ve onceki raporlarda genelde axis olcutu verilir; burada
**ikisi de** veriliyor.

---

## 4. ANA BULGU — darbogaz kapasite degil, YENI URETICIYE TRANSFER

Ayni sistem, ayni veri, ayni oznitelikler. **Tek fark: degerlendirme
kumesinin nasil bolundugu.**

| kosul | robot F1 |
|---|---|
| **Tanidik manufacturer** (brand egitimde var, part yok) | **0.5603** |
| **Gorulmemis manufacturer** (brand egitimde HIC yok) | **0.3135** |
| **fark** | **0.2468** |

Model yeni bir PARCAYI ogrenebiliyor; yeni bir URETICININ tasarim dilini
ogrenemiyor. Sistem kapasitesi degil, **alan transferi** bagliyor.

Ogrenme egrisi olcumu bunu dogruluyor: 0.80 seviyesi icin **3.8x**,
0.85 icin **17.3x** corpus gerekiyor.

---

## 5. Zorluk ekseni: part yogunlugu

| regime | F1 |
|---|---|
| Dusuk-CP parts (%86) | 0.7701 |
| **Cok-CP parts (%14, n_gt ≥ 8)** | **0.6583** |

Yogun klemensler (24+ CP, siki dizilmis ayni gorunumlu acikliklar) sistemin
zayif halkasi. Gorulmemis manufacturer + dense part birlesince uctan uca
neredeyse sifira iniyor.

**Kok neden measured:** dense parcada model YONU biliyor (AUC 0.8899) ama
HANGI ACIKLIGIN kablo girisi oldugunu bilmiyor (AUC 0.7053; gereken 0.944).

---

## 6. Calisma noktasi AYARLANABILIR

Ayni model, farkli confidence esigi:

| threshold | precision | recall |
|---|---|---|
| yok (varsayilan) | 0.393 | 0.196 |
| ≥ 0.7 | 0.547 | 0.114 |
| ≥ 0.9 | **0.667** | 0.033 |

Robot uygulamasinda "az ama emin" isteniyorsa precision **0.67'ye**
cikarilabilir. F1'i maksimize eden nokta varsayilandir (taranarak
dogrulandi: alternatiflerin en iyisi +0.0008, yani noise).

---

## 7. Bu calismada olculen ve KAPANAN yollar

Iki gunde 50+ arm tek degiskenli measured. Kapananlar (hepsi makbuzlu):

| yol | sonuc |
|---|---|
| Sentetik veri uretimi | uretec calisiyor, segmentasyon sentetik geometride ateslemiyor (0.92x) |
| GT'den segmentasyon etiketi | iki farkli etiket sekliyle denendi: 0.5540 / 0.5511 vs kontrol 0.6232 |
| Icsel girdi oznitelikleri (HKS) | 0.1991 vs 0.6232 |
| Yogun-part uzman modeli | −0.0410 (kahin rejimle bile) |
| Odak kaybi / sinif agirliklandirma | −0.3018 / −0.0058 |
| Zor ornek madenciligi | −0.0195 |
| Havuz kucultme (geometrik suzgecler) | hedef 5x, ulasilan 1.33x |
| Aday havuzu threshold taramasi | mevcut nokta zaten optimum |

**Kapatma hukumleri sondanin kusuru olabilir** ilkesi uygulandi: bir arm
"olu" sayilmadan once measurement yolunun dogrulugu denetlendi. Bu sayede en az
uc yanlis verdict duzeltildi (bkz. Bolum 9).

---

## 8. Elde edilen iyilestirmeler

| kalem | kazanc | durum |
|---|---|---|
| Negatif ornekleme orani 8 → 12 | **+0.0126** (2040 part, 5 brand kati) | **DAGITILDI** |
| Hafif mesh augmentasyonu (segmentasyon) | seg IoU **+0.0197 … +0.0649** | dogrulandi |
| Augmentasyon, uctan uca (tek-vs-tek, VAL) | robot signed **+0.0696** | dogrulandi |
| En iyi olculen yapilandirma (4 aug seed + olculen zincir) | robot signed **0.5162** vs 0.4839 | noise bandinda, dagitilmadi |

### En buyuk son-islem kazanci: HALKA NORMALI ile sign duzeltmesi

Her tahmin edilen CP icin, **mouth cevresindeki halkadan** (3-8 mm) mesh
normallerinin ortalamasi alinir; tahmin yonunun isareti bu normalle ayni
yone cevrilir.

| kural | robot ISARETLI |
|---|---|
| baseline | 0.4839 |
| en yakin tepe normali (YANLIS referans) | 0.4573 (−0.0266) |
| **halka normali** | **0.5168 (+0.0329)** |
| kahin sign (ust sinir) | 0.5732 (+0.0893) |

Egitim yok, model degisikligi yok, **ayarlanmis parametre yok**.
Kazanc, kahin tavaninin %37'si.

**Nasil bulundu -- yontem hikayesi.** Ayni fikir once denendi ve
**−0.0266** verdi. "Kol olu" denip kapatilmak yerine NEDEN coktugu
measured: kullanilan "yerel normal" aslinda **deligin DUVAR normaliydi**
ve kablo girisi eksenine **DIKTI** (tahmin yonuyle arasindaki aci ortanca
**88.9 derece**, 617 CP). Dogru referans, delik duvarinin disindaki
yuzeyden alinan halka normalidir.

**Kanit gucu ve KAPSAM (durustce).** Esli part bootstrap:

| populasyon | yol | fark | %95 GA |
|---|---|---|---|
| **tanidik brand (VAL 100)** | olculen | **+0.0195** | **[+0.0030, +0.0378]** KESIN |
| tanidik brand (VAL 100) | saha | +0.0329 | [−0.0155, +0.0819] |
| zor/gorulmemis brand (150) | olculen | +0.0023 | notr |
| zor/gorulmemis brand (150) | saha | −0.0016 | kucuk negatif |

**Kazanc TANIDIK brand populasyonuna ozgudur**; gorulmemis markada notr.
Celiski degil kapsam siniri -- kural, baseline sistemin zaten calistigi
yerde yardim ediyor (tabanlar: 0.4668 vs 0.1605).

Kuralin davranisi bunu aciklar: 617 CP'nin 95'ini cevirir, 48'ini
DUZELTIR ama 27'sini BOZAR (kararli cevirmelerin %64'u dogru).

**Kapi altinda kalan, ama yonu dogrulanan kollar** (hepsi measured, hicbiri
dagitilmadi):

| kalem | kazanc | dogrulama |
|---|---|---|
| Etiket kalitesi agirliklandirma | +0.0082 | **4 fold tohumu**, tek pozitif kalan |
| Oznitelik budama (162→121 sutun) | +0.0040 | 3 fold tohumu |
| ~~Yardimci gorev (aux-wire)~~ | **−0.0072** | 3 eslesen seed -> **CURUDU** |
| ~~Cok-CP agirligi 2x~~ | −0.0002 | 4 fold tohumu -> **CURUDU** |

Ustteki iki satirin ustunun cizili olmasi, bu calismanin yontem
disiplininin somut ciktisidir: ikisi de ILK olcumde pozitifti.

Augmentasyon notu: agresif ayar (60° donme) daha once denenip
**zarar verdigi** icin arm "olu" sayilmisti. Hafif ayar (9–17°)
kazandiriyor. Kol olu degil, **ayari yanlismis**.

---

## 9. Olcum butunlugu — duzeltilen hatalar

Bu calismanin bir cikti da **kendi olcumlerinin denetimi**:

1. **Isaretli/unsigned karisikligi**: headline metrigi isaretsizdi; robot
   icin gecerli criterion signed. Ikisi de raporlaniyor.
2. **Kirli degerlendirme kumesi**: DEV'in %37'si training kumesindeydi;
   VAL'e gecildi (kesisim sifir).
3. **Olcum yolu ile dagitim yolu ayrimi**: bir blok measurement betiginde
   **+0.0151**, uretim egiticisinde **−0.0138** verdi -> dagitilmadi.
4. **Sessiz no-op'lar**: bir arm tam `+0.0000` verdi; sebep yutulan bir
   `AttributeError` idi (`max_iter_` yerine `n_iter_`).
5. **Eslesme kutusu**: Oklit mesafesi yerine carpim kutusu (lateral 2mm VE
   axial 40mm) -- yanlis kutu bir olcumu 0.593'ten 0.0172'ye dusuruyordu.
6. **Vekil metrik ISARET DEGISTIRDI**: yeniden egitilen bir bilesen,
   cevrimdisi vekilde (kabul kutusuna girme orani) **+0.9 puan
   kazaniyordu**; ayni bilesen tam zincirde **−0.0437** verdi (GA sifiri
   icermiyor, orneklerin %0'i pozitif). Sebep yapisal: vekil ortalama
   hatayi olcer, uctan uca metrik ise ZATEN dogru olan bir tahminin
   bozulmasini **cift** cezalandirir (bir dogru gider, bir yanlis gelir).
   **Kural: vekil karar verdirmez, yalnizca hangi adayin tam olcumu hak
   ettigini secer.**

---

## 9b. KAT GURULTUSU — tek olcumle karar verilemez

Bir arm (etiket kalitesi agirliklandirma) DORT farkli fold bolunmesiyle
tekrarlandi:

| fold tohumu | fark |
|---|---|
| 1 | **+0.0144** ("gate gecti") |
| 2 | **−0.0019** (kaybettiriyor) |
| 3 | +0.0101 |
| 4 | +0.0102 |
| **ortalama** | **+0.0082** |

Ayni arm, ayni veri, ayni kod -- yalnizca katlarin nasil bolundugu farkli.
**Tek tohumla dagitilsaydi olmayan bir kazanc raporlanmis olurdu.**

Bu buyuklukteki kollarda fold gurultusu **±0.008** mertebesinde. Dolayisiyla
+0.01 civarindaki her sonuc COK TOHUMLU dogrulanmadan verdict giymemelidir.

**Ayni measurement segmentasyon tarafinda da yapildi ve noise daha da buyuk
cikti.** Ayni recete, ayni veri, yalnizca seed farkli:

| seed | val Conn_IoU |
|---|---|
| 0 | 0.6528 |
| 1 | **0.6990** |
| 2 | 0.6673 |
| **yayilim** | **0.046** |

Segmentasyon tarafinda seed gurultusu **~0.046**'dir -- olculen kollarin
etki buyuklugunun (0.002-0.030) KAT KAT UZERINDE.

**Bu ilke uygulanınca DORT arm curudu:**

| arm | tek seed | cok seed | verdict |
|---|---|---|---|
| Cok-CP agirligi 2x | +0.0065 | −0.0002 (4 seed) | curudu |
| Oznitelik budama | +0.0097 | +0.0040 (3 seed) | gate alti |
| Etiket kalitesi agirligi | +0.0144 | +0.0082 (4 seed) | gate alti |
| Yardimci gorev (aux-wire) | +0.0083 | **−0.0072** (3 eslesen seed) | **sign DEGISTI** |

Dordu de tek tohumla dagitilsaydi **olmayan dort kazanc** raporlanmis
olurdu. Bu, calismanin en aktarilabilir metodolojik ciktisidir.

## 9c. NEREYE YATIRIM YAPILMALI — baglayici kisit OLCULDU

Kalan hatanin nerede oldugu tahmin edilmedi, **measured**. Aday havuzu
(gate'ten onceki tum candidates) alindi ve kabul kutusunun her ekseni TEK
TEK gevsetildi. Sorulan soru: hangi toleransi gevsetirsem kapsama acilir?

VAL'in tamami: **100 part / 660 CP**.

| gevsetilen axis | pool kapsamasi | degisim |
|---|---|---|
| **hicbiri** (gercek criterion: lateral 2mm, aci 10°) | 0.4242 | — |
| aci 10° → 45° | 0.4409 | +0.017 |
| aci **tamamen yok sayilir** | 0.5939 | **+0.170** |
| lateral 2mm → 5mm | 0.6242 | +0.200 |
| **lateral 2mm → 10mm** | **0.7818** | **+0.358** |
| lateral 2mm → 20mm | 0.9000 | +0.476 |

**Okuma.** Yon tahminini TAMAMEN mukemmel yapsak kazanc **+0.170** ile
sinirli. Oysa GT'lerin **%78'i** icin havuzda, yonu ZATEN 10 derece
icinde dogru olan bir candidate **10 mm yakinda duruyor** -- lateral axis
yonun **iki katindan fazlasini** tasiyor.

> **Darbogaz ne candidate uretimi, ne segmentasyon, ne de direction.
> YANAL KONUM HASSASIYETI.**

Bu, bagimsiz iki olcumle birebir tutarli: konum AUC **0.7053** vs direction AUC
**0.8899**; ve lateral hatanin segmentasyon kalitesiyle aciklanmasi.

Sistemde bu isi yapan bir bilesen zaten var (**pose head**: gate
kararindan sonra lateral duzeltme, uctan uca +0.0428 KANITLI) ve olcumde
kapsama 0.4242'den 0.4682'ye tasiyor. 10 mm bandindaki ceiling **0.7818**.
**Aradaki +0.31,
projenin en buyuk tek acik basligidir ve adresi bellidir.**

### Ve bu acigin NEREDEN ALINAMAYACAGI da measured

Pose head'in duzeltmesi 3 mm'ye kirpiliyordu; ilk akla gelen "kirpmayi
gevset" oldu. Olculdu: **kirpmayi tamamen kaldirmak +0.0000 degistiriyor**
-- cunku modelin onerdigi en buyuk duzeltme zaten **2.87 mm** ve
onerilerin **%0.0'i 3 mm'yi asiyor**. Sinirlayan kirpma degil, modelin
kendisi.

Model, buyuk artiklari daha cok gorecek sekilde yeniden egitildi (eslesme
toleransi 15 mm) ve buzulmesi gevsetildi. Ortak degerlendirme kumesinde
sonuc: genis veri **yardim etmiyor**, buzulmeyi gevsetmek **+0.9 puan**
kazandiriyor -- ve en iyi varyantta bile adaylarin **%99.5'ine** 3 mm'den
kucuk duzeltme oneriliyor.

> **Sonuc:** +0.31'lik acik, gate ozniteliklerinden son-islem
> regresyonuyla ALINAMAZ. Ya candidate uretimi (segmentasyon) duzelecek, ya da
> pose kafasi yerel geometriyi DOGRUDAN goren yeni oznitelikler alacak.

Bu, acigin yok oldugu anlamina gelmez; **hangi yolun onu tasiyamayacagini
olcerek** arama alanini daraltir.

---

## 10. Sonraki adimlar (sure yetmedi, yol acik)

**Oncelik sirasi artik tahmine degil 9c'deki olcume dayaniyor.**

| yol | rationale |
|---|---|
| **1. OFFSET/OY BASI (merkez oylamasi)** | **olculen en buyuk acik: +0.358.** Bugun konum, segmentlenen bolgenin agirlik merkezinden turetiliyor -- literaturde bunun bilinen kusuru "degen ayni nesneleri ayiramamak" (bizim dense-part cokusumuz). Cozum: agin her tepesi kendi CP merkezine bir **kayma vektoru** tahmin etsin, noktalar kaydirilip kumelensin (Panoptic-DeepLab / PVN3D / Spatial Embeddings ailesi). **Yeni etiket gerekmez** -- hedef `(en yakin GT CP - tepe)`. Son-islem regresyonunun bu acigi TASIYAMADIGI uc bagimsiz olcumle gosterildi; bu yontem hasarin OLUSTUGU yerde calisir |
| 2. Yeni manufacturer verisi | alan farki +0.2468 -- ikinci en buyuk sayi; ogrenme egrisi 0.80 icin 3.8x corpus |
| 3. Cok-gorunumlu render fuzyonu | mevcut butun kollar TEK kiplikte (mesh); goruntu sinyali yeni bilgi |
| 4. Self-supervised / contrastive on-training | sentetik veri dustukten sonra temsil yolunun kalan adayi |

**Onerilmeyen yollar** (bu calismada olculup elenmis oldugu icin):
segmentasyon loss fonksiyonu varyantlari, augmentasyon ailesi,
yardimci gorevler, MC dropout -- hepsi seed gurultusunun altinda kaldi.
Yon uzerine yeni arm acmak da 9c'ye gore **tavani +0.170 ile sinirli**.
Mesh cozunurlugunu degistirmek de closed: 5000 tepe **kesin olarak
kotu** (−0.030), 7200 ve uc-cozunurluklu ensemble kapinin altinda kaldi
(+0.007 / +0.011, GA sifiri iciyor, maliyet 3 fold) -- tezden gelen
**6000 hedefi iyi secilmis**.
