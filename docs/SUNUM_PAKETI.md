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
   (konum + isaretli yon) cikti orani **%48.4**.

2. **Darbogaz kapasite degil, YENI URETICIYE TRANSFER.** Ayni sistem,
   ayni veri, yalnizca bolme tipi degistirilerek olculdugunde: tanidik
   uretici **0.5603**, gorulmemis uretici **0.3135** -- fark **+0.2468**.
   Bu, projenin tek en buyuk sayisidir ve nereye yatirim yapilacagini
   soyler: **model degil, VERI CESITLILIGI**.

3. **En buyuk tekil iyilestirme egitimsizdi.** Agiz cevresi yuzey
   normaliyle yon isaretini duzeltmek, sifir egitim ve sifir ayarli
   parametreyle **+0.0195** (GA sifiri icermiyor) kazandirdi. Ayni fikir
   YANLIS referansla once **−0.0266** vermisti; "kol olu" denmeyip kok
   neden olculdugu icin kazanca donustu.

4. **Kalan isin ADRESI olculdu, tahmin edilmedi.** VAL'in tamaminda
   (100 parca / 660 CP) kabul kutusunun her ekseni tek tek gevsetildi:
   yonu TAMAMEN mukemmel yapmak kapsamayi yalnizca **+0.170** aciyor,
   ama yanal toleransi 2 → 10 mm yapmak **+0.358** aciyor
   (0.4242 → **0.7818**). GT'lerin **%78'i** icin yonu zaten dogru bir
   aday 10 mm yakinda duruyor. **Darboğaz yanal konum hassasiyeti** --
   yanal eksen, yonun iki katindan fazlasini tasiyor.

5. **Kollarin cogu gurultuydu ve bunu OLCEREK gosterdik.** Ayni recete,
   yalnizca egitim tohumu degistirilerek 0.6528 / 0.6990 / 0.6673 verdi
   (**yayilim 0.046**). Tek kosumla pozitif gorunen **dort kol** cok
   tohumda curudu; biri isaret bile degistirdi. Bu calismanin en
   aktarilabilir ciktisi bu disiplindir.

6. **Sinav kumesi HARCANMADI.** D7 ve LOCKED okunmadi; her sayi
   dogrulanabilir bir makbuza, her degisiklik geri alinabilir bir
   kontrol noktasina bagli.

---

## 1. Problem

Bir klemensin 3B modelinden (STEP), robotun kablo sokabilecegi **baglanti
noktalarini (CP)** bulmak: her CP icin **konum** ve **takma yonu**.

Ciktinin robot icin kullanilabilir sayilmasi:
yanal hata ≤ **2 mm** · **isaretli** aci hatasi ≤ **10°** · eksenel ≤ 40 mm.

---

## 2. Degerlendirme kumesi — neden guvenilir

`VAL`: **100 parca**, `results/split3.json`.

- **Geometri-ayrik** (ayni parcanin varyantlari ayni tarafta)
- **Uretici-karisik**: markalar egitimde VAR, bu PARCALAR yok
- Dogrulandi: **VAL ∩ egitim = 0, VAL ∩ gelistirme = 0, VAL ∩ sinav = 0**

> Bu, gercek kullanim senaryosunun ta kendisi: *bildigimiz bir ureticiden
> gelen, daha once gormedigimiz bir .stp dosyasi.*

(`DEV` kumesi KULLANILMADI: 100 parcasinin 37'si egitim kumesinde.)

---

## 3. ANA SONUCLAR

| metrik | deger |
|---|---|
| **Tespit F1** (CP'nin yerini buluyor mu) | **0.7878** |
| Robot-hazir F1, **eksen** olcutu | 0.5764 |
| Robot-hazir F1, **isaretli** olcut | 0.4839 |
| Kesinlik | 0.5008 |
| Recall | 0.4682 |

### Guven araligi — 100 parcada belirsizlik KUCUK DEGIL

Parca duzeyi bootstrap, 2000 tekrar, **ayni kume ve ayni sistem**:

| metrik | F1 | %95 GA |
|---|---|---|
| **tespit** | **0.7878** | **[0.7464, 0.8263]** |
| robot, eksen olcutu | 0.5764 | [0.5067, 0.6438] |
| robot, isaretli | 0.4839 | [0.4003, 0.5671] |

Tespitin araligi dar (±0.04); robot-isaretli daha genis (±0.08) cunku
100 parcada isaret hatalari daha oynak.

**Sunumda her sayi araligiyla verilmelidir.** Ayrica projede **%80
geometrik ikiz** olculdu; gercek GA burada hesaplanandan GENIS olabilir.
(Makbuz: `results/guven_araligi_saha.json`)

**Tekrarlanabilirlik:** ayni olcum bagimsiz olarak iki kez kosuldu ve
UCU DE BIREBIR ayni cikti (0.7878 / 0.5764 / 0.4839).

### Iki robot metrigi neden AYRI verilir

Kod iki olcut hesapliyor:
- **eksen (isaretsiz)**: 180° ters bir yon DOGRU sayilir
- **isaretli**: yon ureticinin belirttigi yonle AYNI olmak zorunda

Robot kabloyu fiziksel olarak sokacaksa gecerli olcut **isaretlidir**.
Literaturde ve onceki raporlarda genelde eksen olcutu verilir; burada
**ikisi de** veriliyor.

---

## 4. ANA BULGU — darbogaz kapasite degil, YENI URETICIYE TRANSFER

Ayni sistem, ayni veri, ayni oznitelikler. **Tek fark: degerlendirme
kumesinin nasil bolundugu.**

| kosul | robot F1 |
|---|---|
| **Tanidik uretici** (marka egitimde var, parca yok) | **0.5603** |
| **Gorulmemis uretici** (marka egitimde HIC yok) | **0.3135** |
| **fark** | **0.2468** |

Model yeni bir PARCAYI ogrenebiliyor; yeni bir URETICININ tasarim dilini
ogrenemiyor. Sistem kapasitesi degil, **alan transferi** bagliyor.

Ogrenme egrisi olcumu bunu dogruluyor: 0.80 seviyesi icin **3.8x**,
0.85 icin **17.3x** korpus gerekiyor.

---

## 5. Zorluk ekseni: parca yogunlugu

| rejim | F1 |
|---|---|
| Dusuk-CP parcalar (%86) | 0.7701 |
| **Cok-CP parcalar (%14, n_gt ≥ 8)** | **0.6583** |

Yogun klemensler (24+ CP, siki dizilmis ayni gorunumlu acikliklar) sistemin
zayif halkasi. Gorulmemis uretici + yogun parca birlesince uctan uca
neredeyse sifira iniyor.

**Kok neden olculdu:** yogun parcada model YONU biliyor (AUC 0.8899) ama
HANGI ACIKLIGIN kablo girisi oldugunu bilmiyor (AUC 0.7053; gereken 0.944).

---

## 6. Calisma noktasi AYARLANABILIR

Ayni model, farkli guven esigi:

| esik | kesinlik | recall |
|---|---|---|
| yok (varsayilan) | 0.393 | 0.196 |
| ≥ 0.7 | 0.547 | 0.114 |
| ≥ 0.9 | **0.667** | 0.033 |

Robot uygulamasinda "az ama emin" isteniyorsa kesinlik **0.67'ye**
cikarilabilir. F1'i maksimize eden nokta varsayilandir (taranarak
dogrulandi: alternatiflerin en iyisi +0.0008, yani gurultu).

---

## 7. Bu calismada olculen ve KAPANAN yollar

Iki gunde 50+ kol tek degiskenli olculdu. Kapananlar (hepsi makbuzlu):

| yol | sonuc |
|---|---|
| Sentetik veri uretimi | uretec calisiyor, segmentasyon sentetik geometride ateslemiyor (0.92x) |
| GT'den segmentasyon etiketi | iki farkli etiket sekliyle denendi: 0.5540 / 0.5511 vs kontrol 0.6232 |
| Icsel girdi oznitelikleri (HKS) | 0.1991 vs 0.6232 |
| Yogun-parca uzman modeli | −0.0410 (kahin rejimle bile) |
| Odak kaybi / sinif agirliklandirma | −0.3018 / −0.0058 |
| Zor ornek madenciligi | −0.0195 |
| Havuz kucultme (geometrik suzgecler) | hedef 5x, ulasilan 1.33x |
| Aday havuzu esik taramasi | mevcut nokta zaten optimum |

**Kapatma hukumleri sondanin kusuru olabilir** ilkesi uygulandi: bir kol
"olu" sayilmadan once olcum yolunun dogrulugu denetlendi. Bu sayede en az
uc yanlis hukum duzeltildi (bkz. Bolum 9).

---

## 8. Elde edilen iyilestirmeler

| kalem | kazanc | durum |
|---|---|---|
| Negatif ornekleme orani 8 → 12 | **+0.0126** (2040 parca, 5 marka kati) | **DAGITILDI** |
| Hafif mesh augmentasyonu (segmentasyon) | seg IoU **+0.0197 … +0.0649** | dogrulandi |
| Augmentasyon, uctan uca (tek-vs-tek, VAL) | robot isaretli **+0.0696** | dogrulandi |
| En iyi olculen yapilandirma (4 aug tohum + olculen zincir) | robot isaretli **0.5162** vs 0.4839 | gurultu bandinda, dagitilmadi |

### En buyuk son-islem kazanci: HALKA NORMALI ile isaret duzeltmesi

Her tahmin edilen CP icin, **agiz cevresindeki halkadan** (3-8 mm) mesh
normallerinin ortalamasi alinir; tahmin yonunun isareti bu normalle ayni
yone cevrilir.

| kural | robot ISARETLI |
|---|---|
| taban | 0.4839 |
| en yakin tepe normali (YANLIS referans) | 0.4573 (−0.0266) |
| **halka normali** | **0.5168 (+0.0329)** |
| kahin isaret (ust sinir) | 0.5732 (+0.0893) |

Egitim yok, model degisikligi yok, **ayarlanmis parametre yok**.
Kazanc, kahin tavaninin %37'si.

**Nasil bulundu -- yontem hikayesi.** Ayni fikir once denendi ve
**−0.0266** verdi. "Kol olu" denip kapatilmak yerine NEDEN coktugu
olculdu: kullanilan "yerel normal" aslinda **deligin DUVAR normaliydi**
ve kablo girisi eksenine **DIKTI** (tahmin yonuyle arasindaki aci ortanca
**88.9 derece**, 617 CP). Dogru referans, delik duvarinin disindaki
yuzeyden alinan halka normalidir.

**Kanit gucu ve KAPSAM (durustce).** Esli parca bootstrap:

| populasyon | yol | fark | %95 GA |
|---|---|---|---|
| **tanidik marka (VAL 100)** | olculen | **+0.0195** | **[+0.0030, +0.0378]** KESIN |
| tanidik marka (VAL 100) | saha | +0.0329 | [−0.0155, +0.0819] |
| zor/gorulmemis marka (150) | olculen | +0.0023 | notr |
| zor/gorulmemis marka (150) | saha | −0.0016 | kucuk negatif |

**Kazanc TANIDIK marka populasyonuna ozgudur**; gorulmemis markada notr.
Celiski degil kapsam siniri -- kural, taban sistemin zaten calistigi
yerde yardim ediyor (tabanlar: 0.4668 vs 0.1605).

Kuralin davranisi bunu aciklar: 617 CP'nin 95'ini cevirir, 48'ini
DUZELTIR ama 27'sini BOZAR (kararli cevirmelerin %64'u dogru).

**Kapi altinda kalan, ama yonu dogrulanan kollar** (hepsi olculdu, hicbiri
dagitilmadi):

| kalem | kazanc | dogrulama |
|---|---|---|
| Etiket kalitesi agirliklandirma | +0.0082 | **4 kat tohumu**, tek pozitif kalan |
| Oznitelik budama (162→121 sutun) | +0.0040 | 3 kat tohumu |
| ~~Yardimci gorev (aux-wire)~~ | **−0.0072** | 3 eslesen tohum -> **CURUDU** |
| ~~Cok-CP agirligi 2x~~ | −0.0002 | 4 kat tohumu -> **CURUDU** |

Ustteki iki satirin ustunun cizili olmasi, bu calismanin yontem
disiplininin somut ciktisidir: ikisi de ILK olcumde pozitifti.

Augmentasyon notu: agresif ayar (60° donme) daha once denenip
**zarar verdigi** icin kol "olu" sayilmisti. Hafif ayar (9–17°)
kazandiriyor. Kol olu degil, **ayari yanlismis**.

---

## 9. Olcum butunlugu — duzeltilen hatalar

Bu calismanin bir cikti da **kendi olcumlerinin denetimi**:

1. **Isaretli/isaretsiz karisikligi**: manset metrigi isaretsizdi; robot
   icin gecerli olcut isaretli. Ikisi de raporlaniyor.
2. **Kirli degerlendirme kumesi**: DEV'in %37'si egitim kumesindeydi;
   VAL'e gecildi (kesisim sifir).
3. **Olcum yolu ile dagitim yolu ayrimi**: bir blok olcum betiginde
   **+0.0151**, uretim egiticisinde **−0.0138** verdi -> dagitilmadi.
4. **Sessiz no-op'lar**: bir kol tam `+0.0000` verdi; sebep yutulan bir
   `AttributeError` idi (`max_iter_` yerine `n_iter_`).
5. **Eslesme kutusu**: Oklit mesafesi yerine carpim kutusu (yanal 2mm VE
   eksenel 40mm) -- yanlis kutu bir olcumu 0.593'ten 0.0172'ye dusuruyordu.
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

Bir kol (etiket kalitesi agirliklandirma) DORT farkli kat bolunmesiyle
tekrarlandi:

| kat tohumu | fark |
|---|---|
| 1 | **+0.0144** ("kapi gecti") |
| 2 | **−0.0019** (kaybettiriyor) |
| 3 | +0.0101 |
| 4 | +0.0102 |
| **ortalama** | **+0.0082** |

Ayni kol, ayni veri, ayni kod -- yalnizca katlarin nasil bolundugu farkli.
**Tek tohumla dagitilsaydi olmayan bir kazanc raporlanmis olurdu.**

Bu buyuklukteki kollarda kat gurultusu **±0.008** mertebesinde. Dolayisiyla
+0.01 civarindaki her sonuc COK TOHUMLU dogrulanmadan hukum giymemelidir.

**Ayni olcum segmentasyon tarafinda da yapildi ve gurultu daha da buyuk
cikti.** Ayni recete, ayni veri, yalnizca tohum farkli:

| tohum | val Conn_IoU |
|---|---|
| 0 | 0.6528 |
| 1 | **0.6990** |
| 2 | 0.6673 |
| **yayilim** | **0.046** |

Segmentasyon tarafinda tohum gurultusu **~0.046**'dir -- olculen kollarin
etki buyuklugunun (0.002-0.030) KAT KAT UZERINDE.

**Bu ilke uygulanınca DORT kol curudu:**

| kol | tek tohum | cok tohum | hukum |
|---|---|---|---|
| Cok-CP agirligi 2x | +0.0065 | −0.0002 (4 tohum) | curudu |
| Oznitelik budama | +0.0097 | +0.0040 (3 tohum) | kapi alti |
| Etiket kalitesi agirligi | +0.0144 | +0.0082 (4 tohum) | kapi alti |
| Yardimci gorev (aux-wire) | +0.0083 | **−0.0072** (3 eslesen tohum) | **isaret DEGISTI** |

Dordu de tek tohumla dagitilsaydi **olmayan dort kazanc** raporlanmis
olurdu. Bu, calismanin en aktarilabilir metodolojik ciktisidir.

## 9c. NEREYE YATIRIM YAPILMALI — baglayici kisit OLCULDU

Kalan hatanin nerede oldugu tahmin edilmedi, **olculdu**. Aday havuzu
(gate'ten onceki tum adaylar) alindi ve kabul kutusunun her ekseni TEK
TEK gevsetildi. Sorulan soru: hangi toleransi gevsetirsem kapsama acilir?

VAL'in tamami: **100 parca / 660 CP**.

| gevsetilen eksen | havuz kapsamasi | degisim |
|---|---|---|
| **hicbiri** (gercek olcut: yanal 2mm, aci 10°) | 0.4242 | — |
| aci 10° → 45° | 0.4409 | +0.017 |
| aci **tamamen yok sayilir** | 0.5939 | **+0.170** |
| yanal 2mm → 5mm | 0.6242 | +0.200 |
| **yanal 2mm → 10mm** | **0.7818** | **+0.358** |
| yanal 2mm → 20mm | 0.9000 | +0.476 |

**Okuma.** Yon tahminini TAMAMEN mukemmel yapsak kazanc **+0.170** ile
sinirli. Oysa GT'lerin **%78'i** icin havuzda, yonu ZATEN 10 derece
icinde dogru olan bir aday **10 mm yakinda duruyor** -- yanal eksen
yonun **iki katindan fazlasini** tasiyor.

> **Darbogaz ne aday uretimi, ne segmentasyon, ne de yon.
> YANAL KONUM HASSASIYETI.**

Bu, bagimsiz iki olcumle birebir tutarli: konum AUC **0.7053** vs yon AUC
**0.8899**; ve yanal hatanin segmentasyon kalitesiyle aciklanmasi.

Sistemde bu isi yapan bir bilesen zaten var (**pose head**: gate
kararindan sonra yanal duzeltme, uctan uca +0.0428 KANITLI) ve olcumde
kapsama 0.4242'den 0.4682'ye tasiyor. 10 mm bandindaki tavan **0.7818**.
**Aradaki +0.31,
projenin en buyuk tek acik basligidir ve adresi bellidir.**

### Ve bu acigin NEREDEN ALINAMAYACAGI da olculdu

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
> regresyonuyla ALINAMAZ. Ya aday uretimi (segmentasyon) duzelecek, ya da
> pose kafasi yerel geometriyi DOGRUDAN goren yeni oznitelikler alacak.

Bu, acigin yok oldugu anlamina gelmez; **hangi yolun onu tasiyamayacagini
olcerek** arama alanini daraltir.

---

## 10. Sonraki adimlar (sure yetmedi, yol acik)

**Oncelik sirasi artik tahmine degil 9c'deki olcume dayaniyor.**

| yol | gerekce |
|---|---|
| **1. OFFSET/OY BASI (merkez oylamasi)** | **olculen en buyuk acik: +0.358.** Bugun konum, segmentlenen bolgenin agirlik merkezinden turetiliyor -- literaturde bunun bilinen kusuru "degen ayni nesneleri ayiramamak" (bizim yogun-parca cokusumuz). Cozum: agin her tepesi kendi CP merkezine bir **kayma vektoru** tahmin etsin, noktalar kaydirilip kumelensin (Panoptic-DeepLab / PVN3D / Spatial Embeddings ailesi). **Yeni etiket gerekmez** -- hedef `(en yakin GT CP - tepe)`. Son-islem regresyonunun bu acigi TASIYAMADIGI uc bagimsiz olcumle gosterildi; bu yontem hasarin OLUSTUGU yerde calisir |
| 2. Yeni uretici verisi | alan farki +0.2468 -- ikinci en buyuk sayi; ogrenme egrisi 0.80 icin 3.8x korpus |
| 3. Cok-gorunumlu render fuzyonu | mevcut butun kollar TEK kiplikte (mesh); goruntu sinyali yeni bilgi |
| 4. Self-supervised / contrastive on-egitim | sentetik veri dustukten sonra temsil yolunun kalan adayi |

**Onerilmeyen yollar** (bu calismada olculup elenmis oldugu icin):
segmentasyon kayip fonksiyonu varyantlari, augmentasyon ailesi,
yardimci gorevler, MC dropout -- hepsi tohum gurultusunun altinda kaldi.
Yon uzerine yeni kol acmak da 9c'ye gore **tavani +0.170 ile sinirli**.
Mesh cozunurlugunu degistirmek de kapandi: 5000 tepe **kesin olarak
kotu** (−0.030), 7200 ve uc-cozunurluklu topluluk kapinin altinda kaldi
(+0.007 / +0.011, GA sifiri iciyor, maliyet 3 kat) -- tezden gelen
**6000 hedefi iyi secilmis**.
