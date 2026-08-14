# SUNUM PAKETI — klemens kablo girisi (CP) tespiti

Tarih: 2026-08-13 · Tum sayilar OLCULMUS ve makbuzludur.
Kaynak makbuzlar: `results/baseline_VAL_uc_metric.json`,
`results/chain_paired_compare.json`, `results/uzman_router_d6.json`,
`results/new_position_d6_TANIDIK.json`

---

## 0. BIR SAYFADA — soylenecek alti cumle

1. **Sistem calisiyor.** Bilinen a ureticiden gelen, more before
   GORULMEMIS a .stp dosyasinda kablo girislerinin yerini
   **%78.8** F1 with buluyor; robotun dogrudan kullanabilecegi
   (konum + signed direction) cikti ratio **%48.4**.

2. **Darbogaz kapasite not, YENI URETICIYE TRANSFER.** Ayni sistem,
   same data, yalnizca split tipi degistirilerek olculdugunde: familiar
   manufacturer **0.5603**, unseen manufacturer **0.3135** -- diff **+0.2468**.
   Bu, projenin tek en large sayisidir ve nereye yatirim yapilacagini
   soyler: **model not, VERI CESITLILIGI**.

3. **En large tekil iyilestirme egitimsizdi.** Agiz cevresi yuzey
   normaliyle direction isaretini duzeltmek, sifir training ve sifir ayarli
   parametreyle **+0.0195** (GA sifiri icermiyor) kazandirdi. Ayni fikir
   YANLIS referansla before **−0.0266** vermisti; "arm olu" denmeyip kok
   why olculdugu for kazanca donustu.

4. **Kalan isin ADRESI measured, prediction edilmedi.** VAL'in tamaminda
   (100 part / 660 CP) kabul kutusunun each axis tek tek gevsetildi:
   direction TAMAMEN mukemmel yapmak kapsamayi yalnizca **+0.170** aciyor,
   but lateral toleransi 2 → 10 mm yapmak **+0.358** aciyor
   (0.4242 → **0.7818**). GT'lerin **%78'i** for direction already correct a
   candidate 10 mm yakinda duruyor. **Darboğaz lateral konum hassasiyeti** --
   lateral axis, yonun iki katindan fazlasini tasiyor.

5. **Kollarin cogu gurultuydu ve bunu OLCEREK gosterdik.** Ayni recete,
   yalnizca training seed degistirilerek 0.6528 / 0.6990 / 0.6673 verdi
   (**spread 0.046**). Tek kosumla pozitif gorunen **dort arm** very
   tohumda curudu; biri sign bile degistirdi. Bu calismanin en
   aktarilabilir ciktisi this disiplindir.

6. **Sinav set HARCANMADI.** D7 ve LOCKED okunmadi; each number
   dogrulanabilir a makbuza, each degisiklik geri alinabilir a
   kontrol noktasina bagli.

---

## 1. Problem

Bir klemensin 3B modelinden (STEP), robotun kablo sokabilecegi **baglanti
noktalarini (CP)** bulmak: each CP for **konum** ve **takma direction**.

Ciktinin robot for kullanilabilir sayilmasi:
lateral error ≤ **2 mm** · **signed** aci hatasi ≤ **10°** · axial ≤ 40 mm.

---

## 2. Degerlendirme set — why guvenilir

`VAL`: **100 part**, `results/split3.json`.

- **Geometri-ayrik** (same parcanin varyantlari same tarafta)
- **Uretici-karisik**: markalar egitimde VAR, this PARCALAR none
- Dogrulandi: **VAL ∩ training = 0, VAL ∩ gelistirme = 0, VAL ∩ exam = 0**

> Bu, gercek kullanim senaryosunun ta kendisi: *bildigimiz a ureticiden
> gelen, more before gormedigimiz a .stp dosyasi.*

(`DEV` set KULLANILMADI: 100 parcasinin 37'si training kumesinde.)

---

## 3. ANA SONUCLAR

| metrik | value |
|---|---|
| **Tespit F1** (CP'nin yerini buluyor mu) | **0.7878** |
| Robot-hazir F1, **axis** olcutu | 0.5764 |
| Robot-hazir F1, **signed** criterion | 0.4839 |
| Kesinlik | 0.5008 |
| Recall | 0.4682 |

### Guven araligi — 100 parts belirsizlik KUCUK DEGIL

Parca duzeyi bootstrap, 2000 tekrar, **same cluster ve same sistem**:

| metrik | F1 | %95 GA |
|---|---|---|
| **detection** | **0.7878** | **[0.7464, 0.8263]** |
| robot, axis olcutu | 0.5764 | [0.5067, 0.6438] |
| robot, signed | 0.4839 | [0.4003, 0.5671] |

Tespitin araligi dar (±0.04); robot-signed more genis (±0.08) because
100 parts sign hatalari more oynak.

**Sunumda each number araligiyla verilmelidir.** Ayrica projede **%80
geometrik ikiz** measured; gercek GA burada hesaplanandan GENIS olabilir.
(Makbuz: `results/confidence_interval_saha.json`)

**Tekrarlanabilirlik:** same measurement independent as iki kez kosuldu ve
UCU DE BIREBIR same cikti (0.7878 / 0.5764 / 0.4839).

### Iki robot metrigi why AYRI verilir

Kod iki criterion hesapliyor:
- **axis (unsigned)**: 180° ters a direction DOGRU sayilir
- **signed**: direction ureticinin belirttigi yonle AYNI olmak zorunda

Robot kabloyu fiziksel as sokacaksa valid criterion **isaretlidir**.
Literaturde ve onceki raporlarda genelde axis olcutu verilir; burada
**ikisi de** veriliyor.

---

## 4. ANA FINDING — darbogaz kapasite not, YENI URETICIYE TRANSFER

Ayni sistem, same data, same oznitelikler. **Tek diff: degerlendirme
kumesinin nasil bolundugu.**

| kosul | robot F1 |
|---|---|
| **Tanidik manufacturer** (brand egitimde present, part none) | **0.5603** |
| **Gorulmemis manufacturer** (brand egitimde HIC none) | **0.3135** |
| **diff** | **0.2468** |

Model yeni a PARCAYI ogrenebiliyor; yeni a URETICININ tasarim dilini
ogrenemiyor. Sistem kapasitesi not, **alan transferi** bagliyor.

Ogrenme egrisi olcumu bunu dogruluyor: 0.80 seviyesi for **3.8x**,
0.85 for **17.3x** corpus gerekiyor.

---

## 5. Zorluk axis: part yogunlugu

| regime | F1 |
|---|---|
| Dusuk-CP parts (%86) | 0.7701 |
| **Cok-CP parts (%14, n_gt ≥ 8)** | **0.6583** |

Yogun klemensler (24+ CP, siki dizilmis same gorunumlu acikliklar) sistemin
zayif halkasi. Gorulmemis manufacturer + dense part birlesince uctan uca
neredeyse sifira iniyor.

**Kok why measured:** dense parts model YONU biliyor (AUC 0.8899) but
HANGI ACIKLIGIN kablo girisi oldugunu bilmiyor (AUC 0.7053; gereken 0.944).

---

## 6. Calisma noktasi AYARLANABILIR

Ayni model, different confidence esigi:

| threshold | precision | recall |
|---|---|---|
| none (default) | 0.393 | 0.196 |
| ≥ 0.7 | 0.547 | 0.114 |
| ≥ 0.9 | **0.667** | 0.033 |

Robot uygulamasinda "az but emin" isteniyorsa precision **0.67'ye**
cikarilabilir. F1'i maksimize eden nokta varsayilandir (taranarak
dogrulandi: alternatiflerin en iyisi +0.0008, i.e. noise).

---

## 7. Bu calismada measured_path ve KAPANAN yollar

Iki gunde 50+ arm tek degiskenli measured. Kapananlar (hepsi makbuzlu):

| path | sonuc |
|---|---|
| Sentetik data uretimi | uretec calisiyor, segmentasyon sentetik geometride ateslemiyor (0.92x) |
| GT'den segmentasyon etiketi | iki different etiket sekliyle denendi: 0.5540 / 0.5511 vs kontrol 0.6232 |
| Icsel girdi oznitelikleri (HKS) | 0.1991 vs 0.6232 |
| Yogun-part uzman modeli | −0.0410 (oracle rejimle bile) |
| Odak kaybi / sinif agirliklandirma | −0.3018 / −0.0058 |
| Zor ornek madenciligi | −0.0195 |
| Havuz kucultme (geometrik suzgecler) | hedef 5x, ulasilan 1.33x |
| Aday pool threshold taramasi | mevcut nokta already optimum |

**Kapatma hukumleri sondanin kusuru olabilir** ilkesi uygulandi: a arm
"olu" sayilmadan before measurement yolunun dogrulugu denetlendi. Bu sayede en az
uc wrong verdict duzeltildi (bkz. Bolum 9).

---

## 8. Elde edilen iyilestirmeler

| kalem | kazanc | durum |
|---|---|---|
| Negatif ornekleme ratio 8 → 12 | **+0.0126** (2040 part, 5 brand fold) | **DEPLOYED** |
| Hafif mesh augmentasyonu (segmentasyon) | seg IoU **+0.0197 … +0.0649** | dogrulandi |
| Augmentasyon, uctan uca (tek-vs-tek, VAL) | robot signed **+0.0696** | dogrulandi |
| En iyi measured_path yapilandirma (4 aug seed + measured_path zincir) | robot signed **0.5162** vs 0.4839 | noise bandinda, dagitilmadi |

### En large son-islem kazanci: HALKA NORMALI with sign duzeltmesi

Her prediction edilen CP for, **mouth cevresindeki halkadan** (3-8 mm) mesh
normallerinin ortalamasi alinir; prediction yonunun isareti this normalle same
yone cevrilir.

| rule | robot ISARETLI |
|---|---|
| baseline | 0.4839 |
| en yakin tepe normali (YANLIS referans) | 0.4573 (−0.0266) |
| **halka normali** | **0.5168 (+0.0329)** |
| oracle sign (ust sinir) | 0.5732 (+0.0893) |

Egitim none, model degisikligi none, **ayarlanmis parametre none**.
Kazanc, oracle tavaninin %37'si.

**Nasil bulundu -- yontem hikayesi.** Ayni fikir before denendi ve
**−0.0266** verdi. "Kol olu" denip kapatilmak yerine WHY coktugu
measured: kullanilan "yerel normal" aslinda **deligin DUVAR normaliydi**
ve kablo girisi eksenine **DIKTI** (prediction yonuyle arasindaki aci median
**88.9 derece**, 617 CP). Dogru referans, delik duvarinin disindaki
yuzeyden alinan halka normalidir.

**Kanit gucu ve KAPSAM (durustce).** Esli part bootstrap:

| populasyon | path | diff | %95 GA |
|---|---|---|---|
| **familiar brand (VAL 100)** | measured_path | **+0.0195** | **[+0.0030, +0.0378]** KESIN |
| familiar brand (VAL 100) | field | +0.0329 | [−0.0155, +0.0819] |
| zor/unseen brand (150) | measured_path | +0.0023 | notr |
| zor/unseen brand (150) | field | −0.0016 | small negatif |

**Kazanc TANIDIK brand populasyonuna ozgudur**; unseen markada notr.
Celiski not kapsam siniri -- rule, baseline sistemin already calistigi
yerde yardim ediyor (tabanlar: 0.4668 vs 0.1605).

Kuralin davranisi bunu aciklar: 617 CP'nin 95'ini cevirir, 48'ini
DUZELTIR but 27'sini BOZAR (kararli cevirmelerin %64'u correct).

**Kapi altinda remaining, but direction dogrulanan kollar** (hepsi measured, hicbiri
dagitilmadi):

| kalem | kazanc | dogrulama |
|---|---|---|
| Etiket kalitesi agirliklandirma | +0.0082 | **4 fold seed**, tek pozitif remaining |
| Oznitelik budama (162→121 sutun) | +0.0040 | 3 fold seed |
| ~~Yardimci gorev (aux-wire)~~ | **−0.0072** | 3 matched seed -> **CURUDU** |
| ~~Cok-CP agirligi 2x~~ | −0.0002 | 4 fold seed -> **CURUDU** |

Ustteki iki satirin ustunun cizili olmasi, this calismanin yontem
disiplininin somut ciktisidir: ikisi de ILK olcumde pozitifti.

Augmentasyon notu: agresif setting (60° donme) more before denenip
**zarar verdigi** for arm "olu" sayilmisti. Hafif setting (9–17°)
kazandiriyor. Kol olu not, **ayari yanlismis**.

---

## 9. Olcum butunlugu — duzeltilen hatalar

Bu calismanin a cikti da **kendi olcumlerinin denetimi**:

1. **Isaretli/unsigned karisikligi**: headline metrigi isaretsizdi; robot
   for valid criterion signed. Ikisi de raporlaniyor.
2. **Kirli degerlendirme set**: DEV'in %37'si training kumesindeydi;
   VAL'e gecildi (kesisim sifir).
3. **Olcum yolu with dagitim yolu ayrimi**: a blok measurement betiginde
   **+0.0151**, uretim egiticisinde **−0.0138** verdi -> dagitilmadi.
4. **Sessiz no-op'lar**: a arm tam `+0.0000` verdi; reason yutulan a
   `AttributeError` idi (`max_iter_` yerine `n_iter_`).
5. **Eslesme kutusu**: Oklit mesafesi yerine carpim kutusu (lateral 2mm VE
   axial 40mm) -- wrong kutu a olcumu 0.593'ten 0.0172'ye dusuruyordu.
6. **Vekil metrik ISARET DEGISTIRDI**: yeniden egitilen a bilesen,
   cevrimdisi vekilde (kabul kutusuna girme ratio) **+0.9 puan
   kazaniyordu**; same bilesen tam zincirde **−0.0437** verdi (GA sifiri
   icermiyor, orneklerin %0'i pozitif). Sebep yapisal: vekil mean
   hatayi measures, uctan uca metrik ise ZATEN correct which a tahminin
   bozulmasini **cift** cezalandirir (a correct gider, a wrong gelir).
   **Kural: vekil karar verdirmez, yalnizca hangi candidate's tam olcumu hak
   ettigini secer.**

---

## 9b. KAT GURULTUSU — tek olcumle karar verilemez

Bir arm (etiket kalitesi agirliklandirma) DORT different fold bolunmesiyle
tekrarlandi:

| fold seed | diff |
|---|---|
| 1 | **+0.0144** ("gate gecti") |
| 2 | **−0.0019** (kaybettiriyor) |
| 3 | +0.0101 |
| 4 | +0.0102 |
| **mean** | **+0.0082** |

Ayni arm, same data, same kod -- yalnizca katlarin nasil bolundugu different.
**Tek tohumla dagitilsaydi olmayan a kazanc raporlanmis olurdu.**

Bu buyuklukteki kollarda fold gurultusu **±0.008** mertebesinde. Dolayisiyla
+0.01 civarindaki each sonuc COK TOHUMLU dogrulanmadan verdict giymemelidir.

**Ayni measurement segmentasyon tarafinda da yapildi ve noise more da large
cikti.** Ayni recete, same data, yalnizca seed different:

| seed | val Conn_IoU |
|---|---|
| 0 | 0.6528 |
| 1 | **0.6990** |
| 2 | 0.6673 |
| **spread** | **0.046** |

Segmentasyon tarafinda seed gurultusu **~0.046**'dir -- measured_path kollarin
etki buyuklugunun (0.002-0.030) KAT KAT UZERINDE.

**Bu ilke uygulanınca DORT arm curudu:**

| arm | tek seed | very seed | verdict |
|---|---|---|---|
| Cok-CP agirligi 2x | +0.0065 | −0.0002 (4 seed) | curudu |
| Oznitelik budama | +0.0097 | +0.0040 (3 seed) | gate alti |
| Etiket kalitesi agirligi | +0.0144 | +0.0082 (4 seed) | gate alti |
| Yardimci gorev (aux-wire) | +0.0083 | **−0.0072** (3 matched seed) | **sign DEGISTI** |

Dordu de tek tohumla dagitilsaydi **olmayan dort kazanc** raporlanmis
olurdu. Bu, calismanin en aktarilabilir metodolojik ciktisidir.

## 9c. NEREYE YATIRIM YAPILMALI — baglayici kisit MEASURED

Kalan hatanin nerede oldugu prediction edilmedi, **measured**. Aday pool
(gate'ten onceki tum candidates) alindi ve kabul kutusunun each axis TEK
TEK gevsetildi. Sorulan soru: hangi toleransi gevsetirsem kapsama acilir?

VAL'in tamami: **100 part / 660 CP**.

| gevsetilen axis | pool kapsamasi | degisim |
|---|---|---|
| **hicbiri** (gercek criterion: lateral 2mm, aci 10°) | 0.4242 | — |
| aci 10° → 45° | 0.4409 | +0.017 |
| aci **tamamen none sayilir** | 0.5939 | **+0.170** |
| lateral 2mm → 5mm | 0.6242 | +0.200 |
| **lateral 2mm → 10mm** | **0.7818** | **+0.358** |
| lateral 2mm → 20mm | 0.9000 | +0.476 |

**Okuma.** Yon tahminini TAMAMEN mukemmel yapsak kazanc **+0.170** with
sinirli. Oysa GT'lerin **%78'i** for havuzda, direction ZATEN 10 derece
icinde correct which a candidate **10 mm yakinda duruyor** -- lateral axis
yonun **iki katindan fazlasini** tasiyor.

> **Darbogaz ne candidate uretimi, ne segmentasyon, ne de direction.
> YANAL KONUM HASSASIYETI.**

Bu, independent iki olcumle birebir tutarli: konum AUC **0.7053** vs direction AUC
**0.8899**; ve lateral hatanin segmentasyon kalitesiyle aciklanmasi.

Sistemde this isi yapan a bilesen already present (**pose head**: gate
kararindan after lateral correction, uctan uca +0.0428 KANITLI) ve olcumde
kapsama 0.4242'den 0.4682'ye tasiyor. 10 mm bandindaki ceiling **0.7818**.
**Aradaki +0.31,
projenin en large tek acik basligidir ve adresi bellidir.**

### Ve this acigin NEREDEN ALINAMAYACAGI da measured

Pose head'in duzeltmesi 3 mm'ye kirpiliyordu; ilk akla gelen "kirpmayi
gevset" oldu. Olculdu: **kirpmayi tamamen kaldirmak +0.0000 degistiriyor**
-- because modelin onerdigi en large correction already **2.87 mm** ve
onerilerin **%0.0'i 3 mm'yi asiyor**. Sinirlayan kirpma not, modelin
kendisi.

Model, large artiklari more very gorecek sekilde yeniden egitildi (eslesme
toleransi 15 mm) ve buzulmesi gevsetildi. Ortak degerlendirme kumesinde
sonuc: genis data **yardim etmiyor**, buzulmeyi gevsetmek **+0.9 puan**
kazandiriyor -- ve en iyi varyantta bile adaylarin **%99.5'ine** 3 mm'den
small correction oneriliyor.

> **Sonuc:** +0.31'lik acik, gate ozniteliklerinden son-islem
> regresyonuyla ALINAMAZ. Ya candidate uretimi (segmentasyon) duzelecek, ya da
> pose kafasi yerel geometriyi DOGRUDAN goren yeni oznitelikler alacak.

Bu, acigin none oldugu anlamina gelmez; **hangi yolun onu tasiyamayacagini
olcerek** arama alanini daraltir.

---

## 10. Sonraki adimlar (sure yetmedi, path acik)

**Oncelik sirasi residual tahmine not 9c'deki olcume dayaniyor.**

| path | rationale |
|---|---|
| **1. OFFSET/OY BASI (merkez oylamasi)** | **measured_path en large acik: +0.358.** Bugun konum, segmentlenen bolgenin agirlik merkezinden turetiliyor -- literaturde bunun known kusuru "degen same nesneleri ayiramamak" (bizim dense-part cokusumuz). Cozum: agin each tepesi kendi CP merkezine a **offset vektoru** prediction etsin, noktalar kaydirilip kumelensin (Panoptic-DeepLab / PVN3D / Spatial Embeddings ailesi). **Yeni etiket gerekmez** -- hedef `(en yakin GT CP - tepe)`. Son-islem regresyonunun this acigi TASIYAMADIGI uc independent olcumle gosterildi; this yontem hasarin OLUSTUGU yerde calisir |
| 2. Yeni manufacturer verisi | alan farki +0.2468 -- ikinci en large number; ogrenme egrisi 0.80 for 3.8x corpus |
| 3. Cok-gorunumlu render fuzyonu | mevcut butun kollar TEK kiplikte (mesh); goruntu sinyali yeni bilgi |
| 4. Self-supervised / contrastive on-training | sentetik data dustukten after temsil yolunun remaining adayi |

**Onerilmeyen yollar** (this calismada olculup elenmis oldugu for):
segmentasyon loss fonksiyonu varyantlari, augmentasyon ailesi,
yardimci gorevler, MC dropout -- hepsi seed gurultusunun altinda kaldi.
Yon uzerine yeni arm acmak da 9c'ye per **tavani +0.170 with sinirli**.
Mesh cozunurlugunu degistirmek de closed: 5000 tepe **conclusive as
kotu** (−0.030), 7200 ve uc-cozunurluklu ensemble kapinin altinda kaldi
(+0.007 / +0.011, GA sifiri iciyor, maliyet 3 fold) -- tezden gelen
**6000 hedefi iyi secilmis**.
