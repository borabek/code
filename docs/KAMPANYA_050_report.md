# Robot CP kampanyasi -- 2026-08-11

Hedef: unseen markada **robot F1 = 0.50**, sisik olmayan a olcumle.
Sart: each an kampanya oncesi duruma donebilmek.

---

## 1. Geri donus (sart 1) -- KANITLANDI

| | |
|---|---|
| kontrol noktasi | git etiketi `KONTROL_NOKTASI_2026-08-11`, commit `68a6e856` |
| kod | 1678 file (750 `.py`, 729 receipt `.json`, 83 `.md`) commit'te |
| model | `_KN_2026-08-11/dosyalar/` -- 11 file / 643 MB fiziksel kopya |
| damga | 280 model dosyasi + 4 dizin parmak izi SHA-256 with |
| geri donus | **`python rollback.py`** |

**Tatbikat yapildi:** `product_wide.py`'ye sahte satir eklendi, `cp_config.json`
bozuldu (threshold 0.99 + sahte anahtar), `results/kazanan_hgb_derin.pkl` SILINDI,
iki sahte file olusturuldu. `rollback.py` sonrasi tam kanonik zincir:

| | kampanya oncesi | hasar + geri donus sonrasi |
|---|---|---|
| robot | 0.297956 | **0.297956** |
| detection | 0.476022 | **0.476022** |
| makro | 0.293380 | **0.293380** |
| 12 markanin hepsi | -- | **birebir same** |

`master` dali hep kontrol noktasinda; kampanya isi `kampanya_050` dalinda.
Geri donmek yapilan isi de SILMEZ.

---

## 2. Olcum protokolu -- sayinin why sisik olmadigi

### 2.1 Uc cluster MARKA-AYRIK

| cluster | brand | part | rol |
|---|---|---|---|
| `tam` | TOGI, PXC, WEI, SIE, TE, TKM, WAGO, MDI, ABB | 2583 | training + rule secimi |
| `d6` | SUPU, UPUN, MOR, NIT, UTL, S+S, SE, ONV | 468 | gelistirme, after EGITIME katildi |
| `d7` | CCD, KLM, A-B, EFX, WIE, CWT, DIN, WEG, CEM, DEG, ELMEX, C3 | 835 | **SINAV** |

Nihai model **`tam` + `d6` = 3051 part / 17 brand** with egitilir. D6 exam
degildir; teshis ve arm secimi for already dense kullanildi, therefore ondan
after "temiz D6 sayisi" diye a sey YOKTUR ve raporlanmaz. Egitime katilmasi
D7 for mesrudur ve brand cesitliligini 9 -> 17 yapar. **Kural secimi yine
only `tam` markalarinin katlarinda** (TOGI/PXC/WEI/SIE) yapilir.

Referans as D6'da dagitilan urunun sayisi (egitime katilmadan ONCE, urunun
canli zincirinden): **robot 0.2815 / detection 0.4215**, 468 part, %95 GA
[0.2399, 0.3285]. D7'deki dagitilan number 0.2980 -- i.e. iki cluster dagitilan urun
for benzer zorlukta.

Denetim (`results/bolme_denetimi.json`):

| olcek | tam∩d7 | tam∩d6 | d6∩d7 |
|---|---|---|---|
| part kimligi | 0 | 0 | 0 |
| brand | 0 | 0 | 0 |
| TAM geometri (tepe+yuz+kutu 0.1mm) | 0 | 0 | 0 |
| kaba iz (kutu 0.5mm + GT sayisi) | 94 iz / 132 part | 66 | 14 |

Gercek leakage alt sinir (tam geometri) with ust sinir (kaba iz) ARASINDADIR.
Kaba iz DIN klemenslerinin standart olculu olmasindan FAZLA sayar. Bu yuzden
headline yaninda **kaba iz eslesmesi olmayan 703 parcalik D7 alt set** de
raporlanir.

### 2.2 Kurallarin nerede secildigi
* Esik / NMS / arm secimi YALNIZ `tam` korpusunun MARKA KATLARINDA.
* Kural secim olcutu **makro** (brand basina esit agirlik) -- mikro, GT'si very
  which markanin kuralini secip diger markalari cokertiyordu.
* D7 exam; ona bakarak HICBIR setting secilmedi.

### 2.3 Olcumun urunun kendisi olmasi
Her number `canonical_chain.product_output` uzerinden, i.e. **urunun TEK zincirinden**
gecer. Karar kodu tek modulde (`p6_decision`) ve hem training hem urun ONU cagirir.
Havuz kurulumu (`thin_pool`), mesh esigi ve dedupe degerleri egitimdekiyle
BIREBIR same; `ppos` tanimi (`pb[:,CE]+pb[:,CT]`) korpusu ureten betikle same.

### 2.4 Makbuz
Her measurement `receipt_hash.damga()` with kod/model/config SHA-256'larini yazar.
Damgasiz number number degildir.

---

## 3. Teshis zinciri -- hangi sirayla ne bulundu

1. **Yon a SECIM problemiydi ve cozuldu.**
   Dagitilan urun each konuma TEK direction bagliyordu. `direction_bank` (kendi / komsu /
   silindir axis / ana eksenler) eklendi. Tavan D6'da 0.4497 -> 0.6889.
   *Kalan direction kaybi: +0.0052* (selected adaylarda mukemmel direction selector with diff).
   Yani direction residual darbogaz DEGIL.

2. **NMS hipotezi CURUDU.** D6 GT'lerinin %16.1'inin komsusu 5mm'den yakin
   olmasina ragmen threshold x NMS taramasi each kivrimda 5.0'i sectti.

3. **Asil darbogaz KONUM havuzuydu.**

   | D6 pool | only KONUM | konum + YON |
   |---|---|---|
   | B-rep (dagitilan) | 0.5371 | 0.2900 |
   | + mesh tepeleri | 0.9768 | 0.4854 |
   | + mesh + direction bankasi | 0.8713 (seyreltilmis) | **0.7264** |

   Mesh tepeleri tarihte UC kez zarar vermisti; reason anlasildi: direction bankasi
   olmadan eklendiklerinde yalnizca FP uretiyorlardi. Konum ve direction AYRI iki
   missing, ikisi birden kapanmali.

4. **Seyreltmede KAPSAMA, GUVENI yeniyor.** "En high olasilikli 60 tepe"
   yerine "2.5mm uzamsal seyreltme" same maliyette konum recall'unu
   0.6362 -> 0.8713 yapiyor.

5. **Ikinci kademe KASKAD olmali.** Tum secenekleri yeniden puanlayan ikinci
   model ZARAR verdi (-0.0363). Kisa listeye (birinci kademe skoru >= 0.20)
   odaklanan, birinci kademe skorunu da oznitelik alan version kazandi.

6. **Periyodik yapi gercek.** D6'da >=6 CP'li 1096 parts GT'lerin **%90.8'i**
   parcanin en sik OTELEME VEKTORUYLE baska a GT'ye ulasiyor. `lattice`
   modulu bunu oznitelik as verir; tohumlar HER ZAMAN tahminden gelir,
   GT'den ASLA.

---

## 3b. Kapanmayan cephe: YOGUN parts (NIT ornegi)

NIT: 50 part / 1222 GT / 24.4 CP-part. **Dagitilan urun 1222 GT'den 2'sini
buluyor** (F1 0.0032) -- i.e. this kampanyanin actigi a sorun not, sistemin
sureklilik arz eden kor noktasi. Yeni pool o markada yonlu recall **0.5254**
veriyor, i.e. cevabin yarisi HAVUZDA.

Kayip nerede? Uc measurement:

| soru | measurement | cevap |
|---|---|---|
| threshold mi? | o markadaki EN IYI rule | 0.0349 -- HAYIR |
| siralama rastgele mi? | recall@k / rastgele | 9.4x -- HAYIR |
| ne up to iyilestirilebildi? | C blogu + `zskor=ikisi` | **12.0x** |

Denenen ve measured_path iki mudahale:
* **Segmentasyon oznitelikleri (A blogu) NIT'te ZARAR VERIYOR**: only direction
  bankasi (C) 10.4x, hepsi 9.4x.
* **Parca-ici SIRA donusumu** tek basina kotu (5.9x) but Z-SKORLA BIRLIKTE en
  iyisi (12.0x).

**Neden yetmiyor:** NIT parcasinda 4467 option present ve 24'u correct (%0.69). 12x
siralama top-24'e 2 correct koyar -> F1 ~0.08. Kullanilabilir a number for
~50-100x gerekir; this, mevcut oznitelik uzayinda kapanacak a diff DEGILDIR.
Yeni bilgi kaynagi (o yogunlukta etiketli data ya da different a temsil) gerekir.

**Sinav for baglami:** D7 D6'dan very more SEYREK.

| cluster | ort CP/part | >=8 CP which part | o parcalardaki GT payi |
|---|---|---|---|
| tam | 5.7 | %14.4 | %55.7 |
| d6 | 5.7 | %13.7 | %55.4 |
| **d7** | **3.7** | **%7.4** | **%30.3** |

Yani D6'nin mikro sayisi dense parcalarin egemenliginde; D7'ninki not. D6'da
measured_path mikro kazanc D7 for KOTUMSER a tahmindir.

---

## 4. Yakalanan tuzaklar

| tuzak | belirti | sonuc |
|---|---|---|
| `KAYNAKLAR` iki kez tanimli | `P6_KAYNAK=012` hicbir sey yapmiyor, error YOK | mesh pool no acilmamis |
| isin kesisimi tek cagrida | 8 payin 4'u `MemoryError` | topaklandi, sonuc bit duzeyinde same |
| `_tam_oz` onbellegi config'den eski | segmentasyon adaylari 7 vs 12 | dagitilan modelde de VAR, kiyas adil |
| BASELINE kolunda satir/candidate indeksi karisik | mesh suzgeci gelince wrong konum | duzeltildi |
| lattice 1B sira as modellenmisti | GT'nin only %5'i uyuyor | oteleme vektoru with %90.8 |
| lattice ara adim none | 6mm adimli sirada tohumlar 12mm gorunce aradakiler no ongorulmuyor | yarim VE ucte-a adimlar eklendi |
| `sec_ayrintili` erken cikis dali | normal dal 4 value, erken cikis 3 -> D7 P6 arm `expected 4, got 3` with coktu | dal esitlendi + `tests/test_p6_decision_imza.py` |
| iki kosu same dosyalara yazdi | correction ONCESI baslamis paylar correction SONRASI paylarla same makbuza yaziyordu | eski zincir durduruldu, temiz kosu |
| yelpaze sondasi 64 direction | "arm OLU" hukmu verildi; oysa 64 yonun araligi 25 derece, tolerans 10 | 256 yonde +0.0676 -- **verdict SONDANIN kusuruydu** |

---

## 5. D7 OKUMA #1 -- DECISION KURALLARI (okumadan ONCE yazildi)

Bu bolum D7'ye BAKILMADAN before dolduruldu. Sayiya bakip rule secmek sismenin
ta kendisidir; asagidakiler baglayicidir.

**Okunacak yapilandirma (tek, onceden fixed):**
* model: `results/p6_kademe2_model.pkl` -- `tam`+`d6` (3051 part / 17 brand)
  with egitilmis; arm ve karar kurali `tam`in MARKA KATLARINDA (WEI/PXC/SIE/TOGI)
  MAKRO olcutle secilmis.
* poz kafasi: **P6 kolunda KAPALI** (ilan edilen). Gerekce ILKESEL, olcume
  bakilarak not: model ve karar kurali brand katlarinda poz kafasi OLMADAN
  secildi; uzerine dogrulanmamis a son islem koymak, measured_path seyden baska a
  sey dagitmak olurdu. BASELINE arm kendi DAGITILAN hali which poz-kafasi-OPEN with
  kosar. Ucuncu a arm (P6 + poz kafasi) yalnizca GOZLEM as raporlanir;
  headline ondan SECILMEZ.
  (D6 uzerinden karar verilmedi because D6 egitime katildi ve also D6'nin
  `_tam_oz` onbellegi `_p1_olasilik_g7`'den, benim betiklerim `_p1_olasilik`'ten
  besleniyor -- D6 residual urunu temsil etmiyor. D7 ve `tam` for this uyusmazlik
  YOK, ikisi de kendi onbellegiyle tutarli.)
* iki arm AYNI kosuda: `URUN_P6=0` (dagitilan urun) ve `URUN_P6=1`.
* 8 pay + `merge_receipt.py`; mikro F1 for birlestirme kayipsizdir.

**Manset tanimi:** MIKRO robot F1 (lateral <=2mm, ISARETLI aci <=10, axial
<=40mm, Macar bire-a eslesme), 835 part, urunun TEK kanonik zincirinden.
`f1w` KULLANILMAZ.

**Onceden ilan edilen kesmeler:**
| durum | karar |
|---|---|
| `p6_sayac`: P6 arm parcalarin <%90'inda calisti | headline INVALID; reason bulunur, okuma tekrarlanir (butceden sayilir) |
| P6 robot < BASELINE robot | KOL DAGITILMAZ; baseline korunur, sonuc oyle raporlanir |
| P6 robot >= 0.50 | hedef TUTTU; temiz-703 alt kumesinde de raporlanir |
| 0.35 <= P6 robot < 0.50 | hedef TUTMADI; kazanc dagitilir, remaining path 0.75 paketiyle surer |
| P6 robot < 0.35 | kazanc D6'dan D7'ye TASINMADI; reason analizi (brand kirilimi) sart |

**Ayrica each okumada raporlanir:** %95 bootstrap GA, brand kirilimi, makro,
en kotu brand, recall/precision, ve kaba-iz eslesmesi olmayan 703 parcalik
TEMIZ ALT KUME sayisi. Manset with temiz alt cluster arasindaki diff buyukse number
supheli sayilir.

### 5b. `tam` MARKA KATLARINDA GERCEK KAZANC (D7'den ONCE, exam DEGIL)

3051 part (tam+d6), 4 brand fold (WEI/PXC/SIE/TOGI), threshold/arm katta secildi:

| arm | robot | recall | precision | TP | FP | FN |
|---|---|---|---|---|---|---|
| BASELINE | 0.2861 | 0.1868 | 0.6108 | 2671 | 1702 | 11629 |
| **P6** | **0.3091** | 0.2935 | 0.3265 | 4197 | 8658 | 10103 |
| P6_KAFES | 0.3045 | 0.2587 | 0.3698 | 3700 | 6305 | 10600 |
| P6_GEO | 0.2707 | 0.2566 | 0.2864 | 3670 | 9146 | 10630 |

**P6 - BASELINE = +0.0230.** D6'nin vaat ettigi +0.1681 GERCEK DEGILDI.

Kat kirilimi deseni acikliyor:

| fold | n | BASELINE | P6 | diff |
|---|---|---|---|---|
| TOGI | 810 | 0.0597 | 0.1689 | **+0.1092** |
| WEI | 686 | 0.2797 | 0.3795 | **+0.0998** |
| PXC | 713 | 0.5850 | 0.3972 | **-0.1878** |
| SIE | 262 | 0.6226 | 0.4681 | **-0.1545** |

**P6, tabanin ZAYIF oldugu yerde kazanir; GUCLU oldugu yerde kaybeder.** D6'nin
tabani zayifti, o yuzden orada each sey iyi gorunuyordu. Bu, gelistirme
kumesinden read kazancin why aldatabilecegini gosteren somut ornektir.

**REJIM KAPISI** (this bulgunun cevabi): `n01 >= 90 -> P6, altinda BASELINE`.
Yonlendirme istatistigi taramasi (fold-disi threshold secimi, MAKRO criterion):

| istatistik | fold-disi robot | selected esikler |
|---|---|---|
| **n01** | **0.5269** | 90 / 90 / 90 / 90 |
| n_aday | 0.5139 | 114 x4 |
| n_secenek | 0.5137 | 1628/1344/1344/1628 |
| mesh_oran | 0.5025 | 0.42 x4 |
| taban_ort3 | 0.3660 | -0.51 x4 |
| taban_maks | 0.3515 | -0.75 x3, -0.55 |

Iki not: (1) "tabanin kendi guvenine per yonlendir" hipotezi CURUDU -- baseline
skoru parts arasi kalibre not. (2) **Bu tablodaki sayilar ORNEKLEM-ICIDIR**
(nihai model this parcalari egitimde gordu); yalnizca hangi KURALIN secildigini
gosterirler, kuralin degerini DEGIL. Esigin dort katta da same cikmasi kuralin
kararli oldugunu gosterir.

### 5c. D7 SINAV SONUCU

**BASELINE arm (dagitilan urun, poz kafasi OPEN) -- payli kosuyla yeniden uretildi:**

| | value |
|---|---|
| robot MIKRO | **0.2980** %95 GA [0.2670, 0.3293] |
| detection MIKRO | 0.4760 %95 GA [0.4476, 0.5046] |
| TP / FP / FN | 758 / 1243 / 2329 (recall 0.2455, precision 0.3788) |
| makro / en kotu | 0.2934 / 0.0000 |
| **temiz alt cluster (703 part)** | **0.2558** -- headline farki **-0.0422** |

Iki not:
1. Sekiz paya bolunup birlestirilen measurement, tek islemde kosulan known degeri
   BIREBIR verdi (0.2980 / 0.4760). Birlestirme kayipsiz.
2. **Manset, training parcalariyla kaba geometri benzerliginden 0.042 up to
   besleniyor.** Temiz alt cluster more low; this diff each iki arm for de
   raporlanir.

Marka kirilimi (baseline):

| brand | n | GT payi | robot F1 |
|---|---|---|---|
| CWT | 226 | %37.1 | **0.0370** |
| WIE | 169 | %17.0 | 0.5712 |
| A-B | 124 | %12.3 | 0.4815 |
| CCD | 47 | %4.0 | 0.5496 |
| ELMEX | 27 | %1.9 | 0.4885 |
| DIN | 57 | %5.0 | 0.4667 |
| DEG | 19 | %1.3 | 0.4348 |
| EFX | 34 | %5.3 | 0.1688 |
| KLM | 65 | %5.4 | 0.1441 |
| CEM | 4 | %3.6 | 0.1313 |
| WEG | 58 | %5.3 | 0.0471 |
| C3 | 5 | %1.8 | 0.0000 |

D7'nin mikro sayisini CWT tasiyor (GT'nin %37'si, F1 0.0370). Rejim kapisi
"baseline zayifsa P6" dedigine per asil diff orada gorulecek.

**P6 arm (direction bankasi + mesh pool + regime kapisi, poz kafasi KAPALI):**

| | BASELINE | **P6** | diff |
|---|---|---|---|
| **robot MIKRO** | 0.2980 | **0.3115** | **+0.0135** |
| %95 GA | [0.2670, 0.3293] | [0.2854, 0.3375] | ortusuyor |
| detection | 0.4760 | 0.4619 | -0.0141 |
| makro | 0.2934 | **0.3265** | **+0.0331** |
| **en kotu brand** | 0.0000 | **0.0466** | **+0.0466** |
| TP / FP / FN | 758/1243/2329 | 872/1640/2215 | — |
| recall / precision | 0.2455 / 0.3788 | 0.2825 / 0.3471 | — |
| temiz-703 | 0.2558 | 0.2766 | +0.0208 |

P6 arm 835 parcanin **385'inde (%46.1)** calisti; sessiz geri dusme YOK
(`tablo_yok` 0, `model_yok` 0). Rejim kapisi 450 parts tabana yonlendirdi --
kalibrasyonun ongordugu ~yari oranla uyumlu, i.e. threshold urune DOGRU tasindi.

Marka kirilimi (8/12 markada ARTI):

| brand | baseline | P6 | diff |
|---|---|---|---|
| C3 | 0.0000 | 0.1373 | **+0.1373** |
| CEM | 0.1313 | 0.3038 | **+0.1725** |
| EFX | 0.1688 | 0.2988 | **+0.1300** |
| DIN | 0.4667 | 0.5275 | +0.0608 |
| ELMEX | 0.4885 | 0.5191 | +0.0306 |
| DEG | 0.4348 | 0.4507 | +0.0159 |
| CWT | 0.0370 | 0.0466 | +0.0096 |
| WEG | 0.0471 | 0.0486 | +0.0015 |
| KLM | 0.1441 | 0.1191 | -0.0250 |
| A-B | 0.4815 | 0.4530 | -0.0285 |
| CCD | 0.5496 | 0.5220 | -0.0276 |
| WIE | 0.5712 | 0.4910 | **-0.0802** |

### 5e. DECISION (onceden ilan edilen kurala per)

**TARGET TUTMADI.** Ilan edilen rule "P6 >= 0.50 -> tuttu; 0.35-0.50 -> tutmadi
but dagitilir; < 0.35 -> kazanc tasinmadi, reason analizi sart" diyordu.
Sonuc **0.3115**, i.e. en alt bantta.

Ne oldugu durustce:
* Kazanc GERCEK but KUCUK: +0.0135, ve iki kolun confidence araliklari ORTUSUYOR.
  Tek basina this diff istatistiksel as zayiftir.
* Buna karsilik **makro +0.0331 ve en kotu brand 0.0000 -> 0.0466**: sistem
  markalar arasi more DENGELI. Sifirdan produced a brand (C3) ve iki katina
  produced uc brand (CEM, EFX, C3) present.
* Kaybedilen yer WIE (-0.0802) ve A-B/CCD/KLM: regime kapisi this markalarda
  wrong tarafa yonlendiriyor. Kapi tek a threshold (n01>=90) ve this markalarda
  baseline gucluyken P6'ya gecmis olmali.
* **CWT hala 0.0466** ve D7 GT'sinin %37'si orada. Asil duvar CWT'de ve this
  duvar NIT'le AYNI: dense parts temsil yetersizligi. 0.50'nin onundeki
  tek en large engel budur.

### 5f. GUVEN KAPISI -- field 0.90 sozu VERILEMEZ

D7'nin 2512 tahmini uzerinde measured:

```
ham precision 0.3471
precision hicbir esikte >= 0.90 OLMUYOR -- en high 0.6429
```

Dahasi egri tepe noktasindan after GERI DONUYOR (pay0'da threshold 0.95'te 0.5025,
0.99'da 0.4700). Sebep: karar kurali GORELI (`0.85 x part-maks`), i.e. selected
tahminlerin hepsi already part-maksimumuna yakin; mutlak score parts arasi
kalibre a confidence olcusu DEGIL.

**Sonuc:** "robotun kullandigi isaretler >=0.90 kesinliktedir" sozu BU
SKORLAYICIYLA verilemez. Guven kapisi AYRI a calibration modeli ister
(part-ici goreli konum + lattice tutarliligi + candidate mutabakati gibi sinyaller).
Kapsama sayisi uydurmak yerine this boyle kaydedildi.

**KALIBRASYON MODELI KURULDU VE MEASURED** (`run_calibration.py`; secilmis
tahminler uzerinde 15 sinyalle ikinci model; tam+d6, 8015 prediction, 6 brand
fold, KAT-DISI):

| hedef precision | ham kapsama | **kalibre kapsama** |
|---|---|---|
| 0.60 | 0.2299 | **0.3023** |
| 0.70 | 0.1007 | **0.1584** |
| 0.80 | 0.0403 | **0.0905** |
| **0.90** | 0.0041 | **0.0035** |

Kalibrasyon 0.60-0.80 bandinda kapsamayi **~2 fold** artiriyor. Ama 0.90'da
ikisi de sifira yakin (%0.35).

**SAHA SOZUNUN DURUST HALI:** ">=0.90 kesinlikli sign" bugun GT'nin BINDE
3.5'i for verilebilir -- kullanilabilir a teklif DEGIL. **Bugun
verilebilecek en iyi soz: 0.80 kesinlikte %9 kapsama.** 0.90'a however
tam-otomatik bandin kendisi yukselince ulasilir; kisa yolu none.

### 5.1 Okuma plani (onceden ilan)

D7 butcesi 3 okuma; this birincisi. Ayni kosuda iki arm olculur (baseline URUN_P6=0,
P6 URUN_P6=1), ikisi de urunun TEK kanonik zincirinden gecer, 8 pay paralel.
Manset = MIKRO robot F1 + %95 part-bootstrap araligi + brand tablosu + temiz
alt cluster (703) duyarliligi + P6 geri-dusme sayaci.

DECISION KURALLARI (okumadan ONCE yazildi):
* P6 >= 0.50 ve temiz alt cluster farki kucukse -> hedefe ulasildi; dagitim karari
  ayri konusulur.
* 0.50'nin altinda but baseline (0.2980) uzerinde anlamli artis varsa -> kazanc
  raporlanir, remaining diff error bankasiyla aciklanir; ikinci okuma however SOMUT
  a duzeltmeden after yapilir.
* Taban altinda ya da geri-dusme sayaci yuksekse -> number RAPORLANIR, reason
  bulunur; `rollback.py` with donus each an mumkun.

---

### 5g. D7 SONRASI TESHISLER -- 0.50'ye giden path nerede tikaniyor

**1. Rejim kapisi D7'nin iki large markasinda TERS calisiyor.**

| brand | n01/part | konum recall (tum mesh) | direction recall (banka) | baseline | P6 |
|---|---|---|---|---|---|
| WIE | 141 | 0.8897 | 0.8536 | 0.5712 | 0.4910 |
| CWT | 86 | 0.5532 | 0.4415 | 0.0370 | 0.0466 |

Esik 90 idi: CWT (86) TABANA gidiyor -- oysa baseline orada cokuyor; WIE (141)
P6'ya gidiyor -- oysa baseline orada iyi. D7'deki WIE kaybi (-0.0802) ve CWT'nin
kipirdamamasi AYNI sebepten.

**2. Ogrenilmis router (v2) BASARISIZ.** 13 part ozniteligiyle egitilen
router fold-disi 0.5269 -> 0.4646 (**-0.0624**); training markalarina ozgu
oruntuleri ezberleyip unseen markaya tasimiyor. TOGI'de -0.1137. **Tek threshold
KALIYOR.**

**3. CWT'nin pool segmentasyon esiginin ALTINDA kalmis.** Mesh adaylari
`p_pos >= 0.50` with seciliyor:

| brand | p>=0.5 | p>=0.3 | p>=0.2 | p>=0.1 | p>=0.05 |
|---|---|---|---|---|---|
| **CWT** | 0.5532 | 0.6108 | 0.6501 | 0.7286 | **0.7914** |
| WEG | 0.6768 | 0.7439 | 0.7805 | 0.8232 | 0.8354 |
| WIE | 0.8897 | 0.9278 | 0.9430 | 0.9544 | 0.9620 |

CWT'de konum recall **+0.2382** aciliyor (candidate 267 -> 700).

**METODOLOJIK CIKMAZ:** same tarama `tam` korpusunda WEI 0.9497 -> 0.9954,
PXC 0.9914 -> 0.9914 veriyor. Yani training markalarinda pool ZATEN tavanda ve
orada yapilan a secim "0.50'de kal" der. Kaldirac, egitimden FARKLI markalarda
degerli ve o farki only sinavda gorebiliyorum.

**Cozum ILKESEL olmali, D7'ye bakarak not:** esigi dusurmek candidate EKLER, asla
CIKARMAZ -> pool tavanini MONOTON yukseltir. Tek risk kesinliktir ve o training
katlarinda olculebilir. Yapilacak deney: low esikle korpusu yeniden cikar,
training katlarinda uctan uca ZARARSIZ oldugunu goster, after dagit.

---

## 5d. Bu oturumun EN IMPORTANT iki dersi

**1. Gelistirme kumesinden read kazanc aldatir.** D6'da P6 arm +0.1681
veriyordu; `tam` korpusunun brand katlarinda gercek kazanc **+0.0230**. Sebep:
D6'nin TABANI zayifti. Bir arm "kazandi" derken, tabanin o kumede ne up to iyi
oldugunu da yazmak zorunlu.

**2. "CLOSED" hukumleri sondanin kusuru olabilir.** Yon yelpazesi kolunu 64
yonlu a sondayla olcup OLU ilan ettim. Oysa 64 yonun kure uzerindeki araligi
~25 derece, measurement toleransi 10 derece -- probe metrigi FIZIKSEL OLARAK
tutturamiyordu. 256 yonle same arm NIT'te **+0.0676 recall** verdi. Bu, hafizada
"CLOSED" diye stopped kollarin a kisminin da boyle kapanmis olabilecegi
anlamina gelir; kapatma karari verirken "probe this etkiyi olcebilir miydi?"
sorusu ONCE sorulmalidir.

---

## 5h. D7 SONRASI CALISMA -- ne denendi, ne cikti

| is | sonuc | karar |
|---|---|---|
| Ogrenilmis regime router (13 oznitelik) | fold-disi **-0.0624** | REDDEDILDI, tek threshold kaliyor |
| Rejim esigi kararlilik egrisi | 50-110 bandi tepeden 0.01 icinde | threshold 90 -> **60** (tepe) |
| CWT cephe teshisi | each iki arm da cokuyor; pool p>=0.5 esiginin ALTINDA | mesh esigi arm acildi |
| Mesh esigi taramasi | CWT konum recall 0.5532 -> **0.7914** (p>=0.05) | ilkesel rationale yazildi, corpus yeniden cikarimi bekliyor |
| Guven calibration modeli | 0.80'de kapsama %4 -> **%9**; 0.90'da %0.35 | deployed; 0.90 sozu VERILEMEZ |
| Eksen boyu ornekleme (kapali arm yoklamasi) | NIT'te **+0.0034**, maliyet 1.75x | verdict DOGRUYMUS, kapali kaliyor |
| Yon yelpazesi (256 direction) | NIT'te **+0.0676** recall | hatta baglandi (`YB_FAN`), uctan uca measurement bekliyor |
| Sira damgalama (kademe2) | kosuyor | — |

**Duzeltilen uc sessiz error:**
1. `sira` blogu each KAT x KOL for yeniden hesaplaniyordu -> a kosu 70
   dakikada ilerlemedi (70 dk -> 13 sn).
2. `yelpaze_yonleri` default argumani modul sabitine bagliydi; `FAN_N`
   degistirmek SESSIZCE etkisizdi.
3. `run_p6_feature`de `mesh`/`diag` kullanildiktan SONRA tanimlaniyordu --
   ilk parts NameError, sonrakilerde BIR ONCEKI PARCANIN mesh'i.

---

## 6. Durustluk notlari -- neyin temiz OLMADIGI

1. **D6 temiz okuma degildir.** Teshis, arm secimi ve seyreltme kurali orada
   measured. Temiz okuma yalnizca D7'dir.
2. **Seyreltme kurali D6'ya bakilarak secildi.** Ayni measurement training markalarinda
   tekrarlandi: kurallar orada birbirine very yakin (0.9254-0.9571) ve selected
   rule en iyiden 0.0107 geride, %25 more ucuz. Fark however D6'nin YOGUN
   parcalarinda aciliyor.
3. **`_tam_oz` onbellegi `cp_config.json`'un eski halinde uretildi.** Ayni
   durum dagitilan modelde de present, therefore kiyas adil; but iki taraf da
   bugunku segmentasyon ayariyla YENIDEN turetilse sayilar degisebilir.
4. **D7 bootstrap araligi part birimlidir.** D7 icinde kaba-iz ikiz ratio
   %22.3 ve ikizler same markada; grup bootstrap'i also gerekli gorulmedi,
   but this a tercihtir.

---

## 7. GECE 2026-08-11/12 -- SIRA DAMGALAMA CLOSED

`tam` brand katlari (WEI, PXC, SIE, TOGI), corpus `_p6_oz_u25`, 3051 part.
Makbuzlar: `results/p6_kademe2_sira0.json` (kapali) / `sira1.json` (acik).

| arm | SIRA kapali | SIRA acik | diff |
|---|---|---|---|
| **P6** (urun arm) | 0.309114 | 0.309114 | **0.000000** |
| P6_KAFES | 0.293754 | 0.304059 | +0.010305 |

**DECISION: CLOSED.** Iki rationale:

1. P6 kolunda sonuc BASAMAK BASAMAK same. SIRA damgalama urun koluna no
   dokunmuyor -- damgalama yalnizca lattice kolundan giriyor.
2. Kafes kolundaki +0.0103'luk kazanc arm P6'nin onune GECIREMIYOR
   (0.3041 < 0.3091). Yani en iyi haliyle bile urunde a sey degistirmez.

Bu, "kazanc present but wrong kolda" durumunun ders niteliginde ornegi: a kolun
kendi icinde iyilesmesi, o arm already geride oldugu surece urun kazanci DEGILDIR.

### Marka kirilimi (P6 arm, disarida birakilan brand)

| brand | robot F1 | recall | precision |
|---|---|---|---|
| SIE | 0.4681 | 0.8300 | 0.3259 |
| PXC | 0.3972 | 0.5981 | 0.2973 |
| WEI | 0.3795 | 0.4879 | 0.3105 |
| **TOGI** | **0.1689** | **0.1054** | 0.4240 |

TOGI, D7'deki CWT with same imzayi tasiyor: recall %10.5'e cokuyor but precision
en high value (0.4240). Yani model TOGI'de "az but correct" buluyor --
darbogaz SECIM not, candidate's havuza HIC GIRMEMESI. Bu, ADAY_YOK kovasinin
(%36.5) brand duzeyindeki yuzu ve A1b tam-acik pool kolunun hedefi.

---

## 8. KRITIK FINDING -- OLCTUGUMUZ ZINCIR GLB'YE GIRMIYOR

**Robotun actigi GLB, this kampanyada measured_path zinciri KULLANMIYOR.**

Kanit (2026-08-12, kod taramasi):

| file | zincir |
|---|---|
| `export_robot_glb.py` | `robot_cp.extract` |
| `robot_viz.py` | `robot_cp.extract` |
| `robot_cp.py` icinde `canonical_chain` / `product_p6` / `product_wide` | **hicbiri gecmiyor** |
| `canonical_chain`i cagiranlar | `product_p6.py` + yalnizca SONDA/OLCUM betikleri |

`robot_cp.extract` yolu: inference -> `derive_candidates` -> `wire_gate.apply`.
Kampanyanin butun kazanclari (`product_p6` = direction bankasi + ortak siralayici,
`product_wide` = genisletilmis pool) `canonical_chain.product_output` icinde ve this
fonksiyon ihracatcilarin HICBIRI tarafindan cagrilmiyor.

### Ne anlama geliyor

- Bugun a GLB acilsa, uzerindeki isaretler DAGITILAN TABANIN ciktisidir --
  D7'de robot F1 **0.2980**. Kampanyanin olctugu **0.3115** (ve genis pool
  kolunun 0.2029 -> 0.3090'i) o dosyaya YANSIMIYOR.
- Yani "gercek dunyada robot GLB'yi kullandiginda F1 ne olur?" sorusunun
  bugunku yaniti, olctugumuz number not BASELINE sayisidir.

### Entegrasyon for gereken (YAPILMADI -- dogrulanmadan yapilmaz)

1. `export_robot_glb.py` each model for `pbs` listesini already uretiyor but
   yalnizca ortalamasini (`acc`) tutuyor; listeyi saklayip
   `canonical_chain.product_output(V, F, pbs, step_path, cfg)` cagrilmali.
2. **TIER SORUNU:** `tier` alani `product_output` icinde DEGIL, `robot_cp.extract`
   icinde (satir ~404) atanir. `product_output` ciktisi dogrudan verilirse
   ihracatci `c["tier"]` okurken KeyError alir. Tier atamasi ortak a yere
   tasinmali.
3. Dogrulama: same part for iki yolun CP sayisi/konumu karsilastirilmali;
   entegrasyon "sessizce eski yola dusme" with maskelenmemeli.

**Bu night YAPILMADI.** Robotun tukettigi ciktiyi dogrulamadan degistirmek,
kampanyanin bastan beri kacindigi hatanin ta kendisi olurdu: olculmemis a
degisikligi urun diye teslim etmek.

---

## 9. SAHA TIER KURALI -- IKI OLCUM CELISIYOR GIBI, CELISMIYOR

GLB'deki kirmizi/turuncu (auto/review) ayrimi, ihracatciya gecilen
`robot_conf_auto` / `robot_min_auto_votes` with YAPILMIYOR. Gercek rule
`robot_cp.to_records` icinde:

    AUTO = wire_score >= cp_config.robot_auto_gate_threshold (0.66)

`conf_auto`/`min_auto_votes` yalnizca gate skoru YOKKEN (eski path) devreye
giriyor. Yani ihracatcinin gecirdigi o iki parametre pratikte ATIL.

### Iki number, iki different soru

| measurement | kosul | sonuc |
|---|---|---|
| Kodda yazili (2026-07-29) | kilitli holdout, **brand-ayrik DEGIL** | AUTO kesinligi **0.9508**, CP'lerin %52'si otonom |
| Bu kampanya (2026-08-11) | **D7, unseen MARKA** | hicbir esikte precision >= 0.90 none; en high **0.6429** |

**Celismiyorlar; same soruyu sormuyorlar.** Ilki "gordugum markanin yeni
modelinde", ikincisi "no gormedigim markada". Kullanicinin field akisi IKISINI
DE iceriyor ("elimizdeki markalardan yeni model de gelir, bilmedigimiz markadan
yenisi de").

### Durust field sozu

- **Bilinen brand, yeni model:** AUTO katmani for ~0.95 precision iddiasi
  savunulabilir, but o rakam ESKI olcumdur ve this korpusla YENIDEN dogrulanmali.
- **Gorulmemis brand:** 0.90 SOZ VERILEMEZ. Olculen ceiling 0.6429.
- Bu yuzden `run_saha_gate.py` yazildi: precision-kapsama egrisini
  GORULMEMIS MARKA katlarinda cikarir (D7'yi harcamadan) ve ONAYLI esigini
  olcume baglar. Kuyrukta, corpus tamamlaninca kosacak.

**Acik remaining is:** same egri `wire_score` for de cikarilmali -- sahada tier'i
belirleyen score odur, benim olctugum P6 skoru DEGIL. Iki score AYNI OLCEKTE
DEGILDIR; birinde measured_path esigi digerine takmak sessiz a error olurdu.

### EK FINDING -- dagitilan threshold, measured_path threshold DEGIL

`cp_config.json`'daki gercek degerler:

| anahtar | config | kodda/yorumda anilan |
|---|---|---|
| `robot_auto_gate_threshold` | **0.6** | 0.66 ("threshold 0.66 -> AUTO kesinligi 0.9508") |
| `robot_wire_gate_threshold` | **0.4** | 0.30 (kod varsayilani) |
| `robot_conf_auto` / `robot_min_auto_votes` | 0.5 / 3 | pratikte ATIL (gate skoru varken okunmuyor) |

Yani sahada running AUTO esigi **0.6**, oysa 0.9508 precision **0.66** for
olculmustu. Daha DUSUK threshold more COK isareti otonom yapar ve kesinligi
DUSURUR -- bugunku AUTO kesinligi 0.9508'den az olmalidir, ne up to az oldugu
OLCULMEMISTIR.

Bu, "0.90 field sozu" tartismasinin sessiz kalmis parcasidir: sadece brand
kosulu not, ESIGIN KENDISI de olcumden kaymis durumda.

---

## 10. TIER COKUSU -- "kirmizi isaretlerin kaci correct?" (measured)

Makbuz: `results/tier_cokusu_d7.json` · probe: `probe_tier_cokusu.py`
**Yeni a D7 okumasi DEGIL** -- harcanmis olcumun yeniden analizi; model
secimi ya da setting yapilmadi.

### P6 zinciri, D7 (unseen brand), 835 part / 2512 sign

| threshold | AUTO | AUTO payi | precision | GT kapsama |
|---|---|---|---|---|
| **0.60 (DAGITILAN)** | 2512 | **1.0000** | **0.3471** | 0.2825 |
| 0.80 | 2290 | 0.9116 | 0.3694 | 0.2741 |
| 0.90 | 1714 | 0.6823 | 0.4312 | 0.2394 |
| 0.95 | 1395 | 0.5553 | 0.4652 | 0.2102 |

Skor dagilimi: **min 0.6006**, medyan 0.9621, maks 1.0000.

**Dagitilan threshold (0.6) score tabaninin ALTINDA.** Bu yuzden threshold hicbir seyi
elemiyor: REVIEW katmani BOS, isaretlerin %100'u AUTO isaretleniyor. Robot
unseen a brand parcasinda each isarete kendi basina guvenir, oysa
isaretlerin however **%34.7'si** dogrudur.

**Neden cokuyor:** secim kurali with tier esigi AYNI skoru kullaniyor. Secim
already goreli (part-maksimumunun %85'i) oldugundan hayatta remaining each tahminin
skoru high; threshold "baglamiyor". Ayni cokus 2026-07-29'da a kez yasanmisti
(o zaman tier segmentasyon guvenine bakiyordu, REVIEW yine bos, precision
0.7735). Skor degisti, COKUS BICIMI geri geldi.

**Esigi yukseltmek kurtarmiyor:** 0.95'te bile precision 0.4652.

### DAGITILAN BASELINE ICIN OLCUM YOK (correction)

Raporun 9. bolumunde "same egri `wire_score` for de cikarilmali" yazmistim;
this YANLISTI -- P6 makbuzu already `wire_score` tasiyor. Ama BASELINE for gercekten
measurement none: `d7_baseline.json`'daki 2001 skorun **hepsi tam 1.0**. Sebep
`probe_deploy_verify.py`'nin `c.get("wire_score", 1.0)` varsayilani -- baseline
zinciri wire_score uretmemis, probe 1.0 yazmis. Ilk bakista "each esikte %100
AUTO" gibi gorunuyordu; this a FINDING DEGIL, OLCUM BOSLUGUDUR. Sonda residual this
durumu ayirt edip `OLCULMEMIS` diye isaretliyor.

### Urun onerisi (uygulanmadi)

Gorulmemis brand for AUTO katmani KAPATILMALI (each sey REVIEW), ta ki
parts arasi kalibre a score cikana up to. Bugunku hali, olculmemis a
guvenle otonom davranmaktir.

---

## 11. DUZELTME -- "sessiz olum" teshisi yanlisti

Gece boyunca uc surec cikis kodu 0 with, tek satir error yazmadan oldu. Ilk
teshisim **BELLEK** idi: o sirada tek a surec 16.6 GB tutuyordu ve bos RAM
2.6 GB'a dusmustu, teshis makul gorunuyordu.

**Yanlisti.** Dorduncu vaka (NIT max_sec sondasi) 17.1 GB BOS RAM varken same
sekilde oldu. Ortak payda bellek not, BASLATMA BICIMI:

| baslatma | sonuc |
|---|---|
| `run_in_background: true` which kabuk cagrisi ICINDE `nohup ... &` | dis gorev bitince surec KAPANIYOR |
| ON PLAN kabuk cagrisindan `nohup ... & disown` | YASIYOR (run_night.sh, run_extra_queue.sh boyle) |

Kural: uzun kosan is ON PLAN cagrisindan ve `disown` with baslatilir.

**Ders.** Makul a mekanizma (bellek) with o an gozlenen a olgu (low RAM)
ust uste geldiginde teshis "acikliyor" gibi gorunuyor. Ama aciklama however
KARSI ORNEKLE sinanirsa teshistir. Bu, projedeki "kapali arm" denetimiyle same
ders: a aciklamayi kabul etmeden before onu YANLISLAYACAK durumu aramak gerek.

Bellek yine de gercek a kisittir (each EK kosusu ~6 GB, makinede 31 GB) --
`run_extra_queue.sh`'nin 10 GB kapisi yerinde kaliyor. Ama night yasanan
olumlerin sebebi o degildi.

---

## 12. DECISION ARITMETIGI -- 0.50 NEREDEN GELEBILIR? (belirleyici)

Ayni kumede (`tam`) measured; receipt `results/pool_ceiling__p6_oz_u25_full.json`.

| | value |
|---|---|
| pool F1 TAVANI (mukemmel selector) | **0.8474** |
| GERCEKLESEN (P6 arm) | **0.3091** |
| **selector verimliligi** | **%36.5** |

0.50'ye iki path present ve biri kapali:

1. **Havuzla:** verimlilik fixed kalirsa tavanin **1.3707** olmasi gerekir.
   F1 tavani 1.0'i asamaz -> **HAVUZ KOLU TEK BASINA IMKANSIZ.**
2. **Seciciyle:** ceiling fixed kalirsa verimliligin **%59.0** olmasi gerekir
   (**1.62x** iyilesme).

**Sonuc: onceligi SECICI alir.** Havuz genisletme (A1b) hala degerli --
tavani yukseltir ve gerekli verimlilik carpanini dusurur -- but tek basina
hedefe goturmez. Bu, EK bloklarina (kanonik/topoloji/symmetry/depth) ve
candidate-set modeline (D2) verilen onceligi belirler.

### MAX_SEC tavani BAGLIYOR (measured)

Makbuz `results/max_select_sondasi_UPUN-SUPU30.json` (30 part, yelpaze 256):

| option tavani | yonlu recall | option maliyeti |
|---|---|---|
| 12 (BUGUNKU) | 0.8462 | 1.00x |
| **24** | **0.9077** | 1.20x |
| 40 | 0.9077 | 1.23x |

Tavani 12'den 24'e cikarmak yonlu recall'u **+0.0615** artiriyor ve maliyeti
yalnizca **1.20x**. 40'a cikarmak hicbir sey eklemiyor -> **diz noktasi 24.**

Bu, "yelpaze olu" gorunumunun sebebini de acikliyor: 256 isin onlarca direction
uretiyor but ceiling 12 oldugu for cogu eleniyor; yelpaze yeni direction EKLEMIYOR,
mevcut kaynaklarin yerini ALIYOR.

**OPEN SORU (kosuyor):** this kazanc, tavani asagi ceken markada (NIT: D6
GT'sinin %45.7'si, yonlu recall 0.5254, kaybi tam as YON kaybi) da present mi?
Orneklem markaya per secilmeli -- ilk kosu file sirasi yuzunden yalnizca
UPUN/SUPU'yu ornekliyordu.

### NIT'te ceiling very more sert bagliyor (measured)

Makbuz `results/max_select_sondasi.json` (NIT, 18 part, 482 GT, yelpaze 256):

| option tavani | NIT yonlu recall | option maliyeti |
|---|---|---|
| 12 (BUGUNKU) | 0.5913 | 1.00x |
| **24** | **0.8755** | 1.33x |
| 48 | 0.8755 | 1.35x |

Kolay markalarda kazanc **+0.0615**; tavani asagi ceken NIT'te **+0.2842**.
Yani ceiling tam da en very kanayan yerde bagliyor -- dense parts a candidate's
correct direction, 12 kisilik listeye giremiyor.

**Kaba yansima:** NIT D6 GT'sinin %45.7'si ve bugun 0.5254'te. 0.8755'e
cikarsa D6 total yonlu recall **0.7264 -> ~0.886**, F1 tavani
**0.8415 -> ~0.94**. Yani GATE A (>= 0.85) **GECMEMEKTEN GECMEYE** doner.

**Yapilan:** `_p6_oz_tam4` korpusu ceiling 24 with cikariliyor (`run_tam4.sh`,
5 pay). Diger butun settings `_p6_oz_tam3` with birebir same -- tek variable
ceiling, otherwise kazanc neye ait bilinemez. Bitiminde GATE A hem `d6` hem `tam`
kumesinde olculur.

**Not:** this night kosan butun B fazi olcumleri ceiling-12 korpusu (`tam3`)
uzerindedir; gecerlidirler but DAHA DUSUK a tavanin altinda alinmislardir.

---

## 13. SONRAKI KAMPANYA -- gecenin sayilarindan produced sira

Butun oncelikler tek a aritmetikten cikiyor (bolum 12): **selector
verimliligi %36.5** ve pool arm tek basina 0.50'ye MATEMATIKSEL OLARAK
yetmiyor. Sira buna per:

**1. Tavan-24 korpusu (KOSUYOR).** `_p6_oz_tam4`, `YB_MAX_SEC=24`. Bitince
GATE A hem `d6` hem `tam` kumesinde olculur. Beklenti: NIT yonlu recall
0.5409 -> ~0.87, total 0.7347 -> ~0.88, i.e. GATE A GECER. **Beklenti VAAT
DEGILDIR** -- probe 18 parcalik a NIT orneklemiydi.

**2. Tavan-24'un UCTAN UCA kazanci.** Tavan yalnizca TAVANI yukseltir; gercek
kazanc however `run_p6_kademe2.py` tam4 uzerinde kosunca bilinir. tam3 with
BIREBIR same ayarla kosulmali (tek variable corpus).

**3. EK bloklari tam4 uzerinde tekrarlanmali.** Bu night measured_path bloklar
ceiling-12 korpusundadir; gecerlidirler but more low a tavanin altinda.
Kapiyi passing bloklar en iyi korpusta yeniden dogrulanmali.

**4. SECICI KAPASITESI (asil is).** %36.5 -> %59 for 1.62x gerekiyor; bunu
oznitelik bloklari (blok basina +0.01..+0.03) tek basina veremez. Aday-set
modeli (D2) tek gercek candidate: candidates ARASI baglami (lattice, dizi, rekabet)
noktasal a siniflandirici gormuyor. Kapisi onceden ilan edildi: LOMO'da
HGB'ye **+0.05**.

**5. SAHA (urunun dogrudan isi).** Iki is birbirinden independent:
   - GLB'yi `canonical_chain.product_output`ya baglamak (bolum 8; tier alani ortak
     yere tasinmali).
   - AUTO katmani: unseen markada precision 0.3471 ve REVIEW BOS. Kalibre
     a score cikana up to AUTO **kapatilmali**; bugunku hali, olculmemis a
     guvenle otonom davranmaktir.

### Gecenin summary -- ne DEGISTI, ne DEGISMEDI

**Degismedi:** headline robot F1 hala **0.3091** (`tam` katlari). Bu night
dagitilan urune entering a iyilestirme YOK.

**Degisti:** residual hedefin nereden gelebilecegi OLCULU. Havuz kolunun tek
basina yetmedigi, tavanin nerede bagladigi, sahadaki confidence katmaninin atil
oldugu ve measured_path zincirin GLB'ye no girmedigi -- dordu de this night measured.
Bunlarin ucu (bolum 8, 10, 12) number not, YON degistiren bulgulardir.

### Tek satirda tez: loss KONUM not YON

`tam` set, onceki pool (receipt `pool_ceiling__p6_oz_u25_full.json`):

| olcu | value |
|---|---|
| konum recall | **0.9789** |
| yonlu recall | **0.7352** |

GT'nin %97.9'unun KONUMU havuzda; yalnizca %73.5'inin correct YONU da present.
Aradaki **0.244**, tamamen direction kaybidir. Ve direction secenekleri candidate basina 12
with sinirli, doygun halde (bkz. ceiling doygunlugu uyarisi). Havuzu genisletmek
this farki kapatmaz -- nitekim ceiling-12 tam-acik havuzda konum 0.8713 -> 0.8997
cikarken direction yalnizca 0.7264 -> 0.7347 oynadi.

---

## 14. OLCUM TASARIMI UYARISI -- B1/B6 tek degiskenli DEGIL

Orkestrator `B1_zor_negatif`i **`tam3` korpusunda VE zor-negatif acikken**
kosuyor. Elimizdeki referans (0.3091) ise **`u25` korpusunda ve zor-negatif
KAPALI**. Iki variable same anda degisiyor: a diff cikarsa KORPUSA mi
YONTEME mi ait, ayirt edilemez.

Bu, projenin defalarca yakalandigi hatanin ta kendisidir (bkz. bolum 5d:
"gelistirme set aldatir" -- orada da tabanin ne verdigi yazilmamisti).

**Cozum (kuyruga eklendi):** `tam3` uzerinde DUZ ayarla a baseline kosusu
(`results/p6_kademe2_tam3_baseline.json`). Boylece:

    baseline(u25)  -> baseline(tam3)   = KORPUS etkisi
    baseline(tam3) -> B1(tam3)      = ZOR NEGATIF etkisi

iki etki AYRI okunur. Taban kosusu olmadan B1/B6 sayilari raporlanabilir but
**yorumlanamaz**; makbuzlari o yuzden "tek degiskenli not" notuyla okunmali.

---

## 15. B1 (ZOR NEGATIF) SONUCU -- +0.0104 but TEK DEGISKENLI DEGIL

Makbuz `results/p6_kademe2_B1_zorneg.json` (corpus `tam3`, `tam` brand katlari).

| | u25 baseline | B1 (tam3 + zor negatif) | diff |
|---|---|---|---|
| **robot (MIKRO)** | 0.3091 | **0.3195** | **+0.0104** |
| recall | 0.2935 | 0.2771 | -0.0164 |
| precision | 0.3265 | 0.3772 | +0.0507 |

Kirilim, zor-negatif egitiminin BEKLENEN imzasini tasiyor: precision belirgin
yukseliyor (+0.0507), recall a miktar dusuyor -- more az but more isabetli
prediction.

### Marka kirilimi (P6 arm)

| brand | u25 baseline | B1 | diff |
|---|---|---|---|
| PXC | 0.3972 | 0.4433 | **+0.0461** |
| TOGI | 0.1689 | 0.1856 | +0.0167 |
| WEI | 0.3795 | 0.3726 | -0.0069 |
| SIE | 0.4681 | 0.4504 | -0.0177 |

### WARNING -- this number henuz YORUMLANAMAZ

Iki variable same anda degisti: **corpus** (u25 -> tam3, tam-acik pool) ve
**yontem** (zor negatif). +0.0104'un hangisinden geldigi bilinmiyor. Kuyruktaki
`tam3` BASELINE kosusu (duz setting, same corpus) ikisini ayiracak:

    baseline(u25) -> baseline(tam3)  = KORPUS etkisi
    baseline(tam3) -> B1(tam3)    = ZOR NEGATIF etkisi

Taban kosusu gelene up to B1 **raporlanabilir but kola sayilamaz**. Bu
kampanyanin kurali: a arm however TEK DEGISKENLI olcumle acilir.

---

## 16. BELIRLEYICI OLCUM -- ceiling 24, YONLU recall +0.1665 (paired kiyas)

Makbuz `results/paired_ceiling_d6.json` · probe `probe_paired_ceiling.py`.
**359 ORTAK part**, tek variable `YB_MAX_SEC` (12 -> 24).

| | ceiling 12 | ceiling 24 | diff |
|---|---|---|---|
| konum recall | 0.8988 | 0.8988 | **+0.0000** |
| **yonlu recall** | 0.7270 | **0.8935** | **+0.1665** |
| F1 tavani | 0.8419 | 0.9437 | **+0.1018** |
| option/part | 3217 | 4857 | 1.51x |

**Konum recall'un BASAMAK BASAMAK same cikmasi**, this olcumun en guclu i.e.:
ceiling konumlara dokunmuyor, kazancin TAMAMI yonden geliyor. Teorinin
ongordugu tam as buydu.

**0.8935 > 0.85 -> GATE A GECER.**

### Neden ESLI kiyas sart oldu

tam4 korpusu yarim (359/468) ve biten parts RASTGELE DEGIL -- before biten,
i.e. more small/kolay parts. Yarim tam4'un ham olcumu 0.8952 idi; bunu tam
korpusun 0.7347'siyle kiyaslamak farkin ne kadari TAVANDAN ne kadari KOLAY
ALT KUMEDEN geldigini gizlerdi. Esli kiyas iki korpusu da AYNI 359 parts
measures, alt cluster etkisi ikisinde de same olur ve geriye only ceiling kalir.

(Yarim korpusta yazilmis makbuzlar `KISMI_` onekiyle ayrildi; morning raporunun
tarama desenine residual girmiyorlar.)

### Karar aritmetigi GUNCELLENDI

| | onceki | ceiling 24 with |
|---|---|---|
| pool F1 tavani | 0.8474 | **~0.94** |
| 0.50 for gereken selector verimliligi | %59.0 | **%53.2** |
| gereken iyilesme carpani | 1.62x | **1.46x** |

Havuz arm hala TEK BASINA yetmiyor (0.94 x %36.5 = 0.343), but gereken
selector iyilesmesini 1.62x'ten 1.46x'e indiriyor. **Oncelik hala SECICI**,
fakat ceiling-24 residual dagitilmasi gereken a kazanc.

### Kalan is

Bu a CEILING olcumu; **uctan uca kazanc DEGIL**. Tavan yukselmesi however
selector o yonleri SECEBILIRSE F1'e doner. Sirasiyla: (1) tam4 korpusunu
tamamla, (2) GATE A'yi tam korpusta olc, (3) `run_p6_kademe2.py`'yi tam4 ve
tam3 uzerinde AYNI ayarla kosup uctan uca farki al.

---

## 17. GATE A GECTI -- ceiling-24 korpusunda yonlu recall 0.8926

Makbuz `results/pool_ceiling__p6_oz_tam4_d6.json` (464/468 d6 part).

| brand | GT | konum | **yonlu (t12)** | **yonlu (t24)** | F1 tavani |
|---|---|---|---|---|---|
| **NIT** | 1222 | 0.8429 | 0.5409 | **0.8429** | 0.9147 |
| SUPU | 547 | 0.9506 | 0.8921 | 0.9214 | 0.9591 |
| UPUN | 370 | 0.9649 | 0.8811 | 0.9622 | 0.9807 |
| MOR | 274 | 0.8686 | 0.8686 | 0.8686 | 0.9297 |
| UTL/SE/ONV/S+S | 238 | 1.0000 | 0.92-1.00 | 1.0000 | 1.0000 |
| **TOPLAM** | 2651 | 0.8989 | **0.7347** | **0.8926** | **0.9432** |

**GATE A GECTI (0.8926 >= 0.85).** Havuz F1 tavani **0.8470 -> 0.9432**.

### En anlamli satir NIT

NIT'in yonlu recall'u **konum recall'una ESITLENDI** (0.8429 = 0.8429). Yani
ceiling 24 iken, havuzda konumu found HER GT'nin correct direction de havuzda.
NIT'te direction darbogazi **tamamen closed** -- passing olcumde this brand 0.5409'da
ve total tavani tek basina asagi cekiyordu.

Geriye remaining loss residual saf KONUM kaybi (%10.1) ve o baska a arm.

### Duzeltilen yaniltici uyari

Ilk kosuda "CEILING DOYGUN" uyarisi tetiklendi (17.1 option/candidate). YANLIS
ALARMDI: uyari surecin kendi `MAX_SEC` varsayilanina (12) bakiyordu, oysa
corpus 24 with kurulmustu -- 17.1, 24'un %71'i, doygun not. Korpusun
kuruldugu ceiling npz'de yazili olmadigi for residual `HT_KORPUS_MAXSEC` with
verilir ve uyari hangi tavana per konustugunu YAZAR. Bir receipt logundaki
yaniltici uyari, sonradan wrong kola yatirim yaptirir.

---

## 18. SAHA KAPISI, `tam` KATLARINDA -- 0.70 KESINLIK BILE YOK

Makbuz `results/saha_gate_full.json` (3051 part, `tam` brand katlari =
unseen brand kosulu, 35326 prediction / 14300 GT).

| hedef precision | ulasilan threshold |
|---|---|
| 0.70 | **ULASILMIYOR** |
| 0.80 | ULASILMIYOR |
| 0.90 | ULASILMIYOR |
| 0.95 | ULASILMIYOR |

Ham precision **0.1711** (rule `goreli 0.50/0.05` -- bilerek GENIS tutuldu,
daraltmayi esigin yapmasi for).

**D7'deki bulguyu independent a kumede dogruluyor:** orada en high precision
0.6429 idi; burada 0.70'e bile ulasilamiyor (rule more genis oldugu for
prediction sayisi 2.5 fold).

**Sonuc:** unseen markada "robotun otonom davranabilecegi" a sign alt
set BU SKORLAYICIYLA YOK. `robot_auto_kapali` anahtarinin (S6b) gerekcesi
residual IKI independent kumede olculu.

---

# BOLUM 19 — 2026-08-12: DUVARIN YERI BULUNDU

## 19.1 Gunun net sonucu

**Uctan uca kazanc: +0.0093** (kanonik blogu). Baska hicbir arm uctan uca
kazandirmadi. d6 mikro robot F1 baseline **0.2954**.

Gunun asil urunu number not, **duvarin koordinati**.

## 19.2 Duvar nerede

`results/position_auc_d6.json` — option duzeyi AUC'nin konum ve direction paylari
AYRI measured:

| brand | n_konum | option AUC | **konum AUC** | **direction AUC** | ilk-k | rastgele |
|---|---|---|---|---|---|---|
| NIT | 432 | 0.8854 | **0.7053** | **0.8899** | 0.1135 | 0.0670 |
| MOR | 497 | 0.9657 | 0.9003 | 0.9293 | 0.2446 | 0.0277 |
| SUPU | 131 | 0.9696 | 0.9121 | 0.9596 | 0.5192 | 0.0544 |
| UPUN | 161 | 0.9877 | 0.9419 | 0.9771 | 0.6490 | 0.0615 |

Model dense parts **direction biliyor, hangi acikligin kablo girisi oldugunu
bilmiyor.** Uctan uca NIT F1 = **0.0089**; NIT GT'nin %51'i.

**Hedefin tam sayisi:** gereken konum AUC = `1 − k/n_konum`
→ NIT 0.944 (0.705 present) · MOR 0.982 (0.900 present).
Iki path: AUC'yi yukseltmek **veya pool ~5× kucultmek**.

## 19.3 Olculen ve DUSEN kollar (hepsi gate ONCE ilan edilerek)

| arm | gate | measured_path | verdict |
|---|---|---|---|
| yerel karsitlik (p − yerel median) | NIT AUC +0.05 | −0.033 | DUSTU |
| konum toplama (max→median/ort/q75/say) | +0.01 mikro | −0.014 (en iyi) | DUSTU |
| dik-direction kisiti | bugunkuyu asmak | NIT +0.017, digerlerinde ters | DUSTU |
| A1 isin atma / tup skoru | NIT ≥0.40 | tup 0.0115, **tup_eksen 0.0237** | DUSTU |
| spread (regime kapili) | uctan uca | **−0.0645** | DUSTU |
| B1 bimodal `parca_eksen` | NIT +0.05 | **+0.1277** (direction) | GECTI, uctan uca ~+0.002 |
| kanonik blogu | +0.01 | **+0.0093** | TEK KAZANC |

**A1 notu:** sentetik yetenek testini GECMISTI (uc eksende 0.0 derece deviation,
tup 8.45 / zemin 1.00) but gercek geometriye tasinmadi — unsigned axis
bile 0.0237. Mekanizma correct, uygulama alani wrong.

**Yayilim notu:** izole olcumde +0.0394 gorunuyordu; uctan uca **−0.0645**.
Izole measurement uctan uca yerine GECMEZ.

## 19.4 Yakalanan UC measurement kusuru (hepsi kendi sonucumu duzeltti)

1. **Oklid vs CARPIM kutusu.** Aday-GT eslesmesini Oklid 2mm with yapmak
   `oracle`i 0.593 → 0.0172 dusuruyordu. Kutu carpimdir: lateral ≤2mm **ve**
   axial ≤40mm. *(Bugun ikinci kez.)*
2. **GT konumu sizintisi.** "Disari" sign kurali GT konumunu kullaniyordu;
   candidate konumuna cevirince `eksen_disari` **0.5336 → 0.2668**. Gorunen
   kazancin yarisindan fazlasi sizintiydi.
3. **Denominator hatasi.** Havuz sondasinda brand basina kucultme, brand
   part sayisina not global part sayisina bolunuyordu.

## 19.5 Curutulen onerme

Gun ortasinda written `docs/plan_075_direction_CEPHESI.md` "0.75 = direction problemi"
diyordu. O onerme **option sayisindan turetilmisti** (6322 = 260 konum ×
24 direction), olculmemisti. Dogrudan measurement tersini soyledi. Plan basina uyari
konarak birakildi.

**Ders:** option sayisindan konum/direction payini TAHMIN ETME, OLC.

## 19.6 Saha baglantisi (E2)

`canonical_chain.product_output` saglamlik denetimini GECTI (40/40 part cikti,
cokme none, NaN none, yonler birim). **Ama 181 GT for 309 CP uretiyor** —
fonksiyonel karsiligi robotun olmayan yerlere gitmesi. Bayrak
`glb_kanonik_zincir` baseline zincirle paired kiyas yapilmadan ACILMADI.

## 19.7 GELISTIRME TURU — dogrulanmis kazanc

Olcum fasli kapatildi; only F1'i yukselten kollar kosuldu
(`run_improvement_sweep.py`, 20 yapilandirma, hepsi UCTAN UCA).

**Kazanan tek axis: NEGATIF ORANI.** Sistematik HPO ilk kez yapildi.

| neg ratio | d6 F1 | diff |
|---|---|---|
| 3 | 0.2549 | −0.0498 |
| 6 (baseline) | 0.3047 | — |
| **12** | **0.3135** | **+0.0088** |
| 18 | 0.3089 | +0.0042 |
| 24 | 0.3121 | +0.0074 |
| 36 | 0.3060 | +0.0013 |

12–24 arasi DUZ PLATO — sivri tepe not, i.e. d6 gurultusu not.

**`tam` DOGRULAMASI (1617 part, BES brand fold WEI/PXC/TE/SIE/TOGI):**
baseline 0.3699 → neg=12 **0.3753 (+0.0054)**.
d6'da secilip `tam`'da dogrulandi; d6'ya ezberleme YOK.

**Kapanan uc arm (hepsi ilk kez denendi):**
- part-esitleyici agirlik **0.1927 (−0.1120)** — NIT'in %51'lik baskinligi
  egitime zarar not FAYDA veriyormus
- zor negatif (score-yakin secim) **−0.0195**
- ogrenme hizi / yaprak / L2: notr ya da zararli

## 19.8 Gunun kapanis defteri

| kalem | value |
|---|---|
| kanonik blogu (`tam`) | **+0.0151** |
| negatif ratio 12 (`tam`) | **+0.0054** |
| **dogrulanmis kumulatif** | **~+0.0205** |
| D7 okuma kapisi (+0.10) | GECILMEDI — D7 OKUNMADI |

**Sahaya inme durumu:** this kazanclar P6 zincirinde; ihracatcilar
`robot_cp.extract` cagiriyor. Olculen zincir saglamlik denetimini GECTI
(40/40 part, cokme/NaN none, yonler birim) but **181 GT for 309 CP**
uretiyor. Kesinlik sorunu cozulmeden bayrak ACILMADI.

## 19.9 GECE TURU — yeni bilgi kollari ve D1/D2

Taban: temel + kanonik + neg12 = **0.3135** (d6).

**L1 hedef fonksiyonu** (`results/ranking_dizi_d6.json`):

| arm | F1 | diff |
|---|---|---|
| yerel negatif (part-ici ornekleme) | 0.3168 | +0.0034 |
| lambda agirligi (part-ici sira hatasi) | 0.3159 | +0.0024 |
| dizi uyeligi | 0.3150 | +0.0016 |
| dizi + yerel | 0.3123 | −0.0012 |

**L3/L4/L5 yeni bilgi** (`results/new_position_d6.json`):

| arm | F1 | diff |
|---|---|---|
| **ayna esi** | **0.3210** | **+0.0075** |
| isin-temas (CONTACT'a varan isin) | 0.3178 | +0.0043 |
| hepsi | 0.3172 | +0.0038 |
| vida cifti | 0.3150 | +0.0015 |
| ayna + temas + yerelneg | 0.3192 | +0.0058 |
| ayna + yerelneg | 0.3125 | −0.0010 |
| ayna + temas | 0.3085 | −0.0050 |

**HICBIRI +0.01 KAPISINI GECMEDI.** Ayna esi (+0.0075) en guclu yeni
sinyal ve dogrudan bimodal bulgusundan turedi, but kapinin altinda.
Kapiyi indirmek ya da gecene up to cluster degistirmek YAPILMADI.
Birlesimler de kazandirmiyor -- oznitelik seyrelmesi.

**D1 SENTETIK KORPUS — URETEC CALISIYOR, BORU HATTI KABUL ETMIYOR.**
Parametrik klemens uretildi (2-30 kutup, egimli giris, cift sira, sikma
vidasi, GT'ye GIRMEYEN celdiriciler). Duman testi gecti. Ama dondurulmus
segmentasyon modeli sentetik geometride **ates(le)miyor**:

| | GT agzinda | rastgele yuzey | ratio |
|---|---|---|---|
| duz delik | 0.0001 | 0.0002 | 0.52x |
| + havsa + ic kamara | 0.0003 | 0.0003 | **0.92x** |

Kontrol: same kod yolunda GERCEK part maks CE+CT 0.4985, sentetik 0.2318
-> kusur kodda DEGIL. Model sentetikte wrong yerde not HIC ateslemiyor
= girdi dagilimi kaymasi. `domain-gap-is-the-blocker`'in baska ornegi.
D1 however segmentasyon sentetikle BIRLIKTE yeniden egitilirse ise yarar.

**D2 SIE — TUKENMIS.** `_ds1`'de 331 SIE dosyasi present but **309'u already
korpusta**; yeni which yalnizca **22 part**. Onceki "SIE 310 dokunulmamis"
notu yanlisti, duzeltildi. Kiyas: +110 WEI -> +0.0391; 22 part ihmal
edilebilir.

**Geriye remaining tek canli data arm:** `_p6_oz_tam4` cikariminin bitmesi
(966 part bosta duruyordu; training verisi %60 artacak).

## 19.10 URETIM TABANI DUZELTMESI ve DAGITIM

**Yakalanan kusur.** Gelistirme taramasinin tabani `neg=6` idi, but URETIM
egiticisi (`run_p6_kademe2.py`) `P6_NEG_KAT` varsayilani **8** kullaniyordu.
Yani 6'ya per measured_path +0.0088, dagitilacak number DEGILDI. Dogru baseline
measured:

| cluster | neg=8 (URETIM) | neg=12 | diff |
|---|---|---|---|
| d6 (468 part, 4 fold) | 0.2994 | 0.3135 | **+0.0140** |
| **tam (2040 part, 5 fold)** | **0.3092** | **0.3218** | **+0.0126** |

**DEPLOYED:** `run_p6_kademe2.py` NEG_KAT 8 → 12. Canli model
`results/p6_kademe2_model.pkl.oncesi_neg12` as yedeklendi. Yeniden
training, `_p6_oz_tam4` cikarimi bitince TEK seferde yapilacak (corpus hala
buyuyor; simdi egitmek iki degisikligi birbirine karistirirdi).

**NOISE.** neg egrisi 12-24 arasi duz plato (0.3121-0.3135) ve 6 (0.3047)
with 8 (0.2994) TERS donuyor -> fold gurultusu ~±0.005. +0.0126'nin
belirsizligi gercektir ve headline verilirken yazilmalidir.

**MUTLAK DEGERLER KOSU ARASI KIYASLANAMAZ.** `tam` 1617 → 2040 parcaya
buyudu ve added parts DAHA ZOR (same arm 0.3699 → 0.3092). Yalniz
kosu-ici farklar gecerlidir.

## 19.11 KANONIK BLOK URETIMDE YENIDEN URETILMEDI — NOT DEPLOYED

Gun boyunca "gunun tek passing arm" counted kanonik hizalama blogu, URETIM
egiticisinde A/B kosuldu (`logs/kanon_ab_0.log` / `_1.log`, d6):

| arm | kanonik KAPALI | kanonik OPEN |
|---|---|---|
| **P6 (SECILEN arm)** | **0.2932** | **0.2794 (−0.0138)** |
| P6_KAFES | 0.2627 | 0.2475 |
| P6_GEO | 0.2742 | 0.2815 (+0.0073) |

Olcum betiklerinde `tam` katlarinda **+0.0151** veren arm, dagitilacak kod
yolunda **kaybettiriyor**. Sebep rule secimi: measurement betigi SABIT
`('goreli', 0.85, 0.20)` kullaniyordu; uretim fold icinde rule ARIYOR ve
`('mutlak', 0.97)` seciyor. Yani measured_path zincir with dagitilacak zincir AYNI
DEGILDI.

**NOT DEPLOYED.** `P6_KANONIK` varsayilani 0; kod duruyor.

**LESSON:** a arm "gecti" denip dagitilmadan before DAGITILACAK KOD YOLUNDA
yeniden olculur. Ayri a measurement betigindeki kazanc, uretimde de
kazandiracaginin kaniti degildir.

## 19.12 GUNUN DUZELTILMIS BILANCOSU

| kalem | durum |
|---|---|
| negatif ratio 8→12 | **DEPLOYED**, `tam` 2040 part / 5 fold: **+0.0126** |
| kanonik blok | olcumde +0.0151, URETIMDE −0.0138 → **NOT DEPLOYED** |
| ayna esi | +0.0075, gate (+0.01) GECILMEDI → dagitilmadi |
| isin-temas | +0.0043 → dagitilmadi |
| yerel negatif / lambda / dizi | +0.0034 / +0.0024 / +0.0016 → dagitilmadi |
| D1 sentetik | uretec calisiyor, segmentasyon kabul etmiyor (0.92x) |
| D2 SIE | tukenmis (yeni 22 part) |

**GUNUN TEK DAGITILAN KAZANCI: +0.0126** (noise ~±0.005).
Sabah "+0.0205" denen rakam, kanonik dusunce **duzeldi**.
D7 OKUNMADI (gate +0.10; 2 okuma kaldi).

# BOLUM 20 — 2026-08-13 GECESI: SEGMENTASYON CEPHESI

## 20.1 Neden segmentasyon

+0.30 aritmetigi tek yere cikiyor: NIT tipi dense parts GT'nin %51'i ve
uctan uca **0.0089** uretiyor. Mikroyu +0.30 oynatmak NIT'in tek basina
~0.60'a cikmasini gerektirir.

Ve kampanya boyunca SECICI optimize edildi; altindaki segmentasyon
**dondurulmus** kaldi. Canli kontrol noktalari **27 Temmuz** tarihli ve
kontrol arm yalnizca **189 part** goruyor (118 kismi insan etiketi).

## 20.2 SEG-1: GT'den kismi etiket (YAPILDI)

`run_gt_partial_label.py` — manufacturer GT'sinden corpus olceginde kismi
segmentasyon etiketi: **1556 part**, signed tepe ratio **%0.94**
(elle etiketli korpusta ~%1.5 -- same mertebe).

DONGUSEL DEGIL: etiketler model ciktisindan not URETICI JSON'undan.

**LEAKAGE BEKCISI:** only `tam` parcalari boyandi; **835 d6/d7 part
disarida**. Bolme, oznitelik dosyalarinin onekinden okundu -- ilk yazimda
`d6_record.yukle()` kullanmistim, o a GENEL DEPO (6074 kimlik) ve guard
3418 parcanin hepsini eleyip HIC etiket uretmemisti.

## 20.3 SEG-2: A/B egitimi (KOSUYOR)

| arm | etiket dizinleri | part | epoch |
|---|---|---|---|
| A (kontrol) | elle etiketli 5 dizin | 189 | 200 |
| B (deney) | + `_label_targets_gt` | ~1745 | 40 |

**Epoch secimi durustce ASIMETRIK:** B'yi 200 epoch kosmak ~16 saat
surerdi. 40 x 1745 = 69.800 ornek, A'nin 200 x 189 = 37.800 ornegine per
~1.85 fold. B kendi recetesine per AZ egitilmis; kiyas B'nin ALEYHINE
egimli. B yine de kazanirsa evidence guclu, kaybederse BELIRSIZ.

## 20.4 GECENIN OLCUM DERSI: taze inference onbellegi yeniden uretmiyor

Yeni kontrol noktalarini degerlendirmek for elle `D.predict` cagrisi
yazdim. Uc sonuc uretti ve **UCU DE INVALID CIKTI**:

- "yeni A kontrol noktasi NIT'te 4.87x" (canliya karsi 1.19x)
- "cache bayat; tazelemek kazandirir"
- kontrol noktasi basina AUC tablosu

**Kanit:** same parts, canli dort kontrol noktasindan alinan taze
olasiliklarin `_p1_olasilik` onbellegiyle korelasyonu **~0**
(−0.05 … 0.26). Farkli but valid a model olsaydi korelasyon high
olurdu. `op_cache_dir` vermek sonucu degistirmedi.

**Kural:** yeni segmentasyon kontrol noktasi elle `D.predict` with
DEGERLENDIRILMEZ. Degerlendirme, urunun probability uretme yolundan gecip
`_p1_olasilik`-benzeri a dizine yazilmali ve mevcut sondalarla
okunmalidir. Ayni aile: 19.11 (kanonik) ve `glb-measured_path-zinciri-kullanmiyor`.

**Segmentasyon A/B EGITIMI this sorundan ETKILENMEZ** -- kendi boru hattini
kullanir; etkilenen only degerlendirme yolu.

## 20.5 Yan bulgular

- **B-rep NIT'te VAR** (50/50 part, ort 119.6 silindir). "NIT'te B-rep none"
  hipotezi YANLIS; silindirler mevcut but CP'lerde not (0.011) --
  muhtemelen vida delikleri ve pimler.
- **`_p1_olasilik` onbellegi 6 Agustos tarihli ve 2 MODEL iceriyor**, canli
  yapilandirma ise 4 kontrol noktasi kullaniyor. Tum P6 oznitelik zinciri
  this onbellekten besleniyor. Etkisi 20.4 yuzunden HENUZ OLCULEMEDI.
- **`_p6_oz_tam4` cikarimi BITTI** (2560 file). Uretim modeli tam+d6
  korpusuyla ve neg=12 with yeniden egitiliyor.

## 20.6 SEG-2 SONUCU: GT KORPUSU KOLU DUSTU

| arm | training verisi | epoch | val Conn_IoU |
|---|---|---|---|
| A (kontrol) | 189 part (118 kismi insan) | 200 | **0.6232** |
| B (+GT korpusu) | 689 part (618 kismi) | 60 | **0.5540** |

**B −0.0692 with KAYBETTI.** Uretici GT'sinden turetilen kismi etiketler
segmentasyonu iyilestirmedi.

**DURUST KAYITLAR:**
- B bitiste hala TIRMANIYORDU: 0.3574 → 0.4982 → 0.5412 → 0.5540.
  A'nin 200 epoch'una karsi 60 epoch gordu. Hesap dengeliydi (41.340 vs
  37.800 ornek) but EPOCH asimetrikti -> "more uzun kosuda gecebilirdi"
  ihtimali ELENMIS DEGIL.
- Ilk deneme (1556 part) operatör hazirliginda oldu; 500 parcalik YOGUN
  alt cluster with tekrarlandi (alt cluster rastgele not, signed tepe
  sayisina per secildi -- duvar dense parcalarda).
- **Muhtemel kok why:** etiketler GT noktasi cevresine 2mm KURE
  boyanarak uretildi. Gercek CableEntry bolgesi kure not, deligin AGIZ
  YUZEYIDIR. Model wrong SEKIL ogreniyor olabilir. Bir sonraki deneme
  mouth yuzeyini (yerel normal + delik yaricapi kapili) boyamali.

**TRAP NOTU:** izleme dongusunde `pgrep -f train_seg_extra` kullandim ve
"surec oldu" wrong negatifi verdi (Git Bash Windows sureclerini gormuyor;
bkz. `ps-aux-wrong-negatif`). PowerShell `Get-CimInstance` with dogrulandi:
surecler ayaktaydi. Bu yuzden a arm gereksiz yere yeniden baslatildi.

## 20.7 E2 SAHA BAYRAGI ACILDI — measured_path zincir residual sahaya iniyor

**Sorun.** Ihracatcilar `robot_cp.extract` cagiriyordu; kampanyada measured_path
`product_p6`/`product_wide` zinciri sahaya HIC girmiyordu. Yani measured_path each
kazanc robota ULASMIYORDU (`glb-measured_path-zinciri-kullanmiyor`).

**Esli kiyas** (`probe_chain_paired_compare.py`, 80 part / 7 brand,
STEP'ten TAM zincir, AYNI parcalarda yan yana):

| path | robot F1 | precision | recall | uretilen CP |
|---|---|---|---|---|
| field (`robot_cp.extract`) | 0.1309 | 0.1592 | 0.1111 | 289 |
| **measured_path (`product_output`)** | **0.1677** | **0.5467** | 0.0990 | **75** |

**F1 +0.0368 · KESINLIK +0.3875.** Recall hafif DUSTU (0.1111 → 0.0990)
ve this yazilir.

Kural ONCE ilan edilmisti: *F1 ARTTI **VE** precision 0.02'den excess
GERILEMEDI.* Ikisi de saglandi -> `cp_config.glb_kanonik_zincir = true`.

**Fonksiyonel anlami** (kullanicinin sordugu sey): robot 289 yerine **75**
noktaya gidiyor ve yarisindan fazlasi DOGRU; eskiden 6'da 1'i dogruydu.
Yanlis CP = robotun bos yere hareketi, therefore precision artisi sahada
F1 artisindan more degerlidir.

Geri alma: `cp_config.json.oncesi_glb_zincir`.

## 20.8 SEG GT-ETIKET KOLU CLOSED

| arm | etiket sekli | training part | val Conn_IoU |
|---|---|---|---|
| A (kontrol) | — | 189 | **0.6232** |
| B | GT noktasi cevresi 2mm KURE | 689 | 0.5540 |
| C | GT EKSENI etrafinda SILINDIRIK KABUK | 689 | **0.5511** |

**Kok-why hipotezim YANLIS cikti.** "Kure wrong sekil, kanal yuzeyi
correct" dedim; sekli duzeltince sonuc DEGISMEDI (0.5540 → 0.5511).
Demek ki sorun etiket SEKLI not. Kol kapaniyor; yeniden acmak for
YENI a rationale gerekir.

## 20.9 Y19 GIRDI OZNITELIGI (xyz vs hks) — DUSTU

`train_seg_extra.py` cfg'yi SABIT `input_features="xyz"` yaziyordu; `hks`
DiffusionNet'te destekli oldugu halde no denenemiyordu. Bayrak eklendi
(`--input-features`), tek degiskenli kosuldu.

| arm | girdi | val Conn_IoU |
|---|---|---|
| A (kontrol) | xyz (3 kanal, DISSAL) | **0.6232** |
| Y19 | hks (16 kanal, ICSEL) | **0.1991** (ep 99, DURDURULDU) |

Kayip 8. epoch'defn beri kipirdamadi (1.6396 → 1.6062): model HKS'ten
OGRENEMIYOR. Hipotez makuldu (icsel oznitelik unseen markada more iyi
genellemeli) but measurement tersini soyledi. Kol 99/200'de durduruldu -- duz
loss ve 0.20 vs 0.62 farki, remaining 100 epoch'ta kapanacak a opening
not.

**Bayrak KODDA KALDI** (`--input-features`), default `xyz`.

## 20.10 Y1 HAFIF AUGMENTASYON — GECTI (+0.0197 seg IoU)

| arm | augmentasyon | val Conn_IoU |
|---|---|---|
| A (kontrol) | none (`augment=False`) | 0.6232 |
| **Y1** | **hafif, 0.3 rad (~17 derece)** | **0.6429** |

**+0.0197.** Kampanyanin segmentasyon tarafindaki ILK kazanci.

**Mekanizma ogretici:** `--augment` bayraginin yardim metni "1.05 =
aggressive (the heuristic-tuned one that hurt quality)" diyor -- i.e.
AGRESIF donme (~60 derece) denenmis ve DUSURMUS. Kol "olu" degilmis,
AYARI yanlismis. Bu, `KAPANAN_KOLLAR_DENETIMI`'ndeki desenin a ornegi
more: kapatma hukmu sondanin/ayarin kusuru olabilir.

**HENUZ NOT DEPLOYED.** Bu a segmentasyon IoU kazancidir, robot F1 not.
Kanonik dersi (Bolum 19.11) aynen valid: measurement yerinde kazanan arm,
dagitilacak yolda kaybedebilir. Once uctan uca dogrulanacak.

Kontrol noktasi: `results/seg_extra/y1_aug03_s0.pt`

# BOLUM 21 — SUNUM TABANI (temiz VAL set, bugunku sistem)

## 21.1 Kume why VAL

`results/split3.json` -> VAL, 100 part. Olculdu:
**VAL ∩ tam = 0, VAL ∩ d6 = 0, VAL ∩ d7 = 0.**
Yani hicbir training ya da exam kumesiyle kesismiyor -- independent holdout.
(DEV KULLANILMAZ: 100 parcasinin **37'si** secicinin training kumesinde.)

## 21.2 UC METRIK, same parts, same zincir

| metrik | field zinciri (dagitilan) | measured_path zincir |
|---|---|---|
| **detection F1** (konum, aci serbest) | **0.7878** | 0.6101 |
| robot F1 **ISARETSIZ** (axis) | 0.5764 | 0.5376 |
| robot F1 **ISARETLI** (fiziksel) | **0.4839** | 0.4668 |
| precision | 0.5008 | 0.5605 |
| recall | 0.4682 | 0.4000 |
| uretilen CP | 617 | 471 |

## 21.3 ISARETLI / ISARETSIZ AYRIMI — sunumda mutlaka soylenmeli

`headline.py` manseti IKI olcutle hesapliyor:
```
rob = esle(..., 2.0, 10.0, False)                  # signed VARSAYILAN False
rbi = esle(..., 2.0, 10.0, False, signed=True)   # ISARETLI
```
Yapilandirmadaki **`robot_hazir_F1 = 0.6303` ISARETSIZ olandir** -- i.e.
180 derece TERS a direction DOGRU sayilir. Kodun kendi yorumu isaretliyi
sign ediyor: *"robot for correct criterion budur"*.

**Sunumda ikisi de verilmelidir.** "Robot kabloyu hangi yone sokacagini
biliyor mu" sorusuna yalnizca ISARETLI number cevap verir.

## 21.4 Onceki headline with kiyas

| | yapilandirma (v5, DEV+VAL 194) | this measurement (VAL 100, bugunku sistem) |
|---|---|---|
| detection F1 | 0.7584 | **0.7878** |
| robot (unsigned) | 0.6303 | 0.5764 |

Tespit YUKSEK cikti. Isaretsiz robot metriginde diff present but kumeler same
DEGIL (194 vs 100 part, different urun surumu), therefore "geriledi"
denemez; this measurement DAHA TEMIZ olandir.

## 21.5 Bayrak geri alindi

`glb_kanonik_zincir` ACILDI after REVERTED. Iki populasyonda TERS:
- zor/unseen markalar (80 part): measured_path zincir F1 +0.0368
- **familiar brand (VAL 100): measured_path zincir F1 −0.0171**

Olculen zincir more KESIN (0.5605 vs 0.5008) but more low RECALL'li
(0.4000 vs 0.4682). Zor markada this takas kazandiriyor, familiar markada
kaybettiriyor. Varsayilan FALSE. Dogru cozum regime kapisi OLABILIR --
olculmeden acilmaz.

## 21.6 Y1 AUGMENTASYON UCTAN UCA DOGRULANDI — kampanyanin en large kazanci

**Adil kiyas** (VAL 100 part, STEP'ten TAM zincir, `measured_path` yolu,
IKISI DE TEK kontrol noktasi -- baseline 4'lu toplulukti, o yuzden tek-vs-tek
kosuldu):

| arm | detection | robot ISARETSIZ | robot ISARETLI |
|---|---|---|---|
| A kontrol (augmentasyon none) | 0.5430 | 0.4792 | 0.4135 |
| **Y1 (hafif augment 0.3 rad)** | **0.5808** | **0.5293** | **0.4831** |
| **diff** | **+0.0378** | **+0.0501** | **+0.0696** |

**Segmentasyon IoU'daki +0.0197, robot F1'e +0.0696 as gecti** -- i.e.
uc katiyla. Bu, kanonik dersinin (measurement yerinde kazanan dagitilan yolda
kaybedebilir) TERSI yonde a ornek: burada kazanc BUYUYEREK gecti.

**Ve kritik gozlem:** TEK augmentasyonlu ckpt (robot signed 0.4831),
DAGITILMIS DORT ckpt'lik toplulugun field yoluna (0.4839) DENK. Oyleyse
augmentasyonlu a TOPLULUK ikisini de gecmeli. Tohum 1 ve 2 egitiliyor.

**Kiyas kusuru not edilir:** ilk denemede Y1 (tek ckpt) 4'lu toplulukla
kiyaslanmisti; diff augmentasyona DEGIL ensemble-vs-tek'e ait olabilirdi.
Kontrol kolunun tek ckpt'si also kosuldu ve kiyas tek-vs-tek yapildi.

## 21.7 CALISMA NOKTASI TARAMASI (cevrimdisi) — MEVCUT NOKTA OPTIMUM

`probe_operating_point.py`, prediction dokumunden (VAL 100 part, A kontrol
tek ckpt, field yolu). Uretilen CP'ler SABIT, yalnizca tutma kurali degisir.

| deneme | robot ISARETLI | precision | recall | CP |
|---|---|---|---|---|
| BASELINE (hepsi) | 0.2611 | 0.3933 | 0.1955 | 328 |
| ilk-16 | **0.2619** | 0.3969 | 0.1955 | 325 |
| confidence>=0.5 | 0.2365 | 0.4605 | 0.1591 | 228 |
| confidence>=0.7 | 0.1882 | 0.5474 | 0.1136 | 137 |
| confidence>=0.9 | 0.0635 | **0.6667** | 0.0333 | 33 |

**En iyi alternatif +0.0008 -- noise.** Her threshold F1'i DUSURUYOR: recall
kaybi precision kazancini asiyor. **Y15 (threshold taramasi) CLOSED**, mevcut
"hepsini tut" kurali already optimum.

**Sunumda kullanilabilir yan bulgu:** precision AYARLANABILIR (0.393 →
0.667, recall bedeliyle). Robot uygulamasinda "az but emin" tercih
edilirse this a calisma noktasi secenegidir.

**Altyapi notu:** this tarama 27 dakikalik zincir kosusu yerine SANIYELER
surdu, because `probe_chain_paired_compare.py` residual tahminleri diske dokuyor.

## 21.8 KOLLARI TANIDIK MARKADA YENIDEN DENEME — HEPSI DUSTU

Hipotez: this kollar (ayna esi, yerel negatif, isin-temas) GORULMEMIS
markada kapiyi gecemedi; TANIDIK markada temsil problemi very more small
oldugu for tutabilirler. Ayni data (d6), tek variable KAT TURU:
brand-disi katlar yerine RASTGELE 3 fold (brand-KARISIK).

| arm | familiar brand | diff |
|---|---|---|
| BASELINE | **0.5603** | — |
| ayna esi | 0.5585 | −0.0018 |
| ayna + yerel negatif | 0.5575 | −0.0028 |
| ayna + temas | 0.5550 | −0.0052 |
| ayna + temas + yerel negatif | 0.5555 | −0.0048 |

**Hipotez YANLIS: hepsi familiar markada da kaybettiriyor.** Kollar closed.

## 21.9 ALAN FARKI TEK SAYIYLA — sunumun ana bulgusu

Ayni sistem, same data, same oznitelikler; **tek diff fold turu**:

| kosul | baseline robot F1 |
|---|---|
| GORULMEMIS brand (brand-disi katlar) | **0.3135** |
| TANIDIK brand (rastgele katlar) | **0.5603** |
| **diff** | **+0.2468** |

Bu, `domain-gap-is-the-blocker` bulgusunun selector duzeyindeki BUYUKLUGU.
Sunumda tek basina guclu a bulgudur: sistem kapasitesi not, YENI
URETICIYE TRANSFER bagliyor.

## 21.10 COK-CP UZMANLASMASI — TUM BICIMLERI MEASURED, HICBIRI GECMEDI

TANIDIK brand kosulu (d6, rastgele katlar), baseline 0.5572.
Cok-CP tanimi `headline.py` with AYNI: `n_gt >= 8` (468 parcanin 64'u, %13.7).

| arm | robot F1 | diff |
|---|---|---|
| **agirlik 2x** | 0.5637 | **+0.0065** |
| agirlik 4x | 0.5582 | +0.0009 |
| ince setting (warm-start, router) | 0.5536 | −0.0037 |
| ince setting (regime GT'den) | 0.5532 | −0.0040 |
| agirlik 8x | 0.5473 | −0.0099 |
| uzman (ayri training, router) | 0.5162 | −0.0410 |
| uzman (regime GT'den, UST SINIR) | 0.5154 | −0.0418 |

**YONLENDIRICI SUCLU DEGIL:** dogrulugu **0.962**, ve KAHIN rejimle bile
uzman kaybediyor. Sebep VERI PARCALANMASI.

**INCE AYAR bunu DOGRULADI:** genel modelden `warm_start` with devam edip
yalnizca dense parcalarda ek agac eklemek, bolmenin zararinin **%91'ini**
geri aldi (−0.0410 → −0.0037). Yani teshis dogruydu; but yine de baseline
asilmadi.

**VERDICT:** very-CP uzmanlasmasi SECICI duzeyinde hicbir bicimde calismiyor.
En iyisi hafif agirliklandirma (+0.0065), o da kapinin altinda.

**SESSIZ NO-OP YAKALANDI:** ince setting ilk kosuda tam **+0.0000** verdi.
Sebep: `m.max_iter_` diye a oznitelik YOK (dogrusu `n_iter_`);
AttributeError genis a `except` tarafindan yutuluyor, fonksiyon None
donuyor ve iki arm da sessizce TABANA dusuyordu. "Ince setting ise yaramiyor"
diye rapor edilecekti. Hata residual YUTULMUYOR ve agac sayisi `assert` with
dogrulaniyor.

## 21.11 AUGMENTASYON: ACI ve TOHUM

**Aci taramasi (AYNI seed s0, adil kiyas):**

| aci | val Conn_IoU |
|---|---|
| none (kontrol) | 0.6232 |
| **0.15 rad (~9 derece)** | **0.6528** |
| 0.30 rad (~17 derece) | 0.6429 |
| 1.05 rad (~60 derece) | DUSURUYOR (bayrak belgesi) |

Optimum 0.15 civarinda ya da ALTINDA. Ama:

**Tohum gurultusu ACI farkindan BUYUK.** 0.3 rad'da tohumlar:
s0 0.6429 · s1 0.6667 · s2 0.6881 (spread 0.045). Aci farki ise 0.010.
Bu yuzden tek tohumla "optimum aci budur" DENMEZ; aci kiyaslari same
tohumda yapildi ve this sinirlilik raporlanir.

## 21.12 AUGMENTASYONLU TOPLULUK — NOT DEPLOYED

VAL 100 part, TAM zincir:

| yapilandirma | detection | robot ISARETSIZ | robot ISARETLI |
|---|---|---|---|
| **DAGITILMIS 4'lu (field yolu)** | **0.7878** | **0.5764** | **0.4839** |
| augmentasyonlu 3'lu (field yolu) | 0.6204 | 0.4561 | 0.4035 |
| augmentasyonlu 3'lu (measured_path path) | 0.6151 | 0.5573 | 0.4826 |

**Saha yolunda augmentasyonlu ensemble DAHA KOTU.** Sebep teshis edildi:
`robot_cp.extract` OY SAYISINA ve confidence esigine dayaniyor, bunlar ESKI
kontrol noktalari for kalibre edilmis. Yeni modellerin kalibrasyonu
different -> oylar tutmuyor, detection 0.79'dan 0.62'ye dusuyor.

**Sonuc:** augmentasyon kazanci SEGMENTASYON duzeyinde ve OLCULEN zincirde
gercek (+0.0158 signed), but dagitilan OY TABANLI yola KALIBRASYON
yapilmadan takilamiyor. **NOT DEPLOYED.**

**Karistirici:** dagitilmis sistem 4 ckpt, this 3 ckpt. Adil olmasi for
dorduncu seed egitiliyor ve measurement tekrarlanacak.

## 21.13 SAHA YOLUNUN COKME MEKANIZMASI — teshis ve TAHMIN

Augmentasyonlu 3'lu ensemble field yolunda detection F1'i 0.7878 -> 0.6204
dusurmustu. Mekanizma:

- `robot_cp.extract` `min_votes=1` kullaniyor -> OY ESIGI sorun DEGIL.
- Ama `_votes` (kac modelin independent buldugu) **wire-gate'in
  OZNITELIGIDIR** ve olculmustu ki *durust (geometri) bolmede gate'in
  GENELLESEN TEK ozelligi votes'tur* (`gate-memorizes-not-learns`).
- Gate, ESKI **4** modelli toplulugun oy dagilimiyla egitildi. 3 model
  verilince oylar 1-3'e sikisiyor; gate DAGILIM DISI value goruyor ve
  more very REDDEDIYOR.

**TAHMIN (before yazildi):** 4 augmentasyonlu kontrol noktasiyla oylar yine
1-4 araligina doner ve field yolu TOPARLAR. Tohum 3 egitiliyor; measurement
kuyruga alindi (`results/VAL_aug4_ensemble.json`).

Tahmin tutmazsa ikinci option: wire-gate'i yeni toplulugun oy dagilimiyla
YENIDEN egitmek (iki gate AYNI dagilimda egitilmeli -- `gate-refit-minv4`).

## 21.14 KUYRUK KUSURU YAKALANDI — yari egitilmis kontrol noktasiyla measurement

4'lu ensemble olcumu for kurdugum kuyruk sunu bekliyordu:
`while [ ! -f results/seg_extra/y1_aug03_s3.pt ]`.

**Kusur:** kontrol noktasi each val iyilesmesinde yazilir, i.e. YARI
EGITILMIS halde de VARDIR. Olcum, seed 3 more epoch 20/200'deyken
basladi ve 4'lu ensemble sayisi yarim uyeyle uretilecekti.

Yakalandi ve durduruldu. Dogru kosul: training logunda `DONE best` gormek.

**Ders:** "cikti dosyasi present" with "is bitti" AYNI SEY DEGILDIR. Bu projede
same aileden dordunculuk: sessiz no-op'lar, `pgrep` wrong negatifi,
`data.index()` tuzagi, `max_iter_` yutulan hatasi.

## 21.15 HATA OTOPSISI — son-islem ailesi TEK OLCUMLE closed

`probe_error_otopsisi.py`, VAL 100 part, field yolu (A kontrol tek ckpt).
TP 129 · FP 199 · FN 531 · robot ISARETLI F1 0.2611

**Yanlis pozitif dagilimi (194):**

| tur | number | pay |
|---|---|---|
| **hayalet** (hicbir GT'ye yakin not) | 171 | **%88.1** |
| aci wrong | 14 | %7.2 |
| sign ters | 8 | %4.1 |
| cift kopya | 1 | %0.5 |

**Kacan dagilimi (531):**

| tur | number | pay |
|---|---|---|
| **bos** (yakininda HIC prediction none) | 500 | **%94.2** |
| yakin_var (prediction present, kutuya girmiyor) | 31 | %5.8 |

**VERDICT.** Son-islem onarimlarinin TAVANI:
sign %4.1 · aci %7.2 · cift %0.5 -> total **hatanin %12'si**.
Bugun denenen son-islem kollarinin (disari cevirme, axis cakistirma,
birlestirme) why hicbir sey kazandirmadigi buradan anlasilir.

**Kayip YUKARIDA:** sistem cogunlukla YANLIS YERLERDE uretiyor (%88
hayalet) ve gercek girislerin cogunun YAKININDA HIC BIR SEY uretmiyor
(%94 bos). Bu a son-islem sorunu DEGIL, ADAY URETIMI + KONUM SKORLAMA
sorunudur.

Bu, independent as measured_path konum/direction ayrimini DOGRULAR:
direction AUC 0.8899 · konum AUC 0.7053 (gereken 0.944).

**Yontem notu:** this otopsi, agregat sayilardan yapilan a cikarimin
(detection 0.7878 -> axis 0.5764 farkinin "axis hatasi" oldugu) YANLIS
oldugunu gosterdi. Agregat farklardan mekanizma cikarilmaz; error
SINIFLANDIRILIR.

## 21.16 RECEIPT KIRLILIGI YAKALANDI

`results/VAL_aug4_ensemble.json` "4'lu augmentasyonlu ensemble" adiyla
duruyordu but icerigi 3'lu kosunun degeriydi (field F1 0.4035). Sebep:
yari egitilmis ckpt with baslayan olcumu OLDURDUM, but kuyruk betigi a
sonraki satirdaki `cp` komutunu yine de calistirdi ve ESKI
`chain_paired_compare.json`'i YENI adla kopyaladi.

Karantinaya alindi: `results/_SILINDI_VAL_aug4_YARIM.json`.

**Ders:** kuyruk betiklerinde each adim, ONCEKI adimin BASARIYLA bittigini
DOGRULAMALIDIR (`&&` ya da acik kontrol). Yoksa oldurulen a olcumun
ardindan wrong etiketli receipt uretilir. Bu projede receipt cakismasi
more before de olmustu (`havuz_tavani_*`, `max_select_sondasi*`).

## 21.17 HAVUZ mu SKOR mu — OLCUM INVALID, rapor edilmedi

Hata otopsisi kacanlarin %94'unun yakininda HIC prediction olmadigini
gostermisti. Sebebi ayirmak for pool recall'u with cikti recall'u yan
yana measured:

| olcu | recall |
|---|---|
| HAVUZ (konum) | 0.5939 |
| HAVUZ (yonlu) | 0.4242 |
| CIKTI (konum) | 0.6273 |
| CIKTI (yonlu) | **0.4682** |

**CIKTI recall'u HAVUZ recall'undan YUKSEK.** Cikti havuzun ALT KUMESI
oldugu for this yapisal as IMKANSIZ; "gate'in attigi" degerinin
NEGATIF cikmasi (−0.0439) same seyi soyluyor.

**Sebep:** `robot_cp.derive_candidates(...)` benim cagirdigim bicimde,
`extract`'in gercekte kullandigi pool URETMIYOR -- zincirde asagida
`product_p6`/`product_wide` EK candidate ekliyor ve bunlar yakalanmadi.

**Olcum INVALID sayildi ve rapor edilmedi.**
Makbuz karantinada: `results/_GECERSIZ_pool_vs_output.json`.

Soru OPEN kaliyor: kacanlarin sebebi pool mu score mu. Dogru measurement for
pool, `extract`'in ICINDEN (tum candidate kaynaklari birlestikten SONRA)
alinmalidir.

## 21.18 ISARET ONARIMI — CEILING +0.0893 but HICBIR GEOMETRIK RULE YAKALAMIYOR

VAL 100 part, field yolu, correct vekillerle (mesh merkezi ve YEREL YUZEY
NORMALI dokume eklendikten after):

| arm | robot ISARETLI | diff |
|---|---|---|
| baseline | 0.4839 | — |
| yerel yuzey normali | 0.4573 | −0.0266 |
| mesh merkezi | 0.4448 | −0.0392 |
| prediction merkezi (ilk vekil) | 0.3712 | −0.1128 |
| axis cakistirma | 0.4699 | −0.0141 |
| **oracle sign (UST SINIR)** | **0.5732** | **+0.0893** |

Vekiller BEKLENEN sirada iyilesiyor (yerel normal > mesh merkezi > prediction
merkezi) but **ucu de modelin KENDI isaretinden kotu**. Yani error "model
iceri bakiyor" DEGIL; baska a sey.

**VERDICT:** sign onarimi ailesi (disari kurali, axis cakistirma)
CLOSED. **Tavan +0.0893 kayitta kalir** -- more akilli a yontem for
acik ve olculmus a hedef.

**Yontem notu:** ilk olcumde `disari_mesh` ve `disari_normal` tam
`+0.0000` vermisti; reason dokumde o alanlarin HENUZ olmamasiydi (arm
sessizce hicbir sey yapmiyordu). Tam sifir imzasi residual sessiz no-op
belirtisi as taniniyor.

## 21.19 4'LU AUGMENTASYONLU TOPLULUK — OY HIPOTEZI DOGRULANDI

VAL 100 part, TAM zincir:

| yapilandirma | detection | robot ISARETSIZ | robot ISARETLI |
|---|---|---|---|
| DAGITILMIS (4 eski ckpt, field) | **0.7878** | 0.5764 | 0.4839 |
| 3 aug, field | 0.6204 | 0.4561 | 0.4035 |
| 4 aug, field | 0.6725 | 0.5380 | 0.4613 |
| **4 aug, OLCULEN zincir** | 0.6392 | 0.5761 | **0.5162** |

**OY HIPOTEZI DOGRULANDI.** 21.13'te before yazilmisti: "4 ckpt with oylar
1-4 araligina doner ve field yolu toparlar." Olculdu: 3→4 gecisi field
yolunda robot-isaretliyi 0.4035 → 0.4613 (+0.0578), tespiti
0.6204 → 0.6725 cikardi. Cokusun sebebi gercekten OY DAGILIMIYDI.

**En iyi robot-signed number:** augmentasyonlu 4'lu + OLCULEN zincir =
**0.5162**, dagitilmisin 0.4839'una karsi **+0.0323**.

**IKI DURUST KAYIT:**

1. **Tespit DUSUYOR** (0.7878 → 0.6392). Robot for valid criterion
   isaretlidir, but this TAKAS sunumda soylenmelidir.
2. **+0.0323 NOISE BANDININ ICINDE.** Tabanin %95 GA'si
   [0.4003, 0.5671] ve 0.5162 tam onun icinde. "Iyilesti" denebilir;
   "istatistiksel as ayirt edilebilir a iyilesme" DENEMEZ --
   n=100 buna yetmiyor.

Bu yuzden DAGITIM KARARI verilmedi: kazanc direction correct but evidence gucu
yetersiz. Daha large a degerlendirme set (LOCKED, 100 part more)
ayirt edici olabilir -- but LOCKED harcanmadi ve this karar for
harcanmamalidir.

## 21.20 AUGMENTASYON ACISI — tarama TAMAM

| aci | val Conn_IoU (seed 0) |
|---|---|
| none (kontrol) | 0.6232 |
| **0.15 rad (~9 derece)** | **0.6528** |
| 0.30 rad (~17 derece) | 0.6429 |
| 0.50 rad (~29 derece) | 0.6424 |
| 1.05 rad (~60 derece) | DUSURUYOR (bayrak belgesi) |

Optimum ~0.15; 0.3 ve 0.5 birbirine yakin; hepsi kontrolden IYI. Tohum
gurultusu (0.6429-0.6881) aci farkindan large oldugu for "optimum tam
as 0.15'tir" DENMEZ -- soylenebilecek which: **hafif augmentasyon
kazandirir, agresif kaybettirir**.

## 21.21 TOPLULUK CESITLILIGI (6 ckpt) — training gerektirmeyen arm

Elde 6 augmentasyonlu ckpt present: 4 seed x 0.3 rad + 0.15 + 0.5.
FARKLI aci = different error deseni = ensemble cesitliligi. Yeni training
GEREKTIRMEZ, yalnizca measurement. VAL 100 parts kosuluyor.

## 21.22 Y6 LABEL SMOOTHING — DUSTU

| arm | val Conn_IoU |
|---|---|
| augment 0.15 (baseline) | **0.6528** |
| + label smoothing 0.05 | 0.5244 (**−0.1284**) |

Kismi etiketli maskeli BCE'de already `partial_pos_weight = 20` present
(CableEntry tepelerin ~%1.5'i). Hedefi yumusatmak this dengeyi bozuyor:
pozitif sinyal already 20 fold agirlikliyken hedefi 0.95'e cekmek etkiyi
seyreltiyor. **CLOSED.**

Bayrak kodda kaldi (`--label-smooth`, default 0.0).

## 21.23 TOPLULUK CESITLILIGI (6 ckpt) — DUSTU

VAL 100 part, measured_path zincir, robot ISARETLI:

| ensemble | robot ISARETLI |
|---|---|
| **4 seed x 0.3 rad** | **0.5162** |
| 6 ckpt (4 seed + 0.15 + 0.5) | 0.4859 (**−0.0303**) |

Farkli ACILARLA cesitlilik katmak YARDIM ETMIYOR. Ayni acidaki dort seed
more iyi. Muhtemel reason: 0.15 ve 0.5 kollari TEK tohumlu ve seed
gurultusu (0.6429-0.6881) aci farkindan large; toplulugu zayif uyelerle
seyreltiyorlar.

**CLOSED.** Bugunun en iyisi degismedi: **4 aug seed + measured_path zincir =
0.5162** (dagitilmis 0.4839).

## 21.24 SNAPSHOT TOPLULUGU (Y4) — DUSTU

`best` + `last` kontrol noktalari birlikte (4 seed x 2 = 8 ckpt):

| ensemble | robot ISARETLI (measured_path zincir) |
|---|---|
| **4 seed, only `best`** | **0.5162** |
| 8 ckpt (`best` + `last`) | 0.5039 (**−0.0123**) |
| 6 ckpt (different acilar) | 0.4859 (−0.0303) |
| 3 seed | 0.4826 (−0.0336) |

`last` kontrol noktalari `best`ten more zayif oldugu for toplulugu
SEYRELTIYORLAR. **Y4 CLOSED.**

## 21.25 TOPLULUK DESENI — dort seed, only `best`, AYNI aci

Dort ayri ensemble denemesinin hepsi same sonuca sign ediyor:

| deneme | diff |
|---|---|
| seed sayisi 3 -> 4 | **+0.0336** |
| different acilarla cesitlendirme | −0.0303 |
| `last` snapshot'lari ekleme | −0.0123 |

**Kural:** topluluğu buyutmek not, UYELERI GUCLENDIRMEK kazandiriyor.
Zayif uye eklemek each seferinde zarar verdi. En iyi yapilandirma
**4 seed x hafif augmentasyon (0.3 rad) x only `best` + measured_path
zincir = 0.5162**.

## 21.26 Y14 FOCAL-TVERSKY — NOTR

| arm | val Conn_IoU |
|---|---|
| augment 0.15 (baseline) | 0.6528 |
| + Focal-Tversky (gamma 1.33) | 0.6550 (**+0.0022**) |

Tohum gurultusu ~0.045 oldugu for +0.0022 ANLAMSIZ. Ne kazandirir ne
kaybettirir. Bayrak kodda kaldi (`--tversky-gamma`, default 1.0 =
klasik Tversky, davranis degismez).

**CLOSED (notr).**

## 21.27 Y5 YARDIMCI GOREV (aux-wire) — POZITIF but NOISE ICINDE

`--aux-wire` bayragi kodda VARDI, this kampanyada HIC acilmamisti.
53/189 parts TEL/ALET yardimci etiketi present (agirlik 0.5, pos_w 2.0).

| arm (augment 0.15 tabani uzerine) | val Conn_IoU | diff |
|---|---|---|
| baseline | 0.6528 | — |
| **aux-wire** | **0.6611** | **+0.0083** |
| Focal-Tversky (gamma 1.33) | 0.6550 | +0.0022 |
| label smoothing (0.05) | 0.5244 | −0.1284 |

aux-wire uc kolun EN IYISI but **seed gurultusunun (~0.045) icinde**.
Tek tohumla "kazandi" DENMEZ. Dogru test very tohumlu olurdu (~3 x 50 dk);
this, remaining sureye sigmadi ve boyle yazildi.

**Not:** same sinirlilik Focal-Tversky for de valid. Ikisi de "notr ya
da hafif pozitif" kategorisinde; hicbiri augmentasyon up to (seg IoU
+0.0296, uctan uca +0.0696) net not.

## 21.28 Y7 OZNITELIK SECIMI — KAPIYA EN YAKIN KOL (+0.0097)

Secicinin oznitelik matrisi 162 sutun; sistematik no budanmamisti.
Permutasyon onemine per siralanip budandi (TANIDIK brand, rastgele fold):

| arm | sutun | robot F1 | diff |
|---|---|---|---|
| hepsi | 162 | 0.5489 | — |
| **ilk %75** | **121** | **0.5587** | **+0.0097** |
| ilk %50 | 81 | 0.5471 | −0.0018 |
| ilk %25 | 40 | 0.5417 | −0.0072 |

En onemsiz **41 sutunu atmak +0.0097** kazandiriyor -- kapinin (+0.01)
KIL PAYI altinda.

**Desen, bugunun diger sonuclariyla ORTUSUYOR:**
- hafif augmentasyon iyi (0.15), agresif kotu (1.05)
- 2x agirlik iyi, 8x kotu
- az budama iyi, very budama kotu

Uc independent kolda same sekil: **olculu mudahale kazandirir, asiri
mudahale kaybettirir.**

**Kapiyi gecmis SAYILMADI** (tek seed, rastgele fold). Ama listedeki EN
UMUT VERICI acik arm budur; very tohumlu tekrarda gecebilir.

## 21.29 Y8 ETIKET KALITESI AGIRLIKLANDIRMA — 4 TOHUMDA DOGRULANDI

**Fikir.** Secicinin pozitif etiketleri ESIT agirlikta ogretiliyor. Ama
GT'ye 0.2mm'de oturan candidate with tolerans sinirinda 1.9mm'de oturan candidate
same guvenilirlikte DEGIL. Pozitife, GT'ye YANAL yakinligiyla orantili
agirlik verilir. **Metrik DEGISMEZ** -- yalnizca training agirligi
(this, `metrik-cerrahisi-a1-reddedildi`'den different olmasinin sebebi).

**DORT fold tohumunda:**

| seed | baseline | lineer | karesel |
|---|---|---|---|
| 1 | 0.5572 | +0.0144 | +0.0091 |
| 2 | 0.5635 | −0.0019 | +0.0079 |
| 3 | 0.5321 | +0.0101 | +0.0123 |
| 4 | 0.5491 | +0.0102 | +0.0036 |
| **mean** | — | **+0.0082** | **+0.0082** |

**8 olcumun 7'si POZITIF.** Kol gercek; mean +0.0082, kapinin
(+0.01) ALTINDA.

**METODOLOJIK LESSON -- bugunun en onemlisi.** Tek tohumda +0.0144 with
"GATE GECTI" yazdi; ikinci tohumda −0.0019 verdi. Tek tohumla dagitilsaydi
OLMAYAN a kazanc raporlanmis olurdu. Kat gurultusu this buyuklukteki
kollarda **±0.008** mertebesinde; therefore **+0.01 civari each arm COK
TOHUMLU dogrulanmadan verdict giymemelidir.**

Bu, gun boyunca "noise bandinda" diye isaretledigim tum kollari
(Y7 +0.0097, aux-wire +0.0083, Focal-Tversky +0.0022) da kapsar.

## 21.30 Y7 OZNITELIK SECIMI — 3 TOHUMDA CLOSED

| fold seed | ilk %75 sutun (121/162) |
|---|---|
| 1 | +0.0097 |
| 2 | +0.0022 |
| 3 | +0.0002 |
| **mean** | **+0.0040** |

Kapinin (+0.01) COK altinda. **CLOSED.**

Tek tohumdaki +0.0097'nin YARIDAN FAZLASI gurultuydu -- Y8'de gorulen
desenin same. Iki arm da same sonuca sign ediyor: this buyuklukteki
etkiler tek tohumla OLCULEMEZ.

## 21.31 GUN 2 KAPANIS BILANCOSU

**Dogrulanmis ve dagitilan:**
| kalem | kazanc |
|---|---|
| negatif ratio 8 → 12 | +0.0126 (`tam`, 5 brand fold) |

**Olculmus, dogrulanmis, but gate altinda (dagitilmadi):**
| kalem | kazanc | dogrulama |
|---|---|---|
| etiket kalitesi agirliklandirma | +0.0082 | **4 fold seed**, 7/8 pozitif |
| yardimci gorev (aux-wire) | +0.0083 | tek seed |
| oznitelik budama (%75) | +0.0040 | 3 fold seed |
| Focal-Tversky | +0.0022 | tek seed |
| very-CP agirligi 2x | +0.0065 | tek seed |

**Segmentasyon tarafinda dogrulanmis kazanc:**
| kalem | kazanc |
|---|---|
| hafif augmentasyon (0.15-0.3 rad) | seg IoU +0.0197…+0.0649 |
| uctan uca (tek-vs-tek, VAL) | robot ISARETLI **+0.0696** |
| 4 aug seed + measured_path zincir | 0.5162 vs dagitilmis 0.4839 |

**Kapanan kollar (hepsi makbuzlu):** sign onarimi (ceiling +0.0893 acik
kaliyor) · dense-part uzmani (oracle rejimle bile −0.0418) · calisma
noktasi taramasi · hks girdi ozniteligi · axis cakistirma · GT'den
segmentasyon etiketi (iki sekil) · sentetik corpus · snapshot toplulugu ·
ensemble cesitliligi · label smoothing · Y7 oznitelik budama

**Gecersiz counted olcumler:** pool-vs-score (yapisal as imkansiz
sonuc) · yarim-ckpt toplulugu · receipt kirliligi (karantinada)

## 21.32 ISARET KOLUNUN COKUS MEKANIZMASI — teshis edildi

Secici sign kurallari denendi (only OPEN celiskide cevir):

| rule | diff |
|---|---|
| normal_kesin (yerel normale per) | **+0.0000** (no tetiklenmedi) |
| mesh_kesin 150 derece | +0.0031 |
| mesh_kesin 110 derece | +0.0094 |
| mesh_kesin 120 / 135 | +0.0047 / +0.0063 |
| parca_modal | −0.0094 |

**Esik taramasi MONOTONIK DEGIL** (110 > 135 > 120 > 150). Gercek a
mekanizma duzgun a egri verirdi; this desen 100 parts GURULTUYE UYDURMA
demektir ve bugun measured_path ±0.008'lik fold gurultusuyle same mertebede.
110 dereceyi secmek, Y8'de duselecek tuzagin ta kendisi olurdu.
**SECILMEDI.**

**KOK WHY BULUNDU.** Tahmin edilen direction with yerel yuzey normali
arasindaki aci (617 CP):

| yuzdelik | aci |
|---|---|
| %25 | 60.9 derece |
| **%50** | **88.9 derece** |
| %75 | **90.0 derece** |

Yon, yerel normale **neredeyse DIK**. Sebebi geometrik: CP noktasina EN
YAKIN tepe, deligin DUVARINDADIR ve duvar normali eksene DIKTIR. Yani
"disari" referansi diye deligin duvar normali kullanilmis; eksene dik a
referansla sign atamak rastgeleye yakindir. `disari_normal`in −0.0266
vermesinin sebebi budur.

**Dogru referans:** en yakin tepenin normali DEGIL, **mouth cevresindeki
YUZUN** normali (mouth halkasinin disindaki duz yuzey). Gelecek deneme for
somut correction; remaining surede kurulup gurultuden ayrilamadi.

**Tavan +0.0893 hala OPEN.**

## 21.33 COK-CP AGIRLIGI — O DA DOGRULANMADI

| fold seed | agirlik 2x |
|---|---|
| 1 | +0.0065 |
| 2 | **−0.0052** |

Ucuncu arm da tek tohumda pozitif, ikinci tohumda NEGATIF. (Tohum 3-4
kosuyordu, makine kapatildi.)

**UC BAGIMSIZ KOLDA AYNI RESULT:**

| arm | tek seed | very seed |
|---|---|---|
| Y7 oznitelik budama | +0.0097 | **+0.0040** (3 seed) |
| Y8 etiket kalitesi | +0.0144 | **+0.0082** (4 seed, 7/8 poz) |
| very-CP agirligi 2x | +0.0065 | **sign degistirdi** (2 seed) |

**Bu, gunun en saglam bulgusudur:** +0.005…+0.015 araligindaki HICBIR arm
tek tohumla olculemez. Kat gurultusu ±0.008 ve this kollarin etkisiyle AYNI
mertebede.

Y8 ucu icinde EN saglami (4 seed, 8 olcumun 7'si pozitif, mean
+0.0082) but o da kapinin altinda.

## 21.34 COK TOHUMLU DOGRULAMA AILESI — TAMAMLANDI

| arm | tek seed | very seed | seed sayisi |
|---|---|---|---|
| very-CP agirligi 2x | +0.0065 | **−0.0002** | 4 |
| Y7 oznitelik budama | +0.0097 | **+0.0040** | 3 |
| Y8 etiket kalitesi | +0.0144 | **+0.0082** | 4 |

very-CP agirligi ayrintisi: +0.0065 / −0.0052 / −0.0034 / +0.0012.
**Tam sifir.** Ilk tohumdaki degerin TAMAMI gurultuydu.

**UC KOLDA DA tek tohumlu value SISIKTI.** Yalniz Y8 very tohumda net
pozitif kaldi (+0.0082, 8 olcumun 7'si pozitif) but o da kapinin altinda.

**Bu ailenin sonucu, kampanyanin en saglam metodolojik ciktisi:**
this buyuklukteki (+0.005…+0.015) etkiler tek tohumla OLCULEMEZ; fold
gurultusu ±0.008 ve etkiyle AYNI mertebede. Gun boyunca "kapiya yakin"
gorunen hicbir arm this nedenle dagitilmadi.

## 21.35 HALKA NORMALI ILE ISARET DUZELTMESI — KAPIYI GECTI (+0.0329)

**Teshis (21.32) correct cikti.** "En yakin tepe normali" deligin DUVAR
normaliydi ve eksene DIK; correct referans **mouth cevresindeki halkadan**
(3-8mm) alinan yuz normali.

VAL 100 part, field yolu:

| rule | robot ISARETLI | diff |
|---|---|---|
| baseline | 0.4839 | — |
| en yakin tepe normali (eski) | 0.4573 | −0.0266 |
| **halka normali** | **0.5168** | **+0.0329** |
| oracle sign (UST SINIR) | 0.5732 | +0.0893 |

Kazanc, oracle tavaninin **%37'si** ve fold gurultusunun (±0.008) **4 KATI**.
Saf SON-ISLEM: training none, model degisikligi none, **ayarlanmis parametre
none** (yalnizca "isareti halka normaliyle uyumlu yap").

**ESLI BOOTSTRAP (correct test):** marjinal GA not FARKIN GA'si, because
kiyas AYNI parcalarda.

    diff +0.0329   %95 GA [−0.0155, +0.0819]
    bootstrap orneklerinin %90.6'si pozitif

**GA sifiri ICERIYOR -> evidence YETERSIZ.** 100 part this farki ayirmaya
yetmiyor.

**Kuralin ayarlanmis parametresi OLMADIGI for** d6 parcalarinda test
etmek leakage yaratmaz: baseline F1 ornek-ici oldugu for sisik olur but
ESLI FARK valid kalir. 150 parcalik independent ornek kosuluyor -- LOCKED
harcanmadan evidence gucu artirilir.

## 21.36 HALKA KURALININ DAVRANISI ve THRESHOLD TARAMASI

**Davranis** (VAL 617 CP): rule 95 tanesini cevirir (%15.4) --
**48 DUZELTIR, 27 BOZAR, 20 notr**. Net +21 CP; kararli cevirmelerin
**%64'u correct**. Kural correct yonde but GURULTULU; paired bootstrap GA'sinin
sifiri icermesinin sebebi budur.

**Esik taramasi (only GUCLU celiskide cevir):**

| threshold | diff |
|---|---|
| 95 derece | +0.0345 |
| 105 / 115 | +0.0298 / +0.0298 |
| 125 | +0.0063 |
| 140 / 160 | −0.0063 / −0.0047 |

**Monotonik DEGIL** -> gurultuye uydurma. Ayrica 95 derece already ~90
derece, i.e. SADE kuralin kendisi. **Esik eklemek a sey KATMIYOR;
sade rule kullanilir.**

## 21.37 HALKA KURALI OLCULEN ZINCIRDE **ISTATISTIKSEL OLARAK KESIN**

Ayni rule, same parts, `measured_path` yolu (VAL 100):

| rule | robot ISARETLI |
|---|---|
| baseline | 0.4668 |
| en yakin tepe normali | 0.4226 (−0.0442) |
| **halka normali** | **0.4863 (+0.0195)** |
| oracle sign (UST SINIR) | 0.5323 (+0.0654) |

**ESLI BOOTSTRAP:**

    diff +0.0195   %95 GA [+0.0030, +0.0378]
    bootstrap orneklerinin %98.5'i pozitif

**GA SIFIRI ICERMIYOR -> kazanc ISTATISTIKSEL OLARAK KESIN.**
Bu, kampanyanin ilk istatistiksel as conclusive iyilesmesidir.

**Iki path karsilastirmasi tutarli:**

| path | diff | %95 GA | uretilen CP |
|---|---|---|---|
| field | +0.0329 | [−0.0155, +0.0819] | 617 |
| **measured_path** | **+0.0195** | **[+0.0030, +0.0378]** | 471 |

Saha yolunda etki DAHA BUYUK but more OYNAK; measured_path zincirde more small
but more TEMIZ (more az CP -> more az varyans). Iki independent yolda da
POZITIF, birinde KESIN.

**Kural dagitilabilir:** parametresiz, training gerektirmiyor, gerekli data
(mesh V/F + prediction noktalari) `export_robot_glb` icinde already mevcut.

## 21.38 HALKA KURALI POPULASYONA BAGLI — kapsam sinirlandi

Bagimsiz 150 parcalik ornek (ZOR/unseen markalar: NIT, MOR, SUPU,
UPUN, UTL, ONV, S+S, SE):

| path | baseline | halka | diff | %95 GA | bootstrap poz. |
|---|---|---|---|---|---|
| field | 0.1020 | 0.1003 | **−0.0016** | [−0.0054, +0.0000] | %0.0 |
| measured_path | 0.1605 | 0.1628 | **+0.0023** | [−0.0052, +0.0105] | %61.2 |

**Zor markalarda rule HICBIR SEY kazandirmiyor.** Saha yolunda small but
SISTEMATIK negatif (bootstrap orneklerinin %0'i pozitif).

**Bu a CELISKI DEGIL, KAPSAM SINIRI.** Tabanlara bakildiginda populasyon
tamamen different: VAL'de baseline 0.4668, zor markalarda 0.1605. Kural, baseline
sistemin ZATEN CALISTIGI yerde yardim ediyor.

**DURUST IFADE:**

| populasyon | path | diff | verdict |
|---|---|---|---|
| TANIDIK brand (VAL 100) | measured_path | **+0.0195** | **GA sifiri icermiyor -> KESIN** |
| TANIDIK brand (VAL 100) | field | +0.0329 | GA sifiri iceriyor |
| ZOR brand (150 part) | measured_path | +0.0023 | notr |
| ZOR brand (150 part) | field | −0.0016 | small sistematik negatif |

**Yontem notu:** 21.37'de "iki independent yolda pozitif, birinde conclusive"
yazmistim. O yazildiginda this ucuncu measurement YOKTU. Simdi present ve iddiayi
DARALTIYOR: kazanc TANIDIK brand populasyonuna ozgudur.

**DAGITIM KARARI:** rule familiar markada dogrulanmis kazanc, zor markada
notr/ihmal edilebilir negatif veriyor. Sunumun cerceve populasyonu
TANIDIK brand oldugu for dagitilabilir; but zor markada BEKLENTI
YARATMAMALI.

## 21.39 Y13 SINIR-FARKINDALI KAYIP — CLOSED

Ek terim: komsusu FARKLI sinifta which tepelere (sinif siniri) odaklanan
a odak kaybi. Gerekce: CP fiziksel as a SINIRDIR (mouth cemberi)
but mevcut loss (NLL + Tversky) BOLGEYI hedefler.

| arm | val Conn_IoU | diff |
|---|---|---|
| baseline (augment 0.15) | 0.6528 | — |
| sinir agirligi 0.3 | 0.6415 | −0.0113 |
| sinir agirligi 1.0 | 0.6484 | −0.0044 |

**Ikisi de NEGATIF. CLOSED.** Fikir mekanik as makuldu but measurement
aksini soyledi. Kod kaldi (`--sinir-weight`, default 0 = davranis
degismez); `faces` anahtarinin `ops` icinde GERCEKTEN oldugu ONCEDEN
dogrulandi (sessiz no-op riski elendi).

## 21.40 Y5 AUX-WIRE — ESLESEN TOHUMLARDA CURUDU

Ilk measurement tek tohumdaydi ve tabanla ESLESMIYORDU. Uc tohumda ESLESEN
baseline (augment 0.15) also egitildi:

| seed | aux-wire | matched baseline | diff |
|---|---|---|---|
| 0 | 0.6611 | 0.6528 | +0.0083 |
| 1 | 0.6691 | 0.6990 | **−0.0299** |
| 2 | 0.6672 | 0.6673 | −0.0001 |
| **mean** | | | **−0.0072** |

**Ortalama NEGATIF. CLOSED.** Tek tohumdaki +0.0083 gurultuydu.

## 21.41 SEGMENTASYON TARAFINDA TOHUM GURULTUSU — sayisal evidence

Ayni recete (augment 0.15), yalnizca seed different:

| seed | val Conn_IoU |
|---|---|
| 0 | 0.6528 |
| 1 | **0.6990** |
| 2 | 0.6673 |
| **spread** | **0.046** |

Ayni sey augment 0.3'te de gorulmustu: 0.6429 / 0.6667 / 0.6881.

**Segmentasyon tarafinda seed gurultusu ~0.046'dir.** Bugun measured_path
kollarin etkileri 0.002-0.030 araligindaydi, i.e. gurultunun ALTINDA.
Tek kosumluk hicbir segmentasyon sonucu YORUMLANAMAZ.

**COK TOHUMDA CURUYEN KOLLAR (dort):**

| arm | tek seed | very seed |
|---|---|---|
| very-CP agirligi 2x | +0.0065 | −0.0002 (4 seed) |
| Y7 oznitelik budama | +0.0097 | +0.0040 (3 seed) |
| Y8 etiket kalitesi | +0.0144 | +0.0082 (4 seed) |
| Y5 aux-wire | +0.0083 | **−0.0072** (3 matched seed) |

Yalniz Y8 pozitif kaldi; o da kapinin altinda.

## 21.42 KALAN SEGMENTASYON KOLLARI SURE ICINDE VERDICT GIYEMEZ — durust karar

21.41'de measured_path seed gurultusu (**0.046**) dogrudan a source hesabi
dayatir:

* Bir segmentasyon kolunun verdict giymesi for **>=3 matched seed** sart
  (this night dort arm tek tohumda yaniltti, biri **sign degistirdi**).
* Bir training ~2 saat. Yani **arm basina ~6 saat**.
* Kalan segmentasyon kollari: Y3 (EMA/SWA), Y9 (curriculum), Y11
  (optimizator), Y10 (mixup), Y17 (spektral augment), Y18 (jeodezik
  jitter), Y26 (sinifa-ozel augment) = **7 arm x 6 saat = ~42 saat**.

Kalan sure: birkac saat. **Bu kollari this night kosmak, verdict giydiremeyecek
sayilar uretmekten baska a sey yapmaz** -- ve tek tohumluk a number this
projede dort kez yaniltti.

**DECISION: yeni segmentasyon EGITIMI baslatilmiyor.** Gecenin kalani, verdict
giyebilecek kollara ayriliyor -- i.e. **training gerektirmeyen** olanlara:

| arm | maliyet | why verdict giyebilir |
|---|---|---|
| Y16 remesh-varyant toplulugu | 3 x 30 dk inference | tek modelde, seed gurultusu YOK |
| Y22 MC dropout | ~30 dk inference | same model, deterministik baseline |
| Havuz mu score mu (acik soru) | ~30 dk inference | teshis; arm not |

Bu, kampanyanin kendi kuralinin kendi planina uygulanmasidir:
**olculemeyecek seyi olcmeye kalkma.**

## 21.43 Y22 MC DROPOUT — kanca kuruldu, SESSIZ NO-OP DEGIL (dogrulandi)

**Tuzak.** `diffusionnet.predict()` icinde `model.eval()` cagriliyor; this
dropout'u KAPATIR. Dropout katmanlarini disaridan `train()`'e almak this
yuzden **sessiz no-op** olurdu -- this night same tuzak baska a kolda tam
`+0.0000` uretmisti. Kanca `eval()`'den SONRA, `predict()`'in ICINE
kondu (`mc_dropout=T` parametresi) ve opened katman sayisi `assert` with
dogrulaniyor.

**DUMAN TESTI (sentetik mesh, tek kontrol noktasi):**

| measurement | sonuc |
|---|---|
| MC(8) with baseline arasi mean abs diff | **0.047173** |
| MC(8) iki kosu arasi diff (tekrar uretilebilirlik) | 0.00000042 |
| Taban (MC kapali) iki kosu arasi diff | 0.000001 |
| Taban etiket ayniligi | %100.0000 |

Etki, gurultunun **~10^5 fold**. Kol GERCEKTEN calisiyor ve
tekrar uretilebilir.

**Yan bulgu:** inference bit-same not (ozdeger cozumunden ~1e-6), but
ETIKET duzeyinde %100 ozdes. Yani inference gurultusu, fold gurultusunun
(0.046) kaynagi DEGIL -- source training seed.

## 21.44 OPEN SORU COZULDU (before DIAGNOSIS): pool DISARIDAN uretilemez

Bolum 21.17'de pool olcumu "invalid" diye karantinaya alinmisti: cikti
recall'u (0.4682) HAVUZ recall'undan (0.4242) BUYUK cikiyordu -- cikti
havuzun alt set oldugu for **yapisal as imkansiz**. Sebep bugun
bulundu.

**Teshis** (`probe_pool_diagnosis.py`, mevcut dokumden, yeni inference YOK).
Macar eslestirme yerine **GT basina kapsama** measured (o GT'yi kabul
kutusunda karsilayan HERHANGI a candidate present mi) -- pool sorusunun correct
olcutu budur. Yapisal kontrol: "ciktida VAR, havuzda YOK" sayisi **0
olmali**.

| path | unsigned ihlal | ISARETLI ihlal |
|---|---|---|
| field | **84** | 59 |
| measured_path | **104** | 85 |

Ihlal sifir not -> **kaydedilen pool, ciktiyi ureten pool DEGIL.**

**KOK WHY.** `robot_cp.extract` cikarimi **operator onbellegiyle**
yapar (`op_cache_dir=f"{OP}_k{k_eig}"`); probe ise pool yeniden uretmek
for onbelleksiz TAZE inference kullaniyordu. Olasiliklar different cikiyor,
therefore candidate pool da different. Bu, more before independent as
kaydedilmis which **"taze inference onbellegi yeniden uretmiyor"** bulgusunun
ta kendisidir -- orada uc gecelik sonucu invalid kilmisti, burada pool
olcumunu invalid kildi.

**DUZELTME.** Havuz residual `extract`in **ICINDEN**, wire-gate'ten hemen
before yakalaniyor (`robot_cp.HAVUZ_KANCA`, only `CP_HAVUZ_KANCA` cevre
degiskeniyle dolar; urun yolu DEGISMEZ). Sonda da onu kullaniyor.

**LESSON (genellestirilebilir).** Bir ara degeri "same kodu disaridan
cagirarak" yeniden uretmek, o kodun ONBELLEK/DURUM bagimliligi varsa
sessizce different sonuc verir. Ara degerler URETILDIKLERI YERDEN
yakalanmalidir. Bu tuzak this kampanyada **iki kez** vurdu.

## 21.45 HALKA NORMALI EKSEN OLARAK KULLANILAMAZ — conclusive, monotonik

Halka normali simdiye up to only **ISARET** for kullanildi. Fiziksel
sezgi "duz yuzeydeki a deligin axis o yuzeyin normalidir" der; this
iddia simdiye up to NOT MEASURED (sign kollari `unsigned` metrigi tanim
geregi no degistirmez, axis degistirmek ise ikisini de degistirir).

`probe_ring_axis.py`, VAL 99 part (field yolu), paired part bootstrap:

| arm | detection | robot | robot-ISARETLI | robot-ISR farki |
|---|---|---|---|---|
| baseline | 0.6149 | 0.5773 | 0.4847 | — |
| **sign** (mevcut rule) | 0.6149 | 0.5773 | **0.5176** | **+0.0329** |
| tam (axis = halka) | 0.6149 | 0.1161 | 0.0988 | **−0.3860** * |
| kapili_10 | 0.6149 | 0.1224 | 0.1051 | −0.3797 * |
| kapili_20 | 0.6149 | 0.2212 | 0.1976 | −0.2865 * |
| kapili_30 | 0.6149 | 0.3027 | 0.2792 | −0.2045 * |
| kapili_45 | 0.6149 | 0.3482 | 0.3153 | −0.1681 * |
| harman_0.25 | 0.6149 | 0.3341 | 0.2980 | −0.1856 * |
| harman_0.50 | 0.6149 | 0.2165 | 0.1929 | −0.2911 * |
| harman_0.75 | 0.6149 | 0.1475 | 0.1255 | −0.3591 * |

(* = %95 GA sifiri icermiyor; bootstrap orneklerinin **%0**'i pozitif.)

**MONOTONIK**: halka normali karisima ne up to very girerse sonuc o up to
kotu (harman 0.25 → 0.50 → 0.75 with −0.186 → −0.291 → −0.359).
Gurultuye uydurma DEGIL, sistematik.

**MEKANIZMA / LESSON.** Halka normali **~1 bit** tasiyor: *hangi taraf
disari*. Eksenin kendisini tasimiyor -- pah kirmalari, kavisli body ve
girintili agizlar yuzunden mouth cevresi yuzeyin mean normali burgu
ekseniyle hizali not. Kural this 1 biti kullandiginda **+0.0329**,
tamamini kullanmaya kalktiginda **−0.3860**.

**Bu, kolun why ISARET arm as DAGITIMA candidate olup EKSEN arm as
kapali oldugunun kanitidir.** Ayrica 21.32'deki "wrong referans" dersinin
simetrigi: orada correct fikir wrong referansla olculmustu, burada correct
referans wrong IS for kullanildi.

## 21.46 BAGLAYICI KISIT BULUNDU: **YANAL KONUM**, candidate varligi DEGIL, direction DEGIL

Havuz residual `extract`in icinden yakalandigi for (21.44) tolerans taramasi
VALID. VAL 40 part / 238 GT, field yolu, ISARETLI criterion, axial <=40mm.

**YANAL toleransi gevsetince** (aci 10 derecede SABIT):

| lateral | HAVUZ kapsamasi | CIKTI |
|---|---|---|
| **2 mm (gercek criterion)** | **0.4454** | **0.5168** |
| 3 mm | 0.5252 | 0.5504 |
| 5 mm | 0.6597 | 0.6134 |
| 10 mm | **0.8319** | 0.7311 |
| 20 mm | 0.9454 | 0.8235 |

**ACI toleransini gevsetince** (lateral 2mm'de SABIT):

| aci | HAVUZ | CIKTI |
|---|---|---|
| 10 derece | 0.4454 | 0.5168 |
| 20 derece | 0.4538 | 0.5252 |
| 45 derece | 0.4664 | 0.5378 |
| 180 derece (direction TAMAMEN none sayilir) | 0.5924 | 0.6639 |

### OKUMA — this tablo kampanyanin direction tayinidir

* **Aciyi tamamen none saymak** pool yalnizca 0.4454 -> 0.5924 yapiyor
  (+0.147). Yani direction, remaining hatanin **small** kismi.
* **Yanali 2 -> 10 mm yapmak** pool 0.4454 -> **0.8319** yapiyor
  (**+0.387**). GT'lerin **%83'u** for havuzda, direction ZATEN 10 derece
  icinde correct which a candidate **10 mm yakinda duruyor**.

**Yani darbogaz ne candidate URETIMI ne de YON; YANAL KONUM HASSASIYETI.**

Bu, independent as olculmus iki seyle birebir tutarli: konum AUC 0.7053
vs direction AUC 0.8899, ve "lateral error = segmentasyon kalitesi" bulgusu.

### YAPISAL DUZELTME: pool recall'u CEILING DEGILDIR
Yapisal kontrol ("ciktida present, havuzda none") **31** cikti -- sifir olmasi
gerekirken. Sebep bulundu: **POSE HEAD gate'ten SONRA CP'leri OYNATIR**
(`wire_gate.pose_correct`). Yani a candidate, havuzda kabul kutusunun disinda
olup ciktida icine girebilir. Nitekim cikti (0.5168) havuzun (0.4454)
USTUNDE.

Bu, "pool recall = baglayici kisit" seklindeki eski kaydin duzeltmesidir:
**pose head varken pool recall'u a CEILING degildir.** Dogru ceiling,
pose head'in onarabildigi lateral bandda measured_path pool kapsamasidir --
10 mm'de **0.8319**.

### KALAN EN BUYUK KALDIRACIN ADRESI
Pose head bugun 0.4454 -> 0.5168 tasiyor. 10 mm bandindaki ceiling 0.8319.
Aradaki **+0.31**, kampanyada found EN BUYUK acik basliktir ve
segmentasyonda not, **YANAL KONUM REGRESYONUNDA**.

## 21.47 POSE HEAD KIRPMA SINIRI -- taranmamis a hiperparametre bulundu

21.46'nin dogrudan sonucu as pose head'in kodu okundu:

```
mx = float(m.get("maks_mm", 3.0))        # wire_gate.py
c["point"] = p + dw * (min(n, mx) / n)   # correction 3 mm'ye KIRPILIYOR
```

`maks_mm = 3.0` modelin pkl'ine yazili ve **tarandigina dair kayit YOK**
("audit tavsiyesi: 2-3mm" notu present, measurement none).

**Bunun why onemli oldugu.** Teshis, GT'lerin **%83'u** for havuzda,
direction already 10 derece icinde correct which a candidate's **10 mm** yakinda
oldugunu gosterdi. 3 mm'lik kirpma this bandin ucte birini bile kapsamiyor.

**Egitim tarafi bunu DESTEKLIYOR** (dogrulandi, varsayilmadi):
* `q3_pose_data_grow.py`: eslesme toleransi `tt = max(3.0, 0.06*diag)` --
  i.e. model ZATEN 3 mm'den large yer degistirmeler gormus.
* `q1_pose_head.py`: **hedefler KIRPILMIYOR**; kirpma yalnizca CALISMA
  ANINDA uygulaniyor.

Yani model large duzeltmeleri ogrenmis olabilir but uretimde onlari
uygulamasina IZIN VERILMIYOR.

**OLCUM TASARIMI (verimli).** Her `maks_mm` degeri for zinciri yeniden
kosmak yerine, kirpma ONCESI yer degistirme vektoru
(`wire_gate.POSE_KANCA`, cevre degiskeniyle acilir) dokuluyor; so
**tek kosudan** butun tarama cevrimdisi kuruluyor
(`probe_pose_clip.py`). Betik also mx=3.0'da yeniden kurdugu metrigin
dokumun kendi metrigine ESIT oldugunu dogruluyor -- esit degilse taramayi
INVALID ilan ediyor.

## 21.48 Y22 MC DROPOUT CLOSED — kapinin altinda, maliyeti 8 fold

VAL 100 part, paired part bootstrap, MC(8):

| path | metrik | baseline | MC(8) | diff | %95 GA | poz% |
|---|---|---|---|---|---|---|
| measured_path | detection | 0.5553 | 0.5528 | −0.0024 | [−0.0308,+0.0255] | 44.0 |
| measured_path | robot | 0.5376 | 0.5425 | +0.0052 | [−0.0217,+0.0324] | 65.1 |
| measured_path | robot-ISR | 0.4668 | 0.4738 | **+0.0071** | [−0.0180,+0.0319] | 71.1 |

GA sifiri iciyor, kazanc fold gurultusunun (±0.008) altinda, **maliyeti
8 fold inference**. **CLOSED.**

### Yan bulgu: "+0.0000" this kez BUG DEGIL, KAPSAM
Ilk kiyas `field` yolunda yapildi ve **tam +0.0000** verdi -- this gecenin
sessiz no-op imzasi. Sebep arandi ve bulundu: `field` yolu
`robot_cp.extract` icinde KENDI cikarimini yapar; `EZ_MCDROP` yalnizca
sondanin `pbs`'ini etkiler, o da **measured_path** yolu besler. Yani kanca
`field`ya ULASMIYOR -- arm calismiyor not, o yolda YOK.

Duman testi (21.43) onceden kurulmus oldugu for this ayrim iki dakikada
yapildi: kolun gercekten etkili oldugu (0.047 vs 0.000001) already
biliniyordu, therefore "+0.0000" however kapsam sorunu olabilirdi.
**Duman testinin bedelini burada odedi.**

## 21.49 SONDA KANCALARI YALNIZ `measured_path` YOLUNA ULASIR — kapsam kurali

Bu night IKI arm, `field` yolunda **tam +0.0000** verdi (MC dropout ve
remesh varyanti). Ikisi de sessiz no-op DEGILDI; ikisinin de sebebi
ayniydi ve residual rule as yaziliyor:

**`field` yolu = `robot_cp.extract`, ve `extract` KENDI cikarimini ve KENDI
remesh'ini icinde yapar.** Sondanin cevre degiskenleri (`EZ_MCDROP`,
`EZ_REMESH`, `EZ_CKPT`) sondanin urettigi `pbs` ve `V,F`'yi degistirir --
onlar da yalnizca **`measured_path`** yolunu besler.

**RULE: inference/mesh duzeyindeki each arm `measured_path` yolunda olculur.
`field` yolunda +0.0000 gormek, kolun olu oldugu anlamina GELMEZ.**

Bu ayrimi this night iki dakikada yapabilmemizin sebebi, kollarin ONCE
duman testinden gecirilmis olmasidir (21.43): kolun gercekten etkili
oldugu independent as biliniyordu.

## 21.50 Y16 — REMESH HEDEFI 5000 KESIN OLARAK DAHA KOTU

VAL 100 part, **measured_path** yolu, paired part bootstrap. Taban = 6000
(mevcut davranis):

| metrik | 6000 | 5000 | diff | %95 GA | poz% |
|---|---|---|---|---|---|
| detection | 0.5553 | 0.5198 | **−0.0353** | [−0.0600,−0.0079] * | 0.4 |
| robot | 0.5376 | 0.4982 | **−0.0393** | [−0.0632,−0.0141] * | 0.1 |
| robot-ISR | 0.4668 | 0.4371 | **−0.0300** | [−0.0548,−0.0053] * | 0.8 |

(* = GA sifiri icermiyor.) Yani tezden gelen **6000 hedefi iyi secilmis**;
asagi cozunurluk conclusive as zarar veriyor. Topluluk hukmu for 7200
bekleniyor.

## 21.51 GECE 3 (2026-08-13/14) DENETIM DURUMU

Uc urun dosyasi degistirildi; **hepsi cevre degiskeniyle kapili ve
default davranis BIT-AYNI**:

| file | degisiklik | default |
|---|---|---|
| `diffusionnet.py` | `predict(..., mc_dropout=T)` | `0` = eski kod yolu |
| `wire_gate.py` | `CP_POSE_MAKS_MM` ezmesi + `POSE_KANCA` | ezme none, kanca kapali |
| `robot_cp.py` | `HAVUZ_KANCA` (gate oncesi pool) | kanca kapali |

**DOGRULAMA:**
* `smoke_test.py` -> **GECTI** (gercek STEP, 4 kontrol noktasi, 2 CP
  uretildi, 32 s)
* `rollback.py --kontrol` -> **1 deviation: only `cp_config.json`** (bilincli)
* Uc file da git'te izleniyor (`M robot_cp.py`, `M wire_gate.py`,
  `M diffusionnet.py`) -> geri alinabilir
* **D7 OKUNMADI** (2 okuma hakki duruyor) · **LOCKED harcanmadi**

## 21.52 OLCUM DUZELTMESI: yeni sondalarda TESPIT tanimi kanoniklestirildi

Bu night written dort probe (`probe_dump_compare`, `probe_ring_axis`,
`probe_pose_clip`, `probe_remesh_ensemble`) "detection"i **2.0 mm fixed**
toleransla hesapliyordu. Kanonik tanim different:

```
detection : match_hungarian(..., tol=0.0, am=180.0, pct=True)  -> tt = max(3.0, 0.06*diag)
robot  : match_hungarian(..., tol=2.0,  am=10.0, pct=False) -> 2 mm SABIT
```

Yani **robot metrikleri already kanonikti**, detection ise DAHA SIKI
hesaplaniyordu (measured_path yolda 0.6139 yerine dogrusu 0.6101; farkin isareti
ve hukumler degismedi). Kanonik tanima gecildi.

**Duzeltilmis sayilar:**

| kiyas | detection farki (eski -> yeni) | verdict |
|---|---|---|
| remesh 5000 vs 6000 | −0.0353 -> **−0.0380** * | degismedi (KESIN kotu) |
| MC(8) vs baseline | −0.0024 -> **−0.0007** | degismedi (notr) |

**Onemli:** 21.46'daki baglayici-kisit taramasi **yalnizca robot olcutunu**
(2 mm / 10 derece) kullanir; o tanim bastan kanonikti, therefore
**o bulgu this duzeltmeden ETKILENMEZ.**

## 21.53 POSE KIRPMA SINIRI **BAGLAMIYOR** — sinirlayan MODELIN KENDISI

VAL 100 part / 617 CP, measured_path path, tek kosudan cevrimdisi tarama.
Dogrulama gecti (mx=3.0 yeniden kurulan == dokum).

| maks_mm | detection | robot | robot-ISR | 3.0'a per diff |
|---|---|---|---|---|
| 0.0 (pose KAPALI) | 0.4980 | 0.4605 | 0.3837 | **−0.0799** * |
| 1.0 | 0.5889 | 0.5450 | 0.4495 | −0.0141 * |
| 2.0 | 0.6124 | 0.5685 | 0.4636 | +0.0000 |
| **3.0 (MEVCUT)** | 0.6139 | 0.5685 | 0.4636 | — |
| 4 / 5 / 6 / 8 / 10 / 15 / **SINIRSIZ** | 0.6139 | 0.5685 | 0.4636 | **+0.0000** |

**Kirpmayi kaldirmak HICBIR SEY degistirmiyor.** Sebep dogrudan measured:

> Modelin onerdigi yer degistirme (617 CP): median **0.48 mm**,
> %75 0.93 mm, %90 1.64 mm, **maksimum 2.87 mm**.
> **3 mm'yi asan oneri ratio: %0.0**

Guven kapili varyantlar da aynen +0.0000 verdi (kapinin ustunde/altinda
changed a sey none, because hicbir oneri kirpilmiyor).

### VERDICT VE KOK WHY
`maks_mm` **atil a hiperparametre**; sinirlayan sey **modelin kendisi**.
Pose head large duzeltmeler ONERMIYOR.

Mekanizma a **SECIM YANLILIGI**: training verisi (`q3_pose_data_grow.py`)
yalnizca `tt = max(3.0, 0.06*diag)` icinde **ZATEN ESLESMIS** adaylardan
kuruluyor. Eslesmis a candidate's artigi tanim geregi KUCUKTUR. Yani model,
large artiklari **no gormedi** -- ogrenip de uygulayamadigi not,
**ogrenmedigi** for onermiyor.

### AMA POSE HEAD CALISIYOR — ve tam kapasite kullaniliyor
Kolu kapatmak (mx=0) robot-ISARETLI'yi **−0.0799** dusuruyor
(GA [−0.1079,−0.0557], orneklerin %0'i pozitif). Yani bilesen gercek ve
tasidigi each seyi already tasiyor.

### SIRADAKI ADIM (net ve dar)
21.46'nin +0.31'lik acigini almanin yolu kirpmayi gevsetmek DEGIL,
**pose head'i large artiklari GOREREK yeniden egitmek**: training eslesme
toleransi (bugun ~3-6 mm) 15 mm'ye acilir, so 10 mm bandindaki
candidates da hedefe girer. Veri already diskte ve **ag cikarimi
GEREKTIRMIYOR** (q3 onbellekli npz'lerden calisiyor).

## 21.54 POSE HEAD YENIDEN EGITIMI — iki mekanizma ayristi, IKI OLCUM KUSURU YAKALANDI

21.53 "model large correction onermiyor" dedi. Iki candidate mekanizma vardi;
ikisi de measured.

**Veri before incelendi (varsayilmadi):**

| data | satir | lateral hedef median | %90 | maks | **>3mm ratio** |
|---|---|---|---|---|---|
| baseline (`pose_veri`) | 6330 | 0.97 mm | 3.02 | 12.10 | **%10.2** |
| genis (`Q3_TOL=15`) | 7154 | 1.12 mm | 4.60 | 14.93 | **%18.4** |

**Bu, ilk hipotezi KISMEN CURUTTU.** Taban veride already %10.2 large hedef
vardi; i.e. model large artiklari "no gormemis" not. Demek ki asil
mekanizma **REGRESYON BUZULMESI**: orman yaprak ortalamasi ucdegerleri
iceri ceker. Secim yanliligi ikincil.

### YAKALANAN IKI OLCUM KUSURU (ikisi de ilk kosuda vurdu)

**1. Sizinti kapisi SESSIZCE hicbir seyi elemedi.** `split3.json` icinde
`val` a SOZLUK (`{"n":…, "parts":[…]}`); uzerinde dogrudan donmek
`'n'`,`'parts'` anahtarlarini verir. Kapi "0 VAL grubu cikarildi" dedi ve
this a WARNING as gecti. Artik `["val"]["parts"]` okunuyor ve **iki
`assert`** present (list bos olamaz, gruplar bos olamaz). Bu gate olmasaydi
VAL olcumu KIRLI olurdu.

**2. Iki candidate KENDI data kumesinde puanlandi -- kiyaslanamaz.** Genis data
"more kotu" gorunuyordu (kutuda %72.7 vs %81.3), oysa genis cluster DAHA ZOR
satirlar iceriyor: iki ratio same satirlarda olculmemis. Artik butun
candidates **ORTAK degerlendirme kumesinde** puanlaniyor; training kumesinden
degerlendirme katinin gruplari cikarilarak (grup-disi, sizintisiz).

Bu iki kusur da "arm pozitif/negatif" hukmunu ters cevirebilecek
cinstendi ve **sonuc yazilmadan before** yakalandi.

## 21.55 POSE HEAD YENIDEN EGITIMI — ORTAK KUMEDE RESULT

Sizinti kapisi onarildiktan (87 VAL geometri grubu CIKARILDI) ve butun
candidates **same 5644 satirda** puanlandiktan after:

| training verisi | setting | kutuda (<=2mm) | residual median | oneri maks | >3mm oneri |
|---|---|---|---|---|---|
| baseline | yaprak>=5 (**MEVCUT**) | 76.5% -> 80.8% | 0.68 mm | 5.45 | %0.2 |
| baseline | yaprak>=2 | 76.5% -> 81.6% | 0.68 mm | 6.11 | %0.4 |
| **baseline** | **yaprak>=1** | 76.5% -> **81.7%** | 0.67 mm | 6.33 | %0.5 |
| genis (tol15) | yaprak>=5 | 76.5% -> 80.8% | 0.74 mm | 5.96 | %0.6 |
| genis (tol15) | yaprak>=2 | 76.5% -> 80.8% | 0.74 mm | 6.29 | %0.8 |
| genis (tol15) | yaprak>=1 | 76.5% -> 81.1% | 0.74 mm | 6.49 | %1.0 |

### IKI TEMIZ VERDICT

**1. GENIS VERI YARDIM ETMIYOR.** Ortak kumede baseline veriyle egitilen each
setting, genis veriyle egitilenden esit ya da iyi. Yani "secim yanliligi"
hipotezi **CURUDU**; remaining mekanizma **regresyon buzulmesi**.

**2. BUZULMEYI GEVSETMEK COK KUCUK BIR SEY KAZANDIRIYOR.**
`min_samples_leaf` 5 -> 1: kutuda-ratio +0.9 puan (80.8 -> 81.7),
en large oneri 5.45 -> 6.33 mm. Ama **3 mm'yi asan oneri ratio hala
yalnizca %0.5.**

### BUNUN 21.46'DAKI +0.31 ICIN ANLAMI (durust okuma)
Model sinifi ve oznitelikler **10 mm'lik hatalari onaracak bilgiyi
tasimiyor**: en iyi varyant bile adaylarin %99.5'ine 3 mm'den small
correction oneriyor. Yani 21.46'daki acik, **gate ozniteliklerinden
son-islem regresyonuyla ALINAMAZ**; ya candidate URETIMI (segmentasyon /
candidate konumlari) duzelecek, ya da pose kafasina **yerel geometriyi
dogrudan goren** yeni oznitelikler verilecek.

Bu, acigin present olmadigi anlamina gelmez -- **nereden alinamayacagini**
olcerek daraltir.

Aday model yine de tam zincirde olculuyor
(`CP_POSE_MODEL=results/pose_head_yeni.pkl`); beklenti kapinin altinda.

## 21.56 Y16 REMESH-VARYANT TOPLULUGU CLOSED

VAL 100 part, measured_path path, paired part bootstrap. Taban = 6000 (mevcut).
**Egitim none** -- i.e. this arm seed gurultusune tabi not, tek kosuda
verdict giyebilir.

| varyant | detection | robot | robot-ISR | rbi farki | %95 GA | poz% |
|---|---|---|---|---|---|---|
| **BASELINE (6000)** | 0.5553 | 0.5376 | 0.4668 | — | — | — |
| tek 5000 | 0.5198 | 0.4982 | 0.4371 | −0.0300 | [−0.0548,−0.0053] * | 0.8 |
| tek 7200 | 0.5573 | 0.5453 | 0.4735 | +0.0066 | [−0.0208,+0.0314] | 69.3 |
| **birlesim (oy>=1)** | 0.5667 | 0.5446 | **0.4783** | **+0.0112** | [−0.0073,+0.0314] | 87.9 |
| oylama (oy>=2) | 0.5558 | 0.5363 | 0.4655 | −0.0012 | [−0.0163,+0.0144] | 43.4 |
| oybirligi (oy>=3) | 0.4995 | 0.4796 | 0.4219 | −0.0453 | [−0.0801,−0.0133] * | 0.2 |

### VERDICT: **CLOSED**
* 5000 ve oybirligi **KESIN kotu**.
* 7200 ve birlesim pozitif but **GA sifiri iciyor**; birlesim +0.0112 with
  this gecenin diger "kapiya yakin" kollariyla same profilde -- ve
  **maliyeti 3 fold inference**.
* Oylama tam notr.

**Egilim anlamli:** 5000 < 6000 < 7200 (monotonik), i.e. more high
cozunurluk yardim ediyor. Ama 6000 -> 7200 kazanci kapinin altinda.
Tezden gelen **6000 hedefi iyi secilmis**; asagi inmek conclusive zarar.

## 21.57 Y24 CRF — YANLIS KOVAYA KONMUSTU, DUZELTILDI

Y24 (CRF duzeltmesi) 21.42'de "training gerektiren, sure yetmeyen" kollar
arasina konmustu. **Bu yanlisti:** CRF a SON-ISLEMDIR, training
gerektirmez. Kova duzeltildi ve arm this night kosuldu.

**Kurulum.** Mesh kenarlari uzerinde mean-alan yaklasimi: each turda
each tepenin probability vektoru komsularinin ortalamasiyla `w` agirliginda
harmanlanir, after normalize edilir. **Ek parametre ogrenilmiyor.**
`EZ_CRF="tur,agirlik"`; bos birakilirsa bit-same baseline.

**DUMAN TESTI (sentetik ikosahedron) — GECTI:**

| kontrol | sonuc |
|---|---|
| `tur=0` bit-same mi | **EVET** |
| `tur=3` etkisi (ort. abs diff) | 0.077408 |
| probability satirlari 1'e toplaniyor mu | EVET |
| tekil aykiri etiketli tepe duzeliyor mu | **EVET** |

**Neden this arm darbogaza nisan aliyor:** 21.46 baglayici kisiti YANAL
KONUM as belirledi ve lateral error independent as segmentasyon
kalitesine baglanmisti. CRF, tekil wrong etiketli tepeleri bastirarak
candidate MERKEZLERINI oynatir -- i.e. dogrudan lateral hataya dokunur.

## 21.58 ADAY POSE HEAD UCTAN UCA **KESIN OLARAK KOTU** — vekil metrik sign degistirdi

`results/pose_head_yeni.pkl` (baseline data, yaprak>=1, `maks_mm=10`) tam
zincirde measured. VAL 100 part, **field** yolu (pose head `extract`in
icinde oldugu for dagitilan path budur), paired part bootstrap:

| metrik | dagitilan | candidate | diff | %95 GA | poz% |
|---|---|---|---|---|---|
| detection | 0.7878 | 0.7847 | −0.0031 | [−0.0080,+0.0000] | 0.0 |
| robot | 0.5764 | 0.5059 | **−0.0697** | [−0.0993,−0.0416] * | 0.0 |
| robot-ISR | 0.4839 | 0.4401 | **−0.0437** | [−0.0670,−0.0228] * | 0.0 |

**KESIN KOTU. Aday REDDEDILDI, dagitilan model yerinde kaliyor.**

### BU GECENIN EN SERT DERSI: VEKIL METRIK ISARET DEGISTIRDI
Aday, cevrimdisi vekilde **KAZANIYORDU**: ortak degerlendirme kumesinde
kabul kutusuna girme ratio %80.8 -> **%81.7** (+0.9 puan), grup-disi,
leakage kapisi kapali. Uctan uca ise **−0.0437**.

Yani "lateral artigi more iyi prediction etmek" with "robot metrigini
yukseltmek" AYNI SEY DEGIL. Muhtemel mekanizma: vekil TUM adaylarin
mean artigini measures; uctan uca metrik ise ZATEN kutuda which candidate's
disari itilmesini **cift** cezalandirir (a TP gider, a FP gelir).
Yaprak>=1 more oynak duzeltmeler uretiyor ve `maks_mm=10` bunlarin
buyuklerinin gecmesine izin veriyor.

**Kural: son-islem kollarinda cevrimdisi vekil DECISION VERDIRMEZ; yalnizca
uctan uca measurement verdirir.** Vekil olsa olsa hangi adaylarin uctan uca
olcumu HAK ETTIGINI secer.

### AYRISTIRMA KOSULUYOR
Kaybin sebebi MODEL mi (yaprak>=1) otherwise KIRPMA mi (3 -> 10 mm)?
Ayni candidate model `CP_POSE_MAKS_MM=3.0` with tekrar olculuyor. Eski model
2.87 mm'yi no asmadigi for (21.53) kirpmanin however yeni modelde
baglayici hale geldigi biliniyor -- i.e. suphe before kirpmada.

## 21.59 GECE 3 KAPANIS TABLOSU (2026-08-13/14)

Bu night **YEDI arm** measured ve closed; **a arm** dagitima candidate kaldi
(halka isareti, onceki geceden); **a large teshis** cikti.

| arm | measurement | verdict |
|---|---|---|
| Y5 aux-wire | −0.0072 (3 matched seed) | curudu |
| Y13 sinir kaybi | −0.0113 / −0.0044 | closed |
| Y22 MC dropout | +0.0071 (GA sifiri iciyor), 8x maliyet | closed |
| Y16 remesh 5000 | −0.0300 (GA sifirsiz) | **conclusive kotu** |
| Y16 remesh 7200 | +0.0066 (GA sifiri iciyor) | gate alti |
| Y16 ensemble birlesim | +0.0112 (GA sifiri iciyor), 3x maliyet | gate alti |
| Y16 ensemble oybirligi | −0.0453 (GA sifirsiz) | **conclusive kotu** |
| halka normali = EKSEN | −0.3860, monotonik, %0 poz | **conclusive closed** |
| pose kirpma gevsetme | +0.0000 (kirpma ATIL) | closed |
| candidate pose head | −0.0437 (GA sifirsiz, %0 poz) | **REDDEDILDI** |

**Teshis (arm not):** baglayici kisit **YANAL KONUM** (21.46) --
kampanyanin direction tayini.

### GECENIN DORT METODOLOJIK CIKTISI
1. **Segmentasyon seed gurultusu 0.046** measured; this buyuklugun altindaki
   hicbir seg arm tek kosuyla verdict giymez (21.41).
2. **Sonda kancalari only `measured_path` yoluna ulasir**; `field`da +0.0000
   gormek kolun olu oldugu anlamina gelmez (21.49).
3. **Ara degerler uretildikleri yerden yakalanmali**; same kodu disaridan
   cagirmak cache bagimliligi varsa different sonuc verir (21.44).
4. **Cevrimdisi vekil karar verdirmez**; a arm vekilde +0.9 puan
   kazanip uctan uca −0.0437 verdi (21.58).

## 21.60 BAGLAYICI KISIT — TAM VAL (100 part / 660 GT) with YENIDEN MEASURED

21.46 ilk kez 40 parts olculmustu. `_dump_pose.json` kosusunda pool
kancasi 100 parcanin hepsinde acikti; tarama tam kumede tekrarlandi.
**Bunlar kayda passing sayilardir.**

**YANAL taramasi** (aci 10 derece SABIT, ISARETLI, axial <=40mm):

| lateral | HAVUZ | CIKTI |
|---|---|---|
| **2 mm (gercek criterion)** | **0.4242** | **0.4682** |
| 3 mm | 0.4955 | 0.5106 |
| 5 mm | 0.6242 | 0.5727 |
| 8 mm | 0.7303 | 0.6364 |
| **10 mm** | **0.7818** | 0.6697 |
| 15 mm | 0.8455 | 0.7076 |
| 20 mm | 0.9000 | 0.7591 |

**ACI taramasi** (lateral 2 mm SABIT):

| aci | HAVUZ | CIKTI |
|---|---|---|
| 10 derece | 0.4242 | 0.4682 |
| 20 derece | 0.4364 | 0.4742 |
| 45 derece | 0.4409 | 0.4803 |
| 90 derece | 0.4970 | 0.4985 |
| **180 (direction TAMAMEN none sayilir)** | **0.5939** | 0.6273 |

### VERDICT (40 parcalik ilk olcumle AYNI, buyuklukler biraz more ilimli)

| gevsetme | pool kazanci |
|---|---|
| direction TAMAMEN mukemmel yapmak | **+0.1697** |
| yanali 2 -> 10 mm yapmak | **+0.3576** |

**Yanal, yonun iki katindan fazlasini tasiyor.** Darboğaz **YANAL KONUM**.
Sunumda 100 parcalik this sayilar kullanilir (40 parcalik ilk measurement
0.8319 diyordu; dogrusu **0.7818**).

## 21.61 Y24 CRF — ILK AYAR TAM NOTR

VAL 100 part, measured_path path (kanca only oraya ulasir, bkz. 21.49),
paired part bootstrap. Ayar: **2 tur, agirlik 0.3**.

| metrik | baseline | CRF | diff | %95 GA | poz% |
|---|---|---|---|---|---|
| detection | 0.6101 | 0.6043 | −0.0054 | [−0.0247,+0.0157] | 29.0 |
| robot | 0.5376 | 0.5330 | −0.0043 | [−0.0242,+0.0174] | 32.6 |
| robot-ISR | 0.4668 | 0.4670 | **+0.0004** | [−0.0199,+0.0218] | **50.0** |

**Tam sansa esit** (%50.0 pozitif, GA simetrik). Duman testi kolun
gercekten calistigini gosterdigi for this a no-op not, gercek a
**notr** sonuc.

**KAPATILMADAN ONCE IKINCI AYAR KOSULUYOR** (`docs/KAPANAN_KOLLAR_DENETIMI.md`
kurali: "sondanin cozunurlugu toleransi tutuyor mu" -- 2 tur/0.3 HAFIF a
duzeltmedir). Ikinci setting: **4 tur, agirlik 0.5**.

## 21.62 AYRISTIRMA SONUCU: loss **MODELDEN**, kirpmadan DEGIL

Ayni candidate model, yalnizca kirpma degistirilerek tekrar measured:

| kosu | detection | robot | robot-ISR |
|---|---|---|---|
| candidate model, `maks_mm=10` | 0.7847 | 0.5059 | 0.4401 |
| candidate model, `maks_mm=3` | 0.7847 | 0.5059 | 0.4401 |

**BIREBIR AYNI.** Kirpma yeni modelde de baglamiyor -> loss tamamen
**modelin kendisinden**.

### CALISMA ANINDA ONERI BUYUKLUKLERI (617 CP)

| model | median | %90 | maks | >3mm |
|---|---|---|---|---|
| dagitilan | 0.484 mm | 1.636 | 2.868 | **%0.00** |
| **candidate (yaprak>=1)** | **0.427 mm** | 1.166 | **2.537** | **%0.00** |

Aday model calisma aninda **DAHA KUCUK** duzeltmeler oneriyor -- oysa
cevrimdisi OOF'ta en large onerisi 6.33 mm idi.

### UCUNCU KATMAN: CEVRIMDISI POPULASYON != CALISMA ANI POPULASYONU
OOF degerlendirmesi **pose training satirlarinda** yapiliyor; calisma
anindaki oznitelikler ise `wire_gate.feats_for`in **gate'ten gecmis**
candidates for urettikleri. Bunlar different populasyonlar. Sonuc: modelin
"large correction onerebilme" ozelligi bile calisma anina TASINMADI.

Yani vekil sadece METRIGI not, modelin DAVRANISINI da wrong prediction
etti. **Kol closed; dagitilan pose head yerinde kaliyor.**

**Bilanco:** pose cephesinde ucu de measured ve ucu de closed --
kirpmayi gevsetmek (+0.0000), genis veriyle egitmek (vekilde bile
yardim etmedi), buzulmeyi gevsetmek (uctan uca −0.0437). 21.60'taki
+0.31'lik acik **son-islem regresyonuyla alinamaz**; this residual uc independent
olcume dayaniyor.

## 21.63 Y24 CRF — TEK AYARLA KAPATILMADIGI ICIN KOL DIRILDI

Ilk setting (2 tur / 0.3) tam notrdu (%50.0 pozitif) ve normalde "olu"
denirdi. `KAPANAN_KOLLAR_DENETIMI` kurali geregi ikinci, DAHA GUCLU setting
kosuldu:

| setting | detection | robot | robot-ISR | rbi farki | %95 GA | poz% |
|---|---|---|---|---|---|---|
| baseline | 0.6101 | 0.5376 | 0.4668 | — | — | — |
| 2 tur / 0.3 | 0.6043 | 0.5330 | 0.4670 | +0.0004 | [−0.0199,+0.0218] | 50.0 |
| **4 tur / 0.5** | 0.6043 | 0.5432 | **0.4766** | **+0.0100** | [−0.0128,+0.0335] | **80.5** |

**MONOTONIK VE ANLAMLI YONDE**: correction guclendikce robot metrigi
yukseliyor (+0.0004 -> +0.0100), pozitif ornek ratio 50.0 -> 80.5.
Tespit each iki ayarda da hafif negatif (−0.005) -- i.e. correction
kesinlikten biraz verip robot-uygunlugundan aliyor.

**Bu, gecenin "yelpaze 64 yonde olu, 256 yonde +0.0676" dersinin
tekrarıdır:** ilk probe cozunurlugu yetmiyordu. Kol tek ayarla
kapatilsaydi this egilim gorulmeyecekti.

**UCUNCU AYAR KOSULUYOR (6 tur / 0.7)** -- egilimin devam edip etmedigini,
otherwise tepe noktasinin gecilip gecilmedigini belirlemek for. Henuz
hicbir setting GA kapisini gecmedi; arm DAGITIMA ADAY DEGIL, OPEN.

## 21.64 CRF URUN YOLU KANCASI — kuruldu ve DOGRULANDI

Sonda kancalari only `measured_path` yoluna ulasiyor (21.49). CRF kazanci
oradan olculdugu for, kazanan ayarin **DAGITILACAK** yolda da
olculebilmesi sart. Sebep hafizada: kanonik zincir blogu measurement betiginde
**+0.0151**, uretim egiticisinde **−0.0138** vermisti -- path farki DECISION
DEGISTIRIYOR.

`robot_cp.extract` icine `CP_CRF="tur,agirlik"` kancasi eklendi
(default bos = bit-same mevcut davranis).

**DOGRULAMA:**

| kontrol | sonuc |
|---|---|
| `smoke_test.py` (kanca KAPALI) | **GECTI**, 2 CP |
| `smoke_test.py` (kanca OPEN, 6/0.7) | **GECTI**, same 2 CP |
| kanca kod yolu: env okundu / import / etki | 0.08829 ort. abs diff, tepelerin **%67.6**'sinin etiketi degisiyor |

Duman testinde ciktinin AYNI cikmasi **no-op not**: o part basit ve
iki CP'si duzeltmeye dayanikli. Kanca kod yolu also izole as
kosuldu ve etkisi measured -- this night "tam +0.0000" tuzagina dusmemek
for residual standart adim.

## 21.65 UZLASTIRMA: "%94'unun yakininda no prediction none" with 21.60 CELISMIYOR

Bolum 20 (error otopsisi) kacan GT'lerin **%94.2**'sinin yakininda HIC
prediction olmadigini soyluyor. 21.60 ise havuzda GT'lerin **%78'i** for
10 mm yakinda correct yonlu a candidate oldugunu soyluyor. Ilk bakista
celiski gibi duruyor; not -- **iki different cluster olculuyor**:

| measurement | cluster | anlami |
|---|---|---|
| Bolum 20: %94.2 "bos" | **CIKTI** (gate + secim SONRASI) | urunun verdigi CP'ler |
| Bolum 21.60: %78 kapsama | **HAVUZ** (gate ONCESI) | uretilen tum candidates |

Yani: **candidate havuza GIRIYOR, but cikisa ULASMIYOR ya da ulastiginda
yeterince yakin not.** Ikisi birlikte okununca tablo netlesir:

* Havuz 2 mm'de 0.4242 -> candidate **uretimi** de tam not.
* Havuz 10 mm'de 0.7818 -> but candidate **cogunlukla ORADA**, sadece
  2 mm'lik kutuya girecek hassasiyette not.
* Cikti 10 mm'de 0.6697 -> havuzdaki this adaylarin a kismi also
  cikisa da ulasamiyor.

**Bolum 20'nin "ADAY URETIMI + KONUM SKORLAMA sorunu" hukmu AYAKTA;
21.60 onu more keskin hale getiriyor:** iki bilesenden agir basani
**KONUM HASSASIYETI**. Bolum 20'nin son-islem tavani hesabi (%12) da
ayakta -- nitekim this night uc son-islem arm (halka-axis, CRF ilk setting,
pose yeniden egitimi) this tavani asamadi.

## 21.66 ADAY TURETME ESIGI ACILMADI — kayitli tuzak

Baglayici kisit YANAL KONUM oldugu for akla gelen ucuz arm,
`prediction_postproc` icindeki candidate turetme parametrelerini (ozellikle
`vertex_confidence_mask`) taramaktir: this threshold hangi tepelerin adayi
olusturdugunu belirler, i.e. KONUMU dogrudan oynatir.

**ACILMADI.** Sebep `cp_config.json`'un kendi notunda yazili:

> `vc_035_REVERTED_2026_07_25`: vc0.35 REDDEDILDI -- tarama iki ureticide
> de +0.01 onerdi but TAM dagitim OOF'unda WEI 0.641 -> 0.629 (DUSTU).
> "Bir taramanin tam dagitim kosusuna per yaniltmasi IKINCI kez"
> (digeri k_eig128).

Yani this parametre for **tarama with tam dagitim arasindaki diff more before
IKI KEZ karar degistirmis**. Bu night same ders ucuncu kez yasandi
(pose head vekili, 21.58). Sinirli surede taramaya girmek, this up to
kaydedilmis uyariya ragmen, measurement not kumar olurdu.

**Not:** parametreler already 2026-08-06'da yeniden ayarlanmis
(`p1_2026_08_06`: cluster/dedupe/oy 3/10/5 -> 1/2/2, very-CP +0.0331).
Yani this cephe bayat not.

## 21.67 Y24 CRF **CLOSED** — tepe yapip donuyor, i.e. gurultuye uydurma

Tarama tamamlandi (VAL 100 part, measured_path path, paired part bootstrap):

| setting | detection | robot | robot-ISR | rbi farki | %95 GA | poz% |
|---|---|---|---|---|---|---|
| baseline | 0.6101 | 0.5376 | 0.4668 | — | — | — |
| 2 tur / 0.3 | 0.6043 | 0.5330 | 0.4670 | +0.0004 | [−0.0199,+0.0218] | 50.0 |
| **4 tur / 0.5** | 0.6043 | 0.5432 | **0.4766** | **+0.0100** | [−0.0128,+0.0335] | **80.5** |
| 6 tur / 0.7 | 0.6011 | 0.5390 | 0.4645 | −0.0020 | [−0.0249,+0.0222] | 43.0 |

**VERDICT: CLOSED.**

**Gerekce -- MONOTONIK DEGIL.** Egri +0.0004 -> +0.0100 -> −0.0020, i.e.
tepe yapip donuyor. Bu kampanyada same imza more before halka-sign threshold
taramasinda gorulmustu (95/105/115/125/140/160 derece: +0.0345/+0.0298/
+0.0298/+0.0063/−0.0063/−0.0047) ve **"monotonik not, i.e. gurultuye
uydurma"** denip sade rule tercih edilmisti. Ayni criterion burada da
uygulaniyor.

Ustelik tepe degeri (+0.0100) **GA'si sifiri iceren** a sayidir ve this
gecenin diger "kapiya yakin" kollariyla same banttadir (Y16 birlesim
+0.0112, Y8 etiket kalitesi +0.0082, Y22 MC dropout +0.0071).
**Uc setting arasindan en iyisini secmek, kapiyi gecmeyen a sayiyi
tarama with "gecirmek" olurdu.**

**Tespit each ayarda negatif** (−0.005 … −0.009): correction kesinlikten
veriyor, robot-uygunluguna guvenilir a sey katmiyor.

### YAN KAZANIM: KOL YINE DE TEK AYARLA KAPATILMADI
Ilk setting (2/0.3) tam notrdu. `KAPANAN_KOLLAR_DENETIMI` kurali geregi iki
setting more kosuldu; egrinin sekli however boyle gorulebildi. Kol a ayarla
"olu" denip kapatilsaydi verdict same olurdu but **rationale wrong** olurdu
("etkisiz" yerine "gurultuye uydurma"). Kural ise yaradi.

**`CP_CRF` kancasi kodda KALIYOR** (default kapali, bit-same davranis):
ileride segmentasyon kalitesi degisirse arm yeniden olculebilir.

## 21.68 TOPLULUK GENISLETME KOLU (training YOK) — kurulum

Gecenin olculmus baskın etkisi **segmentasyon seed gurultusu 0.046**
(21.41). Topluluk, this gurultuyu ortalamayla azaltmanin dogrudan yoludur
ve **yeni training GEREKTIRMEZ** -- gereken checkpointler diskte.

Urun bugun **4** checkpoint kullaniyor:
`recall_hard_s2` + `recall_hard_keig96_s0/s1/s2`.

Diskte kullanilmayan iki grup present:

| candidate | rationale |
|---|---|
| **A: 5 uye** = urun 4 + `recall_hard_keig96_s3` | AYNI recetenin 4. seed; urun toplulugunda YOK. Saf varyans azaltma, recete degisikligi SIFIR |
| **B: 8 uye** = A + `y1b_aug0.15_s0/s1/s2` | augmentasyon checkpointleri TEK TEK more guclu (val Conn_IoU 0.6528 / 0.6990 / 0.6673) |

Ikisi de VAL 100 parts, measured_path yolda, paired part bootstrap with
olculuyor. **Bu arm seed gurultusune tabi not** (yeni training none),
therefore tek kosuda verdict giyebilir.

## 21.69 ARA TEST HAZIRLIGI — 5 unseen part, TANIDIK brand

Amac number not **fonksiyonel dogrulama**: urun sahada ne yapiyor.
`probe_ara_test_glb.py`, `robot_cp.extract` (i.e. GLB ihracatcilarinin
BUGUN cagirdigi path) uzerinden part part rapor uretir.

Secim (`_ara_test_parcalar.txt`) **zorluk yelpazesini kasten kapsiyor**:

| part | GT CP | konum |
|---|---|---|
| 3061994 | **24** | en dense -- sistemin zayif halkasi |
| 1020700000 | 20 | dense |
| 1208920000 | 8 | very-CP regime esigi (n_gt>=8) |
| 1058680000 | 4 | tipik |
| 3025176 | 1 | tek girisli |

Hepsi VAL'den: **brand egitimde VAR, this PARCALAR none.** Kolay part secip
test sismesin diye en dense part bilerek dahil edildi.

**DURUSTLUK NOTU (betigin ciktisina da yazildi):** 5 part small a
ornektir, this sayilar MANSET DEGILDIR. Manset VAL 100 parcadir
(detection 0.7878 / robot-ISARETLI 0.4839).

## 21.70 TOPLULUK GENISLETME A (5 uye) — NOTR, CLOSED

VAL 100 part, measured_path path, paired part bootstrap. Urun 4 uye +
`recall_hard_keig96_s3` (AYNI recetenin 4. seed):

| metrik | baseline (4) | 5 uye | diff | %95 GA | poz% |
|---|---|---|---|---|---|
| detection | 0.6101 | 0.6100 | −0.0000 | [−0.0191,+0.0201] | 49.1 |
| robot | 0.5376 | 0.5349 | −0.0024 | [−0.0239,+0.0211] | 40.9 |
| robot-ISR | 0.4668 | 0.4651 | −0.0018 | [−0.0221,+0.0173] | 42.9 |

**Tam notr. CLOSED.** Ayni receteden a uye more eklemek hicbir sey
katmiyor -- 4 uyeli ensemble o recetenin varyansini ZATEN doyurmus
(uyeler high korelasyonlu: only seed different).

Bu, TOPLULUK B'nin (different recete = gercek cesitlilik) why ayri a
soru oldugunu da gosterir.

## 21.71 YOGUN PARCA YOLU IKI KEZ BAYATLAMIS — ONARILDI

Gozle yapilan ara test dense parts agir kacirma gosterdi (GT=22 -> 11
CP). Kodda this is for **ayri a urun yolu** present: `robot_cp.extract_highcp`
(receipt `results/product_f1_receipt.json`, OOF F1 0.807). Ama
`export_robot_glb.py:155` **kosulsuz** `robot_cp.extract` cagiriyor, i.e.
path no kullanilmiyordu. Sebep arandiginda yolun **bugun no
kosamadigi** ortaya cikti -- iki ayri bayatlama:

**(1) Kapi donusumu atlaniyordu.** `highcp_selector.apply` icinde
`wm["clf"].predict_proba(X13)` cagriliyordu; dagitilan gate ise
PARCA-ICI Z-SKOR donusumunden after **116** sutun bekliyor (58'in tam iki
fold -- donusum each ozniteligin yanina part-ici z-skorunu ekler).
`ValueError: X has 58 features, but expecting 116`.
**Onarim:** `wire_gate.decision_score(wm, X13)` -- dosyanin kendi "tek
source" kurali already buydu, burasi ona uymuyordu.

**(2) Secici, o gunku oznitelik genisligiyle egitilmis.** Secici
2026-07-26'da egitildi; O GUN `feats_for` **13** sutun donduruyordu
(13 + 8 lattice + 4 rank = **25**). Bugun 58 donduruyor -> augment **70**
uretiyor. `ValueError: X has 70 features, but expecting 25`.
**Onarim:** seciciye giden matris ilk `n-12` sutuna kirpilir.
**DOGRULANDI, VARSAYILMADI:** `FEAT_NAMES = FEAT_NAMES_13 + ...` i.e. yeni
oznitelikler SONA eklenmis; secicinin sakladigi 25 ismin ilk 13'u
`FEAT_NAMES_13` with BIREBIR same (kod isim isim karsilastirip
uymazsa HATA veriyor -- sessiz kirpma none).

**LESSON:** "makbuzu present" demek "bugun kosuyor" demek DEGIL. Cagrilmayan a
path sessizce curur; iki independent degisiklik (gate donusumu, oznitelik
genisligi) this yolu kullanilamaz hale getirmis ve kimse diff etmemis
because kimse cagirmiyormus.

## 21.72 YOGUN YOL DOGRU ADAYI BULUYOR, DOGRU YONU KOYMUYORDU

Yol onarilip kosunca ilk part (3061994, GT=24) sunu verdi:

| path | uretilen CP | robot-ISARETLI |
|---|---|---|
| BASELINE (`extract`) | 20 | **11** |
| YOGUN (`extract_highcp`) | **24** (count TAM) | **1** |

**Adedi tam tutturuyor but signed dogruluk cokuyor.** Sebep kodda:
`extract_highcp`, secimden after dogrudan `_format_cps` diyip bitiyordu;
`extract`in **gate sonrasi zinciri** (pose head, aci duzeltici, uye direction
selector, ayrik direction selector) ORADA HIC KOSMUYORDU.

Bu same zamanda makbuzu da yerine oturtuyor: **0.807 TESPIT F1'idir**,
robot-signed not.

**FIX:** zincir `_kapi_sonrasi_zincir()` diye ortak fonksiyona cikarildi
ve HER IKI path da onu cagiriyor (kod tabaninin kendi "tek source" kurali).
`extract`in davranisi DEGISMEDI -- duman testi birebir same iki CP'yi same
koordinatlarda uretti.

## 21.73 METADATA-SIZ ADET TAHMINI (metadata bilgisi olmadan)

`extract_highcp` `cp_count` ister (ureticinin CP sayisi) = METADATA.
Kod okundu: `cp_count` yalnizca IKI yerde kullaniliyor -- `rank_feats`
icindeki `nratio` oznitelig i ve son `[:N]` kirpmasi. Havuz uretimi ve
lattice ozellikleri N'den BAGIMSIZ. Yani metadata bagimliligini kaldirmak
for tek gereken a **N tahmini**.

`count_estimate.py`: CP'ler duzenli a IZGARADA durur; iki ana eksende
(SVD) izgara adimi (pitch) ve spread olculur, site sayisi cikarilir.
Tamamen geometrik.

**ILK GATE (inference GEREKTIRMEZ): GT noktalarindan adedi geri bulabiliyor mu?**

| | tum parts (6074) | **YOGUN (GT>=11, 606 part)** |
|---|---|---|
| tam isabet | %68.5 | **%56.8** |
| median mutlak error | 0.0 CP | **0.0 CP** |

Kapi GECTI -> izgara gercek ve sayilabilir. `cp_count=None` verilirse
`extract_highcp` residual adedi havuzun kendi geometrisinden prediction ediyor
(gurultuye karsi only ust %60 skorlu candidates kullanilir).

## 21.74 YOGUN YOL BAGLANMADI — receipt TEMMUZ'a ait, bugune TASINMIYOR

Iki bayatlama onarilip path kosar hale gelince (21.71) ve gate sonrasi
zincir baglaninca (21.72) tam measurement yapildi. **5 dense part, GT with:**

| path | detection | robot (unsigned) | robot-ISARETLI |
|---|---|---|---|
| **BASELINE (`extract`)** | **0.7907** | **0.4651** | **0.4186** |
| YOGUN (cp_count=GT) | 0.3402 | 0.1546 | 0.1134 |
| YOGUN (cp_count−2) | 0.3441 | 0.1505 | 0.1075 |

### ONCEKI CIKARIM CURUDU
Ilk parcadan after "dense path correct ADAYLARI buluyor, baseline path correct
YONLERI koyuyor" denmisti. **Yanlis.** Yogun path TESPITTE de yari yariya
kotu (0.3402 vs 0.7907): adedi tutturuyor but noktalari YANLIS YERLERE
koyuyor. Tek parcadan (ISARETLI 11 vs 1) cikarilan mekanizma, bes parcalik
tam tabloyla curudu.

### UCUNCU BAYATLAMA — asil reason
`cp_config.robot_highcp.derive_6k = {min_v: 30, vertex_conf: 0.5,
cluster_mm: 5.0}`. Bu degerler, `prediction_postproc` icinde
**`_superseded_2026_07_24_values`** as stopped TERK EDILMIS degerlerin
ta kendisi. Urun 2026-08-06'da `min_v 4 / vc 0.3 / cluster 1.0`'a gecti
(`p1_2026_08_06`: very-CP +0.0331); **dense path gecmedi.**

Yani `extract_highcp` "more iyi a path" DEGIL, **Temmuz urununun count
yardimi almis hali**. Makbuzdaki 0.807, Temmuz'da, 24 parts, o gunku
turetme parametreleriyle olculmus a sayidir ve bugunku sisteme
TASINMIYOR.

### DECISION: `export_robot_glb.py`'ye BAGLANMADI
Baglansaydi gozle "dense parts more very CP present" gorunurken measurement
0.79 -> 0.34'e duserdi. **Gorsel iyilesme with measured_path iyilesme ters
yonde olabilir** -- this gecenin vekil dersinin (21.58) gorsel karsiligi.

### YINE DE KAZANC: uc onarim kodda KALIYOR
1. `highcp_selector` residual `decision_score` cagiriyor (gate donusumu)
2. Oznitelik genisligi ISIM DOGRULAMASIYLA kirpiliyor (sessiz kirpma none)
3. `_kapi_sonrasi_zincir()` ortak fonksiyon -- iki path a more ayrisamaz
4. `cp_count=None` -> count geometriden prediction ediliyor (`count_estimate.py`)

Yol residual KOSABILIR durumda. Yeniden candidate olmasi for gereken tek sey,
turetme parametrelerinin bugunku urunle esitlenmesi ve secicinin bugunku
ozniteliklerle YENIDEN EGITILMESI (`highcp_selector.train_and_save`,
pool `results/highcp_pool.json` diskte). Bu a GUNLUK istir, bugune
sigmaz.

## 21.75 LITERATURDEN ILHAM — bizim kusurun ADI present: "sem-seg + connected components"

Aranan sey genel fikir not, **bizim boru hattinin known kusuru**.

**Bizim path:** anlamsal segmentasyon -> bagli bilesenler -> bilesenin
agirlik merkezi = CP konumu.

Literaturde bunun adi *semantic segmentation + connected components* ve
known kusuru aynen su: **birbirine DEGEN, birbirinin AYNI nesneleri
ayiramaz** (Panoptic-DeepLab bunu acikca soyler: kutusuz yontemler degen
nesneleri ayirmakta zorlanir). Bizim dense klemenste cokusumuz tam budur.

**Cozum ailesi** (Panoptic-DeepLab, Spatial Embeddings, PVN3D, EmbedTrack):
each nokta kendi ornek MERKEZINE a **offset vektoru (offset)** prediction
eder; also a **centerness/seed** haritasi ogrenilir (hangi noktanin
oyu guvenilir). Noktalar kaydirilip kumelenir. Merkez bolgeden
TURETILMEZ, dogrudan OYLANIR.

**Bizim yigina birebir oturur:**

| bugun | onerilen |
|---|---|
| DiffusionNet -> 5 sinif probability | + **3 kanal offset** + **1 kanal seed** (same ag, ek bas) |
| bagli bilesen -> merkez | kaydir + kumelen |
| konum cozunurlugu ~ bolge boyu | konum cozunurlugu ~ regresyon hassasiyeti |

**YENI ETIKET GEREKMIYOR:** offset hedefi each tepe for
`(en yakin GT CP - tepe konumu)`; elimizdeki 11.927 manufacturer CP'sinden
bedava cikar.

**Neden this kampanyanin olcumleriyle TUTARLI:**
* 21.60: baglayici kisit YANAL KONUM (+0.358). Offset regresyonu tam
  oraya calisir.
* 21.53-21.62: pose head'in SON-ISLEM regresyonu this acigi TASIYAMADI
  (uc independent measurement). Cunku hasar YUKARIDA olusuyor -- offset basi
  yukarida calisir.
* 21.74: dense parts cokus, "degen same nesneleri ayirma" probleminin
  ta kendisi.

**MALIYET (durust):** training ister (~2 sa/seed x 3 seed + clustering
kalibrasyonu). Bugune SIGMAZ. Ama "sonraki adimlar"in 1. maddesi residual
prediction not, **olculmus a darbogaza oturan somut a mimari**.

Kaynaklar: Panoptic-DeepLab (arXiv 1911.10194) · PVN3D (arXiv 1911.04231)
· Spatial Embeddings (arXiv 1906.11109) · EmbedTrack (arXiv 2204.10713)

## 21.76 OFFSET/OY BASININ TAVANI MEASURED — YESIL ISIK (ve probe cozunurlugu dersi TEKRAR)

Egitime girmeden before ceiling measured (`probe_offset_ceiling.py`, VAL 23-25
part, training YOK). Soru: **offset basi ne up to hassas olmali ki
bugunku sonucu gecsin?** GT offsetlerine gercekci noise eklenip
kaydir+kumele cozucusu kosuldu.

**ILK SONDA (clustering bandi 2.0 mm) — "uygulanamaz" diyordu:**

| offset gurultusu | detection F1 |
|---|---|
| 0.00-0.50 mm | 1.0000 |
| **0.75 mm** | 0.6141 |
| 1.00 mm | 0.3289 |

Pose head'in bugun OOF'ta ulastigi residual median **0.67 mm** -- i.e. tam
ucurumun ustunde. Bu haliyle fikir bugunku 0.7878'i GECMEZDI.

**BANDA DUYARLILIK TARANDI (kapatmadan before sondayi denetle kurali):**

| bant | 0.75 mm | 1.00 mm | 1.50 mm |
|---|---|---|---|
| 2.0 mm | 0.6141 | 0.3289 | 0.2016 |
| 3.0 mm | **1.0000** | 0.8132 | 0.3318 |
| **4.0 mm** | **1.0000** | **0.9933** | 0.6066 |

**Ucurum FIKRIN not, COZUCUNUN kusuruymus.** 4 mm bantla yaklasim
**1.0 mm** offset hatasina up to neredeyse kusursuz; bizim ulasabildigimiz
0.67 mm bunun rahatca icinde.

Bu, `KAPANAN_KOLLAR_DENETIMI`ndeki "yelpaze 64 yonde OLU, 256 yonde
+0.0676" dersinin birebir tekrari: **ilk probe low cozunurlukluydu.**

### DURUSTLUK SINIRI — this a CEILING, prediction DEGIL
Olcumde iki sey KAHINDEN geliyor:
1. **Kim oy verir**: a CP'ye 6 mm'den yakin tepeler (gercekte bunu
   `seed` basi ogrenecek).
2. **Yon**: en yakin GT'nin direction (this betik direction olcmuyor).
Yani gercek sistem also seed haritasini ve direction de ogrenmek zorunda.
Olculen sey sudur: **cozme adimi, ulasabilecegimiz hassasiyette
calisiyor mu?** Cevap EVET.

**DECISION: arm aciliyor.** Sonraki adim offset+seed basini kurup egitmek.

## 21.77 TOPLULUK GENISLETME B (8 uye) — KAPIYI GECEMEDI, TOPLULUK CEPHESI CLOSED

Urun 4 + `recall_hard_keig96_s3` + `y1b_aug0.15_s0/s1/s2` (different recete =
gercek cesitlilik). VAL 100 part, measured_path path, paired bootstrap:

| metrik | baseline (4) | 8 uye | diff | %95 GA | poz% |
|---|---|---|---|---|---|
| detection | 0.6101 | 0.6186 | +0.0086 | [−0.0205,+0.0403] | 70.3 |
| robot | 0.5376 | 0.5522 | +0.0148 | [−0.0167,+0.0499] | 81.3 |
| robot-ISR | 0.4668 | 0.4710 | **+0.0041** | [−0.0250,+0.0328] | 60.8 |

Ucunun de GA'si sifiri iciyor. Isaretli metrik neredeyse no kipirdamiyor
(+0.0041). **Topluluk cephesi closed:**

| arm | robot-ISR farki | verdict |
|---|---|---|
| A: 5 uye (same recete) | −0.0018 (%42.9) | notr |
| B: 8 uye (different recete) | +0.0041 (%60.8) | gate alti |

Ayni receteden uye eklemek bosuna; different receteden eklemek isaretsizde
biraz yardim ediyor (+0.0148, %81.3) but isaretlide not. Bu, gecenin
genel bulgusuyla tutarli: **remaining error direction/sign not KONUM.**

## 21.78 SEED KARISMASI SONDASI **BILGI TASIMADI** — tasarim kusuru, kayda gecirildi

21.76'da "kim oy verir" kahinden geliyordu. Bunu zorlamak for oylarin a
kismi KOMSU CP'ye kaydirilarak "bitisik agizlari karistirma" simule
edilmek istendi. Uc ratio (%5 / %15 / %30) **birebir same** sonucu verdi
(hepsi 1.0000, 1.0 mm'de 0.9920) -- son haneye up to. Bu, no-op imzasidir.

**Sebep bulundu:** bozma, oyun hedefini KOMSU CP'nin TAM MERKEZINE
tasiyor. Yani oy hala VALID a kumeye dusuyor; kumeler ayrik kaldigi
ve each CP bol oy aldigi for cluster merkezleri degismiyor.

**Gercek ag hatasi boyle degildir:** ag offseti iki CP'nin ARASINA yayar
(smear), merkeze not. Dogru probe, prediction edilen offsetin YONUNU/BOYUNU
bozmali (orn. hedefi iki CP arasinda enterpolasyon yapmak), CP kimligini
degistirmemeli.

**Bu sonuc "dayaniklidir" diye RAPORLANMIYOR.** Sonda tasarimi geregi
bilgi tasimiyor; dogrusu yarin kurulacak. Kayit, same tuzaga tekrar
dusulmemesi for burada duruyor.

## 21.79 YOGUN YOL — bugunku turetme parametreleriyle MEASURED (ucuncu bayatlama dogrulandi)

21.74'te ucuncu bayatlama teshis edilmisti: `robot_highcp.derive_6k`
degerleri (30 / 0.5 / 5.0) urunun 2026-07-24'te TERK ETTIGI degerler.
Cevre degiskeniyle bugunku urun degerleri (4 / 0.3 / 1.0) verilip
measured (5 dense part, GT with):

| path | detection | robot | robot-ISARETLI |
|---|---|---|---|
| **BASELINE (`extract`)** | **0.7907** | **0.4651** | **0.4186** |
| YOGUN (Temmuz turetmesi) | 0.3402 | 0.1546 | 0.1134 |
| YOGUN (**bugunku** turetme) | 0.3590 | 0.2051 | **0.1641** |

**Teshis DOGRULANDI but path KURTULMADI.** Turetme duzeltmesi
robot-ISARETLI'yi 0.1134 -> 0.1641 cikardi (+0.0507) -- i.e. ucuncu
bayatlama gercekti. Yine de baseline **2.5 fold** onde.

**Kalan kok why: SECICININ KENDISI.** `highcp_selector.pkl`
2026-07-26'da, **24 parcayla**, o gunku adaylarla ve o gunku
ozniteliklerle egitildi. Turetmeyi guncellemek adaylari degistirir but
selector hala eski dagilima per siralar.

**KISA YOL YOK.** Seciciyi yeniden egitmek for before havuzun
(`results/highcp_pool.json`, Temmuz) bugunku cikarimla yeniden
uretilmesi gerekir -- this a GUNLUK istir. Hizlanmak for kestirme
yapilip path ihracatciya baglansaydi, GOZLE "dense parts more very CP
present" gorunurken OLCUDE detection 0.79 -> 0.36 duserdi.

### YOGUN CEPHESINDE BUGUNKU NET KAZANC
1. Yol **calisir** hale geldi (iki bayatlama onarildi; ucuncusu measured)
2. **Metadata-siz count tahmini** kodda (`cp_count=None`)
3. Kapi sonrasi zincir **ortak fonksiyona** cikti -- iki path a more
   ayrisamaz
4. Yeniden training for gerekenin **tam as ne oldugu** olcumle belli:
   pool yeniden uretimi + selector yeniden egitimi (turetme ayari TEK
   BASINA yetmiyor -- measured)

## 21.80 YOGUN YOLDA **DORDUNCU** BAYATLAMA: oy pool yaricapi 5 mm kalmis

Yogun parts uretilen CP sayisinin yariya dusmesinin a sebebi arandi.
Once a hipotez kuruldu ve **CURUTULDU** (iyi ki measured):

* Hipotez: axis-farkindalikli havuzun lateral yaricapi (3.0 mm) komsu
  kutuplari (adim ~3.5 mm) birlestiriyor.
* **Yanlis:** `robot_eksen_havuz` config'de **False**, i.e. o parametre
  no kullanilmiyor. Urun yolu duz kure mesafesi kullaniyor ve degeri
  `vote_pool_mm = 2.0` -- 3.5 mm adimin ALTINDA, birlestirme none.

Ama same yeri okurken GERCEK kusur bulundu:

```
# robot_cp.py:569 (eski)
cps = _vote2(per, min_votes=1)      # cluster_mm GECILMIYOR -> default 5.0 mm
```

`extract_highcp` oy havuzunu **5.0 mm** with yapiyordu; urun yolu ise
`vote_pool_mm = 2.0` kullaniyor. Bu value 2026-08-06 P1 taramasinda
5 -> 2 mm'ye cekilmisti ve o taramanin **suclusu tam da** "komsu iki
gercek girisi tek adaya yutan genis pool"du (very-CP +0.0331).
**Yogun path o duzeltmeyi de almamis.**

**FIX:** `_vote2(per, cluster_mm=vote_pool_mm, min_votes=1)`.

### OLCUM (3061994, GT=24) — each onarim gercek kazanc veriyor

| durum | robot-ISARETLI |
|---|---|
| orijinal (uc bayatlama acik) | 1 |
| + turetme parametreleri bugunku | 4 |
| + oy pool 5 -> 2 mm | **5** |

Taban path same parts 11. Yani dense path hala geride but **each
bayatlama onarimi olculebilir kazanc veriyor** -- this, remaining farkin da
onarilabilir cinsten oldugunun isareti.

**YOGUN YOLDAKI BAYATLAMA SAYISI: DORT.**
1. gate donusumu atlanıyordu (`decision_score` yerine ham `predict_proba`)
2. oznitelik genisligi 25 vs 70
3. turetme parametreleri Temmuz'un terk edilmis degerleri
4. oy pool yaricapi 5 mm (urun 2 mm'ye gecmis)

Hepsinin ortak sebebi same: **this path cagrilmiyordu, therefore urun
gelistikce sessizce geride kaldi.**

## 21.81 SUNUMDAKI VE CONFIG'DEKI GATE ESIKLERI **OLU PARAMETRE**

Yogun parts gate esigi tarandi (0.15 / 0.25 / 0.30 / 0.35 / 0.45) ve
**bes value de BIREBIR same sonucu** verdi (detection 0.7260, robot 0.5721,
robot-ISR 0.4279). Bu gecenin bes numarali "+0.0000" imzasi.

**KOK WHY:** `wire_gate.GORELI_ESIK` 2026-07-31'de deployed ve
`decision_mask` residual fixed esigi KULLANMIYOR:

```
if GORELI_ESIK:
    return (s >= GORELI_ORAN * s.max()) & (s >= GORELI_TABAN)
return s >= threshold          # <- this satira ARTIK GIRILMIYOR
```

Dolayisiyla:
* `cp_config.robot_wire_gate_threshold` (0.40) -> **OLU**
* `cp_config.robot_wire_gate_threshold_highcp` (0.35) -> **OLU**
* `extract` icindeki regime-kosullu threshold secimi -> **ETKISIZ**
* 2026-07-29 taramasinin "0.25 more iyi" sonucu -> residual **UYGULANAMAZ**
  (o tarama fixed threshold doneminde yapildi)

**SUNUM DA BUNLARI CANLI AYAR DIYE GOSTERIYOR** (slayt 8: "Gate
threshold, low CP density 40% / high CP density 35%"). Duzeltilmeli.

**GERCEK CANLI PARAMETRELER:**
`gate_goreli_oran` (0.50) ve `gate_goreli_taban` (0.20) --
ikisi de cevre degiskeninden okunabiliyor (`WG_GORELI_ORAN`,
`WG_GORELI_TABAN`). Tarama bunlara cevrildi.

**LESSON (this gecenin tekrar eden dersi):** a parametrenin config'de
DURMASI, urunun onu KULLANDIGI anlamina gelmez. Tarama before "parametre
gercekten baglayici mi" diye sinanmali; otherwise saatlerce olu a dugmeyi
cevirmis olursun.

## 21.82 YOGUN YOL — DORT ONARIMDAN SONRAKI TAM OLCUM

5 dense part, GT with, bugunku sistem:

| path | detection | robot | robot-ISARETLI |
|---|---|---|---|
| **BASELINE (`extract`)** | **0.7907** | **0.4651** | **0.4186** |
| YOGUN, orijinal (4 bayatlama acik) | 0.3402 | 0.1546 | 0.1134 |
| YOGUN, turetme onarildi | 0.3590 | 0.2051 | 0.1641 |
| **YOGUN, DORT onarim** | **0.3673** | **0.2143** | 0.1531 |

Onarimlar tespitte **+0.0271**, unsigned robotta **+0.0597** kazandirdi.
Ama baseline hala **2.7 fold** onde.

**DECISION DEGISMEDI: ihracatciya BAGLANMIYOR.** Kalan diff yapisal --
selector 2026-07-26'da **24 parcayla** egitildi ve o gunku candidate
dagilimini varsayiyor. Dort onarim adaylari degistirdi; selector o yeni
dagilima per siralamiyor.

**Dort onarim yine de KODA KALIYOR**, because selector yeniden egitildiginde
this yolun DOGRU davranmasi for hepsi gerekli. Yol residual "bozuk" not,
"eski selector with" calisiyor.

## 21.83 OFFSET BASI — EGITIM TRENDI ve TRIVIAL BASELINE

Egitim: 659 part, 14 epok, leakage kapisi kurulu.

**Once TRIVIAL BASELINE measured** (this olmadan training sayisi yorumlanamaz):
> Model HICBIR SEY ogrenmese, i.e. sifir offset prediction etse, median
> error **4.453 mm** olurdu (seed bandindaki tepelerin CP'ye median
> uzakligi).

| epok | median offset hatasi |
|---|---|
| trivial (sifir prediction) | 4.453 mm |
| 1 | 4.129 mm |
| 2 | 4.005 mm |
| 8 | **3.664 mm** |

**Model OGRENIYOR** (3.664 < 4.453, %18 more iyi) but **very yavas**:
epok basina ~0.06 mm ve yavasliyor. 14 epokta ~3.5 mm'de kalir.

**GEREKEN: 1.0 mm** (21.76 ceiling olcumu, clustering bandi 4 mm).

**VERDICT: this butcede threshold GECILMIYOR.** Kol "olu" not, but this recete
ve this butce with hedefe ulasmiyor.

## 21.84 OFFSET BASI TESHISI — YON OGRENILMIS, BUYUKLUK BUZULMUS

Ortanca error 3.4 mm tek basina "failed" der. Ama hatanin BILESENLERI
ayristirilinca tablo degisiyor (epok 13 checkpoint'i, 6 part):

| measurement | value |
|---|---|
| **hedef** offset buyuklugu (median) | 4.299 mm |
| **prediction** buyuklugu (median) | **2.871 mm** |
| **direction uyumu** (kosinus; 0 = rastgele, 1 = mukemmel) | **+0.675** |

**Model YONU OGRENMIS.** Kosinus +0.675, yaklasik **48 derece** median
aci hatasina karsilik gelir -- rastgele a prediction 0 verirdi. 13 epokta,
700 parcayla, hicbir setting aramasi yapilmadan.

**Sorun BUYUKLUK:** model 4.30 mm'lik hedefe 2.87 mm oneriyor (%33 az).
Bu, this kampanyada **ikinci kez** karsilasilan patoloji: pose head de tam
as boyle davraniyordu (21.53: en large oneri 2.87 mm, hedeflerin
%10.2'si 3 mm'nin ustunde). **Regresyon buzulmesi**, L1/L2 kayiplarinin
known davranisi.

### VERDICT: KOL OPEN KALIYOR, RECETE DEGISMELI
"Bu butcede threshold gecilmiyor" (21.83) correct but missing. Dogrusu:
**direction sinyali VAR ve ogreniliyor; ulasilamayan sey buyukluk.**

Sonraki denemenin degistirmesi gerekenler (olculmus gerekceyle):
1. **Buzulmeye karsi loss**: saf L1 yerine direction + buyukluk AYRI
   ogrenilsin (birim vektor + skaler norm), ya da loss norm'a per
   agirliklandirilsin. Buzulme iki independent kolda goruldu.
2. **Daha uzun training**: 13 epokta hala monotonik iyilesiyor
   (4.129 -> 3.351), i.e. doymamis.
3. **Seed bandi daralt**: 6 mm yerine 3 mm; uzak tepelerin hedefi hem
   large hem ambiguous, ortalamayi asagi cekiyor.

**Bu, "arm olu" demekten very different a sonuctur** ve bugunku olcumle
desteklenmektedir.

## 21.85 OFFSET BASI BITTI + **SEED BASI GORULMEMIS PARCADA AUC 0.8042**

**Egitim tamamlandi** (700 part / 14 epok / leakage kapisi kurulu):

| epok | median offset hatasi |
|---|---|
| trivial (sifir prediction) | 4.453 mm |
| 1 | 4.129 |
| 8 | 3.664 |
| 13 | 3.351 |
| **14 (son)** | **3.306** |

**Egri DOYMADI** -- 14 epok boyunca monotonik iyilesti. Butce bitti,
ogrenme bitmedi. Hedef 1.0 mm; this butcede GECILMEDI.

### SEED BASI -- beklenmedik ve GUCLU sonuc
Ag 4 kanal uretiyor; dordu `seed` (this tepe a CP'ye yakin mi).
**GORULMEMIS VAL parcalarinda** (egitimde YOK) measured:

> **seed basi AUC = 0.8042** (8 part ortancasi; 0.5 = rastgele)

Yani 14 epok ve 700 parcayla, hicbir setting aramasi yapilmadan egitilen
a yan cikti, unseen parts guclu a TESPIT sinyali veriyor.

**DURUSTLUK -- 0.7053 with DOGRUDAN KIYASLANAMAZ.** Daha before measured_path
"konum AUC 0.7053" **ADAY** basinadir ("this opening CP mi"); buradaki
0.8042 **TEPE** basinadir ("this tepe a CP'ye yakin mi"). Iki different
populasyon, iki different soru. Ayni tabloya konulmaz.

### NE ANLAMA GELIYOR
Kolun iki ciktisi present ve **ikisi different olgunlukta**:
* **offset** (buyukluk): buzulmus, hedefe uzak -> recete degismeli (21.84)
* **seed** (detection): unseen parts ZATEN calisiyor

Yani this mimari "ilerde belki" not; **yarim gunluk a egitimde bile
kullanilabilir a sinyal uretiyor.** Kol OPEN ve onceligi high.

## 21.86 GATE GORELI ORANI 0.50 -> 0.60: IKI REJIMDE DE KESIN POZITIF

21.81'de fixed esiklerin OLU oldugu, gercek canli parametrenin
`gate_goreli_oran` (0.50) oldugu bulunmustu. Taranan this.

**YOGUN parts (n_gt>=8, 14 part):**

| ratio | detection | robot | robot-ISARETLI |
|---|---|---|---|
| 0.30 | 0.6985 | 0.5628 | 0.4121 |
| 0.40 | 0.7113 | 0.5722 | 0.4227 |
| **0.50 (dagitilan)** | 0.7312 | 0.5968 | 0.4409 |
| **0.60** | 0.7187 | **0.6128** | **0.4568** |

**SEYREK parts (n_gt<8, katalogun %86'si, 26 part):**

| ratio | detection | robot | robot-ISARETLI |
|---|---|---|---|
| **0.50 (dagitilan)** | 0.7931 | 0.4828 | 0.4713 |
| **0.60** | 0.7826 | 0.5093 | **0.5093** |

**ESLI BOOTSTRAP (0.50'ye per, robot-ISARETLI):**

| regime | diff | %95 GA | poz% |
|---|---|---|---|
| dense | **+0.0162** | [+0.0046, +0.0328] * | **100.0** |
| sparse | **+0.0374** | [+0.0167, +0.0608] * | **100.0** |

Tespit bedeli iki rejimde de **conclusive not** (−0.0129 ve −0.0100, iki
GA da sifiri iciyor). Yani kazanc signed robot metrigine geliyor,
tespitten conclusive a sey goturmuyor.

Rejim paylariyla agirliklandirilmis beklenen etki:
0.86 x 0.0374 + 0.14 x 0.0162 = **~+0.034**.

### AMA: BU PARAMETRE **VAL UZERINDE** TARANDI
VAL bizim degerlendirme kumemiz. Ayni kumede tarayip same kumede
raporlamak SISME uretir. Bu kampanyada tarama **iki kez** yaniltti:
* `vc0.35`: taramada +0.01, TAM dagitim OOF'unda WEI 0.641 -> 0.629
* `k_eig 128`: same desen

**Bu yuzden BAGIMSIZ ornekte dogrulama kosuluyor** (VAL ve LOCKED DISI
30 part, same parametre). Dogrulanmadan **DAGITILMAZ**.

## 21.87 ALAN FARKI (+0.2468) GATE DEGISIKLIGINDEN ETKILENMIYOR — dogrulandi

`gate_goreli_oran` 0.50 -> 0.60 dagitildiktan after, sunumun ana bulgusu
which alan farkinin (familiar 0.5603 / unseen 0.3135) hala valid olup
olmadigi soruldu. **Yeniden olcmeden ONCE kod yolu kontrol edildi:**

* Alan farki `run_p6_kademe2.py` with measured; this betik **kendi karar
  kuralini** kullaniyor (`KURALLAR` izgarasi: mutlak/goreli, fold ICINDE
  aranir).
* Degistirilen `gate_goreli_oran` ise `wire_gate.decision_mask`'nin
  parametresi. O betikten `wire_gate`'e giden tek cagri
  `crowd_mask` (NMS) -- esikle ilgisi YOK.

**Sonuc: iki measurement AYRI kod yollarinda; +0.2468 VALID kaliyor.**
25 dakikalik gereksiz a yeniden measurement yapilmadi.

Not: this, raporda kayitli "measured_path sey with dagitilacak sey same degildi"
dersinin (Bolum 21.x, kanonik blok: olcumde +0.0151 / uretimde −0.0138)
simetrigi. Orada ayrim ALEYHIMIZE calismisti; burada LEHIMIZE, because iki
measurement birbirini invalid kilmiyor. Her iki durumda da rule same:
**hangi kod yolunda olculdugu before kontrol edilir.**
