# Robot CP kampanyasi -- 2026-08-11

Hedef: gorulmemis markada **robot F1 = 0.50**, sisik olmayan bir olcumle.
Sart: her an kampanya oncesi duruma donebilmek.

---

## 1. Geri donus (sart 1) -- KANITLANDI

| | |
|---|---|
| kontrol noktasi | git etiketi `KONTROL_NOKTASI_2026-08-11`, commit `68a6e856` |
| kod | 1678 dosya (750 `.py`, 729 makbuz `.json`, 83 `.md`) commit'te |
| model | `_KN_2026-08-11/dosyalar/` -- 11 dosya / 643 MB fiziksel kopya |
| damga | 280 model dosyasi + 4 dizin parmak izi SHA-256 ile |
| geri donus | **`python geri_al.py`** |

**Tatbikat yapildi:** `urun_genis.py`'ye sahte satir eklendi, `cp_config.json`
bozuldu (esik 0.99 + sahte anahtar), `results/kazanan_hgb_derin.pkl` SILINDI,
iki sahte dosya olusturuldu. `geri_al.py` sonrasi tam kanonik zincir:

| | kampanya oncesi | hasar + geri donus sonrasi |
|---|---|---|
| robot | 0.297956 | **0.297956** |
| tespit | 0.476022 | **0.476022** |
| makro | 0.293380 | **0.293380** |
| 12 markanin hepsi | -- | **birebir ayni** |

`master` dali hep kontrol noktasinda; kampanya isi `kampanya_050` dalinda.
Geri donmek yapilan isi de SILMEZ.

---

## 2. Olcum protokolu -- sayinin neden sisik olmadigi

### 2.1 Uc kume MARKA-AYRIK

| kume | marka | parca | rol |
|---|---|---|---|
| `tam` | TOGI, PXC, WEI, SIE, TE, TKM, WAGO, MDI, ABB | 2583 | egitim + kural secimi |
| `d6` | SUPU, UPUN, MOR, NIT, UTL, S+S, SE, ONV | 468 | gelistirme, sonra EGITIME katildi |
| `d7` | CCD, KLM, A-B, EFX, WIE, CWT, DIN, WEG, CEM, DEG, ELMEX, C3 | 835 | **SINAV** |

Nihai model **`tam` + `d6` = 3051 parca / 17 marka** ile egitilir. D6 sinav
degildir; teshis ve kol secimi icin zaten yogun kullanildi, dolayisiyla ondan
sonra "temiz D6 sayisi" diye bir sey YOKTUR ve raporlanmaz. Egitime katilmasi
D7 icin mesrudur ve marka cesitliligini 9 -> 17 yapar. **Kural secimi yine
yalniz `tam` markalarinin katlarinda** (TOGI/PXC/WEI/SIE) yapilir.

Referans olarak D6'da dagitilan urunun sayisi (egitime katilmadan ONCE, urunun
canli zincirinden): **robot 0.2815 / tespit 0.4215**, 468 parca, %95 GA
[0.2399, 0.3285]. D7'deki dagitilan sayi 0.2980 -- yani iki kume dagitilan urun
icin benzer zorlukta.

Denetim (`results/bolme_denetimi.json`):

| olcek | tam∩d7 | tam∩d6 | d6∩d7 |
|---|---|---|---|
| parca kimligi | 0 | 0 | 0 |
| marka | 0 | 0 | 0 |
| TAM geometri (tepe+yuz+kutu 0.1mm) | 0 | 0 | 0 |
| kaba iz (kutu 0.5mm + GT sayisi) | 94 iz / 132 parca | 66 | 14 |

Gercek sizinti alt sinir (tam geometri) ile ust sinir (kaba iz) ARASINDADIR.
Kaba iz DIN klemenslerinin standart olculu olmasindan FAZLA sayar. Bu yuzden
manset yaninda **kaba iz eslesmesi olmayan 703 parcalik D7 alt kumesi** de
raporlanir.

### 2.2 Kurallarin nerede secildigi
* Esik / NMS / kol secimi YALNIZ `tam` korpusunun MARKA KATLARINDA.
* Kural secim olcutu **makro** (marka basina esit agirlik) -- mikro, GT'si cok
  olan markanin kuralini secip diger markalari cokertiyordu.
* D7 sinav; ona bakarak HICBIR ayar secilmedi.

### 2.3 Olcumun urunun kendisi olmasi
Her sayi `kanonik_zincir.urun_cikti` uzerinden, yani **urunun TEK zincirinden**
gecer. Karar kodu tek modulde (`p6_karar`) ve hem egitim hem urun ONU cagirir.
Havuz kurulumu (`havuz_seyrelt`), mesh esigi ve dedupe degerleri egitimdekiyle
BIREBIR ayni; `ppos` tanimi (`pb[:,CE]+pb[:,CT]`) korpusu ureten betikle ayni.

### 2.4 Makbuz
Her olcum `makbuz_hash.damga()` ile kod/model/config SHA-256'larini yazar.
Damgasiz sayi sayi degildir.

---

## 3. Teshis zinciri -- hangi sirayla ne bulundu

1. **Yon bir SECIM problemiydi ve cozuldu.**
   Dagitilan urun her konuma TEK yon bagliyordu. `yon_bankasi` (kendi / komsu /
   silindir ekseni / ana eksenler) eklendi. Tavan D6'da 0.4497 -> 0.6889.
   *Kalan yon kaybi: +0.0052* (secilen adaylarda mukemmel yon secici ile fark).
   Yani yon artik darbogaz DEGIL.

2. **NMS hipotezi CURUDU.** D6 GT'lerinin %16.1'inin komsusu 5mm'den yakin
   olmasina ragmen esik x NMS taramasi her kivrimda 5.0'i sectti.

3. **Asil darbogaz KONUM havuzuydu.**

   | D6 havuzu | yalniz KONUM | konum + YON |
   |---|---|---|
   | B-rep (dagitilan) | 0.5371 | 0.2900 |
   | + mesh tepeleri | 0.9768 | 0.4854 |
   | + mesh + yon bankasi | 0.8713 (seyreltilmis) | **0.7264** |

   Mesh tepeleri tarihte UC kez zarar vermisti; sebep anlasildi: yon bankasi
   olmadan eklendiklerinde yalnizca FP uretiyorlardi. Konum ve yon AYRI iki
   eksik, ikisi birden kapanmali.

4. **Seyreltmede KAPSAMA, GUVENI yeniyor.** "En yuksek olasilikli 60 tepe"
   yerine "2.5mm uzamsal seyreltme" ayni maliyette konum recall'unu
   0.6362 -> 0.8713 yapiyor.

5. **Ikinci kademe KASKAD olmali.** Tum secenekleri yeniden puanlayan ikinci
   model ZARAR verdi (-0.0363). Kisa listeye (birinci kademe skoru >= 0.20)
   odaklanan, birinci kademe skorunu da oznitelik alan surum kazandi.

6. **Periyodik yapi gercek.** D6'da >=6 CP'li 1096 parcada GT'lerin **%90.8'i**
   parcanin en sik OTELEME VEKTORUYLE baska bir GT'ye ulasiyor. `kafes`
   modulu bunu oznitelik olarak verir; tohumlar HER ZAMAN tahminden gelir,
   GT'den ASLA.

---

## 3b. Kapanmayan cephe: YOGUN parcalar (NIT ornegi)

NIT: 50 parca / 1222 GT / 24.4 CP-parca. **Dagitilan urun 1222 GT'den 2'sini
buluyor** (F1 0.0032) -- yani bu kampanyanin actigi bir sorun degil, sistemin
sureklilik arz eden kor noktasi. Yeni havuz o markada yonlu recall **0.5254**
veriyor, yani cevabin yarisi HAVUZDA.

Kayip nerede? Uc olcum:

| soru | olcum | cevap |
|---|---|---|
| esik mi? | o markadaki EN IYI kural | 0.0349 -- HAYIR |
| siralama rastgele mi? | recall@k / rastgele | 9.4x -- HAYIR |
| ne kadar iyilestirilebildi? | C blogu + `zskor=ikisi` | **12.0x** |

Denenen ve olculen iki mudahale:
* **Segmentasyon oznitelikleri (A blogu) NIT'te ZARAR VERIYOR**: yalniz yon
  bankasi (C) 10.4x, hepsi 9.4x.
* **Parca-ici SIRA donusumu** tek basina kotu (5.9x) ama Z-SKORLA BIRLIKTE en
  iyisi (12.0x).

**Neden yetmiyor:** NIT parcasinda 4467 secenek var ve 24'u dogru (%0.69). 12x
siralama top-24'e 2 dogru koyar -> F1 ~0.08. Kullanilabilir bir sayi icin
~50-100x gerekir; bu, mevcut oznitelik uzayinda kapanacak bir fark DEGILDIR.
Yeni bilgi kaynagi (o yogunlukta etiketli veri ya da farkli bir temsil) gerekir.

**Sinav icin baglami:** D7 D6'dan cok daha SEYREK.

| kume | ort CP/parca | >=8 CP olan parca | o parcalardaki GT payi |
|---|---|---|---|
| tam | 5.7 | %14.4 | %55.7 |
| d6 | 5.7 | %13.7 | %55.4 |
| **d7** | **3.7** | **%7.4** | **%30.3** |

Yani D6'nin mikro sayisi yogun parcalarin egemenliginde; D7'ninki degil. D6'da
olculen mikro kazanc D7 icin KOTUMSER bir tahmindir.

---

## 4. Yakalanan tuzaklar

| tuzak | belirti | sonuc |
|---|---|---|
| `KAYNAKLAR` iki kez tanimli | `P6_KAYNAK=012` hicbir sey yapmiyor, hata YOK | mesh havuzu hic acilmamis |
| isin kesisimi tek cagrida | 8 payin 4'u `MemoryError` | topaklandi, sonuc bit duzeyinde ayni |
| `_tam_oz` onbellegi config'den eski | segmentasyon adaylari 7 vs 12 | dagitilan modelde de VAR, kiyas adil |
| TABAN kolunda satir/aday indeksi karisik | mesh suzgeci gelince yanlis konum | duzeltildi |
| kafes 1B sira olarak modellenmisti | GT'nin yalniz %5'i uyuyor | oteleme vektoru ile %90.8 |
| kafes ara adim yok | 6mm adimli sirada tohumlar 12mm gorunce aradakiler hic ongorulmuyor | yarim VE ucte-bir adimlar eklendi |
| `sec_ayrintili` erken cikis dali | normal dal 4 deger, erken cikis 3 -> D7 P6 kolu `expected 4, got 3` ile coktu | dal esitlendi + `tests/test_p6_karar_imza.py` |
| iki kosu ayni dosyalara yazdi | duzeltme ONCESI baslamis paylar duzeltme SONRASI paylarla ayni makbuza yaziyordu | eski zincir durduruldu, temiz kosu |
| yelpaze sondasi 64 yon | "kol OLU" hukmu verildi; oysa 64 yonun araligi 25 derece, tolerans 10 | 256 yonde +0.0676 -- **hukum SONDANIN kusuruydu** |

---

## 5. D7 OKUMA #1 -- KARAR KURALLARI (okumadan ONCE yazildi)

Bu bolum D7'ye BAKILMADAN once dolduruldu. Sayiya bakip kural secmek sismenin
ta kendisidir; asagidakiler baglayicidir.

**Okunacak yapilandirma (tek, onceden sabit):**
* model: `results/p6_kademe2_model.pkl` -- `tam`+`d6` (3051 parca / 17 marka)
  ile egitilmis; kol ve karar kurali `tam`in MARKA KATLARINDA (WEI/PXC/SIE/TOGI)
  MAKRO olcutle secilmis.
* poz kafasi: **P6 kolunda KAPALI** (ilan edilen). Gerekce ILKESEL, olcume
  bakilarak degil: model ve karar kurali marka katlarinda poz kafasi OLMADAN
  secildi; uzerine dogrulanmamis bir son islem koymak, olculen seyden baska bir
  sey dagitmak olurdu. TABAN kolu kendi DAGITILAN hali olan poz-kafasi-ACIK ile
  kosar. Ucuncu bir kol (P6 + poz kafasi) yalnizca GOZLEM olarak raporlanir;
  manset ondan SECILMEZ.
  (D6 uzerinden karar verilmedi cunku D6 egitime katildi ve ayrica D6'nin
  `_tam_oz` onbellegi `_p1_olasilik_g7`'den, benim betiklerim `_p1_olasilik`'ten
  besleniyor -- D6 artik urunu temsil etmiyor. D7 ve `tam` icin bu uyusmazlik
  YOK, ikisi de kendi onbellegiyle tutarli.)
* iki kol AYNI kosuda: `URUN_P6=0` (dagitilan urun) ve `URUN_P6=1`.
* 8 pay + `birlestir_makbuz.py`; mikro F1 icin birlestirme kayipsizdir.

**Manset tanimi:** MIKRO robot F1 (yanal <=2mm, ISARETLI aci <=10, eksenel
<=40mm, Macar bire-bir eslesme), 835 parca, urunun TEK kanonik zincirinden.
`f1w` KULLANILMAZ.

**Onceden ilan edilen kesmeler:**
| durum | karar |
|---|---|
| `p6_sayac`: P6 kolu parcalarin <%90'inda calisti | manset GECERSIZ; sebep bulunur, okuma tekrarlanir (butceden sayilir) |
| P6 robot < TABAN robot | KOL DAGITILMAZ; taban korunur, sonuc oyle raporlanir |
| P6 robot >= 0.50 | hedef TUTTU; temiz-703 alt kumesinde de raporlanir |
| 0.35 <= P6 robot < 0.50 | hedef TUTMADI; kazanc dagitilir, kalan yol 0.75 paketiyle surer |
| P6 robot < 0.35 | kazanc D6'dan D7'ye TASINMADI; sebep analizi (marka kirilimi) sart |

**Ayrica her okumada raporlanir:** %95 bootstrap GA, marka kirilimi, makro,
en kotu marka, recall/kesinlik, ve kaba-iz eslesmesi olmayan 703 parcalik
TEMIZ ALT KUME sayisi. Manset ile temiz alt kume arasindaki fark buyukse sayi
supheli sayilir.

### 5b. `tam` MARKA KATLARINDA GERCEK KAZANC (D7'den ONCE, sinav DEGIL)

3051 parca (tam+d6), 4 marka kati (WEI/PXC/SIE/TOGI), esik/kol katta secildi:

| kol | robot | recall | kesinlik | TP | FP | FN |
|---|---|---|---|---|---|---|
| TABAN | 0.2861 | 0.1868 | 0.6108 | 2671 | 1702 | 11629 |
| **P6** | **0.3091** | 0.2935 | 0.3265 | 4197 | 8658 | 10103 |
| P6_KAFES | 0.3045 | 0.2587 | 0.3698 | 3700 | 6305 | 10600 |
| P6_GEO | 0.2707 | 0.2566 | 0.2864 | 3670 | 9146 | 10630 |

**P6 - TABAN = +0.0230.** D6'nin vaat ettigi +0.1681 GERCEK DEGILDI.

Kat kirilimi deseni acikliyor:

| kat | n | TABAN | P6 | fark |
|---|---|---|---|---|
| TOGI | 810 | 0.0597 | 0.1689 | **+0.1092** |
| WEI | 686 | 0.2797 | 0.3795 | **+0.0998** |
| PXC | 713 | 0.5850 | 0.3972 | **-0.1878** |
| SIE | 262 | 0.6226 | 0.4681 | **-0.1545** |

**P6, tabanin ZAYIF oldugu yerde kazanir; GUCLU oldugu yerde kaybeder.** D6'nin
tabani zayifti, o yuzden orada her sey iyi gorunuyordu. Bu, gelistirme
kumesinden okunan kazancin neden aldatabilecegini gosteren somut ornektir.

**REJIM KAPISI** (bu bulgunun cevabi): `n01 >= 90 -> P6, altinda TABAN`.
Yonlendirme istatistigi taramasi (kat-disi esik secimi, MAKRO olcut):

| istatistik | kat-disi robot | secilen esikler |
|---|---|---|
| **n01** | **0.5269** | 90 / 90 / 90 / 90 |
| n_aday | 0.5139 | 114 x4 |
| n_secenek | 0.5137 | 1628/1344/1344/1628 |
| mesh_oran | 0.5025 | 0.42 x4 |
| taban_ort3 | 0.3660 | -0.51 x4 |
| taban_maks | 0.3515 | -0.75 x3, -0.55 |

Iki not: (1) "tabanin kendi guvenine gore yonlendir" hipotezi CURUDU -- taban
skoru parcalar arasi kalibre degil. (2) **Bu tablodaki sayilar ORNEKLEM-ICIDIR**
(nihai model bu parcalari egitimde gordu); yalnizca hangi KURALIN secildigini
gosterirler, kuralin degerini DEGIL. Esigin dort katta da ayni cikmasi kuralin
kararli oldugunu gosterir.

### 5c. D7 SINAV SONUCU

**TABAN kolu (dagitilan urun, poz kafasi ACIK) -- payli kosuyla yeniden uretildi:**

| | deger |
|---|---|
| robot MIKRO | **0.2980** %95 GA [0.2670, 0.3293] |
| tespit MIKRO | 0.4760 %95 GA [0.4476, 0.5046] |
| TP / FP / FN | 758 / 1243 / 2329 (recall 0.2455, kesinlik 0.3788) |
| makro / en kotu | 0.2934 / 0.0000 |
| **temiz alt kume (703 parca)** | **0.2558** -- manset farki **-0.0422** |

Iki not:
1. Sekiz paya bolunup birlestirilen olcum, tek islemde kosulan bilinen degeri
   BIREBIR verdi (0.2980 / 0.4760). Birlestirme kayipsiz.
2. **Manset, egitim parcalariyla kaba geometri benzerliginden 0.042 kadar
   besleniyor.** Temiz alt kume daha dusuk; bu fark her iki kol icin de
   raporlanir.

Marka kirilimi (taban):

| marka | n | GT payi | robot F1 |
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
"taban zayifsa P6" dedigine gore asil fark orada gorulecek.

**P6 kolu (yon bankasi + mesh havuzu + rejim kapisi, poz kafasi KAPALI):**

| | TABAN | **P6** | fark |
|---|---|---|---|
| **robot MIKRO** | 0.2980 | **0.3115** | **+0.0135** |
| %95 GA | [0.2670, 0.3293] | [0.2854, 0.3375] | ortusuyor |
| tespit | 0.4760 | 0.4619 | -0.0141 |
| makro | 0.2934 | **0.3265** | **+0.0331** |
| **en kotu marka** | 0.0000 | **0.0466** | **+0.0466** |
| TP / FP / FN | 758/1243/2329 | 872/1640/2215 | — |
| recall / kesinlik | 0.2455 / 0.3788 | 0.2825 / 0.3471 | — |
| temiz-703 | 0.2558 | 0.2766 | +0.0208 |

P6 kolu 835 parcanin **385'inde (%46.1)** calisti; sessiz geri dusme YOK
(`tablo_yok` 0, `model_yok` 0). Rejim kapisi 450 parcada tabana yonlendirdi --
kalibrasyonun ongordugu ~yari oranla uyumlu, yani esik urune DOGRU tasindi.

Marka kirilimi (8/12 markada ARTI):

| marka | taban | P6 | fark |
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

### 5e. KARAR (onceden ilan edilen kurala gore)

**HEDEF TUTMADI.** Ilan edilen kural "P6 >= 0.50 -> tuttu; 0.35-0.50 -> tutmadi
ama dagitilir; < 0.35 -> kazanc tasinmadi, sebep analizi sart" diyordu.
Sonuc **0.3115**, yani en alt bantta.

Ne oldugu durustce:
* Kazanc GERCEK ama KUCUK: +0.0135, ve iki kolun guven araliklari ORTUSUYOR.
  Tek basina bu fark istatistiksel olarak zayiftir.
* Buna karsilik **makro +0.0331 ve en kotu marka 0.0000 -> 0.0466**: sistem
  markalar arasi daha DENGELI. Sifirdan cikan bir marka (C3) ve iki katina
  cikan uc marka (CEM, EFX, C3) var.
* Kaybedilen yer WIE (-0.0802) ve A-B/CCD/KLM: rejim kapisi bu markalarda
  yanlis tarafa yonlendiriyor. Kapi tek bir esik (n01>=90) ve bu markalarda
  taban gucluyken P6'ya gecmis olmali.
* **CWT hala 0.0466** ve D7 GT'sinin %37'si orada. Asil duvar CWT'de ve bu
  duvar NIT'le AYNI: yogun parcada temsil yetersizligi. 0.50'nin onundeki
  tek en buyuk engel budur.

### 5f. GUVEN KAPISI -- saha 0.90 sozu VERILEMEZ

D7'nin 2512 tahmini uzerinde olculdu:

```
ham kesinlik 0.3471
kesinlik hicbir esikte >= 0.90 OLMUYOR -- en yuksek 0.6429
```

Dahasi egri tepe noktasindan sonra GERI DONUYOR (pay0'da esik 0.95'te 0.5025,
0.99'da 0.4700). Sebep: karar kurali GORELI (`0.85 x parca-maks`), yani secilen
tahminlerin hepsi zaten parca-maksimumuna yakin; mutlak skor parcalar arasi
kalibre bir guven olcusu DEGIL.

**Sonuc:** "robotun kullandigi isaretler >=0.90 kesinliktedir" sozu BU
SKORLAYICIYLA verilemez. Guven kapisi AYRI bir kalibrasyon modeli ister
(parca-ici goreli konum + kafes tutarliligi + aday mutabakati gibi sinyaller).
Kapsama sayisi uydurmak yerine bu boyle kaydedildi.

**KALIBRASYON MODELI KURULDU VE OLCULDU** (`kos_kalibrasyon.py`; secilmis
tahminler uzerinde 15 sinyalle ikinci model; tam+d6, 8015 tahmin, 6 marka
kati, KAT-DISI):

| hedef kesinlik | ham kapsama | **kalibre kapsama** |
|---|---|---|
| 0.60 | 0.2299 | **0.3023** |
| 0.70 | 0.1007 | **0.1584** |
| 0.80 | 0.0403 | **0.0905** |
| **0.90** | 0.0041 | **0.0035** |

Kalibrasyon 0.60-0.80 bandinda kapsamayi **~2 kat** artiriyor. Ama 0.90'da
ikisi de sifira yakin (%0.35).

**SAHA SOZUNUN DURUST HALI:** ">=0.90 kesinlikli isaret" bugun GT'nin BINDE
3.5'i icin verilebilir -- kullanilabilir bir teklif DEGIL. **Bugun
verilebilecek en iyi soz: 0.80 kesinlikte %9 kapsama.** 0.90'a ancak
tam-otomatik bandin kendisi yukselince ulasilir; kisa yolu yok.

### 5.1 Okuma plani (onceden ilan)

D7 butcesi 3 okuma; bu birincisi. Ayni kosuda iki kol olculur (taban URUN_P6=0,
P6 URUN_P6=1), ikisi de urunun TEK kanonik zincirinden gecer, 8 pay paralel.
Manset = MIKRO robot F1 + %95 parca-bootstrap araligi + marka tablosu + temiz
alt kume (703) duyarliligi + P6 geri-dusme sayaci.

KARAR KURALLARI (okumadan ONCE yazildi):
* P6 >= 0.50 ve temiz alt kume farki kucukse -> hedefe ulasildi; dagitim karari
  ayri konusulur.
* 0.50'nin altinda ama taban (0.2980) uzerinde anlamli artis varsa -> kazanc
  raporlanir, kalan fark hata bankasiyla aciklanir; ikinci okuma ancak SOMUT
  bir duzeltmeden sonra yapilir.
* Taban altinda ya da geri-dusme sayaci yuksekse -> sayi RAPORLANIR, sebep
  bulunur; `geri_al.py` ile donus her an mumkun.

---

### 5g. D7 SONRASI TESHISLER -- 0.50'ye giden yol nerede tikaniyor

**1. Rejim kapisi D7'nin iki buyuk markasinda TERS calisiyor.**

| marka | n01/parca | konum recall (tum mesh) | yon recall (banka) | taban | P6 |
|---|---|---|---|---|---|
| WIE | 141 | 0.8897 | 0.8536 | 0.5712 | 0.4910 |
| CWT | 86 | 0.5532 | 0.4415 | 0.0370 | 0.0466 |

Esik 90 idi: CWT (86) TABANA gidiyor -- oysa taban orada cokuyor; WIE (141)
P6'ya gidiyor -- oysa taban orada iyi. D7'deki WIE kaybi (-0.0802) ve CWT'nin
kipirdamamasi AYNI sebepten.

**2. Ogrenilmis yonlendirici (v2) BASARISIZ.** 13 parca ozniteligiyle egitilen
yonlendirici kat-disi 0.5269 -> 0.4646 (**-0.0624**); egitim markalarina ozgu
oruntuleri ezberleyip gorulmemis markaya tasimiyor. TOGI'de -0.1137. **Tek esik
KALIYOR.**

**3. CWT'nin havuzu segmentasyon esiginin ALTINDA kalmis.** Mesh adaylari
`p_pos >= 0.50` ile seciliyor:

| marka | p>=0.5 | p>=0.3 | p>=0.2 | p>=0.1 | p>=0.05 |
|---|---|---|---|---|---|
| **CWT** | 0.5532 | 0.6108 | 0.6501 | 0.7286 | **0.7914** |
| WEG | 0.6768 | 0.7439 | 0.7805 | 0.8232 | 0.8354 |
| WIE | 0.8897 | 0.9278 | 0.9430 | 0.9544 | 0.9620 |

CWT'de konum recall **+0.2382** aciliyor (aday 267 -> 700).

**METODOLOJIK CIKMAZ:** ayni tarama `tam` korpusunda WEI 0.9497 -> 0.9954,
PXC 0.9914 -> 0.9914 veriyor. Yani egitim markalarinda havuz ZATEN tavanda ve
orada yapilan bir secim "0.50'de kal" der. Kaldirac, egitimden FARKLI markalarda
degerli ve o farki yalniz sinavda gorebiliyorum.

**Cozum ILKESEL olmali, D7'ye bakarak degil:** esigi dusurmek aday EKLER, asla
CIKARMAZ -> havuz tavanini MONOTON yukseltir. Tek risk kesinliktir ve o egitim
katlarinda olculebilir. Yapilacak deney: dusuk esikle korpusu yeniden cikar,
egitim katlarinda uctan uca ZARARSIZ oldugunu goster, sonra dagit.

---

## 5d. Bu oturumun EN ONEMLI iki dersi

**1. Gelistirme kumesinden okunan kazanc aldatir.** D6'da P6 kolu +0.1681
veriyordu; `tam` korpusunun marka katlarinda gercek kazanc **+0.0230**. Sebep:
D6'nin TABANI zayifti. Bir kol "kazandi" derken, tabanin o kumede ne kadar iyi
oldugunu da yazmak zorunlu.

**2. "KAPANDI" hukumleri sondanin kusuru olabilir.** Yon yelpazesi kolunu 64
yonlu bir sondayla olcup OLU ilan ettim. Oysa 64 yonun kure uzerindeki araligi
~25 derece, olcum toleransi 10 derece -- sonda metrigi FIZIKSEL OLARAK
tutturamiyordu. 256 yonle ayni kol NIT'te **+0.0676 recall** verdi. Bu, hafizada
"KAPANDI" diye duran kollarin bir kisminin da boyle kapanmis olabilecegi
anlamina gelir; kapatma karari verirken "sonda bu etkiyi olcebilir miydi?"
sorusu ONCE sorulmalidir.

---

## 5h. D7 SONRASI CALISMA -- ne denendi, ne cikti

| is | sonuc | karar |
|---|---|---|
| Ogrenilmis rejim yonlendirici (13 oznitelik) | kat-disi **-0.0624** | REDDEDILDI, tek esik kaliyor |
| Rejim esigi kararlilik egrisi | 50-110 bandi tepeden 0.01 icinde | esik 90 -> **60** (tepe) |
| CWT cephe teshisi | her iki kol da cokuyor; havuz p>=0.5 esiginin ALTINDA | mesh esigi kolu acildi |
| Mesh esigi taramasi | CWT konum recall 0.5532 -> **0.7914** (p>=0.05) | ilkesel gerekce yazildi, korpus yeniden cikarimi bekliyor |
| Guven kalibrasyon modeli | 0.80'de kapsama %4 -> **%9**; 0.90'da %0.35 | dagitildi; 0.90 sozu VERILEMEZ |
| Eksen boyu ornekleme (kapali kol yoklamasi) | NIT'te **+0.0034**, maliyet 1.75x | hukum DOGRUYMUS, kapali kaliyor |
| Yon yelpazesi (256 yon) | NIT'te **+0.0676** recall | hatta baglandi (`YB_FAN`), uctan uca olcum bekliyor |
| Sira damgalama (kademe2) | kosuyor | — |

**Duzeltilen uc sessiz hata:**
1. `sira` blogu her KAT x KOL icin yeniden hesaplaniyordu -> bir kosu 70
   dakikada ilerlemedi (70 dk -> 13 sn).
2. `yelpaze_yonleri` varsayilan argumani modul sabitine bagliydi; `FAN_N`
   degistirmek SESSIZCE etkisizdi.
3. `kos_p6_oznitelik`de `mesh`/`diag` kullanildiktan SONRA tanimlaniyordu --
   ilk parcada NameError, sonrakilerde BIR ONCEKI PARCANIN mesh'i.

---

## 6. Durustluk notlari -- neyin temiz OLMADIGI

1. **D6 temiz okuma degildir.** Teshis, kol secimi ve seyreltme kurali orada
   olculdu. Temiz okuma yalnizca D7'dir.
2. **Seyreltme kurali D6'ya bakilarak secildi.** Ayni olcum egitim markalarinda
   tekrarlandi: kurallar orada birbirine cok yakin (0.9254-0.9571) ve secilen
   kural en iyiden 0.0107 geride, %25 daha ucuz. Fark ancak D6'nin YOGUN
   parcalarinda aciliyor.
3. **`_tam_oz` onbellegi `cp_config.json`'un eski halinde uretildi.** Ayni
   durum dagitilan modelde de var, dolayisiyla kiyas adil; ama iki taraf da
   bugunku segmentasyon ayariyla YENIDEN turetilse sayilar degisebilir.
4. **D7 bootstrap araligi parca birimlidir.** D7 icinde kaba-iz ikiz orani
   %22.3 ve ikizler ayni markada; grup bootstrap'i ayrica gerekli gorulmedi,
   ama bu bir tercihtir.

---

## 7. GECE 2026-08-11/12 -- SIRA DAMGALAMA KAPANDI

`tam` marka katlari (WEI, PXC, SIE, TOGI), korpus `_p6_oz_u25`, 3051 parca.
Makbuzlar: `results/p6_kademe2_sira0.json` (kapali) / `sira1.json` (acik).

| kol | SIRA kapali | SIRA acik | fark |
|---|---|---|---|
| **P6** (urun kolu) | 0.309114 | 0.309114 | **0.000000** |
| P6_KAFES | 0.293754 | 0.304059 | +0.010305 |

**KARAR: KAPANDI.** Iki gerekce:

1. P6 kolunda sonuc BASAMAK BASAMAK ayni. SIRA damgalama urun koluna hic
   dokunmuyor -- damgalama yalnizca kafes kolundan giriyor.
2. Kafes kolundaki +0.0103'luk kazanc kolu P6'nin onune GECIREMIYOR
   (0.3041 < 0.3091). Yani en iyi haliyle bile urunde bir sey degistirmez.

Bu, "kazanc var ama yanlis kolda" durumunun ders niteliginde ornegi: bir kolun
kendi icinde iyilesmesi, o kol zaten geride oldugu surece urun kazanci DEGILDIR.

### Marka kirilimi (P6 kolu, disarida birakilan marka)

| marka | robot F1 | recall | kesinlik |
|---|---|---|---|
| SIE | 0.4681 | 0.8300 | 0.3259 |
| PXC | 0.3972 | 0.5981 | 0.2973 |
| WEI | 0.3795 | 0.4879 | 0.3105 |
| **TOGI** | **0.1689** | **0.1054** | 0.4240 |

TOGI, D7'deki CWT ile ayni imzayi tasiyor: recall %10.5'e cokuyor ama kesinlik
en yuksek deger (0.4240). Yani model TOGI'de "az ama dogru" buluyor --
darbogaz SECIM degil, adayin havuza HIC GIRMEMESI. Bu, ADAY_YOK kovasinin
(%36.5) marka duzeyindeki yuzu ve A1b tam-acik havuz kolunun hedefi.

---

## 8. KRITIK BULGU -- OLCTUGUMUZ ZINCIR GLB'YE GIRMIYOR

**Robotun actigi GLB, bu kampanyada olculen zinciri KULLANMIYOR.**

Kanit (2026-08-12, kod taramasi):

| dosya | zincir |
|---|---|
| `export_robot_glb.py` | `robot_cp.extract` |
| `robot_viz.py` | `robot_cp.extract` |
| `robot_cp.py` icinde `kanonik_zincir` / `urun_p6` / `urun_genis` | **hicbiri gecmiyor** |
| `kanonik_zincir`i cagiranlar | `urun_p6.py` + yalnizca SONDA/OLCUM betikleri |

`robot_cp.extract` yolu: cikarim -> `adaylari_uret` -> `wire_gate.apply`.
Kampanyanin butun kazanclari (`urun_p6` = yon bankasi + ortak siralayici,
`urun_genis` = genisletilmis havuz) `kanonik_zincir.urun_cikti` icinde ve bu
fonksiyon ihracatcilarin HICBIRI tarafindan cagrilmiyor.

### Ne anlama geliyor

- Bugun bir GLB acilsa, uzerindeki isaretler DAGITILAN TABANIN ciktisidir --
  D7'de robot F1 **0.2980**. Kampanyanin olctugu **0.3115** (ve genis havuz
  kolunun 0.2029 -> 0.3090'i) o dosyaya YANSIMIYOR.
- Yani "gercek dunyada robot GLB'yi kullandiginda F1 ne olur?" sorusunun
  bugunku yaniti, olctugumuz sayi degil TABAN sayisidir.

### Entegrasyon icin gereken (YAPILMADI -- dogrulanmadan yapilmaz)

1. `export_robot_glb.py` her model icin `pbs` listesini zaten uretiyor ama
   yalnizca ortalamasini (`acc`) tutuyor; listeyi saklayip
   `kanonik_zincir.urun_cikti(V, F, pbs, step_path, cfg)` cagrilmali.
2. **TIER SORUNU:** `tier` alani `urun_cikti` icinde DEGIL, `robot_cp.extract`
   icinde (satir ~404) atanir. `urun_cikti` ciktisi dogrudan verilirse
   ihracatci `c["tier"]` okurken KeyError alir. Tier atamasi ortak bir yere
   tasinmali.
3. Dogrulama: ayni parca icin iki yolun CP sayisi/konumu karsilastirilmali;
   entegrasyon "sessizce eski yola dusme" ile maskelenmemeli.

**Bu gece YAPILMADI.** Robotun tukettigi ciktiyi dogrulamadan degistirmek,
kampanyanin bastan beri kacindigi hatanin ta kendisi olurdu: olculmemis bir
degisikligi urun diye teslim etmek.

---

## 9. SAHA TIER KURALI -- IKI OLCUM CELISIYOR GIBI, CELISMIYOR

GLB'deki kirmizi/turuncu (auto/review) ayrimi, ihracatciya gecilen
`robot_conf_auto` / `robot_min_auto_votes` ile YAPILMIYOR. Gercek kural
`robot_cp.to_records` icinde:

    AUTO = wire_score >= cp_config.robot_auto_gate_threshold (0.66)

`conf_auto`/`min_auto_votes` yalnizca gate skoru YOKKEN (eski yol) devreye
giriyor. Yani ihracatcinin gecirdigi o iki parametre pratikte ATIL.

### Iki sayi, iki farkli soru

| olcum | kosul | sonuc |
|---|---|---|
| Kodda yazili (2026-07-29) | kilitli holdout, **marka-ayrik DEGIL** | AUTO kesinligi **0.9508**, CP'lerin %52'si otonom |
| Bu kampanya (2026-08-11) | **D7, gorulmemis MARKA** | hicbir esikte kesinlik >= 0.90 yok; en yuksek **0.6429** |

**Celismiyorlar; ayni soruyu sormuyorlar.** Ilki "gordugum markanin yeni
modelinde", ikincisi "hic gormedigim markada". Kullanicinin saha akisi IKISINI
DE iceriyor ("elimizdeki markalardan yeni model de gelir, bilmedigimiz markadan
yenisi de").

### Durust saha sozu

- **Bilinen marka, yeni model:** AUTO katmani icin ~0.95 kesinlik iddiasi
  savunulabilir, ama o rakam ESKI olcumdur ve bu korpusla YENIDEN dogrulanmali.
- **Gorulmemis marka:** 0.90 SOZ VERILEMEZ. Olculen tavan 0.6429.
- Bu yuzden `kos_saha_kapisi.py` yazildi: kesinlik-kapsama egrisini
  GORULMEMIS MARKA katlarinda cikarir (D7'yi harcamadan) ve ONAYLI esigini
  olcume baglar. Kuyrukta, korpus tamamlaninca kosacak.

**Acik kalan is:** ayni egri `wire_score` icin de cikarilmali -- sahada tier'i
belirleyen skor odur, benim olctugum P6 skoru DEGIL. Iki skor AYNI OLCEKTE
DEGILDIR; birinde olculen esigi digerine takmak sessiz bir hata olurdu.

### EK BULGU -- dagitilan esik, olculen esik DEGIL

`cp_config.json`'daki gercek degerler:

| anahtar | config | kodda/yorumda anilan |
|---|---|---|
| `robot_auto_gate_threshold` | **0.6** | 0.66 ("esik 0.66 -> AUTO kesinligi 0.9508") |
| `robot_wire_gate_threshold` | **0.4** | 0.30 (kod varsayilani) |
| `robot_conf_auto` / `robot_min_auto_votes` | 0.5 / 3 | pratikte ATIL (gate skoru varken okunmuyor) |

Yani sahada calisan AUTO esigi **0.6**, oysa 0.9508 kesinlik **0.66** icin
olculmustu. Daha DUSUK esik daha COK isareti otonom yapar ve kesinligi
DUSURUR -- bugunku AUTO kesinligi 0.9508'den az olmalidir, ne kadar az oldugu
OLCULMEMISTIR.

Bu, "0.90 saha sozu" tartismasinin sessiz kalmis parcasidir: sadece marka
kosulu degil, ESIGIN KENDISI de olcumden kaymis durumda.

---

## 10. TIER COKUSU -- "kirmizi isaretlerin kaci dogru?" (olculdu)

Makbuz: `results/tier_cokusu_d7.json` · sonda: `sonda_tier_cokusu.py`
**Yeni bir D7 okumasi DEGIL** -- harcanmis olcumun yeniden analizi; model
secimi ya da ayar yapilmadi.

### P6 zinciri, D7 (gorulmemis marka), 835 parca / 2512 isaret

| esik | AUTO | AUTO payi | kesinlik | GT kapsama |
|---|---|---|---|---|
| **0.60 (DAGITILAN)** | 2512 | **1.0000** | **0.3471** | 0.2825 |
| 0.80 | 2290 | 0.9116 | 0.3694 | 0.2741 |
| 0.90 | 1714 | 0.6823 | 0.4312 | 0.2394 |
| 0.95 | 1395 | 0.5553 | 0.4652 | 0.2102 |

Skor dagilimi: **min 0.6006**, medyan 0.9621, maks 1.0000.

**Dagitilan esik (0.6) skor tabaninin ALTINDA.** Bu yuzden esik hicbir seyi
elemiyor: REVIEW katmani BOS, isaretlerin %100'u AUTO isaretleniyor. Robot
gorulmemis bir marka parcasinda her isarete kendi basina guvenir, oysa
isaretlerin ancak **%34.7'si** dogrudur.

**Neden cokuyor:** secim kurali ile tier esigi AYNI skoru kullaniyor. Secim
zaten goreli (parca-maksimumunun %85'i) oldugundan hayatta kalan her tahminin
skoru yuksek; esik "baglamiyor". Ayni cokus 2026-07-29'da bir kez yasanmisti
(o zaman tier segmentasyon guvenine bakiyordu, REVIEW yine bos, kesinlik
0.7735). Skor degisti, COKUS BICIMI geri geldi.

**Esigi yukseltmek kurtarmiyor:** 0.95'te bile kesinlik 0.4652.

### DAGITILAN TABAN ICIN OLCUM YOK (duzeltme)

Raporun 9. bolumunde "ayni egri `wire_score` icin de cikarilmali" yazmistim;
bu YANLISTI -- P6 makbuzu zaten `wire_score` tasiyor. Ama TABAN icin gercekten
olcum yok: `d7_taban.json`'daki 2001 skorun **hepsi tam 1.0**. Sebep
`sonda_dagitim_dogrula.py`'nin `c.get("wire_score", 1.0)` varsayilani -- taban
zinciri wire_score uretmemis, sonda 1.0 yazmis. Ilk bakista "her esikte %100
AUTO" gibi gorunuyordu; bu bir BULGU DEGIL, OLCUM BOSLUGUDUR. Sonda artik bu
durumu ayirt edip `OLCULMEMIS` diye isaretliyor.

### Urun onerisi (uygulanmadi)

Gorulmemis marka icin AUTO katmani KAPATILMALI (her sey REVIEW), ta ki
parcalar arasi kalibre bir skor cikana kadar. Bugunku hali, olculmemis bir
guvenle otonom davranmaktir.

---

## 11. DUZELTME -- "sessiz olum" teshisi yanlisti

Gece boyunca uc surec cikis kodu 0 ile, tek satir hata yazmadan oldu. Ilk
teshisim **BELLEK** idi: o sirada tek bir surec 16.6 GB tutuyordu ve bos RAM
2.6 GB'a dusmustu, teshis makul gorunuyordu.

**Yanlisti.** Dorduncu vaka (NIT max_sec sondasi) 17.1 GB BOS RAM varken ayni
sekilde oldu. Ortak payda bellek degil, BASLATMA BICIMI:

| baslatma | sonuc |
|---|---|
| `run_in_background: true` olan kabuk cagrisi ICINDE `nohup ... &` | dis gorev bitince surec KAPANIYOR |
| ON PLAN kabuk cagrisindan `nohup ... & disown` | YASIYOR (kos_gece.sh, kos_ek_kuyruk.sh boyle) |

Kural: uzun kosan is ON PLAN cagrisindan ve `disown` ile baslatilir.

**Ders.** Makul bir mekanizma (bellek) ile o an gozlenen bir olgu (dusuk RAM)
ust uste geldiginde teshis "acikliyor" gibi gorunuyor. Ama aciklama ancak
KARSI ORNEKLE sinanirsa teshistir. Bu, projedeki "kapali kol" denetimiyle ayni
ders: bir aciklamayi kabul etmeden once onu YANLISLAYACAK durumu aramak gerek.

Bellek yine de gercek bir kisittir (her EK kosusu ~6 GB, makinede 31 GB) --
`kos_ek_kuyruk.sh`'nin 10 GB kapisi yerinde kaliyor. Ama gece yasanan
olumlerin sebebi o degildi.
