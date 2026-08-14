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

---

## 12. KARAR ARITMETIGI -- 0.50 NEREDEN GELEBILIR? (belirleyici)

Ayni kumede (`tam`) olculdu; makbuz `results/havuz_tavani__p6_oz_u25_tam.json`.

| | deger |
|---|---|
| havuz F1 TAVANI (mukemmel secici) | **0.8474** |
| GERCEKLESEN (P6 kolu) | **0.3091** |
| **secici verimliligi** | **%36.5** |

0.50'ye iki yol var ve biri kapali:

1. **Havuzla:** verimlilik sabit kalirsa tavanin **1.3707** olmasi gerekir.
   F1 tavani 1.0'i asamaz -> **HAVUZ KOLU TEK BASINA IMKANSIZ.**
2. **Seciciyle:** tavan sabit kalirsa verimliligin **%59.0** olmasi gerekir
   (**1.62x** iyilesme).

**Sonuc: onceligi SECICI alir.** Havuz genisletme (A1b) hala degerli --
tavani yukseltir ve gerekli verimlilik carpanini dusurur -- ama tek basina
hedefe goturmez. Bu, EK bloklarina (kanonik/topoloji/simetri/derinlik) ve
aday-kumesi modeline (D2) verilen onceligi belirler.

### MAX_SEC tavani BAGLIYOR (olculdu)

Makbuz `results/max_sec_sondasi_UPUN-SUPU30.json` (30 parca, yelpaze 256):

| secenek tavani | yonlu recall | secenek maliyeti |
|---|---|---|
| 12 (BUGUNKU) | 0.8462 | 1.00x |
| **24** | **0.9077** | 1.20x |
| 40 | 0.9077 | 1.23x |

Tavani 12'den 24'e cikarmak yonlu recall'u **+0.0615** artiriyor ve maliyeti
yalnizca **1.20x**. 40'a cikarmak hicbir sey eklemiyor -> **diz noktasi 24.**

Bu, "yelpaze olu" gorunumunun sebebini de acikliyor: 256 isin onlarca yon
uretiyor ama tavan 12 oldugu icin cogu eleniyor; yelpaze yeni yon EKLEMIYOR,
mevcut kaynaklarin yerini ALIYOR.

**ACIK SORU (kosuyor):** bu kazanc, tavani asagi ceken markada (NIT: D6
GT'sinin %45.7'si, yonlu recall 0.5254, kaybi tam olarak YON kaybi) da var mi?
Orneklem markaya gore secilmeli -- ilk kosu dosya sirasi yuzunden yalnizca
UPUN/SUPU'yu ornekliyordu.

### NIT'te tavan cok daha sert bagliyor (olculdu)

Makbuz `results/max_sec_sondasi.json` (NIT, 18 parca, 482 GT, yelpaze 256):

| secenek tavani | NIT yonlu recall | secenek maliyeti |
|---|---|---|
| 12 (BUGUNKU) | 0.5913 | 1.00x |
| **24** | **0.8755** | 1.33x |
| 48 | 0.8755 | 1.35x |

Kolay markalarda kazanc **+0.0615**; tavani asagi ceken NIT'te **+0.2842**.
Yani tavan tam da en cok kanayan yerde bagliyor -- yogun parcada bir adayin
dogru yonu, 12 kisilik listeye giremiyor.

**Kaba yansima:** NIT D6 GT'sinin %45.7'si ve bugun 0.5254'te. 0.8755'e
cikarsa D6 toplam yonlu recall **0.7264 -> ~0.886**, F1 tavani
**0.8415 -> ~0.94**. Yani KAPI A (>= 0.85) **GECMEMEKTEN GECMEYE** doner.

**Yapilan:** `_p6_oz_tam4` korpusu tavan 24 ile cikariliyor (`kos_tam4.sh`,
5 pay). Diger butun ayarlar `_p6_oz_tam3` ile birebir ayni -- tek degisken
tavan, yoksa kazanc neye ait bilinemez. Bitiminde KAPI A hem `d6` hem `tam`
kumesinde olculur.

**Not:** bu gece kosan butun B fazi olcumleri tavan-12 korpusu (`tam3`)
uzerindedir; gecerlidirler ama DAHA DUSUK bir tavanin altinda alinmislardir.

---

## 13. SONRAKI KAMPANYA -- gecenin sayilarindan cikan sira

Butun oncelikler tek bir aritmetikten cikiyor (bolum 12): **secici
verimliligi %36.5** ve havuz kolu tek basina 0.50'ye MATEMATIKSEL OLARAK
yetmiyor. Sira buna gore:

**1. Tavan-24 korpusu (KOSUYOR).** `_p6_oz_tam4`, `YB_MAX_SEC=24`. Bitince
KAPI A hem `d6` hem `tam` kumesinde olculur. Beklenti: NIT yonlu recall
0.5409 -> ~0.87, toplam 0.7347 -> ~0.88, yani KAPI A GECER. **Beklenti VAAT
DEGILDIR** -- sonda 18 parcalik bir NIT orneklemiydi.

**2. Tavan-24'un UCTAN UCA kazanci.** Tavan yalnizca TAVANI yukseltir; gercek
kazanc ancak `kos_p6_kademe2.py` tam4 uzerinde kosunca bilinir. tam3 ile
BIREBIR ayni ayarla kosulmali (tek degisken korpus).

**3. EK bloklari tam4 uzerinde tekrarlanmali.** Bu gece olculen bloklar
tavan-12 korpusundadir; gecerlidirler ama daha dusuk bir tavanin altinda.
Kapiyi gecen bloklar en iyi korpusta yeniden dogrulanmali.

**4. SECICI KAPASITESI (asil is).** %36.5 -> %59 icin 1.62x gerekiyor; bunu
oznitelik bloklari (blok basina +0.01..+0.03) tek basina veremez. Aday-kumesi
modeli (D2) tek gercek aday: adaylar ARASI baglami (kafes, dizi, rekabet)
noktasal bir siniflandirici gormuyor. Kapisi onceden ilan edildi: LOMO'da
HGB'ye **+0.05**.

**5. SAHA (urunun dogrudan isi).** Iki is birbirinden bagimsiz:
   - GLB'yi `kanonik_zincir.urun_cikti`ya baglamak (bolum 8; tier alani ortak
     yere tasinmali).
   - AUTO katmani: gorulmemis markada kesinlik 0.3471 ve REVIEW BOS. Kalibre
     bir skor cikana kadar AUTO **kapatilmali**; bugunku hali, olculmemis bir
     guvenle otonom davranmaktir.

### Gecenin ozeti -- ne DEGISTI, ne DEGISMEDI

**Degismedi:** manset robot F1 hala **0.3091** (`tam` katlari). Bu gece
dagitilan urune giren bir iyilestirme YOK.

**Degisti:** artik hedefin nereden gelebilecegi OLCULU. Havuz kolunun tek
basina yetmedigi, tavanin nerede bagladigi, sahadaki guven katmaninin atil
oldugu ve olculen zincirin GLB'ye hic girmedigi -- dordu de bu gece olculdu.
Bunlarin ucu (bolum 8, 10, 12) sayi degil, YON degistiren bulgulardir.

### Tek satirda tez: kayip KONUM degil YON

`tam` kumesi, onceki havuz (makbuz `havuz_tavani__p6_oz_u25_tam.json`):

| olcu | deger |
|---|---|
| konum recall | **0.9789** |
| yonlu recall | **0.7352** |

GT'nin %97.9'unun KONUMU havuzda; yalnizca %73.5'inin dogru YONU da var.
Aradaki **0.244**, tamamen yon kaybidir. Ve yon secenekleri aday basina 12
ile sinirli, doygun halde (bkz. tavan doygunlugu uyarisi). Havuzu genisletmek
bu farki kapatmaz -- nitekim tavan-12 tam-acik havuzda konum 0.8713 -> 0.8997
cikarken yon yalnizca 0.7264 -> 0.7347 oynadi.

---

## 14. OLCUM TASARIMI UYARISI -- B1/B6 tek degiskenli DEGIL

Orkestrator `B1_zor_negatif`i **`tam3` korpusunda VE zor-negatif acikken**
kosuyor. Elimizdeki referans (0.3091) ise **`u25` korpusunda ve zor-negatif
KAPALI**. Iki degisken ayni anda degisiyor: bir fark cikarsa KORPUSA mi
YONTEME mi ait, ayirt edilemez.

Bu, projenin defalarca yakalandigi hatanin ta kendisidir (bkz. bolum 5d:
"gelistirme kumesi aldatir" -- orada da tabanin ne verdigi yazilmamisti).

**Cozum (kuyruga eklendi):** `tam3` uzerinde DUZ ayarla bir taban kosusu
(`results/p6_kademe2_tam3_taban.json`). Boylece:

    taban(u25)  -> taban(tam3)   = KORPUS etkisi
    taban(tam3) -> B1(tam3)      = ZOR NEGATIF etkisi

iki etki AYRI okunur. Taban kosusu olmadan B1/B6 sayilari raporlanabilir ama
**yorumlanamaz**; makbuzlari o yuzden "tek degiskenli degil" notuyla okunmali.

---

## 15. B1 (ZOR NEGATIF) SONUCU -- +0.0104 ama TEK DEGISKENLI DEGIL

Makbuz `results/p6_kademe2_B1_zorneg.json` (korpus `tam3`, `tam` marka katlari).

| | u25 taban | B1 (tam3 + zor negatif) | fark |
|---|---|---|---|
| **robot (MIKRO)** | 0.3091 | **0.3195** | **+0.0104** |
| recall | 0.2935 | 0.2771 | -0.0164 |
| kesinlik | 0.3265 | 0.3772 | +0.0507 |

Kirilim, zor-negatif egitiminin BEKLENEN imzasini tasiyor: kesinlik belirgin
yukseliyor (+0.0507), recall bir miktar dusuyor -- daha az ama daha isabetli
tahmin.

### Marka kirilimi (P6 kolu)

| marka | u25 taban | B1 | fark |
|---|---|---|---|
| PXC | 0.3972 | 0.4433 | **+0.0461** |
| TOGI | 0.1689 | 0.1856 | +0.0167 |
| WEI | 0.3795 | 0.3726 | -0.0069 |
| SIE | 0.4681 | 0.4504 | -0.0177 |

### UYARI -- bu sayi henuz YORUMLANAMAZ

Iki degisken ayni anda degisti: **korpus** (u25 -> tam3, tam-acik havuz) ve
**yontem** (zor negatif). +0.0104'un hangisinden geldigi bilinmiyor. Kuyruktaki
`tam3` TABAN kosusu (duz ayar, ayni korpus) ikisini ayiracak:

    taban(u25) -> taban(tam3)  = KORPUS etkisi
    taban(tam3) -> B1(tam3)    = ZOR NEGATIF etkisi

Taban kosusu gelene kadar B1 **raporlanabilir ama kola sayilamaz**. Bu
kampanyanin kurali: bir kol ancak TEK DEGISKENLI olcumle acilir.

---

## 16. BELIRLEYICI OLCUM -- tavan 24, YONLU recall +0.1665 (esli kiyas)

Makbuz `results/esli_tavan_d6.json` · sonda `sonda_esli_tavan.py`.
**359 ORTAK parca**, tek degisken `YB_MAX_SEC` (12 -> 24).

| | tavan 12 | tavan 24 | fark |
|---|---|---|---|
| konum recall | 0.8988 | 0.8988 | **+0.0000** |
| **yonlu recall** | 0.7270 | **0.8935** | **+0.1665** |
| F1 tavani | 0.8419 | 0.9437 | **+0.1018** |
| secenek/parca | 3217 | 4857 | 1.51x |

**Konum recall'un BASAMAK BASAMAK ayni cikmasi**, bu olcumun en guclu yani:
tavan konumlara dokunmuyor, kazancin TAMAMI yonden geliyor. Teorinin
ongordugu tam olarak buydu.

**0.8935 > 0.85 -> KAPI A GECER.**

### Neden ESLI kiyas sart oldu

tam4 korpusu yarim (359/468) ve biten parcalar RASTGELE DEGIL -- once biten,
yani daha kucuk/kolay parcalar. Yarim tam4'un ham olcumu 0.8952 idi; bunu tam
korpusun 0.7347'siyle kiyaslamak farkin ne kadari TAVANDAN ne kadari KOLAY
ALT KUMEDEN geldigini gizlerdi. Esli kiyas iki korpusu da AYNI 359 parcada
olcer, alt kume etkisi ikisinde de ayni olur ve geriye yalniz tavan kalir.

(Yarim korpusta yazilmis makbuzlar `KISMI_` onekiyle ayrildi; sabah raporunun
tarama desenine artik girmiyorlar.)

### Karar aritmetigi GUNCELLENDI

| | onceki | tavan 24 ile |
|---|---|---|
| havuz F1 tavani | 0.8474 | **~0.94** |
| 0.50 icin gereken secici verimliligi | %59.0 | **%53.2** |
| gereken iyilesme carpani | 1.62x | **1.46x** |

Havuz kolu hala TEK BASINA yetmiyor (0.94 x %36.5 = 0.343), ama gereken
secici iyilesmesini 1.62x'ten 1.46x'e indiriyor. **Oncelik hala SECICI**,
fakat tavan-24 artik dagitilmasi gereken bir kazanc.

### Kalan is

Bu bir TAVAN olcumu; **uctan uca kazanc DEGIL**. Tavan yukselmesi ancak
secici o yonleri SECEBILIRSE F1'e doner. Sirasiyla: (1) tam4 korpusunu
tamamla, (2) KAPI A'yi tam korpusta olc, (3) `kos_p6_kademe2.py`'yi tam4 ve
tam3 uzerinde AYNI ayarla kosup uctan uca farki al.

---

## 17. KAPI A GECTI -- tavan-24 korpusunda yonlu recall 0.8926

Makbuz `results/havuz_tavani__p6_oz_tam4_d6.json` (464/468 d6 parcasi).

| marka | GT | konum | **yonlu (t12)** | **yonlu (t24)** | F1 tavani |
|---|---|---|---|---|---|
| **NIT** | 1222 | 0.8429 | 0.5409 | **0.8429** | 0.9147 |
| SUPU | 547 | 0.9506 | 0.8921 | 0.9214 | 0.9591 |
| UPUN | 370 | 0.9649 | 0.8811 | 0.9622 | 0.9807 |
| MOR | 274 | 0.8686 | 0.8686 | 0.8686 | 0.9297 |
| UTL/SE/ONV/S+S | 238 | 1.0000 | 0.92-1.00 | 1.0000 | 1.0000 |
| **TOPLAM** | 2651 | 0.8989 | **0.7347** | **0.8926** | **0.9432** |

**KAPI A GECTI (0.8926 >= 0.85).** Havuz F1 tavani **0.8470 -> 0.9432**.

### En anlamli satir NIT

NIT'in yonlu recall'u **konum recall'una ESITLENDI** (0.8429 = 0.8429). Yani
tavan 24 iken, havuzda konumu bulunan HER GT'nin dogru yonu de havuzda.
NIT'te yon darbogazi **tamamen kapandi** -- gecen olcumde bu marka 0.5409'da
ve toplam tavani tek basina asagi cekiyordu.

Geriye kalan kayip artik saf KONUM kaybi (%10.1) ve o baska bir kol.

### Duzeltilen yaniltici uyari

Ilk kosuda "TAVAN DOYGUN" uyarisi tetiklendi (17.1 secenek/aday). YANLIS
ALARMDI: uyari surecin kendi `MAX_SEC` varsayilanina (12) bakiyordu, oysa
korpus 24 ile kurulmustu -- 17.1, 24'un %71'i, doygun degil. Korpusun
kuruldugu tavan npz'de yazili olmadigi icin artik `HT_KORPUS_MAXSEC` ile
verilir ve uyari hangi tavana gore konustugunu YAZAR. Bir makbuz logundaki
yaniltici uyari, sonradan yanlis kola yatirim yaptirir.

---

## 18. SAHA KAPISI, `tam` KATLARINDA -- 0.70 KESINLIK BILE YOK

Makbuz `results/saha_kapisi_tam.json` (3051 parca, `tam` marka katlari =
gorulmemis marka kosulu, 35326 tahmin / 14300 GT).

| hedef kesinlik | ulasilan esik |
|---|---|
| 0.70 | **ULASILMIYOR** |
| 0.80 | ULASILMIYOR |
| 0.90 | ULASILMIYOR |
| 0.95 | ULASILMIYOR |

Ham kesinlik **0.1711** (kural `goreli 0.50/0.05` -- bilerek GENIS tutuldu,
daraltmayi esigin yapmasi icin).

**D7'deki bulguyu bagimsiz bir kumede dogruluyor:** orada en yuksek kesinlik
0.6429 idi; burada 0.70'e bile ulasilamiyor (kural daha genis oldugu icin
tahmin sayisi 2.5 kat).

**Sonuc:** gorulmemis markada "robotun otonom davranabilecegi" bir isaret alt
kumesi BU SKORLAYICIYLA YOK. `robot_auto_kapali` anahtarinin (S6b) gerekcesi
artik IKI bagimsiz kumede olculu.

---

# BOLUM 19 — 2026-08-12: DUVARIN YERI BULUNDU

## 19.1 Gunun net sonucu

**Uctan uca kazanc: +0.0093** (kanonik blogu). Baska hicbir kol uctan uca
kazandirmadi. d6 mikro robot F1 taban **0.2954**.

Gunun asil urunu sayi degil, **duvarin koordinati**.

## 19.2 Duvar nerede

`results/konum_auc_d6.json` — secenek duzeyi AUC'nin konum ve yon paylari
AYRI olculdu:

| marka | n_konum | secenek AUC | **konum AUC** | **yon AUC** | ilk-k | rastgele |
|---|---|---|---|---|---|---|
| NIT | 432 | 0.8854 | **0.7053** | **0.8899** | 0.1135 | 0.0670 |
| MOR | 497 | 0.9657 | 0.9003 | 0.9293 | 0.2446 | 0.0277 |
| SUPU | 131 | 0.9696 | 0.9121 | 0.9596 | 0.5192 | 0.0544 |
| UPUN | 161 | 0.9877 | 0.9419 | 0.9771 | 0.6490 | 0.0615 |

Model yogun parcada **yonu biliyor, hangi acikligin kablo girisi oldugunu
bilmiyor.** Uctan uca NIT F1 = **0.0089**; NIT GT'nin %51'i.

**Hedefin tam sayisi:** gereken konum AUC = `1 − k/n_konum`
→ NIT 0.944 (0.705 var) · MOR 0.982 (0.900 var).
Iki yol: AUC'yi yukseltmek **veya havuzu ~5× kucultmek**.

## 19.3 Olculen ve DUSEN kollar (hepsi kapi ONCE ilan edilerek)

| kol | kapi | olculen | hukum |
|---|---|---|---|
| yerel karsitlik (p − yerel ortanca) | NIT AUC +0.05 | −0.033 | DUSTU |
| konum toplama (max→ortanca/ort/q75/say) | +0.01 mikro | −0.014 (en iyi) | DUSTU |
| dik-yon kisiti | bugunkuyu asmak | NIT +0.017, digerlerinde ters | DUSTU |
| A1 isin atma / tup skoru | NIT ≥0.40 | tup 0.0115, **tup_eksen 0.0237** | DUSTU |
| yayilim (rejim kapili) | uctan uca | **−0.0645** | DUSTU |
| B1 bimodal `parca_eksen` | NIT +0.05 | **+0.1277** (yon) | GECTI, uctan uca ~+0.002 |
| kanonik blogu | +0.01 | **+0.0093** | TEK KAZANC |

**A1 notu:** sentetik yetenek testini GECMISTI (uc eksende 0.0 derece sapma,
tup 8.45 / zemin 1.00) ama gercek geometriye tasinmadi — isaretsiz eksen
bile 0.0237. Mekanizma dogru, uygulama alani yanlis.

**Yayilim notu:** izole olcumde +0.0394 gorunuyordu; uctan uca **−0.0645**.
Izole olcum uctan uca yerine GECMEZ.

## 19.4 Yakalanan UC olcum kusuru (hepsi kendi sonucumu duzeltti)

1. **Oklid vs CARPIM kutusu.** Aday-GT eslesmesini Oklid 2mm ile yapmak
   `kahin`i 0.593 → 0.0172 dusuruyordu. Kutu carpimdir: yanal ≤2mm **ve**
   eksenel ≤40mm. *(Bugun ikinci kez.)*
2. **GT konumu sizintisi.** "Disari" isaret kurali GT konumunu kullaniyordu;
   aday konumuna cevirince `eksen_disari` **0.5336 → 0.2668**. Gorunen
   kazancin yarisindan fazlasi sizintiydi.
3. **Denominator hatasi.** Havuz sondasinda marka basina kucultme, marka
   parca sayisina degil global parca sayisina bolunuyordu.

## 19.5 Curutulen onerme

Gun ortasinda yazilan `docs/PLAN_075_YON_CEPHESI.md` "0.75 = yon problemi"
diyordu. O onerme **secenek sayisindan turetilmisti** (6322 = 260 konum ×
24 yon), olculmemisti. Dogrudan olcum tersini soyledi. Plan basina uyari
konarak birakildi.

**Ders:** secenek sayisindan konum/yon payini TAHMIN ETME, OLC.

## 19.6 Saha baglantisi (E2)

`kanonik_zincir.urun_cikti` saglamlik denetimini GECTI (40/40 parca cikti,
cokme yok, NaN yok, yonler birim). **Ama 181 GT icin 309 CP uretiyor** —
fonksiyonel karsiligi robotun olmayan yerlere gitmesi. Bayrak
`glb_kanonik_zincir` taban zincirle esli kiyas yapilmadan ACILMADI.

## 19.7 GELISTIRME TURU — dogrulanmis kazanc

Olcum fasli kapatildi; yalniz F1'i yukselten kollar kosuldu
(`kos_gelistirme_taramasi.py`, 20 yapilandirma, hepsi UCTAN UCA).

**Kazanan tek eksen: NEGATIF ORANI.** Sistematik HPO ilk kez yapildi.

| neg orani | d6 F1 | fark |
|---|---|---|
| 3 | 0.2549 | −0.0498 |
| 6 (taban) | 0.3047 | — |
| **12** | **0.3135** | **+0.0088** |
| 18 | 0.3089 | +0.0042 |
| 24 | 0.3121 | +0.0074 |
| 36 | 0.3060 | +0.0013 |

12–24 arasi DUZ PLATO — sivri tepe degil, yani d6 gurultusu degil.

**`tam` DOGRULAMASI (1617 parca, BES marka kati WEI/PXC/TE/SIE/TOGI):**
taban 0.3699 → neg=12 **0.3753 (+0.0054)**.
d6'da secilip `tam`'da dogrulandi; d6'ya ezberleme YOK.

**Kapanan uc kol (hepsi ilk kez denendi):**
- parca-esitleyici agirlik **0.1927 (−0.1120)** — NIT'in %51'lik baskinligi
  egitime zarar degil FAYDA veriyormus
- zor negatif (skor-yakin secim) **−0.0195**
- ogrenme hizi / yaprak / L2: notr ya da zararli

## 19.8 Gunun kapanis defteri

| kalem | deger |
|---|---|
| kanonik blogu (`tam`) | **+0.0151** |
| negatif orani 12 (`tam`) | **+0.0054** |
| **dogrulanmis kumulatif** | **~+0.0205** |
| D7 okuma kapisi (+0.10) | GECILMEDI — D7 OKUNMADI |

**Sahaya inme durumu:** bu kazanclar P6 zincirinde; ihracatcilar
`robot_cp.extract` cagiriyor. Olculen zincir saglamlik denetimini GECTI
(40/40 parca, cokme/NaN yok, yonler birim) ama **181 GT icin 309 CP**
uretiyor. Kesinlik sorunu cozulmeden bayrak ACILMADI.

## 19.9 GECE TURU — yeni bilgi kollari ve D1/D2

Taban: temel + kanonik + neg12 = **0.3135** (d6).

**L1 hedef fonksiyonu** (`results/siralama_dizi_d6.json`):

| kol | F1 | fark |
|---|---|---|
| yerel negatif (parca-ici ornekleme) | 0.3168 | +0.0034 |
| lambda agirligi (parca-ici sira hatasi) | 0.3159 | +0.0024 |
| dizi uyeligi | 0.3150 | +0.0016 |
| dizi + yerel | 0.3123 | −0.0012 |

**L3/L4/L5 yeni bilgi** (`results/yeni_konum_d6.json`):

| kol | F1 | fark |
|---|---|---|
| **ayna esi** | **0.3210** | **+0.0075** |
| isin-temas (CONTACT'a varan isin) | 0.3178 | +0.0043 |
| hepsi | 0.3172 | +0.0038 |
| vida cifti | 0.3150 | +0.0015 |
| ayna + temas + yerelneg | 0.3192 | +0.0058 |
| ayna + yerelneg | 0.3125 | −0.0010 |
| ayna + temas | 0.3085 | −0.0050 |

**HICBIRI +0.01 KAPISINI GECMEDI.** Ayna esi (+0.0075) en guclu yeni
sinyal ve dogrudan bimodal bulgusundan turedi, ama kapinin altinda.
Kapiyi indirmek ya da gecene kadar kume degistirmek YAPILMADI.
Birlesimler de kazandirmiyor -- oznitelik seyrelmesi.

**D1 SENTETIK KORPUS — URETEC CALISIYOR, BORU HATTI KABUL ETMIYOR.**
Parametrik klemens uretildi (2-30 kutup, egimli giris, cift sira, sikma
vidasi, GT'ye GIRMEYEN celdiriciler). Duman testi gecti. Ama dondurulmus
segmentasyon modeli sentetik geometride **ates(le)miyor**:

| | GT agzinda | rastgele yuzey | oran |
|---|---|---|---|
| duz delik | 0.0001 | 0.0002 | 0.52x |
| + havsa + ic kamara | 0.0003 | 0.0003 | **0.92x** |

Kontrol: ayni kod yolunda GERCEK parca maks CE+CT 0.4985, sentetik 0.2318
-> kusur kodda DEGIL. Model sentetikte yanlis yerde degil HIC ateslemiyor
= girdi dagilimi kaymasi. `domain-gap-is-the-blocker`'in baska ornegi.
D1 ancak segmentasyon sentetikle BIRLIKTE yeniden egitilirse ise yarar.

**D2 SIE — TUKENMIS.** `_ds1`'de 331 SIE dosyasi var ama **309'u zaten
korpusta**; yeni olan yalnizca **22 parca**. Onceki "SIE 310 dokunulmamis"
notu yanlisti, duzeltildi. Kiyas: +110 WEI -> +0.0391; 22 parca ihmal
edilebilir.

**Geriye kalan tek canli veri kolu:** `_p6_oz_tam4` cikariminin bitmesi
(966 parca bosta duruyordu; egitim verisi %60 artacak).

## 19.10 URETIM TABANI DUZELTMESI ve DAGITIM

**Yakalanan kusur.** Gelistirme taramasinin tabani `neg=6` idi, ama URETIM
egiticisi (`kos_p6_kademe2.py`) `P6_NEG_KAT` varsayilani **8** kullaniyordu.
Yani 6'ya gore olculen +0.0088, dagitilacak sayi DEGILDI. Dogru taban
olculdu:

| kume | neg=8 (URETIM) | neg=12 | fark |
|---|---|---|---|
| d6 (468 parca, 4 kat) | 0.2994 | 0.3135 | **+0.0140** |
| **tam (2040 parca, 5 kat)** | **0.3092** | **0.3218** | **+0.0126** |

**DAGITILDI:** `kos_p6_kademe2.py` NEG_KAT 8 → 12. Canli model
`results/p6_kademe2_model.pkl.oncesi_neg12` olarak yedeklendi. Yeniden
egitim, `_p6_oz_tam4` cikarimi bitince TEK seferde yapilacak (korpus hala
buyuyor; simdi egitmek iki degisikligi birbirine karistirirdi).

**GURULTU.** neg egrisi 12-24 arasi duz plato (0.3121-0.3135) ve 6 (0.3047)
ile 8 (0.2994) TERS donuyor -> kat gurultusu ~±0.005. +0.0126'nin
belirsizligi gercektir ve manset verilirken yazilmalidir.

**MUTLAK DEGERLER KOSU ARASI KIYASLANAMAZ.** `tam` 1617 → 2040 parcaya
buyudu ve eklenen parcalar DAHA ZOR (ayni kol 0.3699 → 0.3092). Yalniz
kosu-ici farklar gecerlidir.

## 19.11 KANONIK BLOK URETIMDE YENIDEN URETILMEDI — DAGITILMADI

Gun boyunca "gunun tek gecen kolu" sayilan kanonik hizalama blogu, URETIM
egiticisinde A/B kosuldu (`logs/kanon_ab_0.log` / `_1.log`, d6):

| kol | kanonik KAPALI | kanonik ACIK |
|---|---|---|
| **P6 (SECILEN kol)** | **0.2932** | **0.2794 (−0.0138)** |
| P6_KAFES | 0.2627 | 0.2475 |
| P6_GEO | 0.2742 | 0.2815 (+0.0073) |

Olcum betiklerinde `tam` katlarinda **+0.0151** veren kol, dagitilacak kod
yolunda **kaybettiriyor**. Sebep kural secimi: olcum betigi SABIT
`('goreli', 0.85, 0.20)` kullaniyordu; uretim kat icinde kural ARIYOR ve
`('mutlak', 0.97)` seciyor. Yani olculen zincir ile dagitilacak zincir AYNI
DEGILDI.

**DAGITILMADI.** `P6_KANONIK` varsayilani 0; kod duruyor.

**DERS:** bir kol "gecti" denip dagitilmadan once DAGITILACAK KOD YOLUNDA
yeniden olculur. Ayri bir olcum betigindeki kazanc, uretimde de
kazandiracaginin kaniti degildir.

## 19.12 GUNUN DUZELTILMIS BILANCOSU

| kalem | durum |
|---|---|
| negatif orani 8→12 | **DAGITILDI**, `tam` 2040 parca / 5 kat: **+0.0126** |
| kanonik blok | olcumde +0.0151, URETIMDE −0.0138 → **DAGITILMADI** |
| ayna esi | +0.0075, kapi (+0.01) GECILMEDI → dagitilmadi |
| isin-temas | +0.0043 → dagitilmadi |
| yerel negatif / lambda / dizi | +0.0034 / +0.0024 / +0.0016 → dagitilmadi |
| D1 sentetik | uretec calisiyor, segmentasyon kabul etmiyor (0.92x) |
| D2 SIE | tukenmis (yeni 22 parca) |

**GUNUN TEK DAGITILAN KAZANCI: +0.0126** (gurultu ~±0.005).
Sabah "+0.0205" denen rakam, kanonik dusunce **duzeldi**.
D7 OKUNMADI (kapi +0.10; 2 okuma kaldi).

# BOLUM 20 — 2026-08-13 GECESI: SEGMENTASYON CEPHESI

## 20.1 Neden segmentasyon

+0.30 aritmetigi tek yere cikiyor: NIT tipi yogun parcalar GT'nin %51'i ve
uctan uca **0.0089** uretiyor. Mikroyu +0.30 oynatmak NIT'in tek basina
~0.60'a cikmasini gerektirir.

Ve kampanya boyunca SECICI optimize edildi; altindaki segmentasyon
**dondurulmus** kaldi. Canli kontrol noktalari **27 Temmuz** tarihli ve
kontrol kolu yalnizca **189 parca** goruyor (118 kismi insan etiketi).

## 20.2 SEG-1: GT'den kismi etiket (YAPILDI)

`kos_gt_kismi_etiket.py` — uretici GT'sinden korpus olceginde kismi
segmentasyon etiketi: **1556 parca**, isaretli tepe orani **%0.94**
(elle etiketli korpusta ~%1.5 -- ayni mertebe).

DONGUSEL DEGIL: etiketler model ciktisindan degil URETICI JSON'undan.

**SIZINTI BEKCISI:** yalniz `tam` parcalari boyandi; **835 d6/d7 parcasi
disarida**. Bolme, oznitelik dosyalarinin onekinden okundu -- ilk yazimda
`d6_kayit.yukle()` kullanmistim, o bir GENEL DEPO (6074 kimlik) ve bekci
3418 parcanin hepsini eleyip HIC etiket uretmemisti.

## 20.3 SEG-2: A/B egitimi (KOSUYOR)

| kol | etiket dizinleri | parca | epoch |
|---|---|---|---|
| A (kontrol) | elle etiketli 5 dizin | 189 | 200 |
| B (deney) | + `_label_targets_gt` | ~1745 | 40 |

**Epoch secimi durustce ASIMETRIK:** B'yi 200 epoch kosmak ~16 saat
surerdi. 40 x 1745 = 69.800 ornek, A'nin 200 x 189 = 37.800 ornegine gore
~1.85 kat. B kendi recetesine gore AZ egitilmis; kiyas B'nin ALEYHINE
egimli. B yine de kazanirsa kanit guclu, kaybederse BELIRSIZ.

## 20.4 GECENIN OLCUM DERSI: taze cikarim onbellegi yeniden uretmiyor

Yeni kontrol noktalarini degerlendirmek icin elle `D.predict` cagrisi
yazdim. Uc sonuc uretti ve **UCU DE GECERSIZ CIKTI**:

- "yeni A kontrol noktasi NIT'te 4.87x" (canliya karsi 1.19x)
- "onbellek bayat; tazelemek kazandirir"
- kontrol noktasi basina AUC tablosu

**Kanit:** ayni parcada, canli dort kontrol noktasindan alinan taze
olasiliklarin `_p1_olasilik` onbellegiyle korelasyonu **~0**
(−0.05 … 0.26). Farkli ama gecerli bir model olsaydi korelasyon yuksek
olurdu. `op_cache_dir` vermek sonucu degistirmedi.

**Kural:** yeni segmentasyon kontrol noktasi elle `D.predict` ile
DEGERLENDIRILMEZ. Degerlendirme, urunun olasilik uretme yolundan gecip
`_p1_olasilik`-benzeri bir dizine yazilmali ve mevcut sondalarla
okunmalidir. Ayni aile: 19.11 (kanonik) ve `glb-olculen-zinciri-kullanmiyor`.

**Segmentasyon A/B EGITIMI bu sorundan ETKILENMEZ** -- kendi boru hattini
kullanir; etkilenen yalniz degerlendirme yolu.

## 20.5 Yan bulgular

- **B-rep NIT'te VAR** (50/50 parca, ort 119.6 silindir). "NIT'te B-rep yok"
  hipotezi YANLIS; silindirler mevcut ama CP'lerde degil (0.011) --
  muhtemelen vida delikleri ve pimler.
- **`_p1_olasilik` onbellegi 6 Agustos tarihli ve 2 MODEL iceriyor**, canli
  yapilandirma ise 4 kontrol noktasi kullaniyor. Tum P6 oznitelik zinciri
  bu onbellekten besleniyor. Etkisi 20.4 yuzunden HENUZ OLCULEMEDI.
- **`_p6_oz_tam4` cikarimi BITTI** (2560 dosya). Uretim modeli tam+d6
  korpusuyla ve neg=12 ile yeniden egitiliyor.

## 20.6 SEG-2 SONUCU: GT KORPUSU KOLU DUSTU

| kol | egitim verisi | epoch | val Conn_IoU |
|---|---|---|---|
| A (kontrol) | 189 parca (118 kismi insan) | 200 | **0.6232** |
| B (+GT korpusu) | 689 parca (618 kismi) | 60 | **0.5540** |

**B −0.0692 ile KAYBETTI.** Uretici GT'sinden turetilen kismi etiketler
segmentasyonu iyilestirmedi.

**DURUST KAYITLAR:**
- B bitiste hala TIRMANIYORDU: 0.3574 → 0.4982 → 0.5412 → 0.5540.
  A'nin 200 epoch'una karsi 60 epoch gordu. Hesap dengeliydi (41.340 vs
  37.800 ornek) ama EPOCH asimetrikti -> "daha uzun kosuda gecebilirdi"
  ihtimali ELENMIS DEGIL.
- Ilk deneme (1556 parca) operatör hazirliginda oldu; 500 parcalik YOGUN
  alt kume ile tekrarlandi (alt kume rastgele degil, isaretli tepe
  sayisina gore secildi -- duvar yogun parcalarda).
- **Muhtemel kok neden:** etiketler GT noktasi cevresine 2mm KURE
  boyanarak uretildi. Gercek CableEntry bolgesi kure degil, deligin AGIZ
  YUZEYIDIR. Model yanlis SEKIL ogreniyor olabilir. Bir sonraki deneme
  agiz yuzeyini (yerel normal + delik yaricapi kapili) boyamali.

**TUZAK NOTU:** izleme dongusunde `pgrep -f train_seg_extra` kullandim ve
"surec oldu" yanlis negatifi verdi (Git Bash Windows sureclerini gormuyor;
bkz. `ps-aux-yanlis-negatif`). PowerShell `Get-CimInstance` ile dogrulandi:
surecler ayaktaydi. Bu yuzden bir kol gereksiz yere yeniden baslatildi.

## 20.7 E2 SAHA BAYRAGI ACILDI — olculen zincir artik sahaya iniyor

**Sorun.** Ihracatcilar `robot_cp.extract` cagiriyordu; kampanyada olculen
`urun_p6`/`urun_genis` zinciri sahaya HIC girmiyordu. Yani olculen her
kazanc robota ULASMIYORDU (`glb-olculen-zinciri-kullanmiyor`).

**Esli kiyas** (`sonda_zincir_esli_kiyas.py`, 80 parca / 7 marka,
STEP'ten TAM zincir, AYNI parcalarda yan yana):

| yol | robot F1 | kesinlik | recall | uretilen CP |
|---|---|---|---|---|
| saha (`robot_cp.extract`) | 0.1309 | 0.1592 | 0.1111 | 289 |
| **olculen (`urun_cikti`)** | **0.1677** | **0.5467** | 0.0990 | **75** |

**F1 +0.0368 · KESINLIK +0.3875.** Recall hafif DUSTU (0.1111 → 0.0990)
ve bu yazilir.

Kural ONCE ilan edilmisti: *F1 ARTTI **VE** kesinlik 0.02'den fazla
GERILEMEDI.* Ikisi de saglandi -> `cp_config.glb_kanonik_zincir = true`.

**Fonksiyonel anlami** (kullanicinin sordugu sey): robot 289 yerine **75**
noktaya gidiyor ve yarisindan fazlasi DOGRU; eskiden 6'da 1'i dogruydu.
Yanlis CP = robotun bos yere hareketi, dolayisiyla kesinlik artisi sahada
F1 artisindan daha degerlidir.

Geri alma: `cp_config.json.oncesi_glb_zincir`.

## 20.8 SEG GT-ETIKET KOLU KAPANDI

| kol | etiket sekli | egitim parcasi | val Conn_IoU |
|---|---|---|---|
| A (kontrol) | — | 189 | **0.6232** |
| B | GT noktasi cevresi 2mm KURE | 689 | 0.5540 |
| C | GT EKSENI etrafinda SILINDIRIK KABUK | 689 | **0.5511** |

**Kok-neden hipotezim YANLIS cikti.** "Kure yanlis sekil, kanal yuzeyi
dogru" dedim; sekli duzeltince sonuc DEGISMEDI (0.5540 → 0.5511).
Demek ki sorun etiket SEKLI degil. Kol kapaniyor; yeniden acmak icin
YENI bir gerekce gerekir.

## 20.9 Y19 GIRDI OZNITELIGI (xyz vs hks) — DUSTU

`train_seg_extra.py` cfg'yi SABIT `input_features="xyz"` yaziyordu; `hks`
DiffusionNet'te destekli oldugu halde hic denenemiyordu. Bayrak eklendi
(`--input-features`), tek degiskenli kosuldu.

| kol | girdi | val Conn_IoU |
|---|---|---|
| A (kontrol) | xyz (3 kanal, DISSAL) | **0.6232** |
| Y19 | hks (16 kanal, ICSEL) | **0.1991** (ep 99, DURDURULDU) |

Kayip 8. epoch'tan beri kipirdamadi (1.6396 → 1.6062): model HKS'ten
OGRENEMIYOR. Hipotez makuldu (icsel oznitelik gorulmemis markada daha iyi
genellemeli) ama olcum tersini soyledi. Kol 99/200'de durduruldu -- duz
kayip ve 0.20 vs 0.62 farki, kalan 100 epoch'ta kapanacak bir aciklik
degil.

**Bayrak KODDA KALDI** (`--input-features`), varsayilan `xyz`.

## 20.10 Y1 HAFIF AUGMENTASYON — GECTI (+0.0197 seg IoU)

| kol | augmentasyon | val Conn_IoU |
|---|---|---|
| A (kontrol) | yok (`augment=False`) | 0.6232 |
| **Y1** | **hafif, 0.3 rad (~17 derece)** | **0.6429** |

**+0.0197.** Kampanyanin segmentasyon tarafindaki ILK kazanci.

**Mekanizma ogretici:** `--augment` bayraginin yardim metni "1.05 =
aggressive (the heuristic-tuned one that hurt quality)" diyor -- yani
AGRESIF donme (~60 derece) denenmis ve DUSURMUS. Kol "olu" degilmis,
AYARI yanlismis. Bu, `KAPANAN_KOLLAR_DENETIMI`'ndeki desenin bir ornegi
daha: kapatma hukmu sondanin/ayarin kusuru olabilir.

**HENUZ DAGITILMADI.** Bu bir segmentasyon IoU kazancidir, robot F1 degil.
Kanonik dersi (Bolum 19.11) aynen gecerli: olcum yerinde kazanan kol,
dagitilacak yolda kaybedebilir. Once uctan uca dogrulanacak.

Kontrol noktasi: `results/seg_extra/y1_aug03_s0.pt`

# BOLUM 21 — SUNUM TABANI (temiz VAL kumesi, bugunku sistem)

## 21.1 Kume neden VAL

`results/split3.json` -> VAL, 100 parca. Olculdu:
**VAL ∩ tam = 0, VAL ∩ d6 = 0, VAL ∩ d7 = 0.**
Yani hicbir egitim ya da sinav kumesiyle kesismiyor -- bagimsiz holdout.
(DEV KULLANILMAZ: 100 parcasinin **37'si** secicinin egitim kumesinde.)

## 21.2 UC METRIK, ayni parcalar, ayni zincir

| metrik | saha zinciri (dagitilan) | olculen zincir |
|---|---|---|
| **tespit F1** (konum, aci serbest) | **0.7878** | 0.6101 |
| robot F1 **ISARETSIZ** (eksen) | 0.5764 | 0.5376 |
| robot F1 **ISARETLI** (fiziksel) | **0.4839** | 0.4668 |
| kesinlik | 0.5008 | 0.5605 |
| recall | 0.4682 | 0.4000 |
| uretilen CP | 617 | 471 |

## 21.3 ISARETLI / ISARETSIZ AYRIMI — sunumda mutlaka soylenmeli

`manset.py` manseti IKI olcutle hesapliyor:
```
rob = esle(..., 2.0, 10.0, False)                  # isaretli VARSAYILAN False
rbi = esle(..., 2.0, 10.0, False, isaretli=True)   # ISARETLI
```
Yapilandirmadaki **`robot_hazir_F1 = 0.6303` ISARETSIZ olandir** -- yani
180 derece TERS bir yon DOGRU sayilir. Kodun kendi yorumu isaretliyi
isaret ediyor: *"robot icin dogru olcut budur"*.

**Sunumda ikisi de verilmelidir.** "Robot kabloyu hangi yone sokacagini
biliyor mu" sorusuna yalnizca ISARETLI sayi cevap verir.

## 21.4 Onceki manset ile kiyas

| | yapilandirma (v5, DEV+VAL 194) | bu olcum (VAL 100, bugunku sistem) |
|---|---|---|
| tespit F1 | 0.7584 | **0.7878** |
| robot (isaretsiz) | 0.6303 | 0.5764 |

Tespit YUKSEK cikti. Isaretsiz robot metriginde fark var ama kumeler ayni
DEGIL (194 vs 100 parca, farkli urun surumu), dolayisiyla "geriledi"
denemez; bu olcum DAHA TEMIZ olandir.

## 21.5 Bayrak geri alindi

`glb_kanonik_zincir` ACILDI sonra GERI ALINDI. Iki populasyonda TERS:
- zor/gorulmemis markalar (80 parca): olculen zincir F1 +0.0368
- **tanidik marka (VAL 100): olculen zincir F1 −0.0171**

Olculen zincir daha KESIN (0.5605 vs 0.5008) ama daha dusuk RECALL'li
(0.4000 vs 0.4682). Zor markada bu takas kazandiriyor, tanidik markada
kaybettiriyor. Varsayilan FALSE. Dogru cozum rejim kapisi OLABILIR --
olculmeden acilmaz.

## 21.6 Y1 AUGMENTASYON UCTAN UCA DOGRULANDI — kampanyanin en buyuk kazanci

**Adil kiyas** (VAL 100 parca, STEP'ten TAM zincir, `olculen` yolu,
IKISI DE TEK kontrol noktasi -- taban 4'lu toplulukti, o yuzden tek-vs-tek
kosuldu):

| kol | tespit | robot ISARETSIZ | robot ISARETLI |
|---|---|---|---|
| A kontrol (augmentasyon yok) | 0.5430 | 0.4792 | 0.4135 |
| **Y1 (hafif augment 0.3 rad)** | **0.5808** | **0.5293** | **0.4831** |
| **fark** | **+0.0378** | **+0.0501** | **+0.0696** |

**Segmentasyon IoU'daki +0.0197, robot F1'e +0.0696 olarak gecti** -- yani
uc katiyla. Bu, kanonik dersinin (olcum yerinde kazanan dagitilan yolda
kaybedebilir) TERSI yonde bir ornek: burada kazanc BUYUYEREK gecti.

**Ve kritik gozlem:** TEK augmentasyonlu ckpt (robot isaretli 0.4831),
DAGITILMIS DORT ckpt'lik toplulugun saha yoluna (0.4839) DENK. Oyleyse
augmentasyonlu bir TOPLULUK ikisini de gecmeli. Tohum 1 ve 2 egitiliyor.

**Kiyas kusuru not edilir:** ilk denemede Y1 (tek ckpt) 4'lu toplulukla
kiyaslanmisti; fark augmentasyona DEGIL topluluk-vs-tek'e ait olabilirdi.
Kontrol kolunun tek ckpt'si ayrica kosuldu ve kiyas tek-vs-tek yapildi.

## 21.7 CALISMA NOKTASI TARAMASI (cevrimdisi) — MEVCUT NOKTA OPTIMUM

`sonda_calisma_noktasi.py`, tahmin dokumunden (VAL 100 parca, A kontrol
tek ckpt, saha yolu). Uretilen CP'ler SABIT, yalnizca tutma kurali degisir.

| deneme | robot ISARETLI | kesinlik | recall | CP |
|---|---|---|---|---|
| TABAN (hepsi) | 0.2611 | 0.3933 | 0.1955 | 328 |
| ilk-16 | **0.2619** | 0.3969 | 0.1955 | 325 |
| guven>=0.5 | 0.2365 | 0.4605 | 0.1591 | 228 |
| guven>=0.7 | 0.1882 | 0.5474 | 0.1136 | 137 |
| guven>=0.9 | 0.0635 | **0.6667** | 0.0333 | 33 |

**En iyi alternatif +0.0008 -- gurultu.** Her esik F1'i DUSURUYOR: recall
kaybi kesinlik kazancini asiyor. **Y15 (esik taramasi) KAPANDI**, mevcut
"hepsini tut" kurali zaten optimum.

**Sunumda kullanilabilir yan bulgu:** kesinlik AYARLANABILIR (0.393 →
0.667, recall bedeliyle). Robot uygulamasinda "az ama emin" tercih
edilirse bu bir calisma noktasi secenegidir.

**Altyapi notu:** bu tarama 27 dakikalik zincir kosusu yerine SANIYELER
surdu, cunku `sonda_zincir_esli_kiyas.py` artik tahminleri diske dokuyor.

## 21.8 KOLLARI TANIDIK MARKADA YENIDEN DENEME — HEPSI DUSTU

Hipotez: bu kollar (ayna esi, yerel negatif, isin-temas) GORULMEMIS
markada kapiyi gecemedi; TANIDIK markada temsil problemi cok daha kucuk
oldugu icin tutabilirler. Ayni veri (d6), tek degisken KAT TURU:
marka-disi katlar yerine RASTGELE 3 kat (marka-KARISIK).

| kol | tanidik marka | fark |
|---|---|---|
| TABAN | **0.5603** | — |
| ayna esi | 0.5585 | −0.0018 |
| ayna + yerel negatif | 0.5575 | −0.0028 |
| ayna + temas | 0.5550 | −0.0052 |
| ayna + temas + yerel negatif | 0.5555 | −0.0048 |

**Hipotez YANLIS: hepsi tanidik markada da kaybettiriyor.** Kollar kapandi.

## 21.9 ALAN FARKI TEK SAYIYLA — sunumun ana bulgusu

Ayni sistem, ayni veri, ayni oznitelikler; **tek fark kat turu**:

| kosul | taban robot F1 |
|---|---|
| GORULMEMIS marka (marka-disi katlar) | **0.3135** |
| TANIDIK marka (rastgele katlar) | **0.5603** |
| **fark** | **+0.2468** |

Bu, `domain-gap-is-the-blocker` bulgusunun secici duzeyindeki BUYUKLUGU.
Sunumda tek basina guclu bir bulgudur: sistem kapasitesi degil, YENI
URETICIYE TRANSFER bagliyor.

## 21.10 COK-CP UZMANLASMASI — TUM BICIMLERI OLCULDU, HICBIRI GECMEDI

TANIDIK marka kosulu (d6, rastgele katlar), taban 0.5572.
Cok-CP tanimi `manset.py` ile AYNI: `n_gt >= 8` (468 parcanin 64'u, %13.7).

| kol | robot F1 | fark |
|---|---|---|
| **agirlik 2x** | 0.5637 | **+0.0065** |
| agirlik 4x | 0.5582 | +0.0009 |
| ince ayar (warm-start, yonlendirici) | 0.5536 | −0.0037 |
| ince ayar (rejim GT'den) | 0.5532 | −0.0040 |
| agirlik 8x | 0.5473 | −0.0099 |
| uzman (ayri egitim, yonlendirici) | 0.5162 | −0.0410 |
| uzman (rejim GT'den, UST SINIR) | 0.5154 | −0.0418 |

**YONLENDIRICI SUCLU DEGIL:** dogrulugu **0.962**, ve KAHIN rejimle bile
uzman kaybediyor. Sebep VERI PARCALANMASI.

**INCE AYAR bunu DOGRULADI:** genel modelden `warm_start` ile devam edip
yalnizca yogun parcalarda ek agac eklemek, bolmenin zararinin **%91'ini**
geri aldi (−0.0410 → −0.0037). Yani teshis dogruydu; ama yine de taban
asilmadi.

**HUKUM:** cok-CP uzmanlasmasi SECICI duzeyinde hicbir bicimde calismiyor.
En iyisi hafif agirliklandirma (+0.0065), o da kapinin altinda.

**SESSIZ NO-OP YAKALANDI:** ince ayar ilk kosuda tam **+0.0000** verdi.
Sebep: `m.max_iter_` diye bir oznitelik YOK (dogrusu `n_iter_`);
AttributeError genis bir `except` tarafindan yutuluyor, fonksiyon None
donuyor ve iki kol da sessizce TABANA dusuyordu. "Ince ayar ise yaramiyor"
diye rapor edilecekti. Hata artik YUTULMUYOR ve agac sayisi `assert` ile
dogrulaniyor.

## 21.11 AUGMENTASYON: ACI ve TOHUM

**Aci taramasi (AYNI tohum s0, adil kiyas):**

| aci | val Conn_IoU |
|---|---|
| yok (kontrol) | 0.6232 |
| **0.15 rad (~9 derece)** | **0.6528** |
| 0.30 rad (~17 derece) | 0.6429 |
| 1.05 rad (~60 derece) | DUSURUYOR (bayrak belgesi) |

Optimum 0.15 civarinda ya da ALTINDA. Ama:

**Tohum gurultusu ACI farkindan BUYUK.** 0.3 rad'da tohumlar:
s0 0.6429 · s1 0.6667 · s2 0.6881 (yayilim 0.045). Aci farki ise 0.010.
Bu yuzden tek tohumla "optimum aci budur" DENMEZ; aci kiyaslari ayni
tohumda yapildi ve bu sinirlilik raporlanir.

## 21.12 AUGMENTASYONLU TOPLULUK — DAGITILMADI

VAL 100 parca, TAM zincir:

| yapilandirma | tespit | robot ISARETSIZ | robot ISARETLI |
|---|---|---|---|
| **DAGITILMIS 4'lu (saha yolu)** | **0.7878** | **0.5764** | **0.4839** |
| augmentasyonlu 3'lu (saha yolu) | 0.6204 | 0.4561 | 0.4035 |
| augmentasyonlu 3'lu (olculen yol) | 0.6151 | 0.5573 | 0.4826 |

**Saha yolunda augmentasyonlu topluluk DAHA KOTU.** Sebep teshis edildi:
`robot_cp.extract` OY SAYISINA ve guven esigine dayaniyor, bunlar ESKI
kontrol noktalari icin kalibre edilmis. Yeni modellerin kalibrasyonu
farkli -> oylar tutmuyor, tespit 0.79'dan 0.62'ye dusuyor.

**Sonuc:** augmentasyon kazanci SEGMENTASYON duzeyinde ve OLCULEN zincirde
gercek (+0.0158 isaretli), ama dagitilan OY TABANLI yola KALIBRASYON
yapilmadan takilamiyor. **DAGITILMADI.**

**Karistirici:** dagitilmis sistem 4 ckpt, bu 3 ckpt. Adil olmasi icin
dorduncu tohum egitiliyor ve olcum tekrarlanacak.

## 21.13 SAHA YOLUNUN COKME MEKANIZMASI — teshis ve TAHMIN

Augmentasyonlu 3'lu topluluk saha yolunda tespit F1'i 0.7878 -> 0.6204
dusurmustu. Mekanizma:

- `robot_cp.extract` `min_votes=1` kullaniyor -> OY ESIGI sorun DEGIL.
- Ama `_votes` (kac modelin bagimsiz buldugu) **wire-gate'in
  OZNITELIGIDIR** ve olculmustu ki *durust (geometri) bolmede gate'in
  GENELLESEN TEK ozelligi votes'tur* (`gate-memorizes-not-learns`).
- Gate, ESKI **4** modelli toplulugun oy dagilimiyla egitildi. 3 model
  verilince oylar 1-3'e sikisiyor; gate DAGILIM DISI deger goruyor ve
  daha cok REDDEDIYOR.

**TAHMIN (once yazildi):** 4 augmentasyonlu kontrol noktasiyla oylar yine
1-4 araligina doner ve saha yolu TOPARLAR. Tohum 3 egitiliyor; olcum
kuyruga alindi (`results/VAL_aug4_topluluk.json`).

Tahmin tutmazsa ikinci secenek: wire-gate'i yeni toplulugun oy dagilimiyla
YENIDEN egitmek (iki gate AYNI dagilimda egitilmeli -- `gate-refit-minv4`).

## 21.14 KUYRUK KUSURU YAKALANDI — yari egitilmis kontrol noktasiyla olcum

4'lu topluluk olcumu icin kurdugum kuyruk sunu bekliyordu:
`while [ ! -f results/seg_extra/y1_aug03_s3.pt ]`.

**Kusur:** kontrol noktasi her val iyilesmesinde yazilir, yani YARI
EGITILMIS halde de VARDIR. Olcum, tohum 3 daha epoch 20/200'deyken
basladi ve 4'lu topluluk sayisi yarim uyeyle uretilecekti.

Yakalandi ve durduruldu. Dogru kosul: egitim logunda `DONE best` gormek.

**Ders:** "cikti dosyasi var" ile "is bitti" AYNI SEY DEGILDIR. Bu projede
ayni aileden dordunculuk: sessiz no-op'lar, `pgrep` yanlis negatifi,
`veri.index()` tuzagi, `max_iter_` yutulan hatasi.

## 21.15 HATA OTOPSISI — son-islem ailesi TEK OLCUMLE kapandi

`sonda_hata_otopsisi.py`, VAL 100 parca, saha yolu (A kontrol tek ckpt).
TP 129 · FP 199 · FN 531 · robot ISARETLI F1 0.2611

**Yanlis pozitif dagilimi (194):**

| tur | sayi | pay |
|---|---|---|
| **hayalet** (hicbir GT'ye yakin degil) | 171 | **%88.1** |
| aci yanlis | 14 | %7.2 |
| isaret ters | 8 | %4.1 |
| cift kopya | 1 | %0.5 |

**Kacan dagilimi (531):**

| tur | sayi | pay |
|---|---|---|
| **bos** (yakininda HIC tahmin yok) | 500 | **%94.2** |
| yakin_var (tahmin var, kutuya girmiyor) | 31 | %5.8 |

**HUKUM.** Son-islem onarimlarinin TAVANI:
isaret %4.1 · aci %7.2 · cift %0.5 -> toplam **hatanin %12'si**.
Bugun denenen son-islem kollarinin (disari cevirme, eksen cakistirma,
birlestirme) neden hicbir sey kazandirmadigi buradan anlasilir.

**Kayip YUKARIDA:** sistem cogunlukla YANLIS YERLERDE uretiyor (%88
hayalet) ve gercek girislerin cogunun YAKININDA HIC BIR SEY uretmiyor
(%94 bos). Bu bir son-islem sorunu DEGIL, ADAY URETIMI + KONUM SKORLAMA
sorunudur.

Bu, bagimsiz olarak olculen konum/yon ayrimini DOGRULAR:
yon AUC 0.8899 · konum AUC 0.7053 (gereken 0.944).

**Yontem notu:** bu otopsi, agregat sayilardan yapilan bir cikarimin
(tespit 0.7878 -> eksen 0.5764 farkinin "eksen hatasi" oldugu) YANLIS
oldugunu gosterdi. Agregat farklardan mekanizma cikarilmaz; hata
SINIFLANDIRILIR.

## 21.16 MAKBUZ KIRLILIGI YAKALANDI

`results/VAL_aug4_topluluk.json` "4'lu augmentasyonlu topluluk" adiyla
duruyordu ama icerigi 3'lu kosunun degeriydi (saha F1 0.4035). Sebep:
yari egitilmis ckpt ile baslayan olcumu OLDURDUM, ama kuyruk betigi bir
sonraki satirdaki `cp` komutunu yine de calistirdi ve ESKI
`zincir_esli_kiyas.json`'i YENI adla kopyaladi.

Karantinaya alindi: `results/_SILINDI_VAL_aug4_YARIM.json`.

**Ders:** kuyruk betiklerinde her adim, ONCEKI adimin BASARIYLA bittigini
DOGRULAMALIDIR (`&&` ya da acik kontrol). Yoksa oldurulen bir olcumun
ardindan yanlis etiketli makbuz uretilir. Bu projede makbuz cakismasi
daha once de olmustu (`havuz_tavani_*`, `max_sec_sondasi*`).

## 21.17 HAVUZ mu SKOR mu — OLCUM GECERSIZ, rapor edilmedi

Hata otopsisi kacanlarin %94'unun yakininda HIC tahmin olmadigini
gostermisti. Sebebi ayirmak icin havuz recall'u ile cikti recall'u yan
yana olculdu:

| olcu | recall |
|---|---|
| HAVUZ (konum) | 0.5939 |
| HAVUZ (yonlu) | 0.4242 |
| CIKTI (konum) | 0.6273 |
| CIKTI (yonlu) | **0.4682** |

**CIKTI recall'u HAVUZ recall'undan YUKSEK.** Cikti havuzun ALT KUMESI
oldugu icin bu yapisal olarak IMKANSIZ; "gate'in attigi" degerinin
NEGATIF cikmasi (−0.0439) ayni seyi soyluyor.

**Sebep:** `robot_cp.adaylari_uret(...)` benim cagirdigim bicimde,
`extract`'in gercekte kullandigi havuzu URETMIYOR -- zincirde asagida
`urun_p6`/`urun_genis` EK aday ekliyor ve bunlar yakalanmadi.

**Olcum GECERSIZ sayildi ve rapor edilmedi.**
Makbuz karantinada: `results/_GECERSIZ_havuz_vs_cikti.json`.

Soru ACIK kaliyor: kacanlarin sebebi havuz mu skor mu. Dogru olcum icin
havuz, `extract`'in ICINDEN (tum aday kaynaklari birlestikten SONRA)
alinmalidir.

## 21.18 ISARET ONARIMI — TAVAN +0.0893 ama HICBIR GEOMETRIK KURAL YAKALAMIYOR

VAL 100 parca, saha yolu, dogru vekillerle (mesh merkezi ve YEREL YUZEY
NORMALI dokume eklendikten sonra):

| kol | robot ISARETLI | fark |
|---|---|---|
| taban | 0.4839 | — |
| yerel yuzey normali | 0.4573 | −0.0266 |
| mesh merkezi | 0.4448 | −0.0392 |
| tahmin merkezi (ilk vekil) | 0.3712 | −0.1128 |
| eksen cakistirma | 0.4699 | −0.0141 |
| **kahin isaret (UST SINIR)** | **0.5732** | **+0.0893** |

Vekiller BEKLENEN sirada iyilesiyor (yerel normal > mesh merkezi > tahmin
merkezi) ama **ucu de modelin KENDI isaretinden kotu**. Yani hata "model
iceri bakiyor" DEGIL; baska bir sey.

**HUKUM:** isaret onarimi ailesi (disari kurali, eksen cakistirma)
KAPANDI. **Tavan +0.0893 kayitta kalir** -- daha akilli bir yontem icin
acik ve olculmus bir hedef.

**Yontem notu:** ilk olcumde `disari_mesh` ve `disari_normal` tam
`+0.0000` vermisti; sebep dokumde o alanlarin HENUZ olmamasiydi (kol
sessizce hicbir sey yapmiyordu). Tam sifir imzasi artik sessiz no-op
belirtisi olarak taniniyor.

## 21.19 4'LU AUGMENTASYONLU TOPLULUK — OY HIPOTEZI DOGRULANDI

VAL 100 parca, TAM zincir:

| yapilandirma | tespit | robot ISARETSIZ | robot ISARETLI |
|---|---|---|---|
| DAGITILMIS (4 eski ckpt, saha) | **0.7878** | 0.5764 | 0.4839 |
| 3 aug, saha | 0.6204 | 0.4561 | 0.4035 |
| 4 aug, saha | 0.6725 | 0.5380 | 0.4613 |
| **4 aug, OLCULEN zincir** | 0.6392 | 0.5761 | **0.5162** |

**OY HIPOTEZI DOGRULANDI.** 21.13'te once yazilmisti: "4 ckpt ile oylar
1-4 araligina doner ve saha yolu toparlar." Olculdu: 3→4 gecisi saha
yolunda robot-isaretliyi 0.4035 → 0.4613 (+0.0578), tespiti
0.6204 → 0.6725 cikardi. Cokusun sebebi gercekten OY DAGILIMIYDI.

**En iyi robot-isaretli sayi:** augmentasyonlu 4'lu + OLCULEN zincir =
**0.5162**, dagitilmisin 0.4839'una karsi **+0.0323**.

**IKI DURUST KAYIT:**

1. **Tespit DUSUYOR** (0.7878 → 0.6392). Robot icin gecerli olcut
   isaretlidir, ama bu TAKAS sunumda soylenmelidir.
2. **+0.0323 GURULTU BANDININ ICINDE.** Tabanin %95 GA'si
   [0.4003, 0.5671] ve 0.5162 tam onun icinde. "Iyilesti" denebilir;
   "istatistiksel olarak ayirt edilebilir bir iyilesme" DENEMEZ --
   n=100 buna yetmiyor.

Bu yuzden DAGITIM KARARI verilmedi: kazanc yonu dogru ama kanit gucu
yetersiz. Daha buyuk bir degerlendirme kumesi (LOCKED, 100 parca daha)
ayirt edici olabilir -- ama LOCKED harcanmadi ve bu karar icin
harcanmamalidir.

## 21.20 AUGMENTASYON ACISI — tarama TAMAM

| aci | val Conn_IoU (tohum 0) |
|---|---|
| yok (kontrol) | 0.6232 |
| **0.15 rad (~9 derece)** | **0.6528** |
| 0.30 rad (~17 derece) | 0.6429 |
| 0.50 rad (~29 derece) | 0.6424 |
| 1.05 rad (~60 derece) | DUSURUYOR (bayrak belgesi) |

Optimum ~0.15; 0.3 ve 0.5 birbirine yakin; hepsi kontrolden IYI. Tohum
gurultusu (0.6429-0.6881) aci farkindan buyuk oldugu icin "optimum tam
olarak 0.15'tir" DENMEZ -- soylenebilecek olan: **hafif augmentasyon
kazandirir, agresif kaybettirir**.

## 21.21 TOPLULUK CESITLILIGI (6 ckpt) — egitim gerektirmeyen kol

Elde 6 augmentasyonlu ckpt var: 4 tohum x 0.3 rad + 0.15 + 0.5.
FARKLI aci = farkli hata deseni = topluluk cesitliligi. Yeni egitim
GEREKTIRMEZ, yalnizca olcum. VAL 100 parcada kosuluyor.

## 21.22 Y6 LABEL SMOOTHING — DUSTU

| kol | val Conn_IoU |
|---|---|
| augment 0.15 (taban) | **0.6528** |
| + label smoothing 0.05 | 0.5244 (**−0.1284**) |

Kismi etiketli maskeli BCE'de zaten `partial_pos_weight = 20` var
(CableEntry tepelerin ~%1.5'i). Hedefi yumusatmak bu dengeyi bozuyor:
pozitif sinyal zaten 20 kat agirlikliyken hedefi 0.95'e cekmek etkiyi
seyreltiyor. **KAPANDI.**

Bayrak kodda kaldi (`--label-smooth`, varsayilan 0.0).

## 21.23 TOPLULUK CESITLILIGI (6 ckpt) — DUSTU

VAL 100 parca, olculen zincir, robot ISARETLI:

| topluluk | robot ISARETLI |
|---|---|
| **4 tohum x 0.3 rad** | **0.5162** |
| 6 ckpt (4 tohum + 0.15 + 0.5) | 0.4859 (**−0.0303**) |

Farkli ACILARLA cesitlilik katmak YARDIM ETMIYOR. Ayni acidaki dort tohum
daha iyi. Muhtemel sebep: 0.15 ve 0.5 kollari TEK tohumlu ve tohum
gurultusu (0.6429-0.6881) aci farkindan buyuk; toplulugu zayif uyelerle
seyreltiyorlar.

**KAPANDI.** Bugunun en iyisi degismedi: **4 aug tohum + olculen zincir =
0.5162** (dagitilmis 0.4839).

## 21.24 SNAPSHOT TOPLULUGU (Y4) — DUSTU

`best` + `last` kontrol noktalari birlikte (4 tohum x 2 = 8 ckpt):

| topluluk | robot ISARETLI (olculen zincir) |
|---|---|
| **4 tohum, yalniz `best`** | **0.5162** |
| 8 ckpt (`best` + `last`) | 0.5039 (**−0.0123**) |
| 6 ckpt (farkli acilar) | 0.4859 (−0.0303) |
| 3 tohum | 0.4826 (−0.0336) |

`last` kontrol noktalari `best`ten daha zayif oldugu icin toplulugu
SEYRELTIYORLAR. **Y4 KAPANDI.**

## 21.25 TOPLULUK DESENI — dort tohum, yalniz `best`, AYNI aci

Dort ayri topluluk denemesinin hepsi ayni sonuca isaret ediyor:

| deneme | fark |
|---|---|
| tohum sayisi 3 -> 4 | **+0.0336** |
| farkli acilarla cesitlendirme | −0.0303 |
| `last` snapshot'lari ekleme | −0.0123 |

**Kural:** topluluğu buyutmek degil, UYELERI GUCLENDIRMEK kazandiriyor.
Zayif uye eklemek her seferinde zarar verdi. En iyi yapilandirma
**4 tohum x hafif augmentasyon (0.3 rad) x yalniz `best` + olculen
zincir = 0.5162**.

## 21.26 Y14 FOCAL-TVERSKY — NOTR

| kol | val Conn_IoU |
|---|---|
| augment 0.15 (taban) | 0.6528 |
| + Focal-Tversky (gamma 1.33) | 0.6550 (**+0.0022**) |

Tohum gurultusu ~0.045 oldugu icin +0.0022 ANLAMSIZ. Ne kazandirir ne
kaybettirir. Bayrak kodda kaldi (`--tversky-gamma`, varsayilan 1.0 =
klasik Tversky, davranis degismez).

**KAPANDI (notr).**

## 21.27 Y5 YARDIMCI GOREV (aux-wire) — POZITIF ama GURULTU ICINDE

`--aux-wire` bayragi kodda VARDI, bu kampanyada HIC acilmamisti.
53/189 parcada TEL/ALET yardimci etiketi var (agirlik 0.5, pos_w 2.0).

| kol (augment 0.15 tabani uzerine) | val Conn_IoU | fark |
|---|---|---|
| taban | 0.6528 | — |
| **aux-wire** | **0.6611** | **+0.0083** |
| Focal-Tversky (gamma 1.33) | 0.6550 | +0.0022 |
| label smoothing (0.05) | 0.5244 | −0.1284 |

aux-wire uc kolun EN IYISI ama **tohum gurultusunun (~0.045) icinde**.
Tek tohumla "kazandi" DENMEZ. Dogru test cok tohumlu olurdu (~3 x 50 dk);
bu, kalan sureye sigmadi ve boyle yazildi.

**Not:** ayni sinirlilik Focal-Tversky icin de gecerli. Ikisi de "notr ya
da hafif pozitif" kategorisinde; hicbiri augmentasyon kadar (seg IoU
+0.0296, uctan uca +0.0696) net degil.

## 21.28 Y7 OZNITELIK SECIMI — KAPIYA EN YAKIN KOL (+0.0097)

Secicinin oznitelik matrisi 162 sutun; sistematik hic budanmamisti.
Permutasyon onemine gore siralanip budandi (TANIDIK marka, rastgele kat):

| kol | sutun | robot F1 | fark |
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
- az budama iyi, cok budama kotu

Uc bagimsiz kolda ayni sekil: **olculu mudahale kazandirir, asiri
mudahale kaybettirir.**

**Kapiyi gecmis SAYILMADI** (tek tohum, rastgele kat). Ama listedeki EN
UMUT VERICI acik kol budur; cok tohumlu tekrarda gecebilir.

## 21.29 Y8 ETIKET KALITESI AGIRLIKLANDIRMA — 4 TOHUMDA DOGRULANDI

**Fikir.** Secicinin pozitif etiketleri ESIT agirlikta ogretiliyor. Ama
GT'ye 0.2mm'de oturan aday ile tolerans sinirinda 1.9mm'de oturan aday
ayni guvenilirlikte DEGIL. Pozitife, GT'ye YANAL yakinligiyla orantili
agirlik verilir. **Metrik DEGISMEZ** -- yalnizca egitim agirligi
(bu, `metrik-cerrahisi-a1-reddedildi`'den farkli olmasinin sebebi).

**DORT kat tohumunda:**

| tohum | taban | lineer | karesel |
|---|---|---|---|
| 1 | 0.5572 | +0.0144 | +0.0091 |
| 2 | 0.5635 | −0.0019 | +0.0079 |
| 3 | 0.5321 | +0.0101 | +0.0123 |
| 4 | 0.5491 | +0.0102 | +0.0036 |
| **ortalama** | — | **+0.0082** | **+0.0082** |

**8 olcumun 7'si POZITIF.** Kol gercek; ortalama +0.0082, kapinin
(+0.01) ALTINDA.

**METODOLOJIK DERS -- bugunun en onemlisi.** Tek tohumda +0.0144 ile
"KAPI GECTI" yazdi; ikinci tohumda −0.0019 verdi. Tek tohumla dagitilsaydi
OLMAYAN bir kazanc raporlanmis olurdu. Kat gurultusu bu buyuklukteki
kollarda **±0.008** mertebesinde; dolayisiyla **+0.01 civari her kol COK
TOHUMLU dogrulanmadan hukum giymemelidir.**

Bu, gun boyunca "gurultu bandinda" diye isaretledigim tum kollari
(Y7 +0.0097, aux-wire +0.0083, Focal-Tversky +0.0022) da kapsar.

## 21.30 Y7 OZNITELIK SECIMI — 3 TOHUMDA KAPANDI

| kat tohumu | ilk %75 sutun (121/162) |
|---|---|
| 1 | +0.0097 |
| 2 | +0.0022 |
| 3 | +0.0002 |
| **ortalama** | **+0.0040** |

Kapinin (+0.01) COK altinda. **KAPANDI.**

Tek tohumdaki +0.0097'nin YARIDAN FAZLASI gurultuydu -- Y8'de gorulen
desenin ayni. Iki kol da ayni sonuca isaret ediyor: bu buyuklukteki
etkiler tek tohumla OLCULEMEZ.

## 21.31 GUN 2 KAPANIS BILANCOSU

**Dogrulanmis ve dagitilan:**
| kalem | kazanc |
|---|---|
| negatif orani 8 → 12 | +0.0126 (`tam`, 5 marka kati) |

**Olculmus, dogrulanmis, ama kapi altinda (dagitilmadi):**
| kalem | kazanc | dogrulama |
|---|---|---|
| etiket kalitesi agirliklandirma | +0.0082 | **4 kat tohumu**, 7/8 pozitif |
| yardimci gorev (aux-wire) | +0.0083 | tek tohum |
| oznitelik budama (%75) | +0.0040 | 3 kat tohumu |
| Focal-Tversky | +0.0022 | tek tohum |
| cok-CP agirligi 2x | +0.0065 | tek tohum |

**Segmentasyon tarafinda dogrulanmis kazanc:**
| kalem | kazanc |
|---|---|
| hafif augmentasyon (0.15-0.3 rad) | seg IoU +0.0197…+0.0649 |
| uctan uca (tek-vs-tek, VAL) | robot ISARETLI **+0.0696** |
| 4 aug tohum + olculen zincir | 0.5162 vs dagitilmis 0.4839 |

**Kapanan kollar (hepsi makbuzlu):** isaret onarimi (tavan +0.0893 acik
kaliyor) · yogun-parca uzmani (kahin rejimle bile −0.0418) · calisma
noktasi taramasi · hks girdi ozniteligi · eksen cakistirma · GT'den
segmentasyon etiketi (iki sekil) · sentetik korpus · snapshot toplulugu ·
topluluk cesitliligi · label smoothing · Y7 oznitelik budama

**Gecersiz sayilan olcumler:** havuz-vs-skor (yapisal olarak imkansiz
sonuc) · yarim-ckpt toplulugu · makbuz kirliligi (karantinada)

## 21.32 ISARET KOLUNUN COKUS MEKANIZMASI — teshis edildi

Secici isaret kurallari denendi (yalniz ACIK celiskide cevir):

| kural | fark |
|---|---|
| normal_kesin (yerel normale gore) | **+0.0000** (hic tetiklenmedi) |
| mesh_kesin 150 derece | +0.0031 |
| mesh_kesin 110 derece | +0.0094 |
| mesh_kesin 120 / 135 | +0.0047 / +0.0063 |
| parca_modal | −0.0094 |

**Esik taramasi MONOTONIK DEGIL** (110 > 135 > 120 > 150). Gercek bir
mekanizma duzgun bir egri verirdi; bu desen 100 parcada GURULTUYE UYDURMA
demektir ve bugun olculen ±0.008'lik kat gurultusuyle ayni mertebede.
110 dereceyi secmek, Y8'de duselecek tuzagin ta kendisi olurdu.
**SECILMEDI.**

**KOK NEDEN BULUNDU.** Tahmin edilen yon ile yerel yuzey normali
arasindaki aci (617 CP):

| yuzdelik | aci |
|---|---|
| %25 | 60.9 derece |
| **%50** | **88.9 derece** |
| %75 | **90.0 derece** |

Yon, yerel normale **neredeyse DIK**. Sebebi geometrik: CP noktasina EN
YAKIN tepe, deligin DUVARINDADIR ve duvar normali eksene DIKTIR. Yani
"disari" referansi diye deligin duvar normali kullanilmis; eksene dik bir
referansla isaret atamak rastgeleye yakindir. `disari_normal`in −0.0266
vermesinin sebebi budur.

**Dogru referans:** en yakin tepenin normali DEGIL, **agiz cevresindeki
YUZUN** normali (agiz halkasinin disindaki duz yuzey). Gelecek deneme icin
somut duzeltme; kalan surede kurulup gurultuden ayrilamadi.

**Tavan +0.0893 hala ACIK.**

## 21.33 COK-CP AGIRLIGI — O DA DOGRULANMADI

| kat tohumu | agirlik 2x |
|---|---|
| 1 | +0.0065 |
| 2 | **−0.0052** |

Ucuncu kol da tek tohumda pozitif, ikinci tohumda NEGATIF. (Tohum 3-4
kosuyordu, makine kapatildi.)

**UC BAGIMSIZ KOLDA AYNI SONUC:**

| kol | tek tohum | cok tohum |
|---|---|---|
| Y7 oznitelik budama | +0.0097 | **+0.0040** (3 tohum) |
| Y8 etiket kalitesi | +0.0144 | **+0.0082** (4 tohum, 7/8 poz) |
| cok-CP agirligi 2x | +0.0065 | **isaret degistirdi** (2 tohum) |

**Bu, gunun en saglam bulgusudur:** +0.005…+0.015 araligindaki HICBIR kol
tek tohumla olculemez. Kat gurultusu ±0.008 ve bu kollarin etkisiyle AYNI
mertebede.

Y8 ucu icinde EN saglami (4 tohum, 8 olcumun 7'si pozitif, ortalama
+0.0082) ama o da kapinin altinda.

## 21.34 COK TOHUMLU DOGRULAMA AILESI — TAMAMLANDI

| kol | tek tohum | cok tohum | tohum sayisi |
|---|---|---|---|
| cok-CP agirligi 2x | +0.0065 | **−0.0002** | 4 |
| Y7 oznitelik budama | +0.0097 | **+0.0040** | 3 |
| Y8 etiket kalitesi | +0.0144 | **+0.0082** | 4 |

cok-CP agirligi ayrintisi: +0.0065 / −0.0052 / −0.0034 / +0.0012.
**Tam sifir.** Ilk tohumdaki degerin TAMAMI gurultuydu.

**UC KOLDA DA tek tohumlu deger SISIKTI.** Yalniz Y8 cok tohumda net
pozitif kaldi (+0.0082, 8 olcumun 7'si pozitif) ama o da kapinin altinda.

**Bu ailenin sonucu, kampanyanin en saglam metodolojik ciktisi:**
bu buyuklukteki (+0.005…+0.015) etkiler tek tohumla OLCULEMEZ; kat
gurultusu ±0.008 ve etkiyle AYNI mertebede. Gun boyunca "kapiya yakin"
gorunen hicbir kol bu nedenle dagitilmadi.

## 21.35 HALKA NORMALI ILE ISARET DUZELTMESI — KAPIYI GECTI (+0.0329)

**Teshis (21.32) dogru cikti.** "En yakin tepe normali" deligin DUVAR
normaliydi ve eksene DIK; dogru referans **agiz cevresindeki halkadan**
(3-8mm) alinan yuz normali.

VAL 100 parca, saha yolu:

| kural | robot ISARETLI | fark |
|---|---|---|
| taban | 0.4839 | — |
| en yakin tepe normali (eski) | 0.4573 | −0.0266 |
| **halka normali** | **0.5168** | **+0.0329** |
| kahin isaret (UST SINIR) | 0.5732 | +0.0893 |

Kazanc, kahin tavaninin **%37'si** ve kat gurultusunun (±0.008) **4 KATI**.
Saf SON-ISLEM: egitim yok, model degisikligi yok, **ayarlanmis parametre
yok** (yalnizca "isareti halka normaliyle uyumlu yap").

**ESLI BOOTSTRAP (dogru test):** marjinal GA degil FARKIN GA'si, cunku
kiyas AYNI parcalarda.

    fark +0.0329   %95 GA [−0.0155, +0.0819]
    bootstrap orneklerinin %90.6'si pozitif

**GA sifiri ICERIYOR -> kanit YETERSIZ.** 100 parca bu farki ayirmaya
yetmiyor.

**Kuralin ayarlanmis parametresi OLMADIGI icin** d6 parcalarinda test
etmek sizinti yaratmaz: taban F1 ornek-ici oldugu icin sisik olur ama
ESLI FARK gecerli kalir. 150 parcalik bagimsiz ornek kosuluyor -- LOCKED
harcanmadan kanit gucu artirilir.

## 21.36 HALKA KURALININ DAVRANISI ve ESIK TARAMASI

**Davranis** (VAL 617 CP): kural 95 tanesini cevirir (%15.4) --
**48 DUZELTIR, 27 BOZAR, 20 notr**. Net +21 CP; kararli cevirmelerin
**%64'u dogru**. Kural dogru yonde ama GURULTULU; esli bootstrap GA'sinin
sifiri icermesinin sebebi budur.

**Esik taramasi (yalniz GUCLU celiskide cevir):**

| esik | fark |
|---|---|
| 95 derece | +0.0345 |
| 105 / 115 | +0.0298 / +0.0298 |
| 125 | +0.0063 |
| 140 / 160 | −0.0063 / −0.0047 |

**Monotonik DEGIL** -> gurultuye uydurma. Ayrica 95 derece zaten ~90
derece, yani SADE kuralin kendisi. **Esik eklemek bir sey KATMIYOR;
sade kural kullanilir.**

## 21.37 HALKA KURALI OLCULEN ZINCIRDE **ISTATISTIKSEL OLARAK KESIN**

Ayni kural, ayni parcalar, `olculen` yolu (VAL 100):

| kural | robot ISARETLI |
|---|---|
| taban | 0.4668 |
| en yakin tepe normali | 0.4226 (−0.0442) |
| **halka normali** | **0.4863 (+0.0195)** |
| kahin isaret (UST SINIR) | 0.5323 (+0.0654) |

**ESLI BOOTSTRAP:**

    fark +0.0195   %95 GA [+0.0030, +0.0378]
    bootstrap orneklerinin %98.5'i pozitif

**GA SIFIRI ICERMIYOR -> kazanc ISTATISTIKSEL OLARAK KESIN.**
Bu, kampanyanin ilk istatistiksel olarak kesin iyilesmesidir.

**Iki yol karsilastirmasi tutarli:**

| yol | fark | %95 GA | uretilen CP |
|---|---|---|---|
| saha | +0.0329 | [−0.0155, +0.0819] | 617 |
| **olculen** | **+0.0195** | **[+0.0030, +0.0378]** | 471 |

Saha yolunda etki DAHA BUYUK ama daha OYNAK; olculen zincirde daha kucuk
ama daha TEMIZ (daha az CP -> daha az varyans). Iki bagimsiz yolda da
POZITIF, birinde KESIN.

**Kural dagitilabilir:** parametresiz, egitim gerektirmiyor, gerekli veri
(mesh V/F + tahmin noktalari) `export_robot_glb` icinde zaten mevcut.

## 21.38 HALKA KURALI POPULASYONA BAGLI — kapsam sinirlandi

Bagimsiz 150 parcalik ornek (ZOR/gorulmemis markalar: NIT, MOR, SUPU,
UPUN, UTL, ONV, S+S, SE):

| yol | taban | halka | fark | %95 GA | bootstrap poz. |
|---|---|---|---|---|---|
| saha | 0.1020 | 0.1003 | **−0.0016** | [−0.0054, +0.0000] | %0.0 |
| olculen | 0.1605 | 0.1628 | **+0.0023** | [−0.0052, +0.0105] | %61.2 |

**Zor markalarda kural HICBIR SEY kazandirmiyor.** Saha yolunda kucuk ama
SISTEMATIK negatif (bootstrap orneklerinin %0'i pozitif).

**Bu bir CELISKI DEGIL, KAPSAM SINIRI.** Tabanlara bakildiginda populasyon
tamamen farkli: VAL'de taban 0.4668, zor markalarda 0.1605. Kural, taban
sistemin ZATEN CALISTIGI yerde yardim ediyor.

**DURUST IFADE:**

| populasyon | yol | fark | hukum |
|---|---|---|---|
| TANIDIK marka (VAL 100) | olculen | **+0.0195** | **GA sifiri icermiyor -> KESIN** |
| TANIDIK marka (VAL 100) | saha | +0.0329 | GA sifiri iceriyor |
| ZOR marka (150 parca) | olculen | +0.0023 | notr |
| ZOR marka (150 parca) | saha | −0.0016 | kucuk sistematik negatif |

**Yontem notu:** 21.37'de "iki bagimsiz yolda pozitif, birinde kesin"
yazmistim. O yazildiginda bu ucuncu olcum YOKTU. Simdi var ve iddiayi
DARALTIYOR: kazanc TANIDIK marka populasyonuna ozgudur.

**DAGITIM KARARI:** kural tanidik markada dogrulanmis kazanc, zor markada
notr/ihmal edilebilir negatif veriyor. Sunumun cerceve populasyonu
TANIDIK marka oldugu icin dagitilabilir; ama zor markada BEKLENTI
YARATMAMALI.

## 21.39 Y13 SINIR-FARKINDALI KAYIP — KAPANDI

Ek terim: komsusu FARKLI sinifta olan tepelere (sinif siniri) odaklanan
bir odak kaybi. Gerekce: CP fiziksel olarak bir SINIRDIR (agiz cemberi)
ama mevcut kayip (NLL + Tversky) BOLGEYI hedefler.

| kol | val Conn_IoU | fark |
|---|---|---|
| taban (augment 0.15) | 0.6528 | — |
| sinir agirligi 0.3 | 0.6415 | −0.0113 |
| sinir agirligi 1.0 | 0.6484 | −0.0044 |

**Ikisi de NEGATIF. KAPANDI.** Fikir mekanik olarak makuldu ama olcum
aksini soyledi. Kod kaldi (`--sinir-weight`, varsayilan 0 = davranis
degismez); `faces` anahtarinin `ops` icinde GERCEKTEN oldugu ONCEDEN
dogrulandi (sessiz no-op riski elendi).

## 21.40 Y5 AUX-WIRE — ESLESEN TOHUMLARDA CURUDU

Ilk olcum tek tohumdaydi ve tabanla ESLESMIYORDU. Uc tohumda ESLESEN
taban (augment 0.15) ayrica egitildi:

| tohum | aux-wire | eslesen taban | fark |
|---|---|---|---|
| 0 | 0.6611 | 0.6528 | +0.0083 |
| 1 | 0.6691 | 0.6990 | **−0.0299** |
| 2 | 0.6672 | 0.6673 | −0.0001 |
| **ortalama** | | | **−0.0072** |

**Ortalama NEGATIF. KAPANDI.** Tek tohumdaki +0.0083 gurultuydu.

## 21.41 SEGMENTASYON TARAFINDA TOHUM GURULTUSU — sayisal kanit

Ayni recete (augment 0.15), yalnizca tohum farkli:

| tohum | val Conn_IoU |
|---|---|
| 0 | 0.6528 |
| 1 | **0.6990** |
| 2 | 0.6673 |
| **yayilim** | **0.046** |

Ayni sey augment 0.3'te de gorulmustu: 0.6429 / 0.6667 / 0.6881.

**Segmentasyon tarafinda tohum gurultusu ~0.046'dir.** Bugun olculen
kollarin etkileri 0.002-0.030 araligindaydi, yani gurultunun ALTINDA.
Tek kosumluk hicbir segmentasyon sonucu YORUMLANAMAZ.

**COK TOHUMDA CURUYEN KOLLAR (dort):**

| kol | tek tohum | cok tohum |
|---|---|---|
| cok-CP agirligi 2x | +0.0065 | −0.0002 (4 tohum) |
| Y7 oznitelik budama | +0.0097 | +0.0040 (3 tohum) |
| Y8 etiket kalitesi | +0.0144 | +0.0082 (4 tohum) |
| Y5 aux-wire | +0.0083 | **−0.0072** (3 eslesen tohum) |

Yalniz Y8 pozitif kaldi; o da kapinin altinda.

## 21.42 KALAN SEGMENTASYON KOLLARI SURE ICINDE HUKUM GIYEMEZ — durust karar

21.41'de olculen tohum gurultusu (**0.046**) dogrudan bir kaynak hesabi
dayatir:

* Bir segmentasyon kolunun hukum giymesi icin **>=3 eslesen tohum** sart
  (bu gece dort kol tek tohumda yaniltti, biri **isaret degistirdi**).
* Bir egitim ~2 saat. Yani **kol basina ~6 saat**.
* Kalan segmentasyon kollari: Y3 (EMA/SWA), Y9 (curriculum), Y11
  (optimizator), Y10 (mixup), Y17 (spektral augment), Y18 (jeodezik
  jitter), Y26 (sinifa-ozel augment) = **7 kol x 6 saat = ~42 saat**.

Kalan sure: birkac saat. **Bu kollari bu gece kosmak, hukum giydiremeyecek
sayilar uretmekten baska bir sey yapmaz** -- ve tek tohumluk bir sayi bu
projede dort kez yaniltti.

**KARAR: yeni segmentasyon EGITIMI baslatilmiyor.** Gecenin kalani, hukum
giyebilecek kollara ayriliyor -- yani **egitim gerektirmeyen** olanlara:

| kol | maliyet | neden hukum giyebilir |
|---|---|---|
| Y16 remesh-varyant toplulugu | 3 x 30 dk cikarim | tek modelde, tohum gurultusu YOK |
| Y22 MC dropout | ~30 dk cikarim | ayni model, deterministik taban |
| Havuz mu skor mu (acik soru) | ~30 dk cikarim | teshis; kol degil |

Bu, kampanyanin kendi kuralinin kendi planina uygulanmasidir:
**olculemeyecek seyi olcmeye kalkma.**

## 21.43 Y22 MC DROPOUT — kanca kuruldu, SESSIZ NO-OP DEGIL (dogrulandi)

**Tuzak.** `diffusionnet.predict()` icinde `model.eval()` cagriliyor; bu
dropout'u KAPATIR. Dropout katmanlarini disaridan `train()`'e almak bu
yuzden **sessiz no-op** olurdu -- bu gece ayni tuzak baska bir kolda tam
`+0.0000` uretmisti. Kanca `eval()`'den SONRA, `predict()`'in ICINE
kondu (`mc_dropout=T` parametresi) ve acilan katman sayisi `assert` ile
dogrulaniyor.

**DUMAN TESTI (sentetik mesh, tek kontrol noktasi):**

| olcum | sonuc |
|---|---|
| MC(8) ile taban arasi ortalama abs fark | **0.047173** |
| MC(8) iki kosu arasi fark (tekrar uretilebilirlik) | 0.00000042 |
| Taban (MC kapali) iki kosu arasi fark | 0.000001 |
| Taban etiket ayniligi | %100.0000 |

Etki, gurultunun **~10^5 kati**. Kol GERCEKTEN calisiyor ve
tekrar uretilebilir.

**Yan bulgu:** cikarim bit-ayni degil (ozdeger cozumunden ~1e-6), ama
ETIKET duzeyinde %100 ozdes. Yani cikarim gurultusu, kat gurultusunun
(0.046) kaynagi DEGIL -- kaynak egitim tohumu.

## 21.44 ACIK SORU COZULDU (once TESHIS): havuz DISARIDAN uretilemez

Bolum 21.17'de havuz olcumu "gecersiz" diye karantinaya alinmisti: cikti
recall'u (0.4682) HAVUZ recall'undan (0.4242) BUYUK cikiyordu -- cikti
havuzun alt kumesi oldugu icin **yapisal olarak imkansiz**. Sebep bugun
bulundu.

**Teshis** (`sonda_havuz_teshis.py`, mevcut dokumden, yeni cikarim YOK).
Macar eslestirme yerine **GT basina kapsama** olculdu (o GT'yi kabul
kutusunda karsilayan HERHANGI bir aday var mi) -- havuz sorusunun dogru
olcutu budur. Yapisal kontrol: "ciktida VAR, havuzda YOK" sayisi **0
olmali**.

| yol | isaretsiz ihlal | ISARETLI ihlal |
|---|---|---|
| saha | **84** | 59 |
| olculen | **104** | 85 |

Ihlal sifir degil -> **kaydedilen havuz, ciktiyi ureten havuz DEGIL.**

**KOK NEDEN.** `robot_cp.extract` cikarimi **operator onbellegiyle**
yapar (`op_cache_dir=f"{OP}_k{k_eig}"`); sonda ise havuzu yeniden uretmek
icin onbelleksiz TAZE cikarim kullaniyordu. Olasiliklar farkli cikiyor,
dolayisiyla aday havuzu da farkli. Bu, daha once bagimsiz olarak
kaydedilmis olan **"taze cikarim onbellegi yeniden uretmiyor"** bulgusunun
ta kendisidir -- orada uc gecelik sonucu gecersiz kilmisti, burada havuz
olcumunu gecersiz kildi.

**DUZELTME.** Havuz artik `extract`in **ICINDEN**, wire-gate'ten hemen
once yakalaniyor (`robot_cp.HAVUZ_KANCA`, yalniz `CP_HAVUZ_KANCA` cevre
degiskeniyle dolar; urun yolu DEGISMEZ). Sonda da onu kullaniyor.

**DERS (genellestirilebilir).** Bir ara degeri "ayni kodu disaridan
cagirarak" yeniden uretmek, o kodun ONBELLEK/DURUM bagimliligi varsa
sessizce farkli sonuc verir. Ara degerler URETILDIKLERI YERDEN
yakalanmalidir. Bu tuzak bu kampanyada **iki kez** vurdu.

## 21.45 HALKA NORMALI EKSEN OLARAK KULLANILAMAZ — kesin, monotonik

Halka normali simdiye kadar yalniz **ISARET** icin kullanildi. Fiziksel
sezgi "duz yuzeydeki bir deligin ekseni o yuzeyin normalidir" der; bu
iddia simdiye kadar OLCULMEDI (isaret kollari `isaretsiz` metrigi tanim
geregi hic degistirmez, eksen degistirmek ise ikisini de degistirir).

`sonda_halka_eksen.py`, VAL 99 parca (saha yolu), esli parca bootstrap:

| kol | tespit | robot | robot-ISARETLI | robot-ISR farki |
|---|---|---|---|---|
| taban | 0.6149 | 0.5773 | 0.4847 | — |
| **isaret** (mevcut kural) | 0.6149 | 0.5773 | **0.5176** | **+0.0329** |
| tam (eksen = halka) | 0.6149 | 0.1161 | 0.0988 | **−0.3860** * |
| kapili_10 | 0.6149 | 0.1224 | 0.1051 | −0.3797 * |
| kapili_20 | 0.6149 | 0.2212 | 0.1976 | −0.2865 * |
| kapili_30 | 0.6149 | 0.3027 | 0.2792 | −0.2045 * |
| kapili_45 | 0.6149 | 0.3482 | 0.3153 | −0.1681 * |
| harman_0.25 | 0.6149 | 0.3341 | 0.2980 | −0.1856 * |
| harman_0.50 | 0.6149 | 0.2165 | 0.1929 | −0.2911 * |
| harman_0.75 | 0.6149 | 0.1475 | 0.1255 | −0.3591 * |

(* = %95 GA sifiri icermiyor; bootstrap orneklerinin **%0**'i pozitif.)

**MONOTONIK**: halka normali karisima ne kadar cok girerse sonuc o kadar
kotu (harman 0.25 → 0.50 → 0.75 ile −0.186 → −0.291 → −0.359).
Gurultuye uydurma DEGIL, sistematik.

**MEKANIZMA / DERS.** Halka normali **~1 bit** tasiyor: *hangi taraf
disari*. Eksenin kendisini tasimiyor -- pah kirmalari, kavisli govde ve
girintili agizlar yuzunden agiz cevresi yuzeyin ortalama normali burgu
ekseniyle hizali degil. Kural bu 1 biti kullandiginda **+0.0329**,
tamamini kullanmaya kalktiginda **−0.3860**.

**Bu, kolun neden ISARET kolu olarak DAGITIMA aday olup EKSEN kolu olarak
kapali oldugunun kanitidir.** Ayrica 21.32'deki "yanlis referans" dersinin
simetrigi: orada dogru fikir yanlis referansla olculmustu, burada dogru
referans yanlis IS icin kullanildi.

## 21.46 BAGLAYICI KISIT BULUNDU: **YANAL KONUM**, aday varligi DEGIL, yon DEGIL

Havuz artik `extract`in icinden yakalandigi icin (21.44) tolerans taramasi
GECERLI. VAL 40 parca / 238 GT, saha yolu, ISARETLI olcut, eksenel <=40mm.

**YANAL toleransi gevsetince** (aci 10 derecede SABIT):

| yanal | HAVUZ kapsamasi | CIKTI |
|---|---|---|
| **2 mm (gercek olcut)** | **0.4454** | **0.5168** |
| 3 mm | 0.5252 | 0.5504 |
| 5 mm | 0.6597 | 0.6134 |
| 10 mm | **0.8319** | 0.7311 |
| 20 mm | 0.9454 | 0.8235 |

**ACI toleransini gevsetince** (yanal 2mm'de SABIT):

| aci | HAVUZ | CIKTI |
|---|---|---|
| 10 derece | 0.4454 | 0.5168 |
| 20 derece | 0.4538 | 0.5252 |
| 45 derece | 0.4664 | 0.5378 |
| 180 derece (yon TAMAMEN yok sayilir) | 0.5924 | 0.6639 |

### OKUMA — bu tablo kampanyanin yon tayinidir

* **Aciyi tamamen yok saymak** havuzu yalnizca 0.4454 -> 0.5924 yapiyor
  (+0.147). Yani yon, kalan hatanin **kucuk** kismi.
* **Yanali 2 -> 10 mm yapmak** havuzu 0.4454 -> **0.8319** yapiyor
  (**+0.387**). GT'lerin **%83'u** icin havuzda, yonu ZATEN 10 derece
  icinde dogru olan bir aday **10 mm yakinda duruyor**.

**Yani darbogaz ne aday URETIMI ne de YON; YANAL KONUM HASSASIYETI.**

Bu, bagimsiz olarak olculmus iki seyle birebir tutarli: konum AUC 0.7053
vs yon AUC 0.8899, ve "yanal hata = segmentasyon kalitesi" bulgusu.

### YAPISAL DUZELTME: havuz recall'u TAVAN DEGILDIR
Yapisal kontrol ("ciktida var, havuzda yok") **31** cikti -- sifir olmasi
gerekirken. Sebep bulundu: **POSE HEAD gate'ten SONRA CP'leri OYNATIR**
(`wire_gate.pose_duzelt`). Yani bir aday, havuzda kabul kutusunun disinda
olup ciktida icine girebilir. Nitekim cikti (0.5168) havuzun (0.4454)
USTUNDE.

Bu, "havuz recall = baglayici kisit" seklindeki eski kaydin duzeltmesidir:
**pose head varken havuz recall'u bir TAVAN degildir.** Dogru tavan,
pose head'in onarabildigi yanal bandda olculen havuz kapsamasidir --
10 mm'de **0.8319**.

### KALAN EN BUYUK KALDIRACIN ADRESI
Pose head bugun 0.4454 -> 0.5168 tasiyor. 10 mm bandindaki tavan 0.8319.
Aradaki **+0.31**, kampanyada bulunan EN BUYUK acik basliktir ve
segmentasyonda degil, **YANAL KONUM REGRESYONUNDA**.

## 21.47 POSE HEAD KIRPMA SINIRI -- taranmamis bir hiperparametre bulundu

21.46'nin dogrudan sonucu olarak pose head'in kodu okundu:

```
mx = float(m.get("maks_mm", 3.0))        # wire_gate.py
c["point"] = p + dw * (min(n, mx) / n)   # duzeltme 3 mm'ye KIRPILIYOR
```

`maks_mm = 3.0` modelin pkl'ine yazili ve **tarandigina dair kayit YOK**
("denetim tavsiyesi: 2-3mm" notu var, olcum yok).

**Bunun neden onemli oldugu.** Teshis, GT'lerin **%83'u** icin havuzda,
yonu zaten 10 derece icinde dogru olan bir adayin **10 mm** yakinda
oldugunu gosterdi. 3 mm'lik kirpma bu bandin ucte birini bile kapsamiyor.

**Egitim tarafi bunu DESTEKLIYOR** (dogrulandi, varsayilmadi):
* `q3_pose_veri_buyut.py`: eslesme toleransi `tt = max(3.0, 0.06*diag)` --
  yani model ZATEN 3 mm'den buyuk yer degistirmeler gormus.
* `q1_pose_head.py`: **hedefler KIRPILMIYOR**; kirpma yalnizca CALISMA
  ANINDA uygulaniyor.

Yani model buyuk duzeltmeleri ogrenmis olabilir ama uretimde onlari
uygulamasina IZIN VERILMIYOR.

**OLCUM TASARIMI (verimli).** Her `maks_mm` degeri icin zinciri yeniden
kosmak yerine, kirpma ONCESI yer degistirme vektoru
(`wire_gate.POSE_KANCA`, cevre degiskeniyle acilir) dokuluyor; boylece
**tek kosudan** butun tarama cevrimdisi kuruluyor
(`sonda_pose_kirpma.py`). Betik ayrica mx=3.0'da yeniden kurdugu metrigin
dokumun kendi metrigine ESIT oldugunu dogruluyor -- esit degilse taramayi
GECERSIZ ilan ediyor.

## 21.48 Y22 MC DROPOUT KAPANDI — kapinin altinda, maliyeti 8 kat

VAL 100 parca, esli parca bootstrap, MC(8):

| yol | metrik | taban | MC(8) | fark | %95 GA | poz% |
|---|---|---|---|---|---|---|
| olculen | tespit | 0.5553 | 0.5528 | −0.0024 | [−0.0308,+0.0255] | 44.0 |
| olculen | robot | 0.5376 | 0.5425 | +0.0052 | [−0.0217,+0.0324] | 65.1 |
| olculen | robot-ISR | 0.4668 | 0.4738 | **+0.0071** | [−0.0180,+0.0319] | 71.1 |

GA sifiri iciyor, kazanc kat gurultusunun (±0.008) altinda, **maliyeti
8 kat cikarim**. **KAPANDI.**

### Yan bulgu: "+0.0000" bu kez BUG DEGIL, KAPSAM
Ilk kiyas `saha` yolunda yapildi ve **tam +0.0000** verdi -- bu gecenin
sessiz no-op imzasi. Sebep arandi ve bulundu: `saha` yolu
`robot_cp.extract` icinde KENDI cikarimini yapar; `EZ_MCDROP` yalnizca
sondanin `pbs`'ini etkiler, o da **olculen** yolu besler. Yani kanca
`saha`ya ULASMIYOR -- kol calismiyor degil, o yolda YOK.

Duman testi (21.43) onceden kurulmus oldugu icin bu ayrim iki dakikada
yapildi: kolun gercekten etkili oldugu (0.047 vs 0.000001) zaten
biliniyordu, dolayisiyla "+0.0000" ancak kapsam sorunu olabilirdi.
**Duman testinin bedelini burada odedi.**

## 21.49 SONDA KANCALARI YALNIZ `olculen` YOLUNA ULASIR — kapsam kurali

Bu gece IKI kol, `saha` yolunda **tam +0.0000** verdi (MC dropout ve
remesh varyanti). Ikisi de sessiz no-op DEGILDI; ikisinin de sebebi
ayniydi ve artik kural olarak yaziliyor:

**`saha` yolu = `robot_cp.extract`, ve `extract` KENDI cikarimini ve KENDI
remesh'ini icinde yapar.** Sondanin cevre degiskenleri (`EZ_MCDROP`,
`EZ_REMESH`, `EZ_CKPT`) sondanin urettigi `pbs` ve `V,F`'yi degistirir --
onlar da yalnizca **`olculen`** yolunu besler.

**KURAL: cikarim/mesh duzeyindeki her kol `olculen` yolunda olculur.
`saha` yolunda +0.0000 gormek, kolun olu oldugu anlamina GELMEZ.**

Bu ayrimi bu gece iki dakikada yapabilmemizin sebebi, kollarin ONCE
duman testinden gecirilmis olmasidir (21.43): kolun gercekten etkili
oldugu bagimsiz olarak biliniyordu.

## 21.50 Y16 — REMESH HEDEFI 5000 KESIN OLARAK DAHA KOTU

VAL 100 parca, **olculen** yolu, esli parca bootstrap. Taban = 6000
(mevcut davranis):

| metrik | 6000 | 5000 | fark | %95 GA | poz% |
|---|---|---|---|---|---|
| tespit | 0.5553 | 0.5198 | **−0.0353** | [−0.0600,−0.0079] * | 0.4 |
| robot | 0.5376 | 0.4982 | **−0.0393** | [−0.0632,−0.0141] * | 0.1 |
| robot-ISR | 0.4668 | 0.4371 | **−0.0300** | [−0.0548,−0.0053] * | 0.8 |

(* = GA sifiri icermiyor.) Yani tezden gelen **6000 hedefi iyi secilmis**;
asagi cozunurluk kesin olarak zarar veriyor. Topluluk hukmu icin 7200
bekleniyor.

## 21.51 GECE 3 (2026-08-13/14) DENETIM DURUMU

Uc urun dosyasi degistirildi; **hepsi cevre degiskeniyle kapili ve
varsayilan davranis BIT-AYNI**:

| dosya | degisiklik | varsayilan |
|---|---|---|
| `diffusionnet.py` | `predict(..., mc_dropout=T)` | `0` = eski kod yolu |
| `wire_gate.py` | `CP_POSE_MAKS_MM` ezmesi + `POSE_KANCA` | ezme yok, kanca kapali |
| `robot_cp.py` | `HAVUZ_KANCA` (gate oncesi havuz) | kanca kapali |

**DOGRULAMA:**
* `duman_testi.py` -> **GECTI** (gercek STEP, 4 kontrol noktasi, 2 CP
  uretildi, 32 s)
* `geri_al.py --kontrol` -> **1 sapma: yalniz `cp_config.json`** (bilincli)
* Uc dosya da git'te izleniyor (`M robot_cp.py`, `M wire_gate.py`,
  `M diffusionnet.py`) -> geri alinabilir
* **D7 OKUNMADI** (2 okuma hakki duruyor) · **LOCKED harcanmadi**

## 21.52 OLCUM DUZELTMESI: yeni sondalarda TESPIT tanimi kanoniklestirildi

Bu gece yazilan dort sonda (`sonda_dokum_kiyas`, `sonda_halka_eksen`,
`sonda_pose_kirpma`, `sonda_remesh_toplulugu`) "tespit"i **2.0 mm sabit**
toleransla hesapliyordu. Kanonik tanim farkli:

```
tespit : esle_macar(..., tol=0.0, am=180.0, pct=True)  -> tt = max(3.0, 0.06*diag)
robot  : esle_macar(..., tol=2.0,  am=10.0, pct=False) -> 2 mm SABIT
```

Yani **robot metrikleri zaten kanonikti**, tespit ise DAHA SIKI
hesaplaniyordu (olculen yolda 0.6139 yerine dogrusu 0.6101; farkin isareti
ve hukumler degismedi). Kanonik tanima gecildi.

**Duzeltilmis sayilar:**

| kiyas | tespit farki (eski -> yeni) | hukum |
|---|---|---|
| remesh 5000 vs 6000 | −0.0353 -> **−0.0380** * | degismedi (KESIN kotu) |
| MC(8) vs taban | −0.0024 -> **−0.0007** | degismedi (notr) |

**Onemli:** 21.46'daki baglayici-kisit taramasi **yalnizca robot olcutunu**
(2 mm / 10 derece) kullanir; o tanim bastan kanonikti, dolayisiyla
**o bulgu bu duzeltmeden ETKILENMEZ.**

## 21.53 POSE KIRPMA SINIRI **BAGLAMIYOR** — sinirlayan MODELIN KENDISI

VAL 100 parca / 617 CP, olculen yol, tek kosudan cevrimdisi tarama.
Dogrulama gecti (mx=3.0 yeniden kurulan == dokum).

| maks_mm | tespit | robot | robot-ISR | 3.0'a gore fark |
|---|---|---|---|---|
| 0.0 (pose KAPALI) | 0.4980 | 0.4605 | 0.3837 | **−0.0799** * |
| 1.0 | 0.5889 | 0.5450 | 0.4495 | −0.0141 * |
| 2.0 | 0.6124 | 0.5685 | 0.4636 | +0.0000 |
| **3.0 (MEVCUT)** | 0.6139 | 0.5685 | 0.4636 | — |
| 4 / 5 / 6 / 8 / 10 / 15 / **SINIRSIZ** | 0.6139 | 0.5685 | 0.4636 | **+0.0000** |

**Kirpmayi kaldirmak HICBIR SEY degistirmiyor.** Sebep dogrudan olculdu:

> Modelin onerdigi yer degistirme (617 CP): ortanca **0.48 mm**,
> %75 0.93 mm, %90 1.64 mm, **maksimum 2.87 mm**.
> **3 mm'yi asan oneri orani: %0.0**

Guven kapili varyantlar da aynen +0.0000 verdi (kapinin ustunde/altinda
degisen bir sey yok, cunku hicbir oneri kirpilmiyor).

### HUKUM VE KOK NEDEN
`maks_mm` **atil bir hiperparametre**; sinirlayan sey **modelin kendisi**.
Pose head buyuk duzeltmeler ONERMIYOR.

Mekanizma bir **SECIM YANLILIGI**: egitim verisi (`q3_pose_veri_buyut.py`)
yalnizca `tt = max(3.0, 0.06*diag)` icinde **ZATEN ESLESMIS** adaylardan
kuruluyor. Eslesmis bir adayin artigi tanim geregi KUCUKTUR. Yani model,
buyuk artiklari **hic gormedi** -- ogrenip de uygulayamadigi degil,
**ogrenmedigi** icin onermiyor.

### AMA POSE HEAD CALISIYOR — ve tam kapasite kullaniliyor
Kolu kapatmak (mx=0) robot-ISARETLI'yi **−0.0799** dusuruyor
(GA [−0.1079,−0.0557], orneklerin %0'i pozitif). Yani bilesen gercek ve
tasidigi her seyi zaten tasiyor.

### SIRADAKI ADIM (net ve dar)
21.46'nin +0.31'lik acigini almanin yolu kirpmayi gevsetmek DEGIL,
**pose head'i buyuk artiklari GOREREK yeniden egitmek**: egitim eslesme
toleransi (bugun ~3-6 mm) 15 mm'ye acilir, boylece 10 mm bandindaki
adaylar da hedefe girer. Veri zaten diskte ve **ag cikarimi
GEREKTIRMIYOR** (q3 onbellekli npz'lerden calisiyor).

## 21.54 POSE HEAD YENIDEN EGITIMI — iki mekanizma ayristi, IKI OLCUM KUSURU YAKALANDI

21.53 "model buyuk duzeltme onermiyor" dedi. Iki aday mekanizma vardi;
ikisi de olculdu.

**Veri once incelendi (varsayilmadi):**

| veri | satir | yanal hedef ortanca | %90 | maks | **>3mm orani** |
|---|---|---|---|---|---|
| taban (`pose_veri`) | 6330 | 0.97 mm | 3.02 | 12.10 | **%10.2** |
| genis (`Q3_TOL=15`) | 7154 | 1.12 mm | 4.60 | 14.93 | **%18.4** |

**Bu, ilk hipotezi KISMEN CURUTTU.** Taban veride zaten %10.2 buyuk hedef
vardi; yani model buyuk artiklari "hic gormemis" degil. Demek ki asil
mekanizma **REGRESYON BUZULMESI**: orman yaprak ortalamasi ucdegerleri
iceri ceker. Secim yanliligi ikincil.

### YAKALANAN IKI OLCUM KUSURU (ikisi de ilk kosuda vurdu)

**1. Sizinti kapisi SESSIZCE hicbir seyi elemedi.** `split3.json` icinde
`val` bir SOZLUK (`{"n":…, "parts":[…]}`); uzerinde dogrudan donmek
`'n'`,`'parts'` anahtarlarini verir. Kapi "0 VAL grubu cikarildi" dedi ve
bu bir UYARI olarak gecti. Artik `["val"]["parts"]` okunuyor ve **iki
`assert`** var (liste bos olamaz, gruplar bos olamaz). Bu kapi olmasaydi
VAL olcumu KIRLI olurdu.

**2. Iki aday KENDI veri kumesinde puanlandi -- kiyaslanamaz.** Genis veri
"daha kotu" gorunuyordu (kutuda %72.7 vs %81.3), oysa genis kume DAHA ZOR
satirlar iceriyor: iki oran ayni satirlarda olculmemis. Artik butun
adaylar **ORTAK degerlendirme kumesinde** puanlaniyor; egitim kumesinden
degerlendirme katinin gruplari cikarilarak (grup-disi, sizintisiz).

Bu iki kusur da "kol pozitif/negatif" hukmunu ters cevirebilecek
cinstendi ve **sonuc yazilmadan once** yakalandi.

## 21.55 POSE HEAD YENIDEN EGITIMI — ORTAK KUMEDE SONUC

Sizinti kapisi onarildiktan (87 VAL geometri grubu CIKARILDI) ve butun
adaylar **ayni 5644 satirda** puanlandiktan sonra:

| egitim verisi | ayar | kutuda (<=2mm) | artik ortanca | oneri maks | >3mm oneri |
|---|---|---|---|---|---|
| taban | yaprak>=5 (**MEVCUT**) | 76.5% -> 80.8% | 0.68 mm | 5.45 | %0.2 |
| taban | yaprak>=2 | 76.5% -> 81.6% | 0.68 mm | 6.11 | %0.4 |
| **taban** | **yaprak>=1** | 76.5% -> **81.7%** | 0.67 mm | 6.33 | %0.5 |
| genis (tol15) | yaprak>=5 | 76.5% -> 80.8% | 0.74 mm | 5.96 | %0.6 |
| genis (tol15) | yaprak>=2 | 76.5% -> 80.8% | 0.74 mm | 6.29 | %0.8 |
| genis (tol15) | yaprak>=1 | 76.5% -> 81.1% | 0.74 mm | 6.49 | %1.0 |

### IKI TEMIZ HUKUM

**1. GENIS VERI YARDIM ETMIYOR.** Ortak kumede taban veriyle egitilen her
ayar, genis veriyle egitilenden esit ya da iyi. Yani "secim yanliligi"
hipotezi **CURUDU**; kalan mekanizma **regresyon buzulmesi**.

**2. BUZULMEYI GEVSETMEK COK KUCUK BIR SEY KAZANDIRIYOR.**
`min_samples_leaf` 5 -> 1: kutuda-oran +0.9 puan (80.8 -> 81.7),
en buyuk oneri 5.45 -> 6.33 mm. Ama **3 mm'yi asan oneri orani hala
yalnizca %0.5.**

### BUNUN 21.46'DAKI +0.31 ICIN ANLAMI (durust okuma)
Model sinifi ve oznitelikler **10 mm'lik hatalari onaracak bilgiyi
tasimiyor**: en iyi varyant bile adaylarin %99.5'ine 3 mm'den kucuk
duzeltme oneriyor. Yani 21.46'daki acik, **gate ozniteliklerinden
son-islem regresyonuyla ALINAMAZ**; ya aday URETIMI (segmentasyon /
aday konumlari) duzelecek, ya da pose kafasina **yerel geometriyi
dogrudan goren** yeni oznitelikler verilecek.

Bu, acigin var olmadigi anlamina gelmez -- **nereden alinamayacagini**
olcerek daraltir.

Aday model yine de tam zincirde olculuyor
(`CP_POSE_MODEL=results/pose_head_yeni.pkl`); beklenti kapinin altinda.

## 21.56 Y16 REMESH-VARYANT TOPLULUGU KAPANDI

VAL 100 parca, olculen yol, esli parca bootstrap. Taban = 6000 (mevcut).
**Egitim yok** -- yani bu kol tohum gurultusune tabi degil, tek kosuda
hukum giyebilir.

| varyant | tespit | robot | robot-ISR | rbi farki | %95 GA | poz% |
|---|---|---|---|---|---|---|
| **TABAN (6000)** | 0.5553 | 0.5376 | 0.4668 | — | — | — |
| tek 5000 | 0.5198 | 0.4982 | 0.4371 | −0.0300 | [−0.0548,−0.0053] * | 0.8 |
| tek 7200 | 0.5573 | 0.5453 | 0.4735 | +0.0066 | [−0.0208,+0.0314] | 69.3 |
| **birlesim (oy>=1)** | 0.5667 | 0.5446 | **0.4783** | **+0.0112** | [−0.0073,+0.0314] | 87.9 |
| oylama (oy>=2) | 0.5558 | 0.5363 | 0.4655 | −0.0012 | [−0.0163,+0.0144] | 43.4 |
| oybirligi (oy>=3) | 0.4995 | 0.4796 | 0.4219 | −0.0453 | [−0.0801,−0.0133] * | 0.2 |

### HUKUM: **KAPANDI**
* 5000 ve oybirligi **KESIN kotu**.
* 7200 ve birlesim pozitif ama **GA sifiri iciyor**; birlesim +0.0112 ile
  bu gecenin diger "kapiya yakin" kollariyla ayni profilde -- ve
  **maliyeti 3 kat cikarim**.
* Oylama tam notr.

**Egilim anlamli:** 5000 < 6000 < 7200 (monotonik), yani daha yuksek
cozunurluk yardim ediyor. Ama 6000 -> 7200 kazanci kapinin altinda.
Tezden gelen **6000 hedefi iyi secilmis**; asagi inmek kesin zarar.

## 21.57 Y24 CRF — YANLIS KOVAYA KONMUSTU, DUZELTILDI

Y24 (CRF duzeltmesi) 21.42'de "egitim gerektiren, sure yetmeyen" kollar
arasina konmustu. **Bu yanlisti:** CRF bir SON-ISLEMDIR, egitim
gerektirmez. Kova duzeltildi ve kol bu gece kosuldu.

**Kurulum.** Mesh kenarlari uzerinde ortalama-alan yaklasimi: her turda
her tepenin olasilik vektoru komsularinin ortalamasiyla `w` agirliginda
harmanlanir, sonra normalize edilir. **Ek parametre ogrenilmiyor.**
`EZ_CRF="tur,agirlik"`; bos birakilirsa bit-ayni taban.

**DUMAN TESTI (sentetik ikosahedron) — GECTI:**

| kontrol | sonuc |
|---|---|
| `tur=0` bit-ayni mi | **EVET** |
| `tur=3` etkisi (ort. abs fark) | 0.077408 |
| olasilik satirlari 1'e toplaniyor mu | EVET |
| tekil aykiri etiketli tepe duzeliyor mu | **EVET** |

**Neden bu kol darbogaza nisan aliyor:** 21.46 baglayici kisiti YANAL
KONUM olarak belirledi ve yanal hata bagimsiz olarak segmentasyon
kalitesine baglanmisti. CRF, tekil yanlis etiketli tepeleri bastirarak
aday MERKEZLERINI oynatir -- yani dogrudan yanal hataya dokunur.

## 21.58 ADAY POSE HEAD UCTAN UCA **KESIN OLARAK KOTU** — vekil metrik isaret degistirdi

`results/pose_head_yeni.pkl` (taban veri, yaprak>=1, `maks_mm=10`) tam
zincirde olculdu. VAL 100 parca, **saha** yolu (pose head `extract`in
icinde oldugu icin dagitilan yol budur), esli parca bootstrap:

| metrik | dagitilan | aday | fark | %95 GA | poz% |
|---|---|---|---|---|---|
| tespit | 0.7878 | 0.7847 | −0.0031 | [−0.0080,+0.0000] | 0.0 |
| robot | 0.5764 | 0.5059 | **−0.0697** | [−0.0993,−0.0416] * | 0.0 |
| robot-ISR | 0.4839 | 0.4401 | **−0.0437** | [−0.0670,−0.0228] * | 0.0 |

**KESIN KOTU. Aday REDDEDILDI, dagitilan model yerinde kaliyor.**

### BU GECENIN EN SERT DERSI: VEKIL METRIK ISARET DEGISTIRDI
Aday, cevrimdisi vekilde **KAZANIYORDU**: ortak degerlendirme kumesinde
kabul kutusuna girme orani %80.8 -> **%81.7** (+0.9 puan), grup-disi,
sizinti kapisi kapali. Uctan uca ise **−0.0437**.

Yani "yanal artigi daha iyi tahmin etmek" ile "robot metrigini
yukseltmek" AYNI SEY DEGIL. Muhtemel mekanizma: vekil TUM adaylarin
ortalama artigini olcer; uctan uca metrik ise ZATEN kutuda olan adayin
disari itilmesini **cift** cezalandirir (bir TP gider, bir FP gelir).
Yaprak>=1 daha oynak duzeltmeler uretiyor ve `maks_mm=10` bunlarin
buyuklerinin gecmesine izin veriyor.

**Kural: son-islem kollarinda cevrimdisi vekil KARAR VERDIRMEZ; yalnizca
uctan uca olcum verdirir.** Vekil olsa olsa hangi adaylarin uctan uca
olcumu HAK ETTIGINI secer.

### AYRISTIRMA KOSULUYOR
Kaybin sebebi MODEL mi (yaprak>=1) yoksa KIRPMA mi (3 -> 10 mm)?
Ayni aday model `CP_POSE_MAKS_MM=3.0` ile tekrar olculuyor. Eski model
2.87 mm'yi hic asmadigi icin (21.53) kirpmanin ancak yeni modelde
baglayici hale geldigi biliniyor -- yani suphe once kirpmada.

## 21.59 GECE 3 KAPANIS TABLOSU (2026-08-13/14)

Bu gece **YEDI kol** olculdu ve kapandi; **bir kol** dagitima aday kaldi
(halka isareti, onceki geceden); **bir buyuk teshis** cikti.

| kol | olcum | hukum |
|---|---|---|
| Y5 aux-wire | −0.0072 (3 eslesen tohum) | curudu |
| Y13 sinir kaybi | −0.0113 / −0.0044 | kapandi |
| Y22 MC dropout | +0.0071 (GA sifiri iciyor), 8x maliyet | kapandi |
| Y16 remesh 5000 | −0.0300 (GA sifirsiz) | **kesin kotu** |
| Y16 remesh 7200 | +0.0066 (GA sifiri iciyor) | kapi alti |
| Y16 topluluk birlesim | +0.0112 (GA sifiri iciyor), 3x maliyet | kapi alti |
| Y16 topluluk oybirligi | −0.0453 (GA sifirsiz) | **kesin kotu** |
| halka normali = EKSEN | −0.3860, monotonik, %0 poz | **kesin kapandi** |
| pose kirpma gevsetme | +0.0000 (kirpma ATIL) | kapandi |
| aday pose head | −0.0437 (GA sifirsiz, %0 poz) | **REDDEDILDI** |

**Teshis (kol degil):** baglayici kisit **YANAL KONUM** (21.46) --
kampanyanin yon tayini.

### GECENIN DORT METODOLOJIK CIKTISI
1. **Segmentasyon tohum gurultusu 0.046** olculdu; bu buyuklugun altindaki
   hicbir seg kolu tek kosuyla hukum giymez (21.41).
2. **Sonda kancalari yalniz `olculen` yoluna ulasir**; `saha`da +0.0000
   gormek kolun olu oldugu anlamina gelmez (21.49).
3. **Ara degerler uretildikleri yerden yakalanmali**; ayni kodu disaridan
   cagirmak onbellek bagimliligi varsa farkli sonuc verir (21.44).
4. **Cevrimdisi vekil karar verdirmez**; bir kol vekilde +0.9 puan
   kazanip uctan uca −0.0437 verdi (21.58).

## 21.60 BAGLAYICI KISIT — TAM VAL (100 parca / 660 GT) ile YENIDEN OLCULDU

21.46 ilk kez 40 parcada olculmustu. `_dokum_pose.json` kosusunda havuz
kancasi 100 parcanin hepsinde acikti; tarama tam kumede tekrarlandi.
**Bunlar kayda gecen sayilardir.**

**YANAL taramasi** (aci 10 derece SABIT, ISARETLI, eksenel <=40mm):

| yanal | HAVUZ | CIKTI |
|---|---|---|
| **2 mm (gercek olcut)** | **0.4242** | **0.4682** |
| 3 mm | 0.4955 | 0.5106 |
| 5 mm | 0.6242 | 0.5727 |
| 8 mm | 0.7303 | 0.6364 |
| **10 mm** | **0.7818** | 0.6697 |
| 15 mm | 0.8455 | 0.7076 |
| 20 mm | 0.9000 | 0.7591 |

**ACI taramasi** (yanal 2 mm SABIT):

| aci | HAVUZ | CIKTI |
|---|---|---|
| 10 derece | 0.4242 | 0.4682 |
| 20 derece | 0.4364 | 0.4742 |
| 45 derece | 0.4409 | 0.4803 |
| 90 derece | 0.4970 | 0.4985 |
| **180 (yon TAMAMEN yok sayilir)** | **0.5939** | 0.6273 |

### HUKUM (40 parcalik ilk olcumle AYNI, buyuklukler biraz daha ilimli)

| gevsetme | havuz kazanci |
|---|---|
| yonu TAMAMEN mukemmel yapmak | **+0.1697** |
| yanali 2 -> 10 mm yapmak | **+0.3576** |

**Yanal, yonun iki katindan fazlasini tasiyor.** Darboğaz **YANAL KONUM**.
Sunumda 100 parcalik bu sayilar kullanilir (40 parcalik ilk olcum
0.8319 diyordu; dogrusu **0.7818**).

## 21.61 Y24 CRF — ILK AYAR TAM NOTR

VAL 100 parca, olculen yol (kanca yalniz oraya ulasir, bkz. 21.49),
esli parca bootstrap. Ayar: **2 tur, agirlik 0.3**.

| metrik | taban | CRF | fark | %95 GA | poz% |
|---|---|---|---|---|---|
| tespit | 0.6101 | 0.6043 | −0.0054 | [−0.0247,+0.0157] | 29.0 |
| robot | 0.5376 | 0.5330 | −0.0043 | [−0.0242,+0.0174] | 32.6 |
| robot-ISR | 0.4668 | 0.4670 | **+0.0004** | [−0.0199,+0.0218] | **50.0** |

**Tam sansa esit** (%50.0 pozitif, GA simetrik). Duman testi kolun
gercekten calistigini gosterdigi icin bu bir no-op degil, gercek bir
**notr** sonuc.

**KAPATILMADAN ONCE IKINCI AYAR KOSULUYOR** (`docs/KAPANAN_KOLLAR_DENETIMI.md`
kurali: "sondanin cozunurlugu toleransi tutuyor mu" -- 2 tur/0.3 HAFIF bir
duzeltmedir). Ikinci ayar: **4 tur, agirlik 0.5**.

## 21.62 AYRISTIRMA SONUCU: kayip **MODELDEN**, kirpmadan DEGIL

Ayni aday model, yalnizca kirpma degistirilerek tekrar olculdu:

| kosu | tespit | robot | robot-ISR |
|---|---|---|---|
| aday model, `maks_mm=10` | 0.7847 | 0.5059 | 0.4401 |
| aday model, `maks_mm=3` | 0.7847 | 0.5059 | 0.4401 |

**BIREBIR AYNI.** Kirpma yeni modelde de baglamiyor -> kayip tamamen
**modelin kendisinden**.

### CALISMA ANINDA ONERI BUYUKLUKLERI (617 CP)

| model | ortanca | %90 | maks | >3mm |
|---|---|---|---|---|
| dagitilan | 0.484 mm | 1.636 | 2.868 | **%0.00** |
| **aday (yaprak>=1)** | **0.427 mm** | 1.166 | **2.537** | **%0.00** |

Aday model calisma aninda **DAHA KUCUK** duzeltmeler oneriyor -- oysa
cevrimdisi OOF'ta en buyuk onerisi 6.33 mm idi.

### UCUNCU KATMAN: CEVRIMDISI POPULASYON != CALISMA ANI POPULASYONU
OOF degerlendirmesi **pose egitim satirlarinda** yapiliyor; calisma
anindaki oznitelikler ise `wire_gate.feats_for`in **gate'ten gecmis**
adaylar icin urettikleri. Bunlar farkli populasyonlar. Sonuc: modelin
"buyuk duzeltme onerebilme" ozelligi bile calisma anina TASINMADI.

Yani vekil sadece METRIGI degil, modelin DAVRANISINI da yanlis tahmin
etti. **Kol kapandi; dagitilan pose head yerinde kaliyor.**

**Bilanco:** pose cephesinde ucu de olculdu ve ucu de kapandi --
kirpmayi gevsetmek (+0.0000), genis veriyle egitmek (vekilde bile
yardim etmedi), buzulmeyi gevsetmek (uctan uca −0.0437). 21.60'taki
+0.31'lik acik **son-islem regresyonuyla alinamaz**; bu artik uc bagimsiz
olcume dayaniyor.

## 21.63 Y24 CRF — TEK AYARLA KAPATILMADIGI ICIN KOL DIRILDI

Ilk ayar (2 tur / 0.3) tam notrdu (%50.0 pozitif) ve normalde "olu"
denirdi. `KAPANAN_KOLLAR_DENETIMI` kurali geregi ikinci, DAHA GUCLU ayar
kosuldu:

| ayar | tespit | robot | robot-ISR | rbi farki | %95 GA | poz% |
|---|---|---|---|---|---|---|
| taban | 0.6101 | 0.5376 | 0.4668 | — | — | — |
| 2 tur / 0.3 | 0.6043 | 0.5330 | 0.4670 | +0.0004 | [−0.0199,+0.0218] | 50.0 |
| **4 tur / 0.5** | 0.6043 | 0.5432 | **0.4766** | **+0.0100** | [−0.0128,+0.0335] | **80.5** |

**MONOTONIK VE ANLAMLI YONDE**: duzeltme guclendikce robot metrigi
yukseliyor (+0.0004 -> +0.0100), pozitif ornek orani 50.0 -> 80.5.
Tespit her iki ayarda da hafif negatif (−0.005) -- yani duzeltme
kesinlikten biraz verip robot-uygunlugundan aliyor.

**Bu, gecenin "yelpaze 64 yonde olu, 256 yonde +0.0676" dersinin
tekrarıdır:** ilk sonda cozunurlugu yetmiyordu. Kol tek ayarla
kapatilsaydi bu egilim gorulmeyecekti.

**UCUNCU AYAR KOSULUYOR (6 tur / 0.7)** -- egilimin devam edip etmedigini,
yoksa tepe noktasinin gecilip gecilmedigini belirlemek icin. Henuz
hicbir ayar GA kapisini gecmedi; kol DAGITIMA ADAY DEGIL, ACIK.

## 21.64 CRF URUN YOLU KANCASI — kuruldu ve DOGRULANDI

Sonda kancalari yalniz `olculen` yoluna ulasiyor (21.49). CRF kazanci
oradan olculdugu icin, kazanan ayarin **DAGITILACAK** yolda da
olculebilmesi sart. Sebep hafizada: kanonik zincir blogu olcum betiginde
**+0.0151**, uretim egiticisinde **−0.0138** vermisti -- yol farki KARAR
DEGISTIRIYOR.

`robot_cp.extract` icine `CP_CRF="tur,agirlik"` kancasi eklendi
(varsayilan bos = bit-ayni mevcut davranis).

**DOGRULAMA:**

| kontrol | sonuc |
|---|---|
| `duman_testi.py` (kanca KAPALI) | **GECTI**, 2 CP |
| `duman_testi.py` (kanca ACIK, 6/0.7) | **GECTI**, ayni 2 CP |
| kanca kod yolu: env okundu / import / etki | 0.08829 ort. abs fark, tepelerin **%67.6**'sinin etiketi degisiyor |

Duman testinde ciktinin AYNI cikmasi **no-op degil**: o parca basit ve
iki CP'si duzeltmeye dayanikli. Kanca kod yolu ayrica izole olarak
kosuldu ve etkisi olculdu -- bu gece "tam +0.0000" tuzagina dusmemek
icin artik standart adim.

## 21.65 UZLASTIRMA: "%94'unun yakininda hic tahmin yok" ile 21.60 CELISMIYOR

Bolum 20 (hata otopsisi) kacan GT'lerin **%94.2**'sinin yakininda HIC
tahmin olmadigini soyluyor. 21.60 ise havuzda GT'lerin **%78'i** icin
10 mm yakinda dogru yonlu bir aday oldugunu soyluyor. Ilk bakista
celiski gibi duruyor; degil -- **iki farkli kume olculuyor**:

| olcum | kume | anlami |
|---|---|---|
| Bolum 20: %94.2 "bos" | **CIKTI** (gate + secim SONRASI) | urunun verdigi CP'ler |
| Bolum 21.60: %78 kapsama | **HAVUZ** (gate ONCESI) | uretilen tum adaylar |

Yani: **aday havuza GIRIYOR, ama cikisa ULASMIYOR ya da ulastiginda
yeterince yakin degil.** Ikisi birlikte okununca tablo netlesir:

* Havuz 2 mm'de 0.4242 -> aday **uretimi** de tam degil.
* Havuz 10 mm'de 0.7818 -> ama aday **cogunlukla ORADA**, sadece
  2 mm'lik kutuya girecek hassasiyette degil.
* Cikti 10 mm'de 0.6697 -> havuzdaki bu adaylarin bir kismi ayrica
  cikisa da ulasamiyor.

**Bolum 20'nin "ADAY URETIMI + KONUM SKORLAMA sorunu" hukmu AYAKTA;
21.60 onu daha keskin hale getiriyor:** iki bilesenden agir basani
**KONUM HASSASIYETI**. Bolum 20'nin son-islem tavani hesabi (%12) da
ayakta -- nitekim bu gece uc son-islem kolu (halka-eksen, CRF ilk ayar,
pose yeniden egitimi) bu tavani asamadi.

## 21.66 ADAY TURETME ESIGI ACILMADI — kayitli tuzak

Baglayici kisit YANAL KONUM oldugu icin akla gelen ucuz kol,
`prediction_postproc` icindeki aday turetme parametrelerini (ozellikle
`vertex_confidence_mask`) taramaktir: bu esik hangi tepelerin adayi
olusturdugunu belirler, yani KONUMU dogrudan oynatir.

**ACILMADI.** Sebep `cp_config.json`'un kendi notunda yazili:

> `vc_035_REVERTED_2026_07_25`: vc0.35 REDDEDILDI -- tarama iki ureticide
> de +0.01 onerdi ama TAM dagitim OOF'unda WEI 0.641 -> 0.629 (DUSTU).
> "Bir taramanin tam dagitim kosusuna gore yaniltmasi IKINCI kez"
> (digeri k_eig128).

Yani bu parametre icin **tarama ile tam dagitim arasindaki fark daha once
IKI KEZ karar degistirmis**. Bu gece ayni ders ucuncu kez yasandi
(pose head vekili, 21.58). Sinirli surede taramaya girmek, bu kadar
kaydedilmis uyariya ragmen, olcum degil kumar olurdu.

**Not:** parametreler zaten 2026-08-06'da yeniden ayarlanmis
(`p1_2026_08_06`: cluster/dedupe/oy 3/10/5 -> 1/2/2, cok-CP +0.0331).
Yani bu cephe bayat degil.

## 21.67 Y24 CRF **KAPANDI** — tepe yapip donuyor, yani gurultuye uydurma

Tarama tamamlandi (VAL 100 parca, olculen yol, esli parca bootstrap):

| ayar | tespit | robot | robot-ISR | rbi farki | %95 GA | poz% |
|---|---|---|---|---|---|---|
| taban | 0.6101 | 0.5376 | 0.4668 | — | — | — |
| 2 tur / 0.3 | 0.6043 | 0.5330 | 0.4670 | +0.0004 | [−0.0199,+0.0218] | 50.0 |
| **4 tur / 0.5** | 0.6043 | 0.5432 | **0.4766** | **+0.0100** | [−0.0128,+0.0335] | **80.5** |
| 6 tur / 0.7 | 0.6011 | 0.5390 | 0.4645 | −0.0020 | [−0.0249,+0.0222] | 43.0 |

**HUKUM: KAPANDI.**

**Gerekce -- MONOTONIK DEGIL.** Egri +0.0004 -> +0.0100 -> −0.0020, yani
tepe yapip donuyor. Bu kampanyada ayni imza daha once halka-isaret esik
taramasinda gorulmustu (95/105/115/125/140/160 derece: +0.0345/+0.0298/
+0.0298/+0.0063/−0.0063/−0.0047) ve **"monotonik degil, yani gurultuye
uydurma"** denip sade kural tercih edilmisti. Ayni olcut burada da
uygulaniyor.

Ustelik tepe degeri (+0.0100) **GA'si sifiri iceren** bir sayidir ve bu
gecenin diger "kapiya yakin" kollariyla ayni banttadir (Y16 birlesim
+0.0112, Y8 etiket kalitesi +0.0082, Y22 MC dropout +0.0071).
**Uc ayar arasindan en iyisini secmek, kapiyi gecmeyen bir sayiyi
tarama ile "gecirmek" olurdu.**

**Tespit her ayarda negatif** (−0.005 … −0.009): duzeltme kesinlikten
veriyor, robot-uygunluguna guvenilir bir sey katmiyor.

### YAN KAZANIM: KOL YINE DE TEK AYARLA KAPATILMADI
Ilk ayar (2/0.3) tam notrdu. `KAPANAN_KOLLAR_DENETIMI` kurali geregi iki
ayar daha kosuldu; egrinin sekli ancak boyle gorulebildi. Kol bir ayarla
"olu" denip kapatilsaydi hukum ayni olurdu ama **gerekce yanlis** olurdu
("etkisiz" yerine "gurultuye uydurma"). Kural ise yaradi.

**`CP_CRF` kancasi kodda KALIYOR** (varsayilan kapali, bit-ayni davranis):
ileride segmentasyon kalitesi degisirse kol yeniden olculebilir.

## 21.68 TOPLULUK GENISLETME KOLU (egitim YOK) — kurulum

Gecenin olculmus baskın etkisi **segmentasyon tohum gurultusu 0.046**
(21.41). Topluluk, bu gurultuyu ortalamayla azaltmanin dogrudan yoludur
ve **yeni egitim GEREKTIRMEZ** -- gereken checkpointler diskte.

Urun bugun **4** checkpoint kullaniyor:
`recall_hard_s2` + `recall_hard_keig96_s0/s1/s2`.

Diskte kullanilmayan iki grup var:

| aday | gerekce |
|---|---|
| **A: 5 uye** = urun 4 + `recall_hard_keig96_s3` | AYNI recetenin 4. tohumu; urun toplulugunda YOK. Saf varyans azaltma, recete degisikligi SIFIR |
| **B: 8 uye** = A + `y1b_aug0.15_s0/s1/s2` | augmentasyon checkpointleri TEK TEK daha guclu (val Conn_IoU 0.6528 / 0.6990 / 0.6673) |

Ikisi de VAL 100 parcada, olculen yolda, esli parca bootstrap ile
olculuyor. **Bu kol tohum gurultusune tabi degil** (yeni egitim yok),
dolayisiyla tek kosuda hukum giyebilir.

## 21.69 ARA TEST HAZIRLIGI — 5 gorulmemis parca, TANIDIK marka

Amac sayi degil **fonksiyonel dogrulama**: urun sahada ne yapiyor.
`sonda_ara_test_glb.py`, `robot_cp.extract` (yani GLB ihracatcilarinin
BUGUN cagirdigi yol) uzerinden parca parca rapor uretir.

Secim (`_ara_test_parcalar.txt`) **zorluk yelpazesini kasten kapsiyor**:

| parca | GT CP | konum |
|---|---|---|
| 3061994 | **24** | en yogun -- sistemin zayif halkasi |
| 1020700000 | 20 | yogun |
| 1208920000 | 8 | cok-CP rejim esigi (n_gt>=8) |
| 1058680000 | 4 | tipik |
| 3025176 | 1 | tek girisli |

Hepsi VAL'den: **marka egitimde VAR, bu PARCALAR yok.** Kolay parca secip
test sismesin diye en yogun parca bilerek dahil edildi.

**DURUSTLUK NOTU (betigin ciktisina da yazildi):** 5 parca kucuk bir
ornektir, bu sayilar MANSET DEGILDIR. Manset VAL 100 parcadir
(tespit 0.7878 / robot-ISARETLI 0.4839).

## 21.70 TOPLULUK GENISLETME A (5 uye) — NOTR, KAPANDI

VAL 100 parca, olculen yol, esli parca bootstrap. Urun 4 uye +
`recall_hard_keig96_s3` (AYNI recetenin 4. tohumu):

| metrik | taban (4) | 5 uye | fark | %95 GA | poz% |
|---|---|---|---|---|---|
| tespit | 0.6101 | 0.6100 | −0.0000 | [−0.0191,+0.0201] | 49.1 |
| robot | 0.5376 | 0.5349 | −0.0024 | [−0.0239,+0.0211] | 40.9 |
| robot-ISR | 0.4668 | 0.4651 | −0.0018 | [−0.0221,+0.0173] | 42.9 |

**Tam notr. KAPANDI.** Ayni receteden bir uye daha eklemek hicbir sey
katmiyor -- 4 uyeli topluluk o recetenin varyansini ZATEN doyurmus
(uyeler yuksek korelasyonlu: yalniz tohum farkli).

Bu, TOPLULUK B'nin (farkli recete = gercek cesitlilik) neden ayri bir
soru oldugunu da gosterir.

## 21.71 YOGUN PARCA YOLU IKI KEZ BAYATLAMIS — ONARILDI

Gozle yapilan ara test yogun parcada agir kacirma gosterdi (GT=22 -> 11
CP). Kodda bu is icin **ayri bir urun yolu** var: `robot_cp.extract_highcp`
(makbuz `results/product_f1_receipt.json`, OOF F1 0.807). Ama
`export_robot_glb.py:155` **kosulsuz** `robot_cp.extract` cagiriyor, yani
yol hic kullanilmiyordu. Sebep arandiginda yolun **bugun hic
kosamadigi** ortaya cikti -- iki ayri bayatlama:

**(1) Kapi donusumu atlaniyordu.** `highcp_selector.apply` icinde
`wm["clf"].predict_proba(X13)` cagriliyordu; dagitilan gate ise
PARCA-ICI Z-SKOR donusumunden sonra **116** sutun bekliyor (58'in tam iki
kati -- donusum her ozniteligin yanina parca-ici z-skorunu ekler).
`ValueError: X has 58 features, but expecting 116`.
**Onarim:** `wire_gate.karar_skoru(wm, X13)` -- dosyanin kendi "tek
kaynak" kurali zaten buydu, burasi ona uymuyordu.

**(2) Secici, o gunku oznitelik genisligiyle egitilmis.** Secici
2026-07-26'da egitildi; O GUN `feats_for` **13** sutun donduruyordu
(13 + 8 lattice + 4 rank = **25**). Bugun 58 donduruyor -> augment **70**
uretiyor. `ValueError: X has 70 features, but expecting 25`.
**Onarim:** seciciye giden matris ilk `n-12` sutuna kirpilir.
**DOGRULANDI, VARSAYILMADI:** `FEAT_NAMES = FEAT_NAMES_13 + ...` yani yeni
oznitelikler SONA eklenmis; secicinin sakladigi 25 ismin ilk 13'u
`FEAT_NAMES_13` ile BIREBIR ayni (kod isim isim karsilastirip
uymazsa HATA veriyor -- sessiz kirpma yok).

**DERS:** "makbuzu var" demek "bugun kosuyor" demek DEGIL. Cagrilmayan bir
yol sessizce curur; iki bagimsiz degisiklik (gate donusumu, oznitelik
genisligi) bu yolu kullanilamaz hale getirmis ve kimse fark etmemis
cunku kimse cagirmiyormus.

## 21.72 YOGUN YOL DOGRU ADAYI BULUYOR, DOGRU YONU KOYMUYORDU

Yol onarilip kosunca ilk parca (3061994, GT=24) sunu verdi:

| yol | uretilen CP | robot-ISARETLI |
|---|---|---|
| TABAN (`extract`) | 20 | **11** |
| YOGUN (`extract_highcp`) | **24** (adet TAM) | **1** |

**Adedi tam tutturuyor ama isaretli dogruluk cokuyor.** Sebep kodda:
`extract_highcp`, secimden sonra dogrudan `_format_cps` diyip bitiyordu;
`extract`in **kapi sonrasi zinciri** (pose head, aci duzeltici, uye yon
secici, ayrik yon secici) ORADA HIC KOSMUYORDU.

Bu ayni zamanda makbuzu da yerine oturtuyor: **0.807 TESPIT F1'idir**,
robot-isaretli degil.

**ONARIM:** zincir `_kapi_sonrasi_zincir()` diye ortak fonksiyona cikarildi
ve HER IKI yol da onu cagiriyor (kod tabaninin kendi "tek kaynak" kurali).
`extract`in davranisi DEGISMEDI -- duman testi birebir ayni iki CP'yi ayni
koordinatlarda uretti.

## 21.73 METADATA-SIZ ADET TAHMINI (kunye bilgisi olmadan)

`extract_highcp` `cp_count` ister (ureticinin CP sayisi) = METADATA.
Kod okundu: `cp_count` yalnizca IKI yerde kullaniliyor -- `rank_feats`
icindeki `nratio` oznitelig i ve son `[:N]` kirpmasi. Havuz uretimi ve
lattice ozellikleri N'den BAGIMSIZ. Yani metadata bagimliligini kaldirmak
icin tek gereken bir **N tahmini**.

`adet_tahmini.py`: CP'ler duzenli bir IZGARADA durur; iki ana eksende
(SVD) izgara adimi (pitch) ve yayilim olculur, site sayisi cikarilir.
Tamamen geometrik.

**ILK KAPI (cikarim GEREKTIRMEZ): GT noktalarindan adedi geri bulabiliyor mu?**

| | tum parcalar (6074) | **YOGUN (GT>=11, 606 parca)** |
|---|---|---|
| tam isabet | %68.5 | **%56.8** |
| ortanca mutlak hata | 0.0 CP | **0.0 CP** |

Kapi GECTI -> izgara gercek ve sayilabilir. `cp_count=None` verilirse
`extract_highcp` artik adedi havuzun kendi geometrisinden tahmin ediyor
(gurultuye karsi yalniz ust %60 skorlu adaylar kullanilir).

## 21.74 YOGUN YOL BAGLANMADI — makbuz TEMMUZ'a ait, bugune TASINMIYOR

Iki bayatlama onarilip yol kosar hale gelince (21.71) ve kapi sonrasi
zincir baglaninca (21.72) tam olcum yapildi. **5 yogun parca, GT ile:**

| yol | tespit | robot (isaretsiz) | robot-ISARETLI |
|---|---|---|---|
| **TABAN (`extract`)** | **0.7907** | **0.4651** | **0.4186** |
| YOGUN (cp_count=GT) | 0.3402 | 0.1546 | 0.1134 |
| YOGUN (cp_count−2) | 0.3441 | 0.1505 | 0.1075 |

### ONCEKI CIKARIM CURUDU
Ilk parcadan sonra "yogun yol dogru ADAYLARI buluyor, taban yol dogru
YONLERI koyuyor" denmisti. **Yanlis.** Yogun yol TESPITTE de yari yariya
kotu (0.3402 vs 0.7907): adedi tutturuyor ama noktalari YANLIS YERLERE
koyuyor. Tek parcadan (ISARETLI 11 vs 1) cikarilan mekanizma, bes parcalik
tam tabloyla curudu.

### UCUNCU BAYATLAMA — asil sebep
`cp_config.robot_highcp.derive_6k = {min_v: 30, vertex_conf: 0.5,
cluster_mm: 5.0}`. Bu degerler, `prediction_postproc` icinde
**`_superseded_2026_07_24_values`** olarak duran TERK EDILMIS degerlerin
ta kendisi. Urun 2026-08-06'da `min_v 4 / vc 0.3 / cluster 1.0`'a gecti
(`p1_2026_08_06`: cok-CP +0.0331); **yogun yol gecmedi.**

Yani `extract_highcp` "daha iyi bir yol" DEGIL, **Temmuz urununun adet
yardimi almis hali**. Makbuzdaki 0.807, Temmuz'da, 24 parcada, o gunku
turetme parametreleriyle olculmus bir sayidir ve bugunku sisteme
TASINMIYOR.

### KARAR: `export_robot_glb.py`'ye BAGLANMADI
Baglansaydi gozle "yogun parcada daha cok CP var" gorunurken olcum
0.79 -> 0.34'e duserdi. **Gorsel iyilesme ile olculen iyilesme ters
yonde olabilir** -- bu gecenin vekil dersinin (21.58) gorsel karsiligi.

### YINE DE KAZANC: uc onarim kodda KALIYOR
1. `highcp_selector` artik `karar_skoru` cagiriyor (gate donusumu)
2. Oznitelik genisligi ISIM DOGRULAMASIYLA kirpiliyor (sessiz kirpma yok)
3. `_kapi_sonrasi_zincir()` ortak fonksiyon -- iki yol bir daha ayrisamaz
4. `cp_count=None` -> adet geometriden tahmin ediliyor (`adet_tahmini.py`)

Yol artik KOSABILIR durumda. Yeniden aday olmasi icin gereken tek sey,
turetme parametrelerinin bugunku urunle esitlenmesi ve secicinin bugunku
ozniteliklerle YENIDEN EGITILMESI (`highcp_selector.train_and_save`,
havuz `results/highcp_pool.json` diskte). Bu bir GUNLUK istir, bugune
sigmaz.

## 21.75 LITERATURDEN ILHAM — bizim kusurun ADI var: "sem-seg + connected components"

Aranan sey genel fikir degil, **bizim boru hattinin bilinen kusuru**.

**Bizim yol:** anlamsal segmentasyon -> bagli bilesenler -> bilesenin
agirlik merkezi = CP konumu.

Literaturde bunun adi *semantic segmentation + connected components* ve
bilinen kusuru aynen su: **birbirine DEGEN, birbirinin AYNI nesneleri
ayiramaz** (Panoptic-DeepLab bunu acikca soyler: kutusuz yontemler degen
nesneleri ayirmakta zorlanir). Bizim yogun klemenste cokusumuz tam budur.

**Cozum ailesi** (Panoptic-DeepLab, Spatial Embeddings, PVN3D, EmbedTrack):
her nokta kendi ornek MERKEZINE bir **kayma vektoru (offset)** tahmin
eder; ayrica bir **centerness/seed** haritasi ogrenilir (hangi noktanin
oyu guvenilir). Noktalar kaydirilip kumelenir. Merkez bolgeden
TURETILMEZ, dogrudan OYLANIR.

**Bizim yigina birebir oturur:**

| bugun | onerilen |
|---|---|
| DiffusionNet -> 5 sinif olasilik | + **3 kanal offset** + **1 kanal seed** (ayni ag, ek bas) |
| bagli bilesen -> merkez | kaydir + kumelen |
| konum cozunurlugu ~ bolge boyu | konum cozunurlugu ~ regresyon hassasiyeti |

**YENI ETIKET GEREKMIYOR:** offset hedefi her tepe icin
`(en yakin GT CP - tepe konumu)`; elimizdeki 11.927 uretici CP'sinden
bedava cikar.

**Neden bu kampanyanin olcumleriyle TUTARLI:**
* 21.60: baglayici kisit YANAL KONUM (+0.358). Offset regresyonu tam
  oraya calisir.
* 21.53-21.62: pose head'in SON-ISLEM regresyonu bu acigi TASIYAMADI
  (uc bagimsiz olcum). Cunku hasar YUKARIDA olusuyor -- offset basi
  yukarida calisir.
* 21.74: yogun parcada cokus, "degen ayni nesneleri ayirma" probleminin
  ta kendisi.

**MALIYET (durust):** egitim ister (~2 sa/tohum x 3 tohum + kumeleme
kalibrasyonu). Bugune SIGMAZ. Ama "sonraki adimlar"in 1. maddesi artik
tahmin degil, **olculmus bir darbogaza oturan somut bir mimari**.

Kaynaklar: Panoptic-DeepLab (arXiv 1911.10194) · PVN3D (arXiv 1911.04231)
· Spatial Embeddings (arXiv 1906.11109) · EmbedTrack (arXiv 2204.10713)

## 21.76 OFFSET/OY BASININ TAVANI OLCULDU — YESIL ISIK (ve sonda cozunurlugu dersi TEKRAR)

Egitime girmeden once tavan olculdu (`sonda_offset_tavani.py`, VAL 23-25
parca, egitim YOK). Soru: **offset basi ne kadar hassas olmali ki
bugunku sonucu gecsin?** GT offsetlerine gercekci gurultu eklenip
kaydir+kumele cozucusu kosuldu.

**ILK SONDA (kumeleme bandi 2.0 mm) — "uygulanamaz" diyordu:**

| offset gurultusu | tespit F1 |
|---|---|
| 0.00-0.50 mm | 1.0000 |
| **0.75 mm** | 0.6141 |
| 1.00 mm | 0.3289 |

Pose head'in bugun OOF'ta ulastigi artik ortanca **0.67 mm** -- yani tam
ucurumun ustunde. Bu haliyle fikir bugunku 0.7878'i GECMEZDI.

**BANDA DUYARLILIK TARANDI (kapatmadan once sondayi denetle kurali):**

| bant | 0.75 mm | 1.00 mm | 1.50 mm |
|---|---|---|---|
| 2.0 mm | 0.6141 | 0.3289 | 0.2016 |
| 3.0 mm | **1.0000** | 0.8132 | 0.3318 |
| **4.0 mm** | **1.0000** | **0.9933** | 0.6066 |

**Ucurum FIKRIN degil, COZUCUNUN kusuruymus.** 4 mm bantla yaklasim
**1.0 mm** offset hatasina kadar neredeyse kusursuz; bizim ulasabildigimiz
0.67 mm bunun rahatca icinde.

Bu, `KAPANAN_KOLLAR_DENETIMI`ndeki "yelpaze 64 yonde OLU, 256 yonde
+0.0676" dersinin birebir tekrari: **ilk sonda dusuk cozunurlukluydu.**

### DURUSTLUK SINIRI — bu bir TAVAN, tahmin DEGIL
Olcumde iki sey KAHINDEN geliyor:
1. **Kim oy verir**: bir CP'ye 6 mm'den yakin tepeler (gercekte bunu
   `seed` basi ogrenecek).
2. **Yon**: en yakin GT'nin yonu (bu betik yonu olcmuyor).
Yani gercek sistem ayrica seed haritasini ve yonu de ogrenmek zorunda.
Olculen sey sudur: **cozme adimi, ulasabilecegimiz hassasiyette
calisiyor mu?** Cevap EVET.

**KARAR: kol aciliyor.** Sonraki adim offset+seed basini kurup egitmek.

## 21.77 TOPLULUK GENISLETME B (8 uye) — KAPIYI GECEMEDI, TOPLULUK CEPHESI KAPANDI

Urun 4 + `recall_hard_keig96_s3` + `y1b_aug0.15_s0/s1/s2` (farkli recete =
gercek cesitlilik). VAL 100 parca, olculen yol, esli bootstrap:

| metrik | taban (4) | 8 uye | fark | %95 GA | poz% |
|---|---|---|---|---|---|
| tespit | 0.6101 | 0.6186 | +0.0086 | [−0.0205,+0.0403] | 70.3 |
| robot | 0.5376 | 0.5522 | +0.0148 | [−0.0167,+0.0499] | 81.3 |
| robot-ISR | 0.4668 | 0.4710 | **+0.0041** | [−0.0250,+0.0328] | 60.8 |

Ucunun de GA'si sifiri iciyor. Isaretli metrik neredeyse hic kipirdamiyor
(+0.0041). **Topluluk cephesi kapandi:**

| kol | robot-ISR farki | hukum |
|---|---|---|
| A: 5 uye (ayni recete) | −0.0018 (%42.9) | notr |
| B: 8 uye (farkli recete) | +0.0041 (%60.8) | kapi alti |

Ayni receteden uye eklemek bosuna; farkli receteden eklemek isaretsizde
biraz yardim ediyor (+0.0148, %81.3) ama isaretlide degil. Bu, gecenin
genel bulgusuyla tutarli: **kalan hata yon/isaret degil KONUM.**

## 21.78 SEED KARISMASI SONDASI **BILGI TASIMADI** — tasarim kusuru, kayda gecirildi

21.76'da "kim oy verir" kahinden geliyordu. Bunu zorlamak icin oylarin bir
kismi KOMSU CP'ye kaydirilarak "bitisik agizlari karistirma" simule
edilmek istendi. Uc oran (%5 / %15 / %30) **birebir ayni** sonucu verdi
(hepsi 1.0000, 1.0 mm'de 0.9920) -- son haneye kadar. Bu, no-op imzasidir.

**Sebep bulundu:** bozma, oyun hedefini KOMSU CP'nin TAM MERKEZINE
tasiyor. Yani oy hala GECERLI bir kumeye dusuyor; kumeler ayrik kaldigi
ve her CP bol oy aldigi icin kume merkezleri degismiyor.

**Gercek ag hatasi boyle degildir:** ag offseti iki CP'nin ARASINA yayar
(smear), merkeze degil. Dogru sonda, tahmin edilen offsetin YONUNU/BOYUNU
bozmali (orn. hedefi iki CP arasinda enterpolasyon yapmak), CP kimligini
degistirmemeli.

**Bu sonuc "dayaniklidir" diye RAPORLANMIYOR.** Sonda tasarimi geregi
bilgi tasimiyor; dogrusu yarin kurulacak. Kayit, ayni tuzaga tekrar
dusulmemesi icin burada duruyor.

## 21.79 YOGUN YOL — bugunku turetme parametreleriyle OLCULDU (ucuncu bayatlama dogrulandi)

21.74'te ucuncu bayatlama teshis edilmisti: `robot_highcp.derive_6k`
degerleri (30 / 0.5 / 5.0) urunun 2026-07-24'te TERK ETTIGI degerler.
Cevre degiskeniyle bugunku urun degerleri (4 / 0.3 / 1.0) verilip
olculdu (5 yogun parca, GT ile):

| yol | tespit | robot | robot-ISARETLI |
|---|---|---|---|
| **TABAN (`extract`)** | **0.7907** | **0.4651** | **0.4186** |
| YOGUN (Temmuz turetmesi) | 0.3402 | 0.1546 | 0.1134 |
| YOGUN (**bugunku** turetme) | 0.3590 | 0.2051 | **0.1641** |

**Teshis DOGRULANDI ama yol KURTULMADI.** Turetme duzeltmesi
robot-ISARETLI'yi 0.1134 -> 0.1641 cikardi (+0.0507) -- yani ucuncu
bayatlama gercekti. Yine de taban **2.5 kat** onde.

**Kalan kok neden: SECICININ KENDISI.** `highcp_selector.pkl`
2026-07-26'da, **24 parcayla**, o gunku adaylarla ve o gunku
ozniteliklerle egitildi. Turetmeyi guncellemek adaylari degistirir ama
secici hala eski dagilima gore siralar.

**KISA YOL YOK.** Seciciyi yeniden egitmek icin once havuzun
(`results/highcp_pool.json`, Temmuz) bugunku cikarimla yeniden
uretilmesi gerekir -- bu bir GUNLUK istir. Hizlanmak icin kestirme
yapilip yol ihracatciya baglansaydi, GOZLE "yogun parcada daha cok CP
var" gorunurken OLCUDE tespit 0.79 -> 0.36 duserdi.

### YOGUN CEPHESINDE BUGUNKU NET KAZANC
1. Yol **calisir** hale geldi (iki bayatlama onarildi; ucuncusu olculdu)
2. **Metadata-siz adet tahmini** kodda (`cp_count=None`)
3. Kapi sonrasi zincir **ortak fonksiyona** cikti -- iki yol bir daha
   ayrisamaz
4. Yeniden egitim icin gerekenin **tam olarak ne oldugu** olcumle belli:
   havuz yeniden uretimi + secici yeniden egitimi (turetme ayari TEK
   BASINA yetmiyor -- olculdu)

## 21.80 YOGUN YOLDA **DORDUNCU** BAYATLAMA: oy havuzu yaricapi 5 mm kalmis

Yogun parcada uretilen CP sayisinin yariya dusmesinin bir sebebi arandi.
Once bir hipotez kuruldu ve **CURUTULDU** (iyi ki olculdu):

* Hipotez: eksen-farkindalikli havuzun yanal yaricapi (3.0 mm) komsu
  kutuplari (adim ~3.5 mm) birlestiriyor.
* **Yanlis:** `robot_eksen_havuz` config'de **False**, yani o parametre
  hic kullanilmiyor. Urun yolu duz kure mesafesi kullaniyor ve degeri
  `vote_pool_mm = 2.0` -- 3.5 mm adimin ALTINDA, birlestirme yok.

Ama ayni yeri okurken GERCEK kusur bulundu:

```
# robot_cp.py:569 (eski)
cps = _vote2(per, min_votes=1)      # cluster_mm GECILMIYOR -> varsayilan 5.0 mm
```

`extract_highcp` oy havuzunu **5.0 mm** ile yapiyordu; urun yolu ise
`vote_pool_mm = 2.0` kullaniyor. Bu deger 2026-08-06 P1 taramasinda
5 -> 2 mm'ye cekilmisti ve o taramanin **suclusu tam da** "komsu iki
gercek girisi tek adaya yutan genis havuz"du (cok-CP +0.0331).
**Yogun yol o duzeltmeyi de almamis.**

**ONARIM:** `_vote2(per, cluster_mm=vote_pool_mm, min_votes=1)`.

### OLCUM (3061994, GT=24) — her onarim gercek kazanc veriyor

| durum | robot-ISARETLI |
|---|---|
| orijinal (uc bayatlama acik) | 1 |
| + turetme parametreleri bugunku | 4 |
| + oy havuzu 5 -> 2 mm | **5** |

Taban yol ayni parcada 11. Yani yogun yol hala geride ama **her
bayatlama onarimi olculebilir kazanc veriyor** -- bu, kalan farkin da
onarilabilir cinsten oldugunun isareti.

**YOGUN YOLDAKI BAYATLAMA SAYISI: DORT.**
1. gate donusumu atlanıyordu (`karar_skoru` yerine ham `predict_proba`)
2. oznitelik genisligi 25 vs 70
3. turetme parametreleri Temmuz'un terk edilmis degerleri
4. oy havuzu yaricapi 5 mm (urun 2 mm'ye gecmis)

Hepsinin ortak sebebi ayni: **bu yol cagrilmiyordu, dolayisiyla urun
gelistikce sessizce geride kaldi.**

## 21.81 SUNUMDAKI VE CONFIG'DEKI GATE ESIKLERI **OLU PARAMETRE**

Yogun parcada gate esigi tarandi (0.15 / 0.25 / 0.30 / 0.35 / 0.45) ve
**bes deger de BIREBIR ayni sonucu** verdi (tespit 0.7260, robot 0.5721,
robot-ISR 0.4279). Bu gecenin bes numarali "+0.0000" imzasi.

**KOK NEDEN:** `wire_gate.GORELI_ESIK` 2026-07-31'de dagitildi ve
`karar_maskesi` artik sabit esigi KULLANMIYOR:

```
if GORELI_ESIK:
    return (s >= GORELI_ORAN * s.max()) & (s >= GORELI_TABAN)
return s >= esik          # <- bu satira ARTIK GIRILMIYOR
```

Dolayisiyla:
* `cp_config.robot_wire_gate_threshold` (0.40) -> **OLU**
* `cp_config.robot_wire_gate_threshold_highcp` (0.35) -> **OLU**
* `extract` icindeki rejim-kosullu esik secimi -> **ETKISIZ**
* 2026-07-29 taramasinin "0.25 daha iyi" sonucu -> artik **UYGULANAMAZ**
  (o tarama sabit esik doneminde yapildi)

**SUNUM DA BUNLARI CANLI AYAR DIYE GOSTERIYOR** (slayt 8: "Gate
threshold, low CP density 40% / high CP density 35%"). Duzeltilmeli.

**GERCEK CANLI PARAMETRELER:**
`gate_goreli_oran` (0.50) ve `gate_goreli_taban` (0.20) --
ikisi de cevre degiskeninden okunabiliyor (`WG_GORELI_ORAN`,
`WG_GORELI_TABAN`). Tarama bunlara cevrildi.

**DERS (bu gecenin tekrar eden dersi):** bir parametrenin config'de
DURMASI, urunun onu KULLANDIGI anlamina gelmez. Tarama once "parametre
gercekten baglayici mi" diye sinanmali; yoksa saatlerce olu bir dugmeyi
cevirmis olursun.

## 21.82 YOGUN YOL — DORT ONARIMDAN SONRAKI TAM OLCUM

5 yogun parca, GT ile, bugunku sistem:

| yol | tespit | robot | robot-ISARETLI |
|---|---|---|---|
| **TABAN (`extract`)** | **0.7907** | **0.4651** | **0.4186** |
| YOGUN, orijinal (4 bayatlama acik) | 0.3402 | 0.1546 | 0.1134 |
| YOGUN, turetme onarildi | 0.3590 | 0.2051 | 0.1641 |
| **YOGUN, DORT onarim** | **0.3673** | **0.2143** | 0.1531 |

Onarimlar tespitte **+0.0271**, isaretsiz robotta **+0.0597** kazandirdi.
Ama taban hala **2.7 kat** onde.

**KARAR DEGISMEDI: ihracatciya BAGLANMIYOR.** Kalan fark yapisal --
secici 2026-07-26'da **24 parcayla** egitildi ve o gunku aday
dagilimini varsayiyor. Dort onarim adaylari degistirdi; secici o yeni
dagilima gore siralamiyor.

**Dort onarim yine de KODA KALIYOR**, cunku secici yeniden egitildiginde
bu yolun DOGRU davranmasi icin hepsi gerekli. Yol artik "bozuk" degil,
"eski secici ile" calisiyor.

## 21.83 OFFSET BASI — EGITIM TRENDI ve TRIVIAL TABAN

Egitim: 659 parca, 14 epok, sizinti kapisi kurulu.

**Once TRIVIAL TABAN olculdu** (bu olmadan egitim sayisi yorumlanamaz):
> Model HICBIR SEY ogrenmese, yani sifir offset tahmin etse, ortanca
> hata **4.453 mm** olurdu (seed bandindaki tepelerin CP'ye ortanca
> uzakligi).

| epok | ortanca offset hatasi |
|---|---|
| trivial (sifir tahmin) | 4.453 mm |
| 1 | 4.129 mm |
| 2 | 4.005 mm |
| 8 | **3.664 mm** |

**Model OGRENIYOR** (3.664 < 4.453, %18 daha iyi) ama **cok yavas**:
epok basina ~0.06 mm ve yavasliyor. 14 epokta ~3.5 mm'de kalir.

**GEREKEN: 1.0 mm** (21.76 tavan olcumu, kumeleme bandi 4 mm).

**HUKUM: bu butcede esik GECILMIYOR.** Kol "olu" degil, ama bu recete
ve bu butce ile hedefe ulasmiyor.

## 21.84 OFFSET BASI TESHISI — YON OGRENILMIS, BUYUKLUK BUZULMUS

Ortanca hata 3.4 mm tek basina "basarisiz" der. Ama hatanin BILESENLERI
ayristirilinca tablo degisiyor (epok 13 checkpoint'i, 6 parca):

| olcum | deger |
|---|---|
| **hedef** offset buyuklugu (ortanca) | 4.299 mm |
| **tahmin** buyuklugu (ortanca) | **2.871 mm** |
| **yon uyumu** (kosinus; 0 = rastgele, 1 = mukemmel) | **+0.675** |

**Model YONU OGRENMIS.** Kosinus +0.675, yaklasik **48 derece** ortanca
aci hatasina karsilik gelir -- rastgele bir tahmin 0 verirdi. 13 epokta,
700 parcayla, hicbir ayar aramasi yapilmadan.

**Sorun BUYUKLUK:** model 4.30 mm'lik hedefe 2.87 mm oneriyor (%33 az).
Bu, bu kampanyada **ikinci kez** karsilasilan patoloji: pose head de tam
olarak boyle davraniyordu (21.53: en buyuk oneri 2.87 mm, hedeflerin
%10.2'si 3 mm'nin ustunde). **Regresyon buzulmesi**, L1/L2 kayiplarinin
bilinen davranisi.

### HUKUM: KOL ACIK KALIYOR, RECETE DEGISMELI
"Bu butcede esik gecilmiyor" (21.83) dogru ama eksik. Dogrusu:
**yon sinyali VAR ve ogreniliyor; ulasilamayan sey buyukluk.**

Sonraki denemenin degistirmesi gerekenler (olculmus gerekceyle):
1. **Buzulmeye karsi kayip**: saf L1 yerine yon + buyukluk AYRI
   ogrenilsin (birim vektor + skaler norm), ya da kayip norm'a gore
   agirliklandirilsin. Buzulme iki bagimsiz kolda goruldu.
2. **Daha uzun egitim**: 13 epokta hala monotonik iyilesiyor
   (4.129 -> 3.351), yani doymamis.
3. **Seed bandi daralt**: 6 mm yerine 3 mm; uzak tepelerin hedefi hem
   buyuk hem belirsiz, ortalamayi asagi cekiyor.

**Bu, "kol olu" demekten cok farkli bir sonuctur** ve bugunku olcumle
desteklenmektedir.

## 21.85 OFFSET BASI BITTI + **SEED BASI GORULMEMIS PARCADA AUC 0.8042**

**Egitim tamamlandi** (700 parca / 14 epok / sizinti kapisi kurulu):

| epok | ortanca offset hatasi |
|---|---|
| trivial (sifir tahmin) | 4.453 mm |
| 1 | 4.129 |
| 8 | 3.664 |
| 13 | 3.351 |
| **14 (son)** | **3.306** |

**Egri DOYMADI** -- 14 epok boyunca monotonik iyilesti. Butce bitti,
ogrenme bitmedi. Hedef 1.0 mm; bu butcede GECILMEDI.

### SEED BASI -- beklenmedik ve GUCLU sonuc
Ag 4 kanal uretiyor; dordu `seed` (bu tepe bir CP'ye yakin mi).
**GORULMEMIS VAL parcalarinda** (egitimde YOK) olculdu:

> **seed basi AUC = 0.8042** (8 parca ortancasi; 0.5 = rastgele)

Yani 14 epok ve 700 parcayla, hicbir ayar aramasi yapilmadan egitilen
bir yan cikti, gorulmemis parcada guclu bir TESPIT sinyali veriyor.

**DURUSTLUK -- 0.7053 ile DOGRUDAN KIYASLANAMAZ.** Daha once olculen
"konum AUC 0.7053" **ADAY** basinadir ("bu aciklik CP mi"); buradaki
0.8042 **TEPE** basinadir ("bu tepe bir CP'ye yakin mi"). Iki farkli
populasyon, iki farkli soru. Ayni tabloya konulmaz.

### NE ANLAMA GELIYOR
Kolun iki ciktisi var ve **ikisi farkli olgunlukta**:
* **offset** (buyukluk): buzulmus, hedefe uzak -> recete degismeli (21.84)
* **seed** (tespit): gorulmemis parcada ZATEN calisiyor

Yani bu mimari "ilerde belki" degil; **yarim gunluk bir egitimde bile
kullanilabilir bir sinyal uretiyor.** Kol ACIK ve onceligi yuksek.

## 21.86 GATE GORELI ORANI 0.50 -> 0.60: IKI REJIMDE DE KESIN POZITIF

21.81'de sabit esiklerin OLU oldugu, gercek canli parametrenin
`gate_goreli_oran` (0.50) oldugu bulunmustu. Taranan bu.

**YOGUN parcalar (n_gt>=8, 14 parca):**

| oran | tespit | robot | robot-ISARETLI |
|---|---|---|---|
| 0.30 | 0.6985 | 0.5628 | 0.4121 |
| 0.40 | 0.7113 | 0.5722 | 0.4227 |
| **0.50 (dagitilan)** | 0.7312 | 0.5968 | 0.4409 |
| **0.60** | 0.7187 | **0.6128** | **0.4568** |

**SEYREK parcalar (n_gt<8, katalogun %86'si, 26 parca):**

| oran | tespit | robot | robot-ISARETLI |
|---|---|---|---|
| **0.50 (dagitilan)** | 0.7931 | 0.4828 | 0.4713 |
| **0.60** | 0.7826 | 0.5093 | **0.5093** |

**ESLI BOOTSTRAP (0.50'ye gore, robot-ISARETLI):**

| rejim | fark | %95 GA | poz% |
|---|---|---|---|
| yogun | **+0.0162** | [+0.0046, +0.0328] * | **100.0** |
| seyrek | **+0.0374** | [+0.0167, +0.0608] * | **100.0** |

Tespit bedeli iki rejimde de **kesin degil** (−0.0129 ve −0.0100, iki
GA da sifiri iciyor). Yani kazanc isaretli robot metrigine geliyor,
tespitten kesin bir sey goturmuyor.

Rejim paylariyla agirliklandirilmis beklenen etki:
0.86 x 0.0374 + 0.14 x 0.0162 = **~+0.034**.

### AMA: BU PARAMETRE **VAL UZERINDE** TARANDI
VAL bizim degerlendirme kumemiz. Ayni kumede tarayip ayni kumede
raporlamak SISME uretir. Bu kampanyada tarama **iki kez** yaniltti:
* `vc0.35`: taramada +0.01, TAM dagitim OOF'unda WEI 0.641 -> 0.629
* `k_eig 128`: ayni desen

**Bu yuzden BAGIMSIZ ornekte dogrulama kosuluyor** (VAL ve LOCKED DISI
30 parca, ayni parametre). Dogrulanmadan **DAGITILMAZ**.

## 21.87 ALAN FARKI (+0.2468) GATE DEGISIKLIGINDEN ETKILENMIYOR — dogrulandi

`gate_goreli_oran` 0.50 -> 0.60 dagitildiktan sonra, sunumun ana bulgusu
olan alan farkinin (tanidik 0.5603 / gorulmemis 0.3135) hala gecerli olup
olmadigi soruldu. **Yeniden olcmeden ONCE kod yolu kontrol edildi:**

* Alan farki `kos_p6_kademe2.py` ile olculdu; bu betik **kendi karar
  kuralini** kullaniyor (`KURALLAR` izgarasi: mutlak/goreli, kat ICINDE
  aranir).
* Degistirilen `gate_goreli_oran` ise `wire_gate.karar_maskesi`'nin
  parametresi. O betikten `wire_gate`'e giden tek cagri
  `kalabalik_maskesi` (NMS) -- esikle ilgisi YOK.

**Sonuc: iki olcum AYRI kod yollarinda; +0.2468 GECERLI kaliyor.**
25 dakikalik gereksiz bir yeniden olcum yapilmadi.

Not: bu, raporda kayitli "olculen sey ile dagitilacak sey ayni degildi"
dersinin (Bolum 21.x, kanonik blok: olcumde +0.0151 / uretimde −0.0138)
simetrigi. Orada ayrim ALEYHIMIZE calismisti; burada LEHIMIZE, cunku iki
olcum birbirini gecersiz kilmiyor. Her iki durumda da kural ayni:
**hangi kod yolunda olculdugu once kontrol edilir.**
