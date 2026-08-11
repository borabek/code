# ROTA PLANI — ölçülmüş tavanların üstüne kurulmuş

> Kaynak ölçümler: `results/rb_bayes.json`, `rb_bayes_dogrulama.json`, `p1_ogrenme_egrisi.json`,
> `q1_cesitlilik.json`, `x1_bosluk.json`, `p2_ablasyon.json`, `q3_fn_taksonomi.json`.
> Tez çizgisi korunur: aynı ağ, aynı remesh, aynı v_o türetmesi. Hiçbir rota bunları değiştirmez.

## 1. Nerede duruyoruz — üç ayrı tavan

| tavan | değer | ne sınırlıyor |
|---|---|---|
| aday havuzu recall'ı | **0.8579** | segmentasyon 188/1323 GT'ye hiç aday üretmiyor |
| kâhin gate (mükemmel seçim) | 0.9370 | — |
| **73-sütunluk uzayda ayrılabilirlik** | **~0.86** | adayların %29-32'si ayrılamaz bölgede |
| şu an | **0.7584** | |

**Bağlayıcı olan üçüncüsü.** 0.85 hedefi, mevcut öznitelik uzayının Bayes sınırının
~0.01 altında oturuyor — yani "neredeyse kusursuz çıkarım" demek. Gürbüz bir ürün hedefi değil.

**Ama bu tavan 73 elle-yapılmış sütuna aittir, ağa değil.** Segmentasyon ağı ham mesh'ten
çok daha zengin yerel geometri görür. Rota B bu tavanın *dışındadır*.

## 2. Hedefin yeniden konuşulması

Robot metriği tespitten türer: `robot ≤ tespit` her zaman, ve poz+açı mükemmelse `robot = tespit`.

| hedef | gerektirdiği | değerlendirme |
|---|---|---|
| tespit 0.85 | ayrılabilirlik tavanının %99'u | gürbüz değil |
| **tespit 0.80** | tavanın %93'ü; veriyle 3.8x korpus | zor ama savunulabilir |
| robot 0.75 | tespit ≥0.75 **ve** çiftlerin %98.9'u iki ölçütü geçmeli | ulaşılamaz |
| **robot 0.65** | çiftlerin %85.7'si (şu an %80.3) | **yakın vadede en gerçekçi** |

**Öneri: yakın vade tespit 0.80 / robot 0.65; 0.85/0.75 yeni bir bilgi kipine bağlı uzun ufuk.**

## 3. Rotalar

### Rota C — YÖN/KONUM (önce bu, çünkü tespite dokunmadan en çok payı taşıyor)

Mevcut tespitle (0.7584) poz ve açı mükemmel olsa robot **0.7584** olurdu; şu an 0.5893.
Yani tespite hiç dokunmadan **0.169** pay duruyor. Kırılım:

| başarısızlık | pay |
|---|---|
| yalnız yanal (>2mm) | %6.4 |
| **yalnız açı (>10°)** | **%8.0** |
| ikisi | %5.3 |

Açı hatalarının %56'sı ≥60° = **yanlış eksen sınıfı**, ve o sınıf AUC 0.930 ile tahmin
edilebiliyor. R2'de sürekli regresörle düzeltmeyi denedim, **her eşikte düştü** — bu,
ayrık yaklaşımın lehine kanıttır.

**İlk kapı (ucuz, kararı verir):** sonlu bir **yön sözlüğü** kur — mevcut yön, üye
yönleri, B-rep analitik eksenleri, kanal merkez hattı, dik düzlem adayları. Her eşleşen
çift için sözlükteki EN İYİ yönü seç (kâhin) ve robot metriğini hesapla.
**Kill: kâhin robot < 0.70 ise seçici eğitilmez, rota kapanır.**
Geçerse: vektör regresyonu değil **ayrık seçici** eğit (R2'nin dersi).

Beklenen: açı hatalarının 2/3'ü düzelirse robot ~0.65.

### Rota A — VERİ (ölçülmüş, çalışıyor, pahalı)

`F1 = 0.5295 + 0.0330·ln(grup)`, doymuyor. Şu an 953 grup.

| hedef | gereken grup | kat |
|---|---|---|
| 0.78 | ~1.900 | 2.0x |
| **0.80** | ~3.630 | 3.8x |
| 0.85 | ~16.515 | 17.3x |

Ve **bileşim önemli**: aynı hacimde karışık veri tek üreticiyi her boyutta yendi
(+0.033 / +0.051 / +0.049, ort **+0.0443**). Üçüncü üretici, aynı sayıda PXC/WEI
parçasından belirgin şekilde değerli.

**Talep (net):** *terminal block* + STEP + **kesin ConnectionPoints** kaydı olan üçüncü
üretici; en az **300**, tercihen **900+** yeni geometri grubu.
**Bölme protokolü: indirmeden ÖNCE 60/20/20; son %20 mühürlenir ve tek atış olarak saklanır.**
Kör STEP indirme yapılmaz — CP'si olmayan parça korpusa girmez.

### Rota B — SEGMENTASYON / ONTOLOJİ (tek yüksek-tavan rotası)

188 GT'ye (%14.2) hiç aday üretilmiyor. Bu, öznitelik tavanının *dışındaki* tek açık.

**İlk kapı (ucuz, kararı verir):** o 188 CP'nin konumunda segmentasyon ne diyor?
* **(a) CableEntry/Contact demiyor** → ontoloji/etiket sorunu; hedefli etiket kampanyası haklı
* **(b) diyor ama türetme eliyor** → ucuz eşik düzeltmesi, etiket gerekmez

**Kill:** (b) baskınsa etiket kampanyası açılmaz, türetme düzeltilir.

**Uyarı (ölçüldü):** *geometrik* tipleme işe yaramıyor — negatifler çok-modlu değil
(siluet 0.098-0.134) ve kaba 4'lü tipleme −0.0018 verdi. Yani etiketin değeri
"negatifleri tiplemek" değil, **segmentasyonun kaçırdığı açıklıkları öğretmek** olmalı.

## 4. Sıra

1. **Rota C ilk kapı** (yön sözlüğü kâhini) — saatler, tespite dokunmaz, robot için en büyük yakın pay
2. **Rota B ilk kapı** (188 CP teşhisi) — saatler, etiket kampanyasının gerekip gerekmediğini söyler
3. İkisinin sonucuna göre Rota A talebi yazılır (kaç grup, hangi üretici)
4. LOCKED sınavı **yalnızca** geliştirmede tespit ≥0.80 ve iki üretici-dışı kol da pozitifken;
   ve mutlaka `olcum_kumesi.sinav_egitim_maskesi()` ile eğitilmiş gate ile

## 5. Açılmayacaklar (ölçülüp kapandı)

multires · eşik taraması · ham/sıra dönüşümü · RF/GBM/lojistik değişimi · grup/parça ağırlığı ·
öznitelik seçimi (sızıntılı ve sızıntısız, ikisi de) · kutup örgüsü · B-rep boşluk grafı ·
ağ-kanal betimleyicisi · aday-tipi yönlendirme · tip-içi eşik · geometrik negatif tipleme ·
77→103 genel etiket · GT-tamlık düzeltmesi
