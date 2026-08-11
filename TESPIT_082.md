# TESPİT 0.7584 → 0.82 — OPERASYON LİSTESİ

> Tez çizgisi: ağ mimarisi, tezin remesh'i, `v_o` türetmesi ve 5-sınıf kaybı **hiçbir
> maddede değişmez**. Her madde önceden yazılmış kill ile ölçülür; sonucu gördükten sonra
> bar değişmez. Ölçüm: 194 parça / 171 grup, üretici hakemi, grup bootstrap, sızıntı bekçili.

---

## 0. NEDEN 0.82 MÜMKÜN, 0.87 DEĞİL

| hedef | dengeli recall | aday havuzu tavanı 0.8579 | ayrılabilirlik tavanı ~0.86 |
|---|---|---|---|
| **0.82** | 0.820 | **altında ✓** | **altında ✓** |
| 0.85 | 0.850 | altında (kıl payı) | sınırda |
| 0.87 | 0.870 | **üstünde ✗** | üstünde ✗ |

**0.82 yeni aday gerektirmez.** Havuz zaten 1135 GT'ye ulaşıyor; gate 873'ünü alıyor.
Gereken: **havuzda duran ama gate'in reddettiği CP'lerin %55'ini kurtarmak** ve kesinliği
0.745 → 0.820 çıkarmak.

Ve o CP'ler somut: **gate'in reddettiği 248 doğru aday.** Profilleri ölçüldü —
az oylu (1.33 vs 1.98), düşük güvenli (0.404 vs 0.574), küçük (137 vs 202 tepe),
**yuvarlak olmayan** (%51 vs %81), **yoğun parçalarda** (%77).

Hedef: o 248'in ~136'sını kurtar, aynı anda 313 yanlış pozitifin ~90'ını at.

---

## 1 · TEL/ALET SİNYALİ *(eğitiliyor — en büyük tek kol)*

**Teşhis:** tezin `Contact` sınıfı tanım gereği *"Kontaktierung bzw. Werkzeugeinschub"* —
tel girişi ve alet ağzı **tek sınıf**. Ağ, ayırmak istediğimiz iki şeyi **birleştirmek
üzere** eğitildi. Dokuz gate mekanizmasının aynı duvara çarpmasının kökü bu.

Paylaşılan gövdeye ikinci kafa eklendi; ana 5-sınıf kafası ve kaybı `diff` ile doğrulandı
(**bit düzeyinde aynı**). Etiket üreticinin kendi listesinden: eşleşen açıklık = tel,
eşleşmeyen = alet (53 parça, 30.433 / 56.547 tepe).

**Bu, 73 sütunda OLMAYAN tek bilgi.** Doğrudan 313 yanlış pozitifi hedefler.
**Beklenen:** kesinlik +0.02…+0.06 → F1 +0.015…+0.045
**Kill:** tespit +0.01 ve GA sıfırı dışlamalı; üretici-dışı düşmemeli.

## 2 · 8 ÜYELİ TOPLULUK *(aynı koşuda)*

Kayıtta yazılı ve bugün doğrulandı: **gate'in 13 özelliğinden yalnız `votes` dürüst
geometri bölmesinde transfer ediyor** (AUC düşüşü 0.018; diğerleri 0.12–0.18). Ve
reddedilen 248 doğru adayın ayırt edici özelliği tam olarak **düşük oy**.

4 üyeyle oy ∈ {1,2,3,4}; 8 üyeyle {1..8} → genelleşen tek sinyalin çözünürlüğü **ikiye
katlanır**. Ayrıca birleşim daha çok aday bulur.
**Beklenen:** F1 +0.01…+0.03 · **Kill:** aynı.

## 3 · GÜVEN-AĞIRLIKLI OY *(15 dk — en ucuz untried kol)*

Oy şu an tamsayı: 0.92 güvenle bulan üye ile 0.31 ile bulan **aynı** sayılıyor. Sürekli,
güven-ağırlıklı sürüm genelleşen tek özelliğin **rafine hâli** — transfer riski düşük,
maliyeti neredeyse sıfır (türetme zaten var, yalnız yeni sütun).
**Beklenen:** F1 +0.005…+0.015 · **Kill:** aynı.

## 4 · RECALL TABANI *(15 dk)*

Top-K başarısız oldu çünkü *tam* K seçmeye zorluyordu (K yanlışsa eşikten kötü).
**Taban** farklı: "hiçbir parçada N'den az aday kabul etme". Yanlış K'nın cezası yok;
yalnız aşırı-sessiz parçalar kurtulur — ve reddedilen 248'in %77'si yoğun parçalarda.
**Beklenen:** F1 +0.005…+0.01 · **Kill:** aynı.

## 5 · ÜYE BAŞINA EŞİK KALİBRASYONU *(30 dk)*

Tüm üyeler aynı `min_v`/`vertex_conf` kullanıyor. Sistematik olarak daha temkinli bir üye
daha az aday üretir ve **yalnız onun gördüğü** açıklıklar kaybolur. Her üyenin eşiği aday
sayısını eşitleyecek şekilde kalibre edilir.
**Beklenen:** F1 +0.005…+0.02 · **Kill:** aynı.

## 6 · EKSEN-FARKINDALIKLI OY HAVUZLAMASI *(koşuyor)*

`_vote2` düz Öklid 5mm kullanıyor; ürünün puanlayıcısı `big_arbiter.greedy` zaten
eksen-farkındalıklı — ürün kendi içinde tutarsız. Ölçüldü: birleşme %67.7 → %70.7.
Yanal 3mm sınırı komşu kutupları (adım 3.5–6mm) korur, derinliği serbest bırakır.
**Beklenen:** F1 +0.003…+0.01 (marjinal ama kural daha doğru)

## 7 · ÜRETİCİ-DEĞİŞMEZ GATE *(2 sa — 1-6 bitince)*

Ölçülen patoloji: havuzlanmışı iyileştiren her kol görülmemiş üreticiyi çökertiyor — gate
**ezberliyor**. Çare düzenlileştirme değil, **alan-değişmezliği**: gradyan-tersleme ile
üreticiyi tahmin edemeyen temsil. RF gradyan taşımadığı için sinirsel gate gerekir.
**Beklenen:** havuzlanmışta +0.01, üretici-dışında daha çok · **Ön koşul:** taban sabitlensin.

## 8 · VERİ *(dış bağımlılık, ölçülmüş fiyat)*

`F1 = 0.5295 + 0.0330·ln(grup)`, doymuyor. **0.82 için ~2.500 grup (2.6×).**
Bileşim de önemli: aynı hacimde karışık üretici **+0.0443**.
**Talep:** terminal block + STEP + **kesin ConnectionPoints**; en az 300, tercihen 900+
yeni geometri grubu; **indirmeden ÖNCE 60/20/20, son %20 mühürlü.**

---

## TOPLAM VE DÜRÜST BEKLENTİ

| # | kol | maliyet | beklenen katkı |
|---|---|---|---|
| 1 | tel/alet sinyali | eğitim (koşuyor) | +0.015…+0.045 |
| 2 | 8 üyeli topluluk | aynı koşu | +0.010…+0.030 |
| 3 | güven-ağırlıklı oy | 15 dk | +0.005…+0.015 |
| 4 | recall tabanı | 15 dk | +0.005…+0.010 |
| 5 | üye eşik kalibrasyonu | 30 dk | +0.005…+0.020 |
| 6 | eksen-farkındalıklı havuz | aynı koşu | +0.003…+0.010 |
| **toplam (kod)** | | **~1.5 sa + eğitim** | **+0.043…+0.130** |
| 7 | üretici-değişmez gate | 2 sa | +0.010 |
| 8 | veri 2.6× | dış | +0.023 |

**Gereken: +0.062.** Kod kollarının iyimser toplamı bunu **iki kat** aşıyor, kötümser
toplamı (+0.043) ise altında kalıyor.

**Dürüst hüküm:** 0.82, kod kollarının **çoğunun tutması** hâlinde erişilebilir — ve
0.87'den farklı olarak **yapısal bir engeli yok** (hem havuz tavanının hem ayrılabilirlik
tavanının altında). Tek gerçek risk, ölçülen **"cephe yasası"**: bugüne kadar havuzlanmışı
iyileştiren her kol görülmemiş üreticiyi çökertti. Bu yüzden her kolun kill'inde
*"üretici-dışı düşmemeli"* şartı var ve kaldırılmayacak.

**Merkez tahmin: 0.79–0.82. İyimser: 0.85.**

---

## MAKİNE DİSİPLİNİ

Aynı anda **tek ağır iş**. Durum kontrolü **PowerShell `Get-CimInstance Win32_Process`**
ile — Git Bash `ps` komut satırını göstermez, `grep` hep 0 döndürür ve `pkill -f` hiçbir
şey öldürmez (bugün bu yüzden koşan bir işi üç kez yeniden başlattım).

## AÇILMAYACAKLAR (ölçülüp kapandı)

multires · eşik taraması · ham/sıra dönüşümü · RF/GBM/lojistik değişimi · grup/parça
ağırlığı · öznitelik seçimi (sızıntılı **ve** sızıntısız) · kutup örgüsü · B-rep boşluk
grafı · ağ-kanal betimleyicisi · aday-tipi yönlendirme · tip-içi eşik · top-K sayım seçimi ·
geometrik negatif tipleme · agresif aday havuzu · aile transferi · uyarlanabilir çözünürlük ·
GT-tamlık düzeltmesi · val Conn-IoU ile checkpoint seçimi (bugün Spearman 0.000 ölçüldü)
