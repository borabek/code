# TESPİT 0.7584 → 0.87 — YOL HARİTASI

> Tez çizgisi: ağ mimarisi, tezin remesh'i, `v_o` türetmesi ve 5-sınıf kaybı **hiçbir
> maddede değişmez**. Her madde önceden yazılmış kill ile ölçülür.
> Ölçüm: 194 parça / 171 geometri grubu, üretici hakemi, grup bootstrap, sızıntı bekçili.

---

## 0. HEDEFİN ARİTMETİĞİ — listeyi bu belirliyor

Bugün: **TP 873 · FP 313 · FN 450** → kesinlik 0.745 · recall 0.774 · **F1 0.7584**

F1 = 0.87 için kesinlik ve recall'ın **ikisi birden** ~0.87 olmalı. Aday havuzu en fazla
**1135** GT'ye ulaşıyor (recall tavanı 0.8579), dolayısıyla tek geçerli çözüm:

| büyüklük | bugün | 0.87 için gereken | fark |
|---|---|---|---|
| doğru pozitif | 873 | **1135** (havuz tavanı) | **+262** |
| yanlış pozitif | 313 | **152** | **−161** |
| kesinlik | 0.745 | **0.882** | +0.137 |
| recall | 0.774 | **0.858** (tavan) | +0.084 |

**Okuma:** 0.87, "gate her ulaşılabilir CP'yi yakalayacak **ve** yanlış pozitiflerin
yarısını atacak" demek. İkisi aynı anda. Ve ölçülen üç tavan bunu kısıtlıyor:

- aday havuzu recall'ı **0.8579** → TP 1135'in üstüne çıkılamaz (havuz büyümeden)
- el-yapımı 73 sütunun ayrılabilirlik tavanı **~0.86** → 0.87 bunun **üstünde**
- adayların **%32.5'i ayrılamaz bölgede** (komşularının çoğu farklı etiketli)

**Sonuç: 0.87, mevcut aday havuzu ve mevcut temsille ULAŞILAMAZ.** Liste bu yüzden iki
cephede birden çalışır: **havuzu büyüt** ve **temsili değiştir**.

---

## CEPHE I — HAVUZU BÜYÜT (recall tavanını 0.8579'un üstüne çıkar)

### I-1 · 8 üyeli topluluk *(koşuyor)*
Birleşim daha çok açıklık bulur → havuz tavanı yükselir. Ayrıca oy ∈ {1..8} olur ve
**genelleşen tek özelliğin** (`votes`, AUC düşüşü 0.018; diğerleri 0.12–0.18)
çözünürlüğü ikiye katlanır.
**Beklenen:** havuz recall'ı +0.01…+0.03 · **Kill:** tespit +0.01, GA>0, üretici-dışı düşmesin.

### I-2 · Üye başına eşik kalibrasyonu
Tüm üyeler aynı `min_v`/`vertex_conf` kullanıyor. Sistematik olarak daha temkinli bir üye
daha az aday üretir ve **yalnız onun gördüğü** açıklıklar kaybolur. Her üyenin eşiği aday
sayısını eşitleyecek şekilde kalibre edilir.
**Maliyet:** ~30 dk · **Beklenen:** +0.005…+0.02

### I-3 · 188 adaysız CP'nin ulaşılabilirlik teşhisi *(önce bu)*
CP ekseni boyunca dışarıdan ışın at: malzemeye çarpmadan 2mm yaklaşabiliyorsa **açık kanal**
(ağ hatası, kazanılabilir); yaklaşamıyorsa CP **malzemenin arkasında** (üretici iç kontağı
listelemiş — hiçbir yüzey yöntemi bulamaz).
**Neden ilk:** kapalı oran yüksekse havuz tavanı 0.8579 **sahte düşük**, gerçek tavan daha
yukarıda ve 0.87 aritmetiği kolaylaşır. Bu ölçüm hedefin gerçekçiliğini yeniden tanımlar.
**Maliyet:** ~20 dk

---

## CEPHE II — TEMSİLİ DEĞİŞTİR (ayrılabilirlik tavanını ~0.86'nın üstüne çıkar)

### II-1 · Tel/alet yardımcı kafası *(eğitiliyor)*
**Teşhis:** tezin `Contact` sınıfı tanım gereği *"Kontaktierung bzw. Werkzeugeinschub"* —
tel girişi ve alet ağzı **tek sınıf**. Ağ, bizim ayırmak istediğimiz iki şeyi
**birleştirmek üzere** eğitildi. Dokuz gate mekanizmasının aynı cepheye çarpmasının kökü bu.

Paylaşılan gövdeye ikinci kafa eklendi (ana 5-sınıf kafası ve kaybı `diff` ile doğrulandı:
**bit düzeyinde aynı**). Etiket: eşleşen açıklık = tel, eşleşmeyen = alet (53 parça,
30.433/56.547 tepe).

**Bu, 73 sütunda OLMAYAN tek bilgi** — yani ayrılabilirlik tavanını yükseltme ihtimali olan
tek kol. Doğrudan 313 FP'yi hedefliyor.
**Beklenen:** kesinlik +0.02…+0.06 · **Kill:** tespit +0.01, GA>0

### II-2 · Güven-ağırlıklı oy — *en ucuz untried kol*
Oy şu an tamsayı: 0.92 güvenle bulan üye ile 0.31 ile bulan **aynı** sayılıyor.
Güven-ağırlıklı sürüm, genelleşen tek özelliğin **rafine hâli** — transfer riski düşük.
**Maliyet:** ~15 dk · **Beklenen:** +0.005…+0.015

### II-3 · Eksen-farkındalıklı oy havuzlaması *(koşuyor)*
`_vote2` düz Öklid 5mm kullanıyor; ürünün puanlayıcısı zaten eksen-farkındalıklı — ürün
kendi içinde tutarsız. Ölçüldü: birleşme %67.7 → %70.7. Yanal 3mm sınırı komşu kutupları
(adım 3.5–6mm) korur.
**Beklenen:** +0.003…+0.01 (marjinal ama kural daha doğru)

### II-4 · Üretici-değişmez gate
Ölçülen patoloji: havuzlanmışı iyileştiren her kol görülmemiş üreticiyi çökertiyor — gate
**ezberliyor** (13 özellikten yalnız `votes` transfer ediyor). Çare düzenlileştirme değil,
**alan-değişmezliği**: gradyan-tersleme ile üreticiyi tahmin edemeyen temsil. RF gradyan
taşımadığı için sinirsel gate gerekir.
**Maliyet:** ~2 sa · **Ön koşul:** I ve II-1…3 bitmeli (taban değişiyor)

---

## CEPHE III — VERİ (ölçülmüş fiyat etiketi)

`F1 = 0.5295 + 0.0330·ln(grup)`, **doymuyor** (son iki noktanın eğimi 0.0439).

| hedef | gereken grup | kat |
|---|---|---|
| 0.78 | ~1.900 | 2.0× |
| 0.80 | ~3.630 | 3.8× |
| **0.87** | **~34.000** | **36×** |

Ve bileşim hacimden ayrı değer taşıyor: aynı grup sayısında **karışık üretici**, tek
üreticiyi her boyutta yendi (+0.033 / +0.051 / +0.049 → ortalama **+0.0443**).

**Talep:** terminal block + STEP + **kesin ConnectionPoints**; en az 300, tercihen 900+ yeni
geometri grubu; **indirmeden ÖNCE 60/20/20 bölme, son %20 mühürlü.**

---

## SIRA VE DÜRÜST BEKLENTİ

| # | kol | maliyet | beklenen | durum |
|---|---|---|---|---|
| 1 | I-1 + II-1 + II-3 (tek koşuda) | eğitim + 40 dk | +0.02…+0.08 | **koşuyor** |
| 2 | I-3 ulaşılabilirlik teşhisi | 20 dk | tavanı yeniden tanımlar | sırada |
| 3 | II-2 güven-ağırlıklı oy | 15 dk | +0.005…+0.015 | sırada |
| 4 | I-2 üye eşik kalibrasyonu | 30 dk | +0.005…+0.02 | I-3'e bağlı |
| 5 | II-4 üretici-değişmez gate | 2 sa | +0.01 (çoğu üretici-dışında) | 1–4 bitince |
| 6 | III veri | dış | +0.042 (3.8× ile) | talep yazıldı |

**Toplam iyimser:** +0.08…+0.12 → tespit **0.84…0.88**
**Merkez beklenti:** +0.02…+0.04 → tespit **0.78…0.80**

**Dürüst hüküm:** 0.87, 1–5 kollarının **hepsinin çalışması ve toplanabilir olması**
hâlinde erişilebilir. Ama ölçülen "cephe yasası" toplanabilirliğe karşı: bugüne kadar
havuzlanmışı iyileştiren her kol görülmemiş üreticiyi çökertti, ve dokuz bağımsız
mekanizma aynı duvara çarptı. **Veri olmadan 0.87 gerçekçi değil; 0.80 gerçekçi.**

---

## MAKİNE DİSİPLİNİ

Aynı anda **tek ağır iş**. Bugün beş süreç aynı anda koştu ve her şeyi yavaşlattı; sebep
Git Bash `ps`'inin komut satırını göstermemesiydi (`grep` hep 0 döndü, `pkill -f` hiçbir
şey öldürmedi). Durum kontrolü **PowerShell `Get-CimInstance Win32_Process`** ile yapılır.

## AÇILMAYACAKLAR (ölçülüp kapandı)

multires · eşik taraması · ham/sıra dönüşümü · RF/GBM/lojistik değişimi · grup/parça ağırlığı ·
öznitelik seçimi (sızıntılı **ve** sızıntısız) · kutup örgüsü · B-rep boşluk grafı · ağ-kanal
betimleyicisi · aday-tipi yönlendirme · tip-içi eşik · top-K sayım seçimi · geometrik negatif
tipleme · agresif aday havuzu · aile transferi · uyarlanabilir çözünürlük · GT-tamlık düzeltmesi ·
val Conn-IoU ile checkpoint seçimi (bugün geçersiz olduğu ölçüldü: Spearman 0.000)
