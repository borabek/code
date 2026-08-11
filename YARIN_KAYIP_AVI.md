# YARIN — kayıp avı listesi

> Bu listede **fikir yok**. Her madde bir DENETİM: ya bozuk/yanlış dağıtılmış bir şey bulur
> (öder), ya mevcut ayarı doğrular (maliyeti bir saat, ama kayıp değil).
>
> Gerekçe: ölçülen 12 "yeni bilgi ekle" kaldıracının **12'si de öldü**. Yaşayan 7 değişikliğin
> **7'si de** bozuk/yanlış dağıtılmış bir şeyin düzeltilmesiydi. Bu boru hattında puanı
> kayıp avı artırıyor.

## Başlangıç noktası (1903 parça, aile-dışı OOF)

| rejim | aday-R | gate-R | precision | F1 | tavan |
|---|---|---|---|---|---|
| düşük-CP (%89.5) | 0.897 | 0.771 | 0.746 | 0.758 | 0.946 |
| çok-CP (%10.5) | 0.587 | 0.513 | 0.932 | 0.662 | 0.740 |

Korpus ağırlıklı **0.748**, tavan **0.924**.

---

## K1 — Eşik değişikliğini ÇALIŞAN HATTA doğrula  *(ilk iş, ~30 dk)*

Dün gece `robot_wire_gate_threshold` 0.35 → 0.45 yapıldı. **+0.0096** gate-veri seviyesinde,
tek türetmeyle ölçüldü. Ürün 4 modelin birleşimini kullanıyor — **fark orada da duruyor mu?**

- [x] 64-parçalık çalışan hat taraması → **0.45 yanlış, −0.0010**
- [x] Kök neden: 0.45, **1903-parçalık başka bir gate** için optimaldi; üründe **1041-parçalık**
      gate var. Farklı sınıflandırıcı → farklı skor dağılımı → farklı optimal eşik.
- [x] Dağıtılan gate ile 1903 parçada doğrulama → **0.40**
- [x] Uygulandı: `robot_wire_gate_threshold` **0.40** (çok-CP 0.25 iki ölçümde de optimal)

**SONUÇ: K1 TAMAM.** Kazanç küçük (+0.003 ağırlıklı) ama asıl değeri **yanlış bir dağıtımı
yakalamasıydı** — dün gece ürüne ölçülmemiş bir değer koymuştum.
Makbuz: `results/k1_esik_dogrulama.json`, `results/k1b_deployed_gate_thr.json`

## K2 — Gate'in attığı 636 GERÇEK adayı otopsi et  *(~1 sa)*

Düşük-CP'de gate 636 gerçek açıklığı öldürüyor (kayıp **0.1126**). Bu adaylar **zaten üretilmiş**.

- [x] **Sistematik dilim VAR ve güçlü: `votes`** (atılan medyan 1.00 vs tutulan 3.00, AUC **0.86**).
      Ardından `conf` (0.63 vs 0.83, AUC 0.83), `size` (7.4 vs 13.0), `depth` (2.0 vs 3.7).
      Gate'in attığı gerçekler = **4 modelden sadece birinin gördüğü, küçük, düşük güvenli** açıklıklar.
- [x] Skor dağılımı: **%42'si 0.30–0.40** (kıl payı kaçan) · %27'si <0.20 (gerçekten yanlış sınıflanan)
- [x] Üretici: **WEI %23 atılıyor, PXC %13** — yay-kelepçe sistematik olarak daha çok kaybediyor

**K2'DEN ÇIKAN ASIL BULGU — bir sonraki iş (K2b):**
`votes` gate'in **en güçlü özelliği** (AUC 0.86) ve doğrudan `min_v`'ye bağlı. K4c'de `min_v`
10 → 4 yapıldı, yani oy dağılımı **değişti**. Dağıtılan gate ise 25 Temmuz'da, **eski `min_v`
ile üretilmiş** adaylarla eğitilmiş. Gate'in en güçlü özelliğinin dağılımı altından kaydı.

Bu, bu gece ölen "gate'i daha çok parçayla eğit" ile **aynı şey değil** (o, aynı boru hattıyla
daha çok veriydi). Bu, **değişmiş boru hattına gate'i uydurmak** = K6'nın "ölçülen ≠ çalışan"
sınıfı, ve o sınıf bu projede her seferinde ödedi.

- [ ] **K2b** `min_v=4` ile aday çıkarımı → gate'i O dağılımla yeniden eğit → gerçek hatta ölç
      (~1 sa çıkarım + 10 dk eğitim). **KILL: < +0.02**

## K3 — Tutulan 1330 YANLIŞ adayı otopsi et  *(~1 sa)*

En büyük tek kayıp (**0.2277**). Eski bir ölçüm "hayatta kalan FP'ler TP'lerle geometrik olarak
aynı" demişti ama o **farklı bir FP kümesiydi**; bu 1330 hiç incelenmedi.

- [x] **Sistematik dilim YOK.** En güçlü ayırıcı `nn_dist` AUC **0.62**; ardından `conf` 0.60,
      `outward` 0.59. Neredeyse kör.
- [x] **Üretici kırılımı yok:** WEI %20, PXC %20 — birebir aynı.
- [x] **Aile yoğunlaşması yok:** 925 yanlış, **568 ayrı aileye** yayılmış, en kötü aile 7 tane.
- [x] Skorların %54'ü 0.40–0.50 arası (eşiğin hemen üstünde), %8'i ≥0.70.

**SONUÇ: K3 "kanıtlanmış olumsuz" ile bitti — ve bu, listenin vaat ettiği geçerli sonuç.**
Düşük-CP precision mevcut özelliklerle çözülemez. Bu, önceki bulguyu **tamamen yeni bir FP
kümesinde** doğruluyor.

**RECALL vs PRECISION karşıtlığı (K2 + K3 birlikte):**

| taraf | en güçlü sinyal | anlamı |
|---|---|---|
| recall — atılan gerçekler (K2) | `votes` AUC **0.86** | yapı VAR → K2b ödemeli |
| precision — tutulan yanlışlar (K3) | `nn_dist` AUC **0.62** | yapı YOK → yeni bilgi şart |

**En büyük kayıp kalemi (0.2277) geometriyle kapanmıyor → K7 artık o kalem için TEK yol.**

## K4 — Adaysız kalan 522 düşük-CP GT'yi ikiye ayır  *(~1 sa)*

Düşük-CP'de 522 GT'nin hiç adayı yok (kayıp **0.0924**). İki bambaşka sebep olabilir:

- [x] **Düşük-CP: %100 (a)** — adaysız GT ağzında CE+CT tepe medyanı **0.05**, ağ oraya hiç
      bakmıyor. Düşük-CP için çıkarım parametresi taramak boşuna → hipotez (b) düşük-CP'de ÖLÜ.
- [x] **Çok-CP: %55 (b)** — tepe medyanı 0.39, %33'ünde ≥0.50. Ağ gördü, çıkarım öldürdü.
- [x] **K4b parametre taraması (çok-CP):** `vertex_conf` ve `cluster_mm` bu rejimde **tamamen
      ölü knob** (hiçbir değerde fark yok). `ct_depth_min_mm` 1.0→0.0 küçük. Kazanan: **`min_v`**.
- [x] **K4c gerçek boru hattı doğrulaması (44 parça, 4-model union):**
      min_v 10 → 4 = **+0.0224 ağırlıklı** — kapıyı geçti, **uygulandı**.
      Kazanç **düşük-CP'den** geldi (0.7164 → 0.7424); tek-türetmeli ölçüm bunun tersini
      söylüyordu. `min_v` her modelin kendi türetmesinde uygulandığı için union yolu
      önbellekten doğrulanamıyor — K1'in dersi burada da geçerliydi.
      Geri alma: `cp_config.json.bak_minv_2026_07_29` · makbuz `results/k4c_minv.json`

**SONUÇ: K4 TAMAM ve ÖDEDİ (+0.0224).** Ayrıca düşük-CP'nin aday oluşumunda çözülecek bir şey
OLMADIĞI kanıtlandı — oradaki kayıp ağın ateşlememesi, yani model sorunu.

## K5 — Çok-CP'de adaysız 1430 GT (%41)  *(~1 sa)*

Çok-CP precision **0.932** — orada yanlış pozitif yok, sorun tamamen aday açlığı.

- [x] **`min_v=4` ile bugünkü ürün:** çok-CP GT 478'in **175'i adaysız (%37)** — önce %41'di.
      Bunların **83'ünde ağ ATEŞLEDİ** (%47, çıkarım kaybı), 92'sinde ateşlemedi (model kaybı).
- [x] **`min_v` artık suçlu değil:** ateşlemiş-ama-adaysız ağızlarda **medyan 23 sıcak vertex**,
      yalnız **%1'i `min_v=4`'ün altında**.
- [x] **K5b `dedupe_mm`:** aritmetik sorun GERÇEK — çok-CP'de GT'ler arası en yakın mesafe
      **medyan 5.1 mm**, %93'ü 10 mm'nin altında, birleştirme yarıçapı ise 10 mm.
      Ama ödemesi küçük: dedupe 10→4 = çok-CP **+0.0046** (ağırlıklı +0.0005), çünkü birleştirme
      yalnız **sınıflar arası** çalışıyor. Kod birden çok yerde sabit → **değiştirilmedi**.
- [x] **K5c `conn_promote` taraması (tek türetme, DOĞRULANMADI):**

| promote | aday-R | aday sayısı | çok-CP F1 |
|---|---|---|---|
| **0.10** | 0.561 | 368 | **0.4720** |
| 0.25 (üründe) | 0.634 | 350 | 0.4215 |

Ters yönlü: promote düşünce aday-recall düşüyor ama F1 **+0.0505** yükseliyor (ağırlıklı +0.0053).
**Kabul EDİLMEDİ** — bugün tek-türetmeli ölçüm hem eşikte hem `min_v`'de gerçek hatta transfer
etmedi, biri yönü bile ters gösterdi. Gerçek boru hattında (4-model union) doğrulanmadan
ürüne girmez.

- [ ] **K5d** `conn_promote` 0.10 vs 0.25 — GERÇEK boru hattı doğrulaması (K4c düzeneği, GPU)

**SONUÇ: K5 TAMAM.** Çok-CP kaybı isimlendirildi: **yarısı model (ağ ateşlemiyor), yarısı
çıkarım.** Çıkarım tarafında `min_v` ve `dedupe` artık tükendi; tek canlı knob `conn_promote`
ve o da doğrulama bekliyor.

## K6 — Dağıtım denetimi: ölçülen ≠ çalışan  *(~45 dk)*

Bu hata bugün **iki kez** çıktı (router girdi uyuşmazlığı, config okunmayan ayar) ve ikisi de
ölçülmüş bir kazancın robota hiç ulaşmadığı anlamına geliyordu.

- [x] 34 anahtarın **17'si okunmuyor**. 16'sı zararsız belge notu (`*_note`, tarihli kayıtlar).
- [x] **BİRİ GERÇEK BUG: `robot_auto_gate_threshold` (0.66) hiçbir yerde okunmuyordu.**

**K6'NIN BULDUĞU KUSUR — bugünün 3. "ölçüldü ama dağıtılmadı" vakası:**
Config'de 28 Temmuz tarihli **kullanıcı kararı** yazılı: *AUTO katmanı = `wire_score ≥ 0.66`*,
ve *"ÖNCEKİ KUSUR: katman segmentasyon güvenine bakıyordu → kilitli holdout'ta REVIEW BOŞ çıktı,
her şey AUTO işaretlendi, precision 0.7735 — 4 yerleştirmede 1 hata, insana hiç sorulmadan."*
Karar verilmiş, config'e yazılmış, **koda hiç bağlanmamış**. `_format_cps` hâlâ eski kusuru
çalıştırıyordu.

- [x] **Bağlandı** (`robot_cp._format_cps`); gate kapalıysa eski kurala düşüyor
- [x] **K6b ölçümü (64 parça, aynı adaylar, yalnız katman kuralı değişiyor):**

| kural | AUTO CP | doğru | precision |
|---|---|---|---|
| eski (segmentasyon güveni) | 107 | 92 | **0.8598** |
| yeni (gate skoru ≥ 0.66) | 43 | 43 | **1.0000** |

**SONUÇ: K6 TAMAM.** CP-F1 değişmez (katman etiketler, filtrelemez) ama robot artık **%86
precision'la otonom davranmıyor**. Ürün güvenilirliği açısından günün en önemli düzeltmesi.
Kapsama seviyesi (%7–15) belgelenen %52 ile kıyaslanamaz — bu örneklem %37 çok-CP, korpus %10.5.

---

## 0.83 hakkında — dürüst hesap

| kaynak | gerçekçi |
|---|---|
| K1 eşik doğrulaması | +0.005 … +0.010 |
| K2 atılan gerçekler | 0 … +0.02 |
| K3 tutulan yanlışlar | 0 … +0.03 |
| K4 düşük-CP çıkarım aritmetiği | 0 … +0.03 |
| K5 çok-CP | +0.005 |
| K6 dağıtım denetimi | 0 … +0.02 |

**Toplam gerçekçi: 0.76 → 0.78–0.83.** Üst uç ancak K3 ve K4'ün İKİSİ birden sistematik bir
dilim bulursa gelir. **0.83 mümkün ama garanti değil**; garanti edebileceğim tek şey bu
maddelerin hiçbirinin "ölçtük, hiçbir şey çıkmadı, emek boşa gitti" ile bitmeyeceği —
her biri ya kazanç ya kesinleşmiş bilgi bırakır.

Eğer K2/K3 "sistematik dilim yok" derse, o zaman düşük-CP precision'ı geometriyle çözülemez
demektir ve tek yol `TODO_STEP_RENK.md` olur. O cevap da bu listeden çıkar.

## K7 — STEP yüz rengi: yeni bilgi kanalı  *(uzun iş, K2/K3 sonucuna bağlı)*

Bu listedeki **tek** "yeni bilgi" maddesi ve bilerek en sona konuldu: K2/K3 "sistematik dilim
yok" derse düşük-CP precision'ın geometriyle çözülemediği kanıtlanmış olur ve bu madde tek yol
haline gelir. Detaylı plan: `TODO_STEP_RENK.md`.

Neden bu, diğer "yeni bilgi" fikirlerinden farklı: hepsi mesh geometrisiydi ve bilgi duvarı
bulgusu (sızıntılı üst sınır +0.003) o aileyi zaten öldürmüştü. Renk **mesh'te yok** — STEP'in
kendi verisi, AP214 `OVER_RIDING_STYLED_ITEM`, 108/108 yüzde doğrulandı. Tel girişi bir
**kontakta** biter, alet ağzı bir kola/yaya: kanal dibinde metal görünmesi tam da geometrinin
veremediği fonksiyonel sinyal.

- [ ] **K7.1** Yüz eşleştirici: B-rep yüzü → mesh üçgeni, geometriyle (ağırlık merkezi + normal
      + alan). Sıra/indis eşleştirmesi denendi, null testinde **2/5** ile çakıldı.
      **KAPI: null testi ≥ 4/5.**
- [ ] **K7.2** `dip_metal` özelliği: ağızdan eksen boyunca ışın, ilk çarpılan yüzün rengi metal mi
- [ ] **K7.3** Ayırt edicilik: TP vs FP, **KAPI: AUC ≥ 0.70** (`dip_CT` zaten 0.61 veriyor;
      renk bunu belirgin geçmiyorsa yeni bilgi getirmiyor demektir → dur)
- [ ] **K7.4** Gate'e ekle, aile-dışı OOF, rejim ayrımlı. **KILL: < +0.02**
- [ ] **K7.5** Üretici-dışı sonda (ExtraTrees dersi: aile-dışı +0.012, üretici-dışı −0.061)
- [ ] **K7.6** Kapsam: renksiz STEP'te özellik **NaN** olmalı, 0 değil (0 = "metal yok" yalanı)

**Neden ölebilir:** bu listedeki tek ölebilen madde. K7.3 kapısı, K7.1'in emeği boşa gitmeden
karar verdirir.
**Beklenen:** bilinmiyor — ve değeri de o. 0.83+ buradan gelirse gelir.

---

## Sıra (güncel)

**K1 → K4 → K2 → K3 → K6 → K5 → (K2/K3 sonucuna göre) K7**
(K1 önce çünkü ürün şu an doğrulanmamış bir değerle çalışıyor. K4 ikinci çünkü en ucuz
"yapılandırma aritmetiği" adayı ve o sınıf bu projede iki kez ödedi.)
