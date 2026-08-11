# D2 — Ölçüm protokolü: bu projede hatalardan öğrenilen kurallar

> Teze girecek bölüm taslağı. Buradaki her kural bir **yanlış ölçümden** doğdu; her birinin
> yanında onu doğuran somut olay var. Yöntem bölümüne değil, **tartışma/dersler** bölümüne aittir.

## 1. Rejim ayrımı — tek ortalama iki farklı problemi gizler

Terminal blokları CP sayısına göre **iki ayrı rejim** oluşturuyor ve bunların darboğazları farklı:

| | düşük-CP (<8 CP) | çok-CP (≥8 CP) |
|---|---|---|
| korpus payı | **%89.5** | %10.5 |
| CP-F1 | 0.82 | 0.55 |
| aday-recall | 0.905 | 0.676 |
| darboğaz | **precision** (gate) | **recall** (aday üretimi) |

Tek bir ortalama raporlamak, %10'luk bir dilimin çöküşünü %90'lık dilimin başarısıyla maskeliyor —
ve emeği yanlış yere yönlendiriyor. **Kural: her CP-F1 rejim ayrımlı raporlanır.**

## 2. Korpus ağırlıklandırma — çarpık örneklemde düz ortalama alma

Bir ölçümde skorlanan küme %18.6 çok-CP içeriyordu; gerçek korpusta oran **%10.5**. Düz ortalama
**0.691** verdi, korpus oranıyla ağırlıklandırılmış doğru sayı **0.79**. Aradaki 0.10'luk fark
tamamen örnekleme çarpıklığındandı.

**Kural: skorlanan kümenin bileşimi korpusunkinden farklıysa, rejim F1'leri gerçek oranla
ağırlıklandırılır.**

## 3. Kill kriteri — baştan yazılır ve ÜRÜN metriği üzerinden olur

Bir özellik ailesi gate AUC'sini **+0.024** yükseltti (etki gerçek; eşli bootstrap sıfırı dışlıyor)
ama CP-F1'e katkısı **+0.003** oldu — tohum gürültüsünün içinde. Kriter AUC'ye yazılmıştı ve
"yaşıyor" dedi; ürün metriğine yazılsaydı "ölü" derdi.

**Kural: kriter ürün metriği üzerinden ve ölçüm başlamadan önce yazılır. Sonucu gördükten sonra
eşiği değiştirmek, kriterin var olma sebebini yok eder.**

Alt kural: birden çok kaldıraç tek tek eşiği geçemiyorsa, kriter **bileşime** uygulanabilir — ama
bu, eşiği düşürmek değil **doğru birimi ölçmek** olmalı ve öyle belgelenmeli.

## 4. Zor bölünmeyle sağlamlık — holdout harcanmadan ÖNCE

Bir sınıflandırıcı değişikliği aile-dışı CV'de **+0.012** kazandırdı ve ürüne alındı. Kilitli
holdout onu **−0.029** ile geri çevirdi. Sonradan koşulan **üretici-dışı** sonda sebebi gösterdi:
aynı değişiklik dağılım kaydığında **−0.061**.

**Kural: bir kaldıracın dağılım kaymasına dayanıp dayanmadığı, üretici-dışı (ya da benzeri zor)
bölünmeyle holdout harcanmadan önce sınanır.** O sonda dakikalar sürüyor ve bu gerilemeyi
önceden söylerdi.

## 5. Geliştirme ölçümü ~3 kat iyimser — ürün rakamı holdout'tur

| | WORK OOF | kilitli holdout |
|---|---|---|
| dondurulmuş ürün | ~0.789 | **0.7418** |
| aday yapılandırma | 0.8423 | **0.7601** |
| iyileşme | +0.053 | **+0.0183** |

**Kural: ürün rakamı olarak holdout kullanılır; geliştirme sayısı her zaman o çerçeveyle birlikte
anılır.**

## 6. Tek atış disiplini — ve harcandıktan sonra iterasyon yok

Kilitli holdout **iki önceden tanımlı kolla, tek bakışta** harcandı. Sonrasında ortaya çıkan
düşük-CP gerilemesi WORK üzerinde teşhis edildi; düzeltme önerisi **doğrulanmamış** olarak
kaydedildi. Holdout'ta iterasyon yapmak, tek atışın koruduğu şeyi yok ederdi.

## 7. Kaldıraç etkileşimi — tek başına ölçüm yanıltabilir

Post-işleme gevşetmesi (`min_v10`) **tek başına −0.001**, yani hiçbir şey. Aynı değişiklik korpus
iki katına çıkarıldığında çiftin katkısı **+0.037** (toplamsal olsaydı +0.022).

**Kural: aday arzını değiştiren bir müdahale, küçük korpusta değerlendirilmez** — orada nötr veya
zararlı okunur.

## 8. Teşhis kodu, veriden önce şüphelenilecek taraftır

Bir gecede **beş ölçüm artefaktı** çıktı ve hepsi teşhis kodundandı, veriden değil:

- tek uçlu ışın testi açıklık kaybını %17→%26 şişirdi (iki uç: gerçek kayıp %8–12)
- yer gerçeğinden türetilen eksen bir geometrik süzgece sızdı (+0.047 sahte kazanç)
- yuvaya düz Öklid mesafesi imkânsız bir "%10 aday" sayısı üretti (doğrusu %86)
- etiketler yeniden-remesh'e kıyaslandı, kayıtlı mesh yerine — "yanlış yer" sanıldı
- iki farklı ağız tanımı karşılaştırıldı, "16 mm sapma" sanıldı

Altıncı artefakt aynı geceden, **ayrı bir imzayla**: teşhisi hızlandırmak için aday CP'ler diske
önbelleklendi, ama yalnız `point`/`direction`/`confidence` alanları saklandı. Üretimdeki
`wire_gate.feats_for` bunlara ek olarak `area`, `insertion_depth_mm` ve `_votes` okuyor; eksik
alanlar `c.get(..., 0.0)` ile sessizce sıfırlandı. 13 özelliğin 3'ü ölünce sınıflandırıcı skorları
sistematik olarak düştü ve **bütün eşik tablosu kaydı** (düşük-CP F1 0.714 → 0.482; optimum eşik
0.45 → 0.15 göründü). Hiçbir hata mesajı yoktu — çıktı makul görünüyordu, sadece yanlıştı.

**Alt kural: bir teşhis önbelleği, üretim kodunun okuduğu kaydın TAMAMINI saklar. Alt küme
saklanacaksa, geri yüklemede alan varlığı denetlenir ve eksikse ÇÖKER** (`c.get()` varsayılanı
bir önbellek sınırında sessiz veri kaybına dönüşür).

**Kural: yeni bir teşhis, tahminlerle GT'yi karşılaştırıyorsa mevcut kanonik eşleşme
konvansiyonunu (eksene duyarlı: dik mesafe + eksenel pencere) **aynen** kullanır, kendi mesafesini
uydurmaz. Üretim kodu ile teşhis kodu aynı geometrik soruya farklı cevap veriyorsa, önce teşhisten
şüphelenilir.**

## 9. Sessizce başarısız olan kod, çöken koddan tehlikelidir

Üç ayrı kök hata aynı imzayı taşıyordu: **hata vermeden yanlış çalışmak.**

- `rtree` yokluğunda `trimesh.contains()` her çağrıda istisna atıyordu; `except` onu yutunca iki
  geometri fonksiyonu girdisini geri döndürüp "başarılı" göründü
- `gmsh.finalize()` `finally` bloğunda değildi; tek bir meshlenemeyen STEP sonrası **bütün**
  parçalar sürünmeye başladı — 3 saat sıfır ilerleme, tek hata mesajı yok
- görselleştirmenin kendi doğrulaması aynı bozuk çağrıyı kullandığı için "8/8 geçti" yazdı

**Kural: bir geometri primitifi asla `except: continue` ile sarılmaz. İşini yapamayan bir yardımcı
girdisini geri döndürmez, **söyler**. Ve doğrulama, doğruladığı kodun içinde yaşayamaz —
bağımsız bir denetçi çıktıyı diskten geri okumalıdır.**

## 10. Determinizm testi süreç sınırını geçmelidir

Remesh üç kez **aynı süreçte** çağrıldı, aynı sonuç geldi, "deterministik" denildi. Ayrı
süreçlerde farklı sonuç veriyordu ve bu, aşağı akışta farklı CP sayısına yol açıyordu.

**Kural: süreç sınırını geçmeyen bir determinizm testi hiçbir şey test etmez.**
