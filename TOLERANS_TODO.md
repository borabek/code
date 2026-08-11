# TOLERANS SORUNU — kökten çözüm listesi (2026-07-30)

## Sorun tek cümlede

Eşleşme toleransı **parça boyutuyla ölçekleniyor** (`max(3mm, %6 × köşegen)`) ve eksenel pencere
**±40 mm**. Robotun gereksinimi ise sabit: **<2 mm**. Büyük ve yoğun parçalarda tolerans 15 mm'ye
çıkıyor; 14 mm ötedeki bir tahmin "doğru" sayılabiliyor. 5 parçalık sondada kesinlik %6-toleransla
**1.000**, 2 mm ile **0.455** çıktı.

Bu, projenin daha önce kaydettiği şişme hatasının **kardeşi**: eski "%98 kesinlik" alet ağızlarını
doğru sayıyordu. Orada **tanım** yanlıştı, burada **ölçüt** gevşek. İkisinin ortak deseni:
*metrik, ürünün gerçek işini değil kendi kolaylığını ölçüyor.*

---

## T1 — Gerçek sayıyı öğren *(KOŞUYOR)*

`tolerans_gercegi.py` — P9 ile **birebir** protokol (aynı 100 parça, tohum 202, test aileleri
gate eğitiminden dışarıda, eşikler büyük veride aile-dışı OOF). **Tek çıkarım, dört ölçüt**:
mevcut %6 · sabit 5 mm · sabit 3 mm · sabit 2 mm. Fark böylece modelden değil yalnız ölçütten gelir.

**Çıktı:** tolerans eğrisi + hizalama residual dağılımı. Makbuz `results/tolerans_gercegi.json`.

---

## T2 — Hata bütçesini ayrıştır: hizalama mı, model mi?

2 mm bütçenin bir kısmını **STEP↔JSON çerçeve hatası** yiyor (ölçülmüş: 0.3–1.8 mm residual).
Modeli 2 mm'de yargılamak, kendi ölçüm hatamızı modele fatura etmek olur.

- [ ] Her parça için `align_frames` residual'ı ile eşleşme mesafesini **birlikte** kaydet
- [ ] Model hatasını residual'dan arındır: `model_err² ≈ eşleşme_mesafesi² − residual²`
- [ ] **Karar:** residual medyanı >1 mm ise adil tolerans 2 mm değil `2mm + residual` olmalı ve
      bu **açıkça** yazılmalı

---

## T3 — Eksenel pencereyi kaldır, tanımı düzelt *(asıl kök neden)*

±40 mm pencere bir tolerans değil, **tanım uyuşmazlığının yaması**: üreticinin CP noktası eksende
başka bir derinlikte, bizimki ağızda. Yamayı büyütmek yerine tanımı eşitle.

- [ ] Üretici CP'sinin ağza göre eksenel ofsetini **ölç** (dağılım, üretici kırılımıyla)
- [ ] İki tarafı da **aynı referansa** taşı (ikisini de ağız düzlemine izdüşür, ya da ikisini de
      eksende aynı derinliğe)
- [ ] Pencereyi ölçülen dağılımdan seç (ör. %99'u kapsayan), **40 mm'yi elle yazma**
- [ ] **Kill:** tanım eşitlendikten sonra gereken pencere hâlâ >10 mm ise, üretici CP'si ağzı
      göstermiyor demektir — bunu ayrı bir bulgu olarak yaz

---

## T4 — Eşleştirmeyi açgözlüden optimale çevir + kararlılık testi

`greedy_match` en yakın çifti sırayla kilitliyor. Yoğun CP'de bu **yanlış eşleştirebilir**:
doğru çift daha sonra sıraya geldiğinde partneri çoktan tükenmiş olur.

- [ ] Hungarian (`scipy.optimize.linear_sum_assignment`) ile değiştir
- [ ] **Kararlılık testi:** toleransı yarıya indir; eşleşen ÇİFTLERİN kimliği değişiyor mu?
      Değişiyorsa o eşleşmeler zaten güvenilir değildi — oranını raporla
- [ ] Açgözlü vs optimal farkını ölç (küçükse açgözlü kalabilir, ama **ölçülmüş** olur)

---

## T5 — Fiziğe dayalı toleransı benimse ve HER sayıyı yeniden temellendir

- [ ] T1–T4'ten çıkan ölçütü tek yerde tanımla (`cp_config.json > matching`), her betik oradan okusun
- [ ] `big_arbiter`, `robot_e2e`, `robot_viz`, `f1_sweep`, `gate_regrow` hepsi aynı fonksiyonu çağırsın
      *(şu an her biri kendi kopyasını taşıyor — bu yüzden bir kez düzeltmek yetmiyor)*
- [ ] Ürün sayısını yeni ölçütle yeniden ölç, **eskisiyle birlikte** raporla
- [ ] `PRODUCT_MODEL.md` + `cp_config.current_product` + slaytlar güncellensin

---

## T6 — Bir daha sessizce dönmesin: regresyon bekçisi

Bu sınıfın tehlikesi sessiz olması. Ölçüt gevşerse hiçbir hata çıkmaz, sadece sayı yükselir.

- [ ] `tests/test_matching_tolerance.py`: sentetik parça, bilinen CP'ler, **kasıtlı 10 mm sapmış**
      bir tahmin → gevşek ölçütte geçmeli, sıkı ölçütte **kalmalı**
- [ ] Ürün metriği iki ölçütle birden hesaplansın; **aradaki fark makbuza yazılsın**
      (fark büyükse rapor bunu kendisi söyler, ben fark etmesem de)
- [ ] `CIZIM_SOZLESMESI` mantığı: ölçüt kodun içinde gömülü sabit olmasın, tek kaynaktan gelsin

---

## T7 — Dürüst raporlama *(süpervizör sunumu için, hemen)*

- [ ] Slayda tek satır: *"tolerance: 6% of part diagonal; at the robot's 2 mm requirement,
      precision on dense parts drops to 46%"*
- [ ] Sunumda önce **senin** söylemen, sorulunca yakalanmandan iyi

---

## Sıra ve gerekçe

1. **T1** (koşuyor) — gerçek sayı olmadan hiçbir karar verilemez
2. **T2** — 2 mm hedefi adil mi, önce bunu bil
3. **T3** — asıl kök neden burada; T4/T5 bunun üstüne oturur
4. **T4** → **T5** → **T6**
5. **T7** paralel, ölçüm beklemez

**Yapılmayacaklar:** toleransı sonuca göre seçmek (ölçütü sayı güzel çıksın diye ayarlamak, tam da
bu sorunun nasıl oluştuğu); eski sayıları sessizce silmek (ikisi yan yana raporlanır).
