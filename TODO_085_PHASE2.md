# CP F1 → 0.85 — FAZ 2 TO-DO (kullanıcı listesi, 2026-07-28)
> Faz A bittikten SONRA otonom yürütülecek. Bu liste, benim ROAD_TO_085_FBI planımın üstüne gelen
> metodolojik sıkılaştırmadır ve 3 noktada onu DÜZELTİR: (1) aile tanımı STEP PRODUCT'tan (benimki
> bbox-proxy idi), (2) bir kez açılan KİLİTLİ holdout (bu gece aynı veride çok karar verildi =
> seçim-aşırı-uyumu riski), (3) öğrenilmiş EMBEDDING'ler (benim feature'larım el-yapımıydı).

## P0 — Ölçümü Kilitle
- [ ] `P` ve `direction` aynı koordinat frame'inde olacak şekilde candidate verisini doğrula.
- [ ] Ürün ailelerini STEP `PRODUCT` bilgisinden belirle.
- [ ] Geometry-hash tekrarlarını tespit et.
- [ ] Part-out, geometry-out ve family-out splitlerini sabitle.
- [ ] Son değerlendirme için dokunulmayacak aile holdout'u ayır.
- [ ] Baseline'ları yeniden doğrula: ALL `0.750`, metadata `0.775`, WEI `0.696`, PXC-tipik `0.823`.
- [ ] Candidate recall ile oracle-F1 kavramlarını raporda ayır.

## P1 — Gizli Model Sinyalini Ölç (embeddings)
- [ ] DiffusionNet'in son sınıflandırma katmanı öncesindeki vertex embedding'lerini çıkar.
- [ ] Her CP adayı için opening-region embedding ortalama/maksimum değerlerini oluştur.
- [ ] 3/6/12/24 mm çevre halkalarından semantik bağlam özellikleri çıkar.
- [ ] Insert-channel boyunca sınıf olasılığı profili çıkar.
- [ ] Global parça embedding'ini candidate özelliklerine ekle.
- [ ] Linear probe, RF ve küçük MLP'yi aynı OOF splitlerinde karşılaştır.
- [ ] GO: ALL en az `+0.015`, WEI en az `+0.020`, PXC kaybı en fazla `0.005`.
- [ ] GO başarısızsa embedding kolunu kapat.

## P2 — Part-Level Set/Graph Seçici
- [ ] Her parçayı candidate CP düğümlerinden oluşan graph/set olarak temsil et.
- [ ] Göreli konum, yön açısı, aynı yüz, aynı axis, pitch ve sıra ilişkilerini edge özelliği yap.
- [ ] Eksen/yüz özelliklerini yalnız küçük yardımcı sinyal olarak kullan.
- [ ] Binary sınıflandırma yerine part-içi listwise/top-N ranking loss dene.
- [ ] CP-count bilinen ve bilinmeyen modları ayrı eğit ve ölç.
- [ ] Part-out ve family-out sonuçlarını ayrı raporla.
- [ ] GO: P1 sonucunun üstüne family-out en az `+0.010`.
- [ ] Üç seed'de kararlı değilse set/graph kolunu kapat.

## P3 — Agresif Aday Havuzu
- [ ] Yeni seçiciyi önce mevcut WEI agresif havuzunda test et.
- [ ] WEI metadata tabanı `0.723` ile karşılaştır.
- [ ] End-to-end kazanım en az `+0.020` değilse ALL multires koşusunu iptal et.
- [ ] Başarılıysa 6k+9k+12k candidate havuzunu PXC ve ALL'a genişlet.
- [ ] Precision çöküşü, duplicate oranı ve inference maliyetini ölç.
- [ ] AUTO'ya yalnız yüksek güvenli adayları geçir; diğerlerini REVIEW'da tut.

## P4 — Family Transfer'ı Dürüst Yeniden Ölç
- [ ] Hedef parçanın GT CP noktalarını hizalamada kullanma.
- [ ] Template ve hedef STEP meshlerini yalnız CAD geometrisiyle hizala.
- [ ] Template CP'lerini sadece train-fold'dan aktar.
- [ ] Singleton aileleri aggregate hesapta sıfır katkıyla dahil et.
- [ ] Known-family ve unseen-family sonuçlarını ayrı raporla.
- [ ] Weighted ALL kazanımı `+0.010` altında kalırsa family-transfer'ı kapat.

## P5 — Metadata Modu
- [ ] CP-count bilgisinin inference sırasında gerçekten mevcut olduğunu doğrula.
- [ ] Giriş yönü bilgisinin benchmark GT'den bağımsız katalog kaynağını doğrula.
- [ ] Kaynak yoksa direction-prior kullanma.
- [ ] Metadata sonucunu base/geometri-only sonuçtan ayrı raporla.

## P6 — İnsan Adımı, En Son
- [ ] Kalan FP'lerin aile-açıklık tipi yoğunlaşmasını ölç.
- [ ] FP'lerin en az `%60`ı en fazla 40 tipte toplanmıyorsa adjudication yapma.
- [ ] 50–100 hedefli `wire / tool / unsure` sorusu hazırla.
- [ ] Eski "gerçek açıklık mı?" sorusunu kullanma.
- [ ] İnsan etiketli ailelerle final test ailelerini ayır.
- [ ] Önce candidate auxiliary head eğit.
- [ ] Yoğun ToolOpening segmentasyonu için gerçek bölge etiketi olmadan otomatik vertex boyama yapma.

## P7 — Final Doğrulama
- [ ] Tüm model ve eşikleri final holdout açılmadan kilitle.
- [ ] Final holdout'u yalnız bir kez çalıştır.
- [ ] ALL, WEI, PXC, high-CP ve family-out F1'larını ayrı raporla.
- [ ] Üç seed ortalaması ve bootstrap güven aralığı hesapla.
- [ ] AUTO precision ve AUTO+REVIEW F1'ı ayrı göster.
- [ ] Başarı kapısı: locked holdout ALL CP-F1 `≥0.85`.

---
### Faz A'dan devralınan durum (bu listeye girdi)
- Ürün: base 0.750 → **zengin gate 0.789 (parça-out) / 0.769 (aile-out, bbox-proxy aile)**; WEI aile-out 0.717.
- Baseline reprodüksiyonu: ALL 0.7535 ✓, metadata top-N 0.7757 ✓ (P0'ın doğrulama maddesi kısmen hazır).
- Öldürülenler: axis-kümeleme, skor-havuzlama, huni/taper (B-rep+mesh), bağlam feature'ları, "%60 FP öldürülebilir" (artefakt).
- P6 ön-koşulu ÖLÇÜLDÜ: FP kütlesinin %86'sı 40 ailede, aile-içi std 0.001 (ama bbox-proxy aile ile — P0'da PRODUCT-ailesiyle YENİLENECEK).
- P3 girdisi hazırlanıyor: `build_aggr_rich.py WEI` (agresif pool + zengin feature) koşuyor.
