# GEOMETRIK CP DEDEKTORU — ML YOK, EGITIM YOK (2026-07-27)

> AYRI görev: CP'leri SADECE geometriden algoritmik bul. Eğitilmiş model/AI YOK. Klasik geometrik
> feature detection (oyuk/çukur/delik tespiti). ML pipeline'dan TAMAMEN bağımsız (CPU, GPU'ya dokunmaz,
> paralel çalışır).

## Neden ayrı ve neden değerli
- **Zero-shot her üretici:** eğitim olmadığı için ABB/WEI/PXC/görülmemiş her parçada çalışır. ML'in domain-gap zayıflığını kapatır.
- **Yorumlanabilir:** kara kutu değil, her CP'nin neden bulunduğu geometrik olarak açık.
- **Hizalama sorunu YOK:** JSON mesh (Graphic3d) üstünde çalışır; CP'ler de aynı JSON frame'inde → align_frames'e gerek yok (ML'deki ~17% hizalama artefaktı burada yok).

## CP'nin geometrik tanımı
Cable-entry/contact = dış yüzeydeki bir OYUK/GİRİNTİ (delik). Geometrik imzası:
- **Konkav** (yüzey içeri döner) — çevresinde yüksek eğrilikli ağız-kenarı
- **Ağız** (dış yüzeyde) + **derinlik** (içeri kanal, ~contact seat'e)
- Eksen = ağza dik (içeri doğru insertion yönü)

## Yöntem (ray-casting'siz, hızlı)
1. **Mesh yükle:** JSON Graphic3d.Points + Indices → trimesh (üreticinin kendi geometrisi)
2. **Konkavlik sinyali** (birden fazla dene, en iyiyi seç):
   - a) discrete mean curvature (trimesh) — konkav bölgeler
   - b) yerel-oyukluk proxy: KDTree ile komşuların normal-üstü/altı dağılımı (oyuk = çevrili)
   - c) convex-hull mesafesi (proximity) — derin girintiler
3. **Kümele:** konkav vertexleri DBSCAN (pozisyon) → aday açıklıklar
4. **CP çıkar:** her küme → ağız-merkezi + eksen (ortalama içeri-normal) + derinlik
5. **Geometrik filtre:** derinlik [2,20]mm, ağız-yarıçap [1,8]mm, küme-boyu eşiği (kablo açıklığı priorleri)
6. **Çıktı:** CP = ağız-merkezi + insertion ekseni

## Eval (ML ile aynı hakem, karşılaştırılabilir)
- JSON CP'lerine (aynı frame) eşleştir → F1/P/R
- **Üretici bazında:** WEI, PXC, ve **ABB (görülmemiş — asıl test)**
- ML modeliyle karşılaştır: geometri nerede kazanır (yeni üretici), nerede kaybeder (precision)

## Fazlar
- **G0:** iskelet + 1 konkavlik sinyali + kümele + eşleştir → ilk F1 (birkaç parça)
- **G1:** CP çıkarma (ağız+eksen+derinlik) + geometrik filtre tuning
- **G2:** çoklu sinyal birleştir, eşikleri fiziksel prior'larla ayarla (öğrenme YOK)
- **G3:** tam eval — WEI/PXC/ABB; ML ile karşılaştır
- **G4 (opsiyonel):** ML ile ensemble (geometrik aday + ML skor) VEYA görülmemiş-üretici fallback

## Prensip
- Öğrenme YOK — tüm eşikler geometrik/fiziksel prior (kablo çapı, açıklık derinliği)
- Eşikler tek bir üreticiye fit edilmez; fiziksel aralıklar (her üretici için geçerli)
- ML'den bağımsız ölçülür; sonra "birleştirmek F1 artırıyor mu" ayrı sorulur

---

# YARIN İÇİN DETAYLI RUNBOOK (geometric model direkt CP bulsun)

## Ne öğrendik (ilk deneme, 2026-07-27)
- Naif konkavlık (yerel-oyukluk proxy): PXC F1 0.076, WEI 0.000 -> ÇOK FP, yetersiz
- KÖK: (a) naif sinyal kaba, (b) WEI açıklıkları mesh'te geometrik oyuk DEĞİL (düz) -> geometri WEI'de zorlanır
- PXC'de %60 parça CP-yakın konkav -> **geometri PXC-stilde ÇALIŞABİLİR, iyi yöntemle**
- Mesh kaynağı = JSON Graphic3d (Points+Indices), CP ile AYNI frame -> HİZALAMA YOK ✅

## Yöntem: çok-ipuçlu opening dedektörü (naif konkavlık DEĞİL)

### G1 — Daha iyi opening sinyali (3 ipucu birleştir)
1. **Ambient occlusion (ışın-tabanlı):** oyuk içindeki vertex = yüksek occlusion. trimesh ray (embree yok -> numpy fallback yavaş; ALTERNATIF: hemisphere-sample KDTree proxy). Kod: `ao_signal(mesh, n_rays=32)`
2. **Konkav eğrilik:** `trimesh.curvature.discrete_mean_curvature_measure` -> negatif = oyuk kenarı/duvarı
3. **Convex-hull mesafesi:** `trimesh.proximity` vertex->hull -> derin girinti = büyük mesafe
-> Üçünü normalize+birleştir (ağırlıklı veya voting) -> aday opening vertexleri (naif tek-sinyalden çok daha temiz)

### G2 — Opening segmentasyonu
- Aday vertexleri DBSCAN (pozisyon, eps~4mm) -> aday açıklıklar
- Küme boyu eşiği (gürültü kümelerini at)

### G3 — Silindir/prizma fitting (ASIL kalite adımı)
- Her küme için: **RANSAC silindir fit** (yön+yarıçap) VEYA oriented-bbox (prizma açıklıklar için)
- Fit residual düşükse = gerçek silindirik/prizmatik açıklık (kablo deliği); yüksekse = gürültü, at
- CP = fit'in AĞIZ merkezi (dış yüzeydeki uç) + eksen (insertion yönü)
- Bu adım FP'yi çok azaltır (sadece silindir/prizma-uyan açıklıklar kalır)

### G4 — Geometrik filtre (fiziksel prior, ÖĞRENME YOK)
- yarıçap/boyut ∈ [1,8]mm (kablo açıklığı)
- derinlik ∈ [2,20]mm
- eksen ≈ dış yüzeye dik
- (opsiyonel outward-gate: ağız iç-gövdede değil)

### G5 — Eval + tune (mevcut hakem, karşılaştırılabilir)
- JSON CP'lerine eşleştir (aynı frame, axis-aware perp), per-üretici: WEI / PXC / **ABB (zero-shot asıl test)**
- Fiziksel eşikleri tune et (öğrenme değil, aralık ayarı)
- ML (recall_hard) ile karşılaştır: geometri nerede kazanır (yeni üretici/precision)

### G6 — Dürüst kapsam + ürünleştirme
- Nerede çalışır (PXC-stil, açıklık-oyuk) vs fails (WEI düz) -> açık raporla
- **Ürün seçenekleri:** (a) yeni-üretici zero-shot fallback, (b) ML ile ENSEMBLE (geo aday + ML skor), (c) ML'in 60-zero-recall parçasını geo kurtarıyor mu

## YARIN ÇALIŞTIRMA SIRASI (kopyala-çalıştır)
1. geo_cp.py'ye G1 (ao+curvature+hull birleşik sinyal) ekle -> `--mode validate` ile CP-lokalizasyon kontrol (>+0.1 hedef)
2. G2+G3 (cluster+silindir fit) ekle -> `--mode eval --mfg PXC --n 40` -> F1 (hedef >0.4, naif 0.076'dan)
3. G4 filtre tune -> PXC F1 maksimize
4. `--mfg WEI` + `--mfg ABB` -> zero-shot kapsam (ABB asıl vaat: 386 parça, ML görmedi)
5. ML ile karşılaştır + ensemble/fallback kararı
- Hepsi CPU-only (GPU yok), ML pipeline'dan bağımsız, paralel çalışır

## Beklenti (dürüst)
- **PXC-stil:** geometri iyi yöntemle F1 0.4-0.6 ulaşabilir (açıklık oyuksa)
- **WEI:** düşük kalır (açıklık mesh'te yok) — bu ML'in de zorlandığı yer
- **ABB:** ZERO-SHOT vaat — ML hiç görmedi, geometri görsel açıklıkları bulabilir
- Değer: yeni-üretici genelleme + ML-tamamlayıcı (60-zero-recall kurtarma denemesi)

---

# GEO R&D DURUM (2026-07-27 aksam) — mevcut yontem BASARISIZ, yeni yon

## Denenen ve BASARISIZ
- G0 naif konkavlik-kumeleme: PXC 0.076
- G3 tup-fit (konkavlik + hull + derinlik/genislik/aspect filtre): gevsek 0.089, siki 0.000
- KOK-NEDEN: (1) konkavlik sinyali cable-acikligini DIGER konkavliktan (snap/kenar/label) ayirmiyor
              (2) JSON mesh COK KABA (~2078 nokta) -> kucuk kablo acikliklari cozulmiyor

## YENI R&D YONU (cok-oturumlu, tonight-fix DEGIL)
1. **FINE MESH kullan:** JSON'un 2078'i yerine STEP-remesh (6000-8325 nokta) -> acikliklar cozunur
   - Bedeli: align_frames ile CP'leri mesh frame'e tasi (ML pipeline'daki gibi)
2. **PROPER HOLE DETECTION** (konkavlik degil):
   - Isin-derinligi: her yuzey noktasindan iceri isin, tunel-derinligi (delik=derin, kati=sig)
   - VEYA boundary-loop: gercek delikler mesh kenar-donguleri
   - VEYA RANSAC silindir fit (proper, mevcut kaba fit degil)
3. **2D-render kolu** (alternatif): 6-view render + acikliklar gorsel belirgin -> CV/tespit
4. Eval: her zaman WEI/PXC/**ABB zero-shot** (bagimsiz, ML'siz)

## Durum: BAGIMSIZ R&D item, terk edilmedi. Ana model oncelikli; geo fine-mesh sonraki oturum.
