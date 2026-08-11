# Robot stratejisi — nihai amaç (2026-07-23)

> 📊 **GÜNCEL F1 (2026-07-26, kanon `results/product_f1_receipt.json`):** base 0.750 · metadata 0.775 ·
> full-stack high-CP 0.807 part-out / 0.799 family-out. GB-dönemi 0.693 EMEKLİ. Aşağıdaki strateji/kısıt geçerli.

## AMAÇ
Master tezi + BİZİM datasetlerimiz + BİZİM kodumuzla modeli eğitip, ardından **bir WSCAD dosyasını
alıp CP'lerini (nokta + yön) bulan bir ROBOT/boru hattı** tasarlamak. F1 önemli çünkü robot her CP
üzerinde FİZİKSEL eylem yapıyor — teli o açıklığa sokuyor. Güvenilir nokta şart.

## 🚫 SERT KISIT — eğitim kaynağı (2-3 hafta önce buradan patladık)
Eğitim verisi YALNIZCA "certain sources"tan:
  - `wscad_corpus_scheffler_exact` (71 train + 20 val, tam 5-sınıf insan etiketi)
  - insan kısmi etiketleri (`_label_targets*`, connection-channel)
  - **manufacturer-CP ile konumlanmış + insan-boyalı recall etiketleri** (`_label_targets_recall`)
**RANDOM WSCAD parçalarıyla / pseudo-label / CAD-türevi etiketle EĞİTME.** Geçmiş felaketler:
CAD-pseudo domain gap (F1 1.0 train ama 0.046 STEP), random-wscad patlaması. Recall etiketleri bu
kısıta UYUYOR: konum üreticiden (ground truth), şekil insandan — uydurma değil.

## Robot için doğru metrik dengesi
- **FP (sahte CP)** robot için EN KÖTÜ: olmayan deliğe tel = fiziksel hata
- **FN (kaçan CP)** ikincil: bir bağlantı atlanır
- Bu yüzden post-proc **min_v 30** (gürültülü küçük fragmentleri atar) + **confidence alanı** açık
  (robot: yüksek-güven otonom, düşük-güven insana işaretle) -- iki katmanlı politika.

## Nerede olduğumuz (1811-CP bağımsız hakem)
- Ürün: seed s2 + min_v 30 / vc 0.5.  PXC ~0.62, WEI recall-turu-1 ile 0.40→0.46 (held-out).
- Precision gerçekte ~%98 (hakemleme kanıtı; hakem listesi eksik olduğu için düşük ölçülüyor).
- Gerçek boşluk RECALL; recall boyamayla kımıldadığı ÖLÇÜLDÜ (tek çalışan kaldıraç).

## YOL (robot-odaklı, hepsi certain-source)

### FAZ M — modeli olgunlaştır (recall boyama döngüsü) — 0.90 için ASIL motor
Her tur: kaçan CP'leri çıkar → **NADİR-ÖNCELİKLİ + ÇEŞİTLİ** parçalarda insan boyar → certain-source
etiket → eğit → 1811-CP hakemde ölç (audit_sets + sızıntı koruması + kalite kapısı zorunlu).
KANIT: M1 recall boyama WEI held-out **+0.093** (recall 0.26→0.41, genelliyor). Tek çalışan kaldıraç.
HEDEF: recall'ı 0.83'e taşımak (precision 0.98 sabit → F1 0.90). Gerçekçi ~0.75-0.80.
- [x] M1: WEI recall turu 1 -- WEI 0.347→0.440 (+0.093, audit'li) ✓
- [ ] M2: WEI+PXC birleşik model ölç (koşuyor) -- iki üreticide de kazanç mı
- [ ] M3-M6: recall turlarını DOYMA NOKTASINA kadar sürdür (tur < +0.02 olunca dur)
       * her tur NADİR-ÖNCELİKLİ (yüksek-CP çok-katlı/güç/bıçaklı; veri %57 basit klemens)
       * mümkünse her tur FARKLI ÜRETİCİ karışımı (genelleme = robot amacı)
- [ ] precision hakemliği: 3× DEĞİL, **1× EN SONDA** (precision zaten ~0.98; hakemlik model değil
      ÖLÇÜM doğrular -- eğitim kazancı metrik illüzyonu çıkmıştı). Savunulabilir tez rakamı için.

### FAZ B — ROBOT boru hattı (asıl ürün)
Zaten `infer_step_cp.py` çekirdeği var: STEP → remesh → segmentasyon → CP (nokta+yön). Robota çevir:
- [ ] B1: girdi = herhangi bir WSCAD STEP; çıktı = CP listesi {nokta, yön, confidence, sınıf}
- [ ] B2: iki katmanlı confidence politikası (yüksek→otonom, düşük→işaretle) + eşik kalibrasyonu
- [ ] B3: çıktı formatı (JSON/CSV robot için) + görsel doğrulama (GLB/render)
- [ ] B4: uçtan uca test: görülmemiş üretici STEP'i → CP → doğruluk (üretici CP'siyle)

### FAZ T — tez
- [ ] segmentasyon acc 0.858 (baseline 0.51) = ana sonuç
- [ ] CP F1 iki sayı (ham hakem + precision-düzeltilmiş) dürüstçe
- [ ] üreticiler-arası genelleme + recall-boyama mekanizması = katkı
- [ ] robot boru hattı = uygulama

## Her adımda ZORUNLU disiplin (bu gece pahalıya öğrenildi)
1. Eğitim SADECE certain-source (yukarı) -- random wscad ASLA
2. `audit_sets.py` -- karşılaştırılan iki ölçüm birebir aynı parça setinde
3. Sızıntı koruması -- eğitilen parça hakemde puanlanmaz
4. Etiket kalite kapısı -- boyama kaçan CP'nin <20mm'sinde (bu tur 60/60, medyan 3.4mm)
5. Karar SADECE bağımsız üretici hakeminde (insan held-out metrik illüzyonu üretir)
6. Eşikler (min_v vb) küçük sette değil 1811-CP'de + robot-güvenilirlik merceğiyle
7. Recall boyama ÇEŞİTLİ parçalarda (tek aile = ezberler, genelleme yok)

## ⚠️ RECALL PLATOSU YEDEK PLANI (recall ~0.75'te takılırsa)
Bazı açıklıklar doğası gereği zor (minik/gömük/sıradışı). Recall bir turda +0.02'nin altına düşüp
0.83'e ulaşamadan platoya girerse, SIRAYLA bu HENÜZ DENENMEMİŞ kaldıraçları dene:
1. **İnatçı-kaçan madenciliği:** turlar boyunca SÜREKLİ kaçan CP'leri özellikle boya (en zorlar).
2. **UNION ensemble:** softmax-ortalama recall'ı kesiyordu (öldürüldü); TERSİ hiç denenmedi -- seedlerin
   CP BİRLEŞİMİ (herhangi biri bulduysa al) recall'ı maksimize eder, sonra confidence filtre. Plato silahı.
3. **Çok-ölçekli inference:** 6k + yüksek çözünürlük CP'lerini birleştir (9k EĞİTİMİ reddedildi ama
   inference çok-ölçek farklı -- minik + büyük açıklıkları birlikte yakalar).
4. **k_eig 96:** küçük hakemde recall+ verdi ama büyük 1811-CP'de HİÇ ölçülmedi -- plato anında tam değerlendir.
Bunlar tavanı KIRMA denemeleri; normal akışta değil, SADECE plato gelirse.

## Ölü kaldıraçlar (tekrar deneme)
ensemble · hakemleme-eğitim-kazancı · üretici-etiketli-eğitim · çözünürlük 9k/12k · pos_weight ·
augmentation · çift-açıklık-bölme · rastgele-insan/random-wscad · ABB/A-B indirme (404)

## TEZ TEMELI (robot kodunun teorik/fonksiyonel dayanagi -- Masterarbeit_Scheffler.md)
- **Motivasyon (§1):** pano kablaji montaj suresinin ~%49'u, hala neredeyse tamamen ELLE (Moravec
  paradoksu). Robotun varlik sebebi. CP'ler bu otomasyonu besler + dijital ikize veri saglar.
- **Hedef (Abbildung 1):** CAD'den montaj-ilgili ozelliklerin POZISYON + BOYUT + NORMALVEKTOR'unu
  otomatik bul. robot_cp.py bunu birebir uyguluyor (point + size_mm + direction + depth_mm).
- **CP tureti (§5.3.6 / Abb.44):** segmentlenmis Kabeleinfuehrung'dan 3 schwerpunkt:
  Clusterschwerpunkt (vertex ort = point), Volumenschwerpunkt (konveks-oru merkezi = mesh-yansiz orta),
  Flaechenschwerpunkt (dis vertex iteratif eleme = acikligin DERINLIGI = depth_mm). + Normalvektor = yon.
- **4 Merkmal:** Kontaktierung, Aufrastpunkt, Kabeleinfuehrung, Beschriftungsflaeche = 5-sinif segmentasyon.
robot_cp.py = tezin Abbildung 1'inin implementasyonu; savunulabilir cikti.
