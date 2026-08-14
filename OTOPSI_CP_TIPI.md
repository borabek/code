# OTOPSİ: Robot neden tool/actuator ağzını CP sayıyor? (2026-07-24)

> ✅ **ÇÖZÜLDÜ (2026-07-26):** yapısal wire/tool gate (RF, held-out AUC 0.867) tool ağızlarını eliyor.
> Ürün precision düzeltildi (eski "~0.98" ŞİŞİRİLMİŞTİ — tool'ları doğru sayıyordu). Sonuç:
> base F1 0.693→**0.750**. Kanon: `results/product_f1_receipt.json`. Aşağıdaki kök-neden analizi geçerli.

## Belirti (kullanıcı gözüyle yakalandı)
Robot bir push-in/kelepçe klemenste "her açıklığı / her boşluğu" CP işaretliyor. 6 CP dediği yerde
belki 2-4 gerçek **tel girişi** var; gerisi **tornavida/mandal (Werkzeugeinschub) ağzı** veya montaj
boşluğu — telin GİRMEDİĞİ delikler. Sayı şişiyor, precision düşüyor.

## Kök neden zinciri (kanıtla)

### L1 — TANIM düzeyi (tez, satır 1303) ★ asıl kaynak
Tez, kendi etiket şemasında dört özelliği sayarken birebir şöyle diyor:
> "vier geometrischen Featuregruppen **Kontaktierung bzw. Werkzeugeinschub**, Aufrastpunkt,
>  Kabeleinführung und Beschriftungsfläche"

Yani **Contact sınıfı = "Kontaktierung *ya da* Werkzeugeinschub"** = kontakt noktası VEYA alet-sokma
ağzı — **bilerek aynı sınıf.** §1 (satır 236) de montaj-ilgili özellikler arasında "Kontaktier-
positionen ... sowie ... **Werkzeuge**" diyor: tez tool ağzını bilerek bir özellik sayıyor, ama
**Contact sınıfı olarak, Kabeleinführung değil.** Tool ağzı ile kontakt, tez tasarımı gereği ayrılmamış.

### L2 — ETİKET gerçeği (push-in'de tel-girişi de Contact)
cp-v2 (yalnız CableEntry) manufacturer PXC setinde **F1 = 0.000** almıştı — HİÇBİRİNİ bulamadı, çünkü
**PXC push-in/kelepçe tel-girişleri Contact etiketli** (ayrı bir CableEntry yok). Yani sınıf düzeyinde
**tel-girişi ile tool ağzı AYNI sınıf (Contact)** → segmentasyon ikisini ayıramaz.

### L3 — VERİ sinyali (corpus geometrisi, _autopsy_pairing.py)
91 tam-etiketli corpus parçası: Contact bileşeni **258** vs CableEntry bileşeni **173** → ratio **1.49x**
(Contact'lar tel-girişlerinden ~%50 fazla). Contact→en yakın CableEntry medyanı **17mm** (= ağız→kontakt
axis mesafesi; aynı tel-yolu). Fazladan Contact'ların bir kısmı tool ağzı. (Uyarı: corpus etiketlemesi
tutarsız — bazı parçalarda CableEntry hiç işaretlenmemiş; bu ölçüm yön verir, tek başına kanıt değil.)

### L4 — TÜRETME (cp_openings)
cp-v3 = CableEntry VEYA depth-kapılı Contact. Her depth-kapılı Contact bölgesinden bir CP üretir
→ **tool ağzını da CP üretir.** ct_depth_min_mm=1.0 kapısı düz pad'i eler ama tool ağzının da derinliği
var → geçer.

## Neden basit bir bug değil
Sınıflar **etiket düzeyinde birleşik** (tez tasarımı). Bu yüzden:
- Daha iyi segmentasyon ayırmaz (aynı sınıf).
- "Yalnız CableEntry" kullan → PXC tel-girişlerini kaçırır (F1 0.000).
- "Contact'ı at" → gerçek push-in tel-girişlerini kaçırır.
Ayrım **fonksiyonel** (biri tel alır, biri alet) — yerel geometriden gelmeyebilir. O yüzden karar deneyi
şart: geometri (boyut/depth) bu ikisini ayırıyor mu?

## KARAR DENEYİ (recon_wire_vs_tool.py — KOŞUYOR)
Üretici CP'sini ayraç-etiketi al: robot-CP'si üretici CP'ye eşleşiyorsa TEL, değilse TOOL/fazla.
İki grubun size_mm/depth_mm/votes/confidence AUC-ayrılabilirliğini ölç.
- AUC ~0.5 → ayırt edilemez → **alt-sınıf etiketi + yeniden eğitim** gerekir (tool'u Contact'tan ayır)
- AUC >0.75 → **geometrik kapı** genel fix (cp_openings'e, yeniden eğitim YOK, tüm modeller kazanır)

## KARAR DENEYİ SONUCU (2026-07-24, recon_wire_vs_tool.py, 770 robot-CP: 465 tel / 305 tool-fazla)
| özellik | tel medyan | tool medyan | AUC |
|---|---|---|---|
| size_mm | 12.84 | 11.26 | **0.61** (zayıf) |
| depth_mm | 3.71 | 3.48 | 0.52 (yok) |
| votes | 3.00 | 2.00 | 0.50 (yok) |
| confidence | 0.84 | 0.82 | 0.57 (zayıf) |
**GEOMETRİ AYIRMIYOR → DAL A ÖLÜ.** Ayrım fonksiyonel, yerel geometriden gelmiyor.
**BONUS BULGU (önemli):** bu, önceki "precision ~0.98 / FP'lerin %96'sı gerçek açıklık" hakemliğinin
YANLIŞ soruyu sorduğunu gösteriyor — "gerçek açıklık mı?" (tool=evet) yerine "tel giriyor mu?" (tool=hayır)
sorulmalıydı. Tool ağızları "doğru" sayılmış → robot-etkin ~0.80 sayısı ŞİŞİK. Gerçek wire-precision daha düşük.
NOT: 305 "tool" grubu saf tool değil — üretici listelemediği gerçek telleri de içerir (eksik-GT). Ama
sonuç değişmez: bu iki grup GEOMETRİK olarak ayrılamıyor.

## ⚔️ ASKERİ PLAN (karar deneyi ÖLÜ Dal A -> Dal B'ye geçildi)
Amaç: 2-3 modeli değil, **kodun (cp_openings + tanım) genelini** bu sorunu aşacak hale getirmek.

1. **[KOŞUYOR] RECON** — geometri tel/tool'u ayırıyor mu (AUC).
2. **DAL A (AUC yüksek):** cp_openings'e **wire-vs-tool geometrik kapısı** ekle (boyut/depth/eşlik
   eşiği). Yeniden eğitim yok → her modele uygulanır. Hakemde precision↑ / tel-recall sabit doğrula.
3. **DAL B (AUC düşük):** sınıfı **un-merge et** — bir parça setinde tool ağzını ayrı işaretle (insan),
   6. sınıf/alt-baş olarak öğret; VEYA **eşleme sezgiseli**: kutup başına tel+tool çifti tespit et,
   telini tut (üretici-CP yönü/derinliğiyle hangisi tel öğrenilir). Thesis-faithful (tezin birleştirdiğini ayırmak).
4. **DOĞRULAMA:** manufacturer hakemde robot-CP sayısı üretici-CP sayısına yaklaşmalı; precision
   yükselmeli, gerçek tel-recall düşmemeli. Görsel: tool ağızları artık işaretlenmemeli.
5. **GENELLİK:** fix cp_openings.py'de (türetme) → cp_config tanımına bağlanır → TÜM ürün modelleri
   otomatik kazanır. "Sadece 2-3 model" değil.

## Değişecek/değişmeyecek
- DEĞİŞİR: cp_openings türetmesi (+ belki 6. sınıf etiketi/eğitim), cp_config CP tanımı, robot çıktısı
  (CP'ye "kind: wire/tool" tipi eklenebilir — tez tool'u da istiyor, ama robot yalnız wire'a tel sokar).
- DEĞİŞMEZ: segmentasyon backbone (gerekmezse), vote>=2 ürün mantığı, hakem disiplini.

## F1 OPTIMIZASYON PLANI DURUMU (2026-07-24, kullanici review sonrasi)
YAPILDI (self-executable): P0.1 config freeze (current_product) · P0.2 receipt (product_f1_2026_07_24.json, hash'li) ·
P0.3 OOF-only raporlama kurali · P0.4 PRODUCT_MODEL.md tek current-product basligi · P0.5 failure taxonomy ·
P1.3 per-mfg gate (WEI/PXC ikisi de 0.35 optimum, ayirmak kazanc YOK) · P1.4 CP-count prior (metadata-assisted F1 0.742).
NIHAI URUN: vote1_union + wire-gate 0.35, OOF F1 0.693 (ALL) / WEI 0.641 / PXC 0.703.

BEKLEYEN (insan/pahali, self-executable DEGIL):
- P1.1 recall labeling: en cok FN veren WEI/PXC ailelerinden 40-80 part (cok-CP/push-in). Recall tavani icin.
- P1.2 wire/tool alt-sinif (6. sinif veya aux head): Contact'i wire_contact vs tool_actuator ayir. Precision sicramasi.
  (wire-gate simdilik post-hoc bunu yapiyor; egitimde ayirmak daha guclu ama insan etiketi ister.)
- P2.1 Manifold remesh (current product + OOF ile); P2.2 YOLOv6 2D arm (ciddi proje, klasik CV hack YOK).
- Final holdout korunacak: model/gate/threshold seciminde KULLANILMAZ.
