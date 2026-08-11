# ROAD TO 0.85 — FBI OTOPSİSİ + SALDIRI PLANI (2026-07-27)
> ML CP-F1 hedef: **ALL 0.85** (leakage-free OOF, manufacturer arbiter). Başlangıç: base 0.750 /
> metadata 0.775 (deployed) / ölçülü en iyi 0.788-0.807. Bu belge: neden takıldığının kanıtlı
> otopsisi + ölçüm-kapılı (go/no-go) saldırı planı. Over-promise yok: her adımın mekanizması,
> maliyeti ve öldürme-kriteri yazılı.

---

## 1. OTOPSİ — kanıtlı ölüm zinciri

**Elenen şüpheliler (hepsi bu hafta leakage-free ölçüldü):**
- Aday bulamama → DEĞİL (multires aday-tavan recall 0.969, oracle-F1 0.985)
- Eşik / per-mfg eşik → DEĞİL (nested-CV kazanç ±0.01; 0.35 optimal)
- Etiket-boyama azlığı → DEĞİL (recall-painting saturasyon: +0.006)
- Mesh yoğunluğu → DEĞİL (avg 5712 ≈ 6000)
- B-rep huni/koni feature → DEĞİL (has_cone AUC 0.504; wire/non-wire koni-oranı 0.13/0.13)

**Katil: gate'in LOKAL-GEOMETRİ ontolojisi.**
13 gate feature'ının HEPSİ deliğin kendisine bakar (size/depth/aspect/ce_frac/...). Oysa wire-vs-tool
ayrımı lokal şekil DEĞİL, **ROL**dür: vida deliği ile tel deliği aynı silindirdir; ayıran şey
**eksen/yüz/düzen/tekrar**. Kanıt:
- Surviving-FP (616) feature-uzayında TP ile ÇAKIŞIK (size/depth/nverts/conf/aspect/flat hepsi örtüşür);
  239'u votes≥3 = "gerçek açıklık", sadece non-wire.
- Kullanıcının haftalar önce yakaladığı hata birebir bu: vida deliğinden ok çıkıyor. Vida ÜSTTEN,
  tel ÖNDEN girer — **eksen farkı 13 feature'ın hiçbirinde yok.**
- Tezin kendisi ayırmıyor (Contact = "Kontaktierung bzw. Werkzeugeinschub") — biz tezden zor bir
  hedefe (manufacturer wire-CP) karşı ölçülüyoruz; tezin sınıf şemasında bu sinyal tanımsız.

**Felsefi çekirdek:** lokal bilgi tükendi ≠ bilgi tükendi. Kalan bilgi İLİŞKİSEL:
(a) eksen/yüz aileleri, (b) katman/latis tekrarı, (c) ürün-ailesi tekrarı, (d) artikel metadata,
(e) agresif aday rezervi. Beşi de bugüne dek MODELLENMEDİ.

---

## 2. SALDIRI PLANI — P0→P5, her biri ölçüm-kapılı

> **FAZLAMA (kullanıcı kararı 2026-07-27): insan-emeği EN SONA.**
> **FAZ A (makine-only, insan girdisi SIFIR):** P0-a/b/c/d · P1-a · P1-b · P2 · P3 · P4-ara-ölçüm.
> **KARAR KAPISI:** Faz A sonunda OOF ALL ölçülür. 0.85 geldiyse → P1-c ve P5 HİÇ YAPILMAZ (tasarruf).
> Gelmediyse → P1-c artık **hedefli ve ucuz**: P0-a + P1-a çıktısı hangi ~40 aile-açıklık-tipinin FP
> kütlesini taşıdığını söylemiş olur (kör 100 soru yerine 40 isabetli soru, aynı gün).
> **FAZ B (yalnız gerekirse):** P1-c → P5.
> Gerekçe: (1) tek insan-maliyetli kalem odur, bedava kaldıraçlar önce tüketilir; (2) P1-c'nin NE
> soracağını Faz A ölçümleri belirler — önce yapılırsa kör adjudikasyon olur; (3) Faz A yeterse
> gereksizleşir. Dürüst kayıt: Faz A yetmezse kalan kaldıraç odur — iptal değil, ertelenmiş.

### P0 — EKSEN-KÜME havuzlanmış gate  [CPU, cached veri, ~1 gün — EN YÜKSEK EV]
Mekanizma: adayları (eksen-yönü × yüz/yükseklik) ile kümele; kümeyi TEK nesne olarak sınıfla
(havuzlanmış feature = √N kanıt-artışı); count-prior küme-seviyesinde uygula ("N tel → hangi kümeler").
Vida/pusher FP'leri ayrı eksen-kümesinde ölür.
- P0-a: `wei_aggr_pool.json` (P, **dir**, X, y hazır!) → FP'lerin ekseni GT InsertDirection'dan
  sapıyor mu? SAYIYI ölç. (2 saat)
- P0-b: kümele → **küme saflığı** ölç. GO-kriteri: kümelerin ≥%85'i saf-wire ya da saf-tool.
- P0-c: küme-havuzlu sınıflama + count → WEI top-N 0.723 baz vs yeni. GO: **+0.03**.
- PXC/ALL için pool'a P/dir alanı ekle (f1_sweep extract'e 5 satır; warm-cache ~30 dk GPU).

### P1 — AİLE-KONSENSÜS + aile-adjudikasyon ekonomisi  [CPU; opsiyonel 1 gün insan]
Mekanizma: aynı gövde-platformu kardeşleri AYNI açıklık tiplerini taşır; bir açıklığın kimliği
ailede SABİTTİR.
- P1-a: 616 surviving-FP'nin aile-yoğunlaşmasını ölç (error_mine aile haritası mevcut).
  GO: FP kütlesinin ≥%60'ı ≤40 aile-açıklık-tipinde.
- P1-b: kardeş-parçalar arası skor-havuzlama (ML-only, insansız) → varyans düşür.
- P1-c: aile başına 1 karar: "bu açıklık-tipi wire mi?" → ~50-100 evet/hayır (1 gün, boyama DEĞİL —
  saturasyon boyama içindi, bu FARKLI ve ucuz kanal) → gate'e kural olarak enjekte.

### P2 — AGRESİF ADAYLARI ALL'A GENİŞLET  [GPU, 1 gece]
WEI'de hazır olan multires havuzu (recall 0.969) PXC/ALL'a kur (build_wei_aggr genelleştir).
Recall tavanı 0.808 → ~0.95. Precision yükünü P0/P1'in yapısal gate'i taşır. GO: yapısal gate'le
birlikte ALL F1 net artı (union-çöküşü yaşanmazsa).

### P3 — YÖN-PRIOR metadata modu  [CPU, ürün-meşru]
CP-sayısı zaten datasheet-meşru (0.775 deployed). Aynı meşruiyetle **giriş yönü** ("front-entry"):
off-axis adayları kes. InsertDirection çoğunluğu = datasheet-yönü proxy'siyle ölç. GO: +0.01-0.02.

### P4 — MERDİVENİ BİRLEŞTİR + RECEIPT
P0+P1+P2+P3 → tek OOF receipt (per-mfg + ALL; per-mfg modeller olsa da TEK aggregate F1).
0.85 kontrolü burada yapılır; hangi bileşen ne kattı tabloyla.

### P5 — (opsiyonel, kökten) TOOL-SINIFI SEGMENTASYONA İN
P1-c adjudikasyonlarını otomatik vertex-etikete çevir (açıklık geometrisi belli → boyama bedava) →
6-sınıf seg (ToolOpening ayrı) retrain. Ayrımı kaynağında çözer; en pahalı, en kalıcı.

---

---

## 2.5 GECE 1 SONUÇLARI (2026-07-27 gece, otonom Faz A) — ÖLÇÜLDÜ

**P0 (ilişkisel eksen/kümeleme kolu) → TAMAMEN KAPANDI, dürüst.**
- P0-a: eksen-kümesi saflığı null'dan farksız (z=+0.09). "Sıfır-TP kümesinde %60 FP" → null-düzeltince
  **+0.031 = şans** (taban-oranı artefaktıydı; iddia edilmeden yakalandı).
- P0-a2/a3: hipotez **DOĞRU** — wire-CP'ler boyut-eşleşmiş rastgele adaylardan fazla ortak eksen
  paylaşıyor (+0.058, parçaların %78'i). **AMA 13 feature bu bilgiyi zaten taşıyor: +0.0036 AUC.**
  → "gate'te eksen bilgisi yok" teşhisim ETKİDE yanlıştı.
- P0-c: LOO skor-havuzlama (√N argümanı) → +0.011 AUC ama **top-N F1 +0.002** (eşik altı). Ölü.

**P1-x (TEMSİL) → GERÇEK LEVER BULUNDU ve AİLE-OUT'ta AYAKTA.**
13 feature'ın kör noktası: adayın **parçada NEREDE** olduğu yok (`outward` sadece kendi ekseni boyunca).
Vida üstte / tel önde ayrımı = kullanıcının "vidadan ok çıkıyor" bulgusu. WEI agresif havuzda:

| Split | 13 feature | + konum | kazanç |
|---|---|---|---|
| Parça-out | top-N F1 0.6562 | **0.6971** | **+0.041** |
| **Aile-out (katı)** | 0.5913 | **0.6226** | **+0.031** |

Aile-out'ta kazancın %76'sı duruyor → **aile-ezberi DEĞİL, genellenen sinyal.** (Ayrıca parça-içi rank
ve medyan-oran blokları katkısız çıktı: kazanç spesifik olarak KONUM'dan geliyor.)

**P1-a (aile yoğunlaşması, Faz B girdisi) → GO kriterini GEÇTİ.**
Surviving-FP kütlesinin **%86'sı en büyük 40 ailede**; çok-parçalı ailelerde FP-oranı std **0.001**
(aile-içi davranış neredeyse birebir sabit) → aile başına TEK karar tüm aileyi temizler. Bu, P1-c'nin
hedef listesidir (insan işi Faz B'ye ertelendi).

**P2 ASKIYA ALINDI (premise ölçülmüş-ölü):** agresif aday havuzu WEI'de top-N F1 **0.662 < baz 0.723**;
receipt de "selector_on_aggressive = baseline ile aynı" diyor. Ayrım gücü artmadan aday çoğaltmak zarar.
Gece GPU'su bu yüzden P2'ye DEĞİL, temsil-zenginleştirmeye verildi.

**Gece GPU işi:** `build_rich_feats.py` — tüm korpus (823 parça, ürün-sadık 4-model union yolu) için
41 zengin feature: konum(9) + çok-yarıçap sınıf-olasılık profili(24) + **taper/huni profili(5, mesh
tabanlı — B-rep koni testi ölmüştü, bu farklı)** + eğrilik(3) → `results/rich_feats.npz`.
Analiz: `analyze_rich.py` (blok-blok katkı, parça-out + AİLE-out, WEI/PXC ayrı, eşik-modu + top-N modu).

---

## 3. DÜRÜST BEKLENTİ MATEMATİĞİ (garanti değil, mekanizma-bazlı aralık)
Başlangıç (metadata modu, deployed): **0.775**
- P0 eksen-küme: +0.02…0.05 (vida/pusher FP sınıfını hedefliyor — ölçülü en büyük FP kütlesi)
- P2 agresif aday: +0.02…0.04 (recall 0.73→0.80+; oracle 0.894→0.985 kapısını açar)
- P1 aile-konsensüs (+adjudikasyonla): +0.01…0.04
- P3 yön-prior: +0.01…0.02 (P0 ile kısmen örtüşür)
**Bant: 0.82 – 0.87; orta yol ~0.84-0.85.** PXC muhtemelen 0.85'i İLK geçer (şimdiden 0.823/0.837).
Her adım go/no-go: mekanizma ölçümde doğrulanmazsa o kol durur, bant daralır — dürüstçe raporlanır.

## 4. SÜPERVİZÖR İÇİN ŞİMDİDEN ELDE OLANLAR (fallback değil, mevcut sonuç)
- Tez-native segmentasyon: **acc 0.864 / Dice 0.790 / IoU 0.681 vs tez 0.514 (+%33)** ✅
- PXC-tipik CP-F1 **0.823** ✅ · count-assisted high-CP **0.807** ✅ · ALL 0.775 (deployed)
- Bu haftanın katkısı: 0.85'in ÖNÜNDEKİ engelin kanıtlı teşhisi + ilişkisel-bilgi saldırı planı.

## 5. DİSİPLİN (değişmez)
Leakage-free OOF tek geçer akçe · GT pozisyonu inference'ta ASLA (count/yön datasheet-prior'ı hariç) ·
her iddia önce ölçüm · in-sample sayı rapor edilmez.
