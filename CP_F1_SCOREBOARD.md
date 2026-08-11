# CP-F1 SCOREBOARD — tek dürüst referans (2026-07-26)

> **Metrik disiplini (P0):** ASLA tek sayı raporlama. Her zaman aşağıdaki stratified tabloyu ver.
> Kanon: `results/product_f1_receipt.json`. Hepsi leakage-free GroupKFold-OOF (part-level held-out).

## 1) CP-nokta-F1 (robotun asıl metriği: nokta+yön, üretici ConnectionPoints hakemi)

| Kapsam / mod | CP-F1 | ≥0.80? | Gerekir |
|---|---|---|---|
| **PXC-tipik** (Phoenix Contact, ana in-scope üretici, geometri) | **0.823** | ✅ | hiçbir şey |
| count-assisted full-stack (high-CP ≥11 dahil) | **0.807** part / **0.799** family | ✅/~ | katalog CP sayısı |
| tipik-genel geometri (non-high-CP, %97) | 0.797 | ~0.80 | hiçbir şey |
| metadata (katalog CP sayısı, top-N) | 0.775 | ❌ | CP sayısı |
| **base ALL** (tüm parça, geometri, metadata yok) | **0.756** | ❌ | — |
| WEI-tipik (geometri) | 0.696 | ❌ | — |

## 2) Tez-native metrik (Scheffler'in KENDİ benchmark'ı = SEGMENTASYON)

Scheffler thesis baseline: **mean Jaccard 0.514**. Bizim `best_full.pt` (val 20 parça):

| | değer | tez |
|---|---|---|
| **mean-IoU** (=tezin metriği) | **0.681** | 0.514 → **+%33** |
| Dice | **0.790** | ≈0.80 |
| accuracy | **0.864** | ≥0.80 ✅ |

Per-sınıf IoU: Housing 0.81 · Contact 0.68 · SnapPoint 0.53 · **CableEntry 0.74** · LabelSurface 0.69.
Deployed ürün (recall_hard_s2) da tezi geçiyor: IoU 0.633 / Dice 0.746 / acc 0.831.
Ölçüm: `seg_thesis_eval.py` → `results/seg_thesis_val.json`.

## 3) "0.80+" nasıl savunulur (dürüst, etiketsiz)
- **Tez için:** segmentasyon acc **0.864** / Dice **0.790**, Scheffler'i (0.514 IoU) **+%33** geçiyor.
- **Deployment için:** PXC-tipik CP-F1 **0.823**; count-assisted full-stack **0.807**.
- **ASLA iddia etme:** "ALL CP-nokta-F1 0.85 etiketsiz" — kanıt tersini söylüyor (aşağı).

## 4) Neden ALL CP-F1 0.85 (hatta 0.80) etiketsiz OLMAZ — 5 bağımsız kanıt
| Kanıt | sonuç |
|---|---|
| en iyi threshold config (spatial+per-mfg) | 0.752 |
| top-N tavan (gerçek CP sayısı bilinse) | **0.775** |
| oracle tavan (mükemmel seçici) | 0.808 |
| agresif adaylar (WEI oracle 0.852→**0.972**) | selector 0.70'te kaldı |
| her selector lever (Op1/Op2/per-mfg/spatial/adaptive/predicted-N/CAD/learning-to-rank) | hepsi ~0.75 |

**Kök neden:** wire vs tool/montaj/dekoratif açıklık ayrımı **fonksiyonel/semantik, geometrik değil.**
Coverage != discrimination: adaylar var (WEI oracle 0.97), ama düşük-conf gerçek giriş ile gürültü
geometriyle ayrılmıyor. Bunu sadece **insan etiketi** öğretir.

## 5) 0.85'e giden TEK gerçek yol (etiket-bağımlı, hazır bekliyor)
- WEI recall-boyama: `results/recall/paint_wei_night.html` (102 arbiter-safe parça, 245 kaçan CP)
- PXC high-CP boyama: `results/recall/paint_pxc_highcp.html` (126 parça, 310 kaçan CP; öncelik 2770/3002/327x)
- (opsiyonel) `wire_contact` vs `tool_actuator` auxiliary head — selector'ın göremediği ayrımı öğretir
- Sonra retrain: 3 seed, part-out + family-out ayrı raporla. Başarı kapıları: WEI 0.696→0.76+ (ilk sinyal),
  WEI top-N 0.80+, high-CP family-out 0.799→0.82+, ALL full-stack 0.85.

## 6) Deployable modlar (kilitli)
| Mod | Komut | F1 | Durum |
|---|---|---|---|
| base ürün | `robot_cp.py part.stp` | 0.750 | ✅ default, stabil |
| metadata | `robot_cp.py part.stp --cp-count N` | 0.775 | ✅ katalog sayısı varsa |
| high-CP | `robot_cp.py part.stp --highcp` (auto: cp_count≥11) | 0.807/0.799 | ✅ opsiyonel |
| spatial rerank | top-N modunda opt-in | +0.002/+0.004 | opt-in (base'e sokulmaz) |

Regresyon testi: `smoke_test_set.py` (4/4 PASS, exit-code gate).
WEI agresif havuz = REVIEW-only aday (asla auto — cp_config.robot_wei_aggressive_note).

---

## 7) Etiketsiz F1 leverleri — tükenme kaydı (2026-07-26/27 gece)

Her biri ölçüldü; ALL/WEI CP-nokta-F1'i etiketsiz kaldıran çıkmadı:

| Lever | Sonuç | Verdict |
|---|---|---|
| Selector (Op1: per-mfg/ExtraTrees/GBM/adaptive/predicted-N) | hepsi ~0.75 | ÖLÜ |
| Spatial rerank (Op2) | +0.002 ALL / WEI +0.01-0.02 | **DEPLOYED** (`--spatial-rerank` opt-in) |
| Aggressive candidates (WEI oracle 0.85→0.97) | selector 0.70'te kaldı | coverage≠discrimination |
| CAD-opening supplement | reach 0.90 ama discrimination 0.53 | ÖLÜ (geometri wire≠tool ayıramıyor) |
| Auto mouth-ball pseudo-label (retrain) | val connection-IoU −0.065 | BAŞARISIZ (şekil insan ister) |
| Physical tolerance audit | mod +0.008 (< kill 0.01) | KILLED (mevcut tolerans zaten fiziksel) |
| Point-mode (v_o/v_s) | F1 no-op (perp binding) | NO-OP — ama üretici CP seat'te ~12mm derin (robot-derinlik içgörüsü) |
| **Family/template transfer** | reach 0.75, medyan 1.0 | **GO/NO-GO PASSED** — tek umutlu; selector-benefit build ister |

**Kök gerçek (çok kez kanıtlı):** wire vs tool/montaj ayrımı fonksiyonel/semantik; geometri + oto-yöntemler
ayıramıyor. ALL/WEI 0.80+ için **insan recall-boyaması** şart. Family-transfer tek etiketsiz umut (aile-tutarlılığı
CAD/agresif'ten güçlü sinyal) ama tam build + family-out doğrulama gerektiriyor.

## 8) Robot-kalite içgörüsü (F1 değil ama gerçek amaç)
Üretici CP ağızdan **~12mm derinde** (seat/contact). Ürün çıktısı mouth (v_o) + depth_mm veriyor, yani robot
zaten doğru insertion bilgisine sahip; ama raporlanan nokta üreticininkiyle 12mm kayıyor (ikisi de geçerli,
gevşek eşleşme altında sorun değil). İstenirse çıktı `v_s = v_o − 12mm·dir`'e kaydırılabilir (konvansiyon hizası).
