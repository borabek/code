# CP-F1 → 0.85 PLANI (2026-07-27)

> Başlangıç: base ALL 0.756. Hedef: **0.85**. Boşluk +0.094.
> Artık 0.85 ULAŞILABİLİR (etiketsiz-imkânsızın aksine) çünkü 3 gerçek lever var: etiket (kanıtlı),
> hizalama (gerçek ölçüm-hatası), aux-head (kök-neden). Ama **stacking** şart — tek lever yetmez.

## Boşluğun anatomisi: TAVAN mı DUVAR mı?
| Engel | Tür | Kıran lever |
|---|---|---|
| wire vs tool ayrımı geometrik değil | **TAVAN** (semantik) | etiket + aux-head |
| align_frames ~17% parçada bozuk → sahte-FN | **DUVAR** (ölçüm) | hizalama recovery |
| remesh ~7% açıklığı siliyor | küçük duvar | hole-preserving remesh |
| high-CP aileler az örnekli | veri | etiket (PXC high-CP) |

---

## FAZ 0 — ŞU AN ÇALIŞIYOR: gerçek konumu ölç
- WEI (95) + PXC high-CP (78) etiket → retrain → eval
- Hizalama-stratified F1 (big_arbiter'a gömüldü)
- **Çıktı:** post-etiket base + hizalama-düzeltilmiş F1 → gerçek başlangıç (~0.80-0.83 tahmini)
- **Bu bittiğinde 0.85'e kalan boşluk netleşir; sonraki fazlar ona göre.**

## FAZ 1 — HİZALAMA RECOVERY (duvar #1, en yüksek ROI)  → hedef +0.03-0.05
Sadece bozuk-hizalı parçaları ÇIKARMAK (ölçüm dürüstlüğü) değil, **DÜZELTMEK** (parçayı geri kazan):
- align_frames şu an sadece 24 eksen-permütasyonu + bbox-merkez arıyor (kaba)
- **Ekle: ICP refinement** (kaba align sonrası ince oturtma) → variant/revizyon uyumsuzluklarını + rotasyon hatalarını düzelt
- Düzelen parçalar hem SKORLANABİLİR hem EĞİTİLEBİLİR olur (çift kazanç)
- Kill: ICP residual'ı düşürmüyorsa (STEP≠JSON gerçekten farklı parça) → o parçaları çıkar (dürüst)
- **Neden en yüksek ROI:** hem ölçümü düzeltir hem veri geri kazanır, model eğitimi gerektirmez

## FAZ 2 — WIRE/TOOL AUXILIARY HEAD (tavan, kök-neden)  → hedef +0.02-0.05
Projenin kanıtladığı TEK tavan = wire vs tool ayrımı. Şu an post-hoc wire_gate.pkl var (AUC 0.867).
- **Model İÇİNE öğrenilmiş bir wire/tool head** ekle (segmentasyon + wire/tool 2. çıktı, multi-task)
- Artık daha büyük etiketli set var (WEI 95 + PXC 78 + eski) → head'i besler
- Geometri ayıramıyor ama YAPISAL context (komşu, simetri, derinlik) ayırıyor (wire_gate kanıtı)
- Kill: held-out wire/tool AUC < 0.85 VEYA e2e F1 kalkmıyorsa
- **Bu, "geometri wire≠tool ayıramaz" tavanını doğrudan deler**

## FAZ 3 — 2. ETİKET TURU (zayıf kalan noktalar)  → hedef +0.02-0.03
Faz 0'dan sonra hangi aile/üretici hâlâ <0.75 ise oraya 2. tur recall-boyama:
- Muhtemel hedef: WEI kalan zayıflar + high-CP singleton aileler (2770/3002)
- Azalan getiri ama gerçek; boyama altyapısı hazır (build_recall_tool + run_after_paint)
- Kill: 2. tur ilk turun <%30'u kadar katıyorsa → doygunluk, dur

## FAZ 4 — HOLE-PRESERVING REMESH (küçük duvar, opsiyonel)  → hedef +0.01-0.02
Sadece Faz 1-3 sonrası hâlâ boşluk varsa:
- ~7% parçada remesh açıklığı siliyor (WEI yay-klemp). Manifold/9k hole-preserving
- **Ama pahalı + domain-gap riski** (tüm korpus re-remesh + retrain). Sadece gerekirse
- Önce ~15 WEI FN-heavy parçada A/B, tüm korpusu riske atmadan

---

## STACKING MATEMATİĞİ (kaba, çakışma var)
| Faz | Lever | Katkı (tahmini) | Kümülatif |
|---|---|---|---|
| 0 | etiket (WEI+PXC) | ölçülecek | ~0.80-0.83 |
| 1 | hizalama recovery | +0.03-0.05 | ~0.82-0.85 |
| 2 | wire/tool aux-head | +0.02-0.04 | ~0.84-0.87 |
| 3 | 2. etiket turu | +0.02 | ~0.85-0.88 |

**Faz 0+1+2 ile 0.85 en olası nokta.** Faz 3 tampon. Faz 4 sadece gerekirse.

## DÜRÜST OLASILIK
- **0.85 base ALL:** Faz 0+1+2 landerse **olası** (%60-70). Hizalama recovery'nin büyüklüğü wildcard.
- **0.85 count-assisted:** daha kolay, muhtemelen Faz 0+1 sonrası.
- **0.90:** bu plan getirmez; 0.85 sonrası ayrı bir seviye (daha çok etiket + mimari).

## SIRA (net)
1. **Faz 0 bitsin** (çalışıyor) → gerçek başlangıç + kalan boşluk
2. **Faz 1 (hizalama recovery)** → en yüksek ROI, model-eğitimsiz, hem ölçüm hem veri
3. **Faz 2 (aux-head)** → tavan-delici, retrain
4. Boşluk varsa **Faz 3** (2. tur), sonra **Faz 4** (remesh)
5. Her fazda: 3-seed, part-out + family-out ayrı, hem tam hem hizalama-filtreli F1

## KİLİT PRENSİP (metric gaming'e karşı)
- Hizalama çıkarması SADECE objektif align residual ile (model performansı DEĞİL)
- Her sayı: leakage-free OOF, hem tam hem filtreli şeffaf
- "Gerçek iyileşme" (etiket/aux-head) vs "ölçüm düzeltmesi" (hizalama) ayrı raporla
