# 0.85'E GIDEN YOL — stacked, each adim OLCULU (2026-07-27 aksam)

> Bu gece CONFIRMED temel: (1) recall-fav + last-epoch recipe WEI 0.525->0.558 (+0.033, full-145),
> (2) BUG-2 checkpoint: last-epoch >> best-val (+0.086). Artik gercek leverler present.
> DURUST cerceve: "conclusive 0.85" garanti not but this, each adimi olçulen en gercek yol.

## Baslangic (gated urun, scoreboard): base ALL 0.75 | WEI 0.696 | PXC 0.823 | count-assisted 0.775

## STACKED LEVERLER (confidence sirasi, kumulatif)

### 1. BUG-2 last-epoch redeploy  [YUKSEK confidence, BEDAVA, +0.05-0.08]
- Deployed urun best-val checkpoint kullaniyor (WEI-underfit, this gece kanitlandi)
- Urun receipsini last-epoch save with YENIDEN egit -> last-epoch sec
- Beklenti: WEI 0.696->~0.75, PXC 0.82->~0.85, ALL 0.75->~0.78-0.80
- Kill: last-epoch best-val'i gecmiyorsa (this gece B_rec'te gecti)

### 2. Recall-fav recipe deploy  [CONFIRMED, +0.03]
- recall-fav (pos-w 40, tversky 0.5, ce-mult 2) + last-epoch = combined (su an egitiliyor)
- WEI +0.033 confirmed; PXC bozmuyorsa deploy

### 3. Bug-tuning: ensemble + postproc  [ORTA, +0.02-0.04]
- BUG-B: ensemble recall dusuruyorsa single-best/optimize
- BUG-E: min_v/vc yeni modele tune
- BUG-A: density-outlier parcalari ayri ele

### 4. Metadata / count-assisted mode  [YUKSEK, +0.03-0.05 (count biliniyorsa)]
- Katalog CP sayisi -> top-N (already 0.775) + iyilesmis model -> ~0.83-0.85
- WSCAD katalog-bagli parts for gercekci

### 5. Hedefli label OLCEK (recall-fav recipe with)  [BUYUK, +0.04-0.08]
- ARTIK etiket YARDIM EDIYOR (correct recipe with, this gece kanitlandi)
- WEI/high-CP zayif noktalarina recall-fav recipe with more very label
- WEI 0.75 -> 0.80+; base ALL -> 0.83+

### 6. Wire/tool precision (RF gate + gerekirse aux-head)  [ceiling lever, +0.02-0.03]
- Precision tavani; RF gate already AUC 0.867

## KUMULATIF TARGET (durust aralik)
| Metrik | Baslangic | 1-4 sonrasi | 5-6 sonrasi |
|---|---|---|---|
| PXC-tipik | 0.823 | **~0.85** | 0.86+ |
| count-assisted ALL | 0.775 | **~0.83-0.85** | 0.85+ |
| base ALL | 0.75 | ~0.80-0.82 | **~0.83-0.85** |
| WEI | 0.696 | ~0.75 | ~0.80 |

## EN ULASILABILIR 0.85 (correct cerceve)
1. **PXC-tipik 0.85** -> lever 1 (BUG-2) with en yakin
2. **count-assisted ALL 0.85** -> lever 1+4 with olasi
3. **base ALL 0.85** -> lever 1-6 hepsi (en zor, en gec)

## SIRA (net)
1. Combined verdict (su an) -> recipe+PXC gecerli mi
2. BUG-2 redeploy (bedava, en high ROI)
3. Bug-tuning (ensemble/postproc)
4. Metadata mode netlestir
5. Hedefli label olcek (recipe with) -> asil 0.85 itici
6. Locked operating point -> held-out arbiter'da 0.85 DOGRULA

## DURUST NOT
"Kesin 0.85" tek adimda not, STACK with. Her lever olculecek; landmiyorsa durust raporla.
Bu gece 3 gunun ilk CONFIRMED pozitifi geldi -> yol residual OPEN, sadece stack + olcek.

---

## DUZELTME (2026-07-27 18:59 — combined eval): PER-MANUFACTURER, combined DEGIL
- Combined (WEI+PXC, recall-fav+last) WEI=0.468 < B_rec WEI-only 0.558 < A 0.525
- PXC etiketi WEI'yi DILUE ediyor (confound-free, same seed) -> COMBINED YANLIS
- **LEVER-2 DUZELTILDI:** each manufacturer KENDI modeli:
   - WEI modeli = B_rec (recall-fav+last, WEI-only) = 0.558 CONFIRMED
   - PXC modeli = ayri (PXC etiketleri + recall-fav, WEI'siz)
- Router: part-manufacturer -> ilgili model. (Robot already manufacturer biliyor.)
- Bu, dilusyon sorununu cozer; each manufacturer kendi zirvesinde

---

## HAM WEI-DAHIL ALL 0.85 SALDIRISI (en zor framing, test edilmemis leverler)
> Bu gece: etiket doyumda (95->+0.006), recipe marjinal, BUG-2 recall_hard'a yardim etmiyor.
> AMA test EDILMEMIS leverler present. Sirasiyla (potansiyel), each biri OLCULECEK:

### W1. GT-TAMLIK AUDIT (ONCE — belki 0.58 PESIMIST) [measurement, large potansiyel]
- WEI FP'lerinin ne kadari GERCEK listelenmemis opening? (PXC'de %96-98'di)
- Ornek WEI FP'yi denetle: gercek acikliksa -> GT missing -> gercek WEI precision YUKSEK -> F1 0.58->0.70+?
- NOTE: sisme tuzagi -> SADECE dogrulanirsa (insan/gorsel denetim). Olcum duzeltmesi, model not.

### W2. YUKSEK-COZUNURLUK / HOLE-PRESERVING REMESH [WEI recall, test edilmemis]
- WEI acikliklari KUCUK/duz; 6000-vertex remesh cozmuyor olabilir
- 9k/12k VEYA Manifold hole-preserving -> model aciklikari GORUR -> recall 0.58->?
- 20 WEI FN-agir parts A/B (tum corpus riske atmadan)

### W3. MULTI-RESOLUTION ENSEMBLE [recall]
- 6k + 9k + 12k different acikliklar yakalar -> union -> recall
- (ensemble softening'e dikkat -> vote/gate with)

### W4. WIRE/TOOL + PRECISION [precision]
- RF gate AUC 0.867; metadata + yapisal ozellik ekle -> precision
- 2D-render arm (WEI DUZ opening gorsel belirgin) -> multimodal

### W5. METADATA-AGIR YAPISALLASTIRMA [precision+recall]
- pole/pitch/row/deck katalog -> structured top-K per row (sadece count not)

### DURUST TARGET
| Lever | Beklenen |
|---|---|
| W1 GT-audit (dogrulanirsa) | 0.58 -> ~0.70 (measurement) |
| +W2 hole-remesh | +recall |
| +W3 multi-res | +recall |
| +W4/W5 | +precision |
| **HAM WEI 0.85** | STACK ILE hedef; each lever olculecek, landmiyorsa DURUST |

### KESIN OLAN: each lever OLCULECEK. 0.85 gelirse evidence; gelmezse durust ceiling.
### "Kesin duzeltecez" = kararlilikla TEST edecegiz, garanti not (over-promise none).
