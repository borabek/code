# WSCAD->CP Selective Accuracy — Olculen Sonuc (2026-07-15)

Model: v31 best (epoch 19/20, YAKINSAMADI). Etiket: v8 CAD. Val: 270 parca (donmus).
Metrik: Jaccard = TP/(TP+FP+FN). Match 5mm, thr 0.30, nms 5mm.

## Full coverage (tum parcalar otomatik)
TP=1593 FP=335 FN=545  ->  Jaccard = 0.6442

## Selective (risk-sirali kabul, gercek risk skoru)
| coverage | parca | kabul-Jaccard |
|---|---|---|
| 50% | 135 | 0.857 |
| 60% | 162 | 0.829 |
| 70% | 189 | 0.798 |
| 100%| 270 | 0.644 |

## Selective TAVANI (mukemmel/oracle risk skoru — ulasilamaz ust sinir)
| coverage | havuz-Jaccard |
|---|---|
| 50% | 0.911  <- 0.90'i ancak burada geciyor |
| 40% | 0.940 |
| 30% | 0.966 |

Tek-parca Jaccard >=0.90 olan: 80/270 (%30). Kusursuz (1.0): 58/270 (%21).

## KARAR
- Full-coverage 0.90: ULASILMAZ (decode +0.004, full-patch +0.002 olctuldu).
- Selective 0.90: gercekte ~%40-50 coverage. Plan kurali: <%50 => "human-review
  routing", "model %90" DEME.
- Coverage'i yukselten tek kaldirac: modeli keskinlestirmek = v31 yakinsamasi (~11h)
  veya v32 regularizasyon. Etiket temizligi full-coverage'i yaklastiran ayri kaldirac.
