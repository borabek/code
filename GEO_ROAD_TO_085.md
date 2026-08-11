# GEOMETRIK MODEL -> 0.85 YOLU (ML YOK, bagimsiz) — 2026-07-27

> DUZELTME: naif konkavlik (0.08) YANLIS yaklasimdi. Dogru foundation = step_openings.py
> (CAD-direct: STEP B-rep'ten CYLINDRICAL_SURFACE -> CP merkez+yaricap). Konkavlik DEGIL.
> DURUST cerceve: ML'in CONFIRMED leverleri var; geometrik 0.85 RESEARCH-BET (discrimination
> tavani gercek: geometri cable-entry'yi tool/mounting/vent'ten fonksiyonel ayirmali).

## Foundation: step_openings.py (CAD-direct)  [baseline ~0.46, YUKSEK recall]
- STEP B-rep -> tum silindirik acikliklar (cable + screw + mounting + vent) -> merkez+yaricap
- Sorun: HEPSINI buluyor (over-mark) -> precision dusuk -> tavan ~0.46 (sunum slayt 22)
- Recall yuksek; asil is = PRECISION (cable-entry'yi digerlerinden ayir)

## 0.85'E ITICI LEVERLER (fiziksel prior, OGRENME YOK)

### L1. Yaricap filtresi (var)  [+recall/precision]
- --rmin/--rmax = kablo capi araligi -> screw/rail-boyu silindirleri at

### L2. EKSEN-YONELIM filtresi  [BUYUK precision]
- Cable-entry YATAY girer (on/erisim yuzu); screw access DIKEY (ust); mounting ARKA
- Her acikligin ekseni -> on-yuz-normali ile hizali olanlari tut, ust/arka olanlari at
- Bu, screw/mounting acikliklarini geometrik olarak eler (fonksiyonel ayrimi YAKALAR)

### L3. YUZ-KONUMU filtresi  [precision]
- Parca bbox'inda acikligin hangi yuzde oldugu (on/ust/arka/yan)
- Terminal blok: cable-entry ON yuzde, montaj ARKA/ALT -> yuze gore filtre

### L4. DERINLIK + KANAL filtresi  [precision]
- Cable-entry: belirli kanal-derinligi (contact seat'e). Sig delik (vent) veya cok-derin (montaj) at

### L5. PATTERN/DIZI tespiti  [precision]
- Terminal blok: cable-entry'ler DUZENLI DIZI/GRID. Tek-basina outlier'lari at, dizi-uyeleri tut

### L6. PRIZMATIK/SLOT tespiti (WEI icin ZORUNLU)  [WEI recall]
- WEI yay-klemp = DIKDORTGEN/SLOT, silindir DEGIL -> step_openings kaciriyor
- PLANAR+PLANAR paralel yuz cifti (slot) tespiti ekle -> WEI acikliklarini yakala

### L7. ABB/yeni-uretici zero-shot  [kapsam]
- Ayni fiziksel priorlar her uretici (egitim yok) -> ABB 386 parca test

## DURUST TAVAN DEGERLENDIRMESI
| Asama | Beklenen F1 |
|---|---|
| step_openings ham (over-mark) | ~0.46 |
| + L1 radius | ~0.50 |
| + L2 eksen + L3 yuz (asil discrimination) | ~0.60-0.68 |
| + L4 derinlik + L5 pattern | ~0.68-0.75 |
| + L6 slot (WEI) | WEI recall duzelir |
| **0.85** | **STRETCH** -- priorlar cable-vs-diger'i NEREDEYSE mukemmel ayirmali |

## DURUST: 0.85 saf-geometrik GARANTI DEGIL
- Realistik tavan (L1-L6): ~0.68-0.75 (PXC silindirik'te daha yuksek, WEI slot'ta dusuk)
- 0.85 = priorlar cable-entry'yi tool/mounting'den ~mukemmel ayirirsa -> RESEARCH BET
- **AMA:** eksen+yuz priorlari (L2/L3) fonksiyonel ayrimi GEOMETRIK yakalayabilir (cable yatay-on,
  tool dikey-ust) -> CAD-direct'in 0.46 tavanini ASMA sansi GERCEK
- Deger: ML'siz, her-uretici zero-shot, yorumlanabilir. ML-tamamlayici (60-zero-recall kurtarma)

## SIRA (sonraki oturum, CPU)
1. step_openings.py'yi WEI/PXC'de test -> gercek ham baseline (naif 0.08 DEGIL, ~0.46 beklenir)
2. L2 eksen-yonelim + L3 yuz-konumu ekle -> asil precision sicramasi
3. L4-L5 (derinlik/pattern), L6 (slot/WEI)
4. ABB zero-shot
5. Her adim WEI/PXC/ABB arbiter'da olç -> 0.85'e ne kadar yaklasti DURUST raporla
