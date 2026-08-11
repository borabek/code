# DURUM — R serisi (2026-08-05 ogle)

## OLCULMUS TABAN (hepsi AYNI kod yolundan, adil)

```
                      ESKI ag+eski gate   YENI ag+gate v4
194'luk tespit            0.6778             0.7410   +0.0632
194'luk ROBOT             0.4607             0.5039   +0.0432
gorulmemis tespit         0.3475             0.5898   +0.2423
gorulmemis ROBOT          0.1185             0.2410   +0.1225
```
YENI ag TEK SEED, eski 4 checkpoint TOPLULUK -- dezavantajli durumda kazaniyor.

## KAYIP AYRISTIRMASI (gorulmemis)

```
tespit -> robot:  yanal<=2mm %64 · aci<=10 %54 · IKISI %41 · DIK>80 %38
194'lukte:        %84 · %83 · %75 · %12
```
Fark KONVANSIYON DEGIL: GT hizaliyken bile basarimiz 194'lukte %88, gorulmemiste %55.
Uretici kirilimi: PXC %2 dik, WEI %36 -- mekanizma PXC'de CALISIYOR.

## EKSEN HATTINDA UC NULL (hepsi olculdu, hicbiri kod degistirmedi)

| deneme | sonuc |
|---|---|
| kanal ekseni (`channel_axis_robust`) | dik vakalarin %13'u |
| agiz duzlemi normali (PCA r=6mm) | %40 (kontrol %21) |
| **yakinlik-oncelikli B-rep** | **194'te +5, gorulmemiste NET -30** -> REDDEDILDI |

Aci kapisi (max_turn_deg=60) bir HATA DEGIL, KORUMA: en yakin silindir cogu zaman
KOMSU delik/vida. Kaldirmak 35 iyi vakayi bozup 5 kurtariyor.
-> Kalan tek yol: UC ADAYI DA uretip SECICI ile birlestirmek (yakinlik + yaricap-agiz
uyumu + disari-bakma tutarliligi). Tek mekanizmaya bel baglamak UC KEZ dustu.

## R3 ESIK DUYARLILIGI (SINAV kumesinde tarandi -> TUNING DEGIL, DUYARLILIK)

```
oran/taban   194 tespit  194 robot   sinav tespit  sinav robot
0.50/0.25      0.7410     0.5039       0.5898       0.2410  <- mevcut
0.55/0.30      0.7333     0.5133       0.5879       0.2549
0.60/0.35      0.7109     0.5166       0.5641       0.2643
0.70/0.40      0.6435     0.4896       0.5212       0.2721
```
Gate'i sikmak tespiti DUSURUP robotu YUKSELTIYOR. Mevcut esik TESPIT icin ayarlanmis.
**Esik DEGISTIRILMEDI** -- sinav kumesinde secmek coklu-karsilastirma tuzagi.
DOGRUSU: DEV'de robot-oncelikli esik sec, sinavda TEK ATISLA dogrula. Beklenen +0.02..0.03.

## SIRADAKI

1. R4 topluluk: seed1 kosuyor (ep ~51/200), sonra seed2 -> 3'lu topluluk olcumu
2. R3 dogru yordamla: DEV'de esik, sinavda tek atis
3. R5 kalabalik: cluster_mm/dedupe DEV'de tara (yeni ag 2.15x aday uretiyor)
4. R1 secici: uc eksen adayini birlestir (tek mekanizma UC KEZ dustu)
5. R8 iki kumeli manset + R9 rapor
