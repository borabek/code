# CP VERİSİ TALEBİ — kanıtlanmış, sayısal temelli
**2026-07-28 · WiringRobot ConnectionPointDetector**

## Tek cümlelik talep
> Elimizde **2838 terminal parçasının 3D geometrisi (STEP)** var, **bağlantı noktası verisi yok**.
> Bu veri **EPLAN Data Portal'da mevcut** (`connection_point_pattern`) ama indirmek **EPLAN Electric P8 /
> Pro Panel** entegrasyonu gerektiriyor. EPLAN kurulu bir iş istasyonundan bu parçaların indirilmesi
> gerekiyor.

## Kanıt (bu oturumda ölçüldü)
| | |
|---|---|
| İhtiyaç listemizden 24 rastgele parça, EPLAN portalında **bulunan** | **21/24 (%88)** |
| **`connection_point_pattern` veri tipi mevcut** | **17/24 (%71)** |
| `3d_graphical_data` mevcut | 21/24 (%88) |
| → 2838 parçanın tahminen | **~2010'unda CP verisi hazır** |

Doğrulanan örnekler (hepsinde CP-pattern + 3D var):
`PXC.0311087` · `PXC.3002162` · `PXC.0421029` · `PXC.3044225` · `WEI.1010100000` · `WEI.1934810000`

## Neden web'den alamıyoruz (denenen tüm kanallar)
| Kanal | Sonuç |
|---|---|
| EPLAN Data Portal — web indirme | ❌ sadece **ticari CSV** + **grafik DXF** (makro readme'si de "graphic files" diyor) |
| WSCAD Universe — `.wspak` | ✅ iniyor ama içi metadata + kaba OBJ(657 vertex) + tescilli sembol → **3D CP yok** |
| WSCAD Universe — EDZ / DWG / diğer partType | ❌ 18/18 parçada yok |
| Phoenix Contact / Weidmüller siteleri | ❌ 403 (tarayıcıyla bile — bu ağdan engelli) |
| Yerel EPLAN kurulumu | ❌ sadece "Smart Production", parça veritabanı yok |
| Yerel WSCAD servisi (port 10384) | ❌ kurulu değil |
| STEP dosyalarının içi | ❌ 43 entity tipinin hepsi saf geometri, `CONNECT/TERMINAL/PORT` **hiç yok** |

**Sonuç:** iki büyük portal da 3D CP verisine sahip ama **yalnızca kendi masaüstü yazılımına** veriyor.
Bu bir eksiklik değil, ticari model. Programatik/web yolu **yok**.

## İstenen format (elimizdeki 1926 parça için zaten mevcut)
```json
{ "PartNr": "PXC.0311087",
  "ConnectionPoints": [
    {"Index":0, "Name":"1",
     "Point":{"X":3.93,"Y":54.82,"Z":23.13},
     "InsertDirection":{"X":0.01,"Y":1.0,"Z":0.0}} ]}
```
Dosya adı deseni: `MFG.PARTNR_ElectricalTerminal_ElectricalTerminal.json`
**Parça listesi:** `results/CP_VERISI_ISTENEN_PARCALAR.csv` (2838 parça — PXC 1787, WEI 557, ABB 25, diğer 469)

## Beklenen kazanç (dürüst)
Mevcut: kilitli holdout CP-F1 **0.748** · WORK family-out **0.760**
İlk gerçek veri-ekleme ölçümü (+220 parça): **+0.003** — yani eğri tahmin ettiğimden **yatık**.
Bu yüzden 2838 parça için dürüst beklenti **+0.02 – 0.05** (önceki "+0.07" iyimserdi, ölçümle düzeltildi).
Kesin sayı, 321 parçalık temiz test bitince güncellenecek.

## Kime
EPLAN Electric P8 / Pro Panel lisansı olan mühendis (şirket EPLAN müşterisi — bu makinede
EPLAN Smart Production Collection kurulu). Portal hesabı ücretsiz, zaten açıldı.
