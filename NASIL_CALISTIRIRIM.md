# Robotu kendim nasıl çalıştırırım

Klasör: `c:\Users\DE00024082\Desktop\code` — terminali burada aç (PowerShell).

---

## 1. Tek parça — rapor + 3D görsel ⭐

**En kolay yol:**

```powershell
.\viz.ps1 0311087
```

Parametresiz çalıştırırsan denemelik parça listesini gösterir:

```powershell
.\viz.ps1
```

**Elle yapmak istersen:**

```powershell
$env:PYTHONPATH = "_diffusion_net_repo/src"
.venv\Scripts\python.exe robot_viz.py 0311087
```

> ⚠️ `PYTHONPATH=... komut` yazımı **bash'e özeldir, PowerShell'de çalışmaz.**
> PowerShell'de önce `$env:PYTHONPATH = "..."` ile ayarlanmalı. `viz.ps1` bunu senin yerine yapıyor.

**Ekran çıktısı:**

```
=== PXC.0311087 | [8.2 71.8 50.4] mm ===
  manufacturer CP 2 | robot 2 | TP 2 FP 0 FN 0 -> F1 1.000
   tahmin1 -> GT2 ('2') dik 0.2mm, axis +8.4mm  [DOGRU]
   tahmin2 -> GT1 ('1') dik 1.1mm, axis +5.5mm  [DOGRU]
   -> results/robot_glb/PXC_0311087_F1_1.00.glb
```

Sonra `results\robot_glb\` klasöründeki `.glb` dosyasına **çift tıkla** → Windows 3D Viewer açılır.

**Renkler:**

| Renk | Anlamı |
|---|---|
| 🟢 yeşil iğne | üretici CP — robot **buldu** |
| 🟡 sarı iğne | üretici CP — robot **kaçırdı** (FN) |
| 🔴 kırmızı iğne | robot tahmini — **doğru** (TP) |
| 🟣 mor iğne | robot tahmini — **fazladan** (FP) |
| 🔵 mavi çizgi | eşleşme (ağız ↔ yuva farkı) |

İğneler gövdeden dışarı taşar, döndürünce her açıdan görünür.

---

## 2. Birden çok parça

```powershell
.\viz.ps1 3002613 3012300 3273246
```

---

## 3. Denemelik parça listeleri

| Ne görmek istersen | Parça numaraları |
|---|---|
| **Mükemmel** (F1 1.00) | `3002613 3002614 3002617 3002618 3210545` |
| **Orta** (F1 0.4–0.7) | `3273246 3273382 3273388 3001871 3270115` |
| **Başarısız** (F1 0.00) | `3012300 3071356 1469110000 1783600000 8670750000` |
| **Zor / çok CP'li** | `2770943 3001879 3270137 3270230 3273112` |

Kendi listeni üretmek için (en kötü 10 parça):

```powershell
.venv\Scripts\python.exe -c "import json; r=json.load(open('results/per_part_f1.json')); r.sort(key=lambda x: x['f1']); print(' '.join(x['pid'] for x in r[:10]))"
```

---

## 4. Ham robot çıktısı (koordinatlar, makine formatı)

```powershell
$env:PYTHONPATH = "_diffusion_net_repo/src"
.venv\Scripts\python.exe robot_e2e.py --limit 5
```

Her CP için dönen alanlar:

| Alan | Anlamı |
|---|---|
| `point` | robot buraya gidecek (x, y, z mm) |
| `direction` | teli bu yöne sokacak |
| `size_mm` / `depth_mm` | açıklık çapı / derinliği |
| `confidence`, `votes` | güven, kaç model hemfikir |
| `wire_score` | 1 = tel girişi, 0 = alet deliği |
| `tier` | `auto` = robot kendi yapsın, `review` = insana sor |

---

## 5. Bilmen gereken tek kavram: **yuva ≠ ağız**

Üreticinin verdiği CP genelde **kontak yuvasındadır** — parçanın **içinde**, katı malzemede.
Ama bu **parçaya göre değişir** (ölçüldü: 3270115'te 16/16 içeride, 3273112'de 0/19 zaten yüzeyde).
Görselleştirme bunu her CP için tek tek sınar; içerideyse ağza taşır, değilse yerinde bırakır.

Delik (**ağız**) oradan **5–25 mm** ötede, yüzeydedir. **Robot ağzı tahmin eder.**

Bu yüzden raporda `axis +8.4mm` gibi bir sayı görürsün — o **error değil**, ağız ile yuva arasındaki
mesafedir. Görselde de GT iğnesi ağza taşınmış çizilir (`cp_geometry.seat_to_mouth`), yoksa gövdenin
içinde kalıp görünmezdi — senin "bu dosyalarda CP yok gibi" demenin sebebi tam olarak buydu.

---

## Not

Bir parça ilk kez işlenirken spektral operatörler hesaplanır (20–60 sn). Aynı parça ikinci kez ~4 sn.
