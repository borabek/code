# YARIN NEREDEN DEVAM

> ## ⚠️ 2026-07-31 GECESİ: MANŞET DEĞİŞTİ
> **tespit 0.625 · robot-hazır 0.416** (geometry_key bölmesi, doğru olan bu).
> Aşağıdaki 0.7784 / 0.5272 sayıları `family_key` bölmesinden ve **şişik**.
> Sebep: parçaların %80.1'inin korpusta geometrik ikizi var; "test aileleri çıkarıldı"
> derken gerçekte yalnız test parçaları çıkarılıyordu. Kontrol yapıldı: −0.132 sızıntı,
> −0.022 veri azlığı. Detay: `geometry-twin-leakage` hafıza kaydı.

# (eski kayıt) 2026-07-30 akşamı

PC kapatıldı. Aşağıdakiler **diskte kalıcı**, hiçbiri yeniden üretilmeyecek.

---

## 1. ÜRÜNÜN BUGÜNKÜ HALİ *(değişmedi, çalışıyor, 99/99 test yeşil)*

| metrik | değer | ölçüm |
|---|---|---|
| **tespit F1** | **0.7784** | 100 parça, tam boru hattı, sızıntısız |
| **robot-hazır F1** (yanal ≤2 mm **ve** eksen ≤10°) | **0.5272** | aynı koşu |
| çok-CP F1 | 0.5582 | |
| düşük-CP F1 | 0.5236 | |
| eksen >15° yanlış | %15.8 | |

Makbuzlar: `results/tolerans_gercegi.json`, `results/p9_hizalama.json`,
`cp_config.json > current_product`.

**Bugün ürüne giren, ölçülmüş değişiklikler:**
B-rep analitik eksen (`brep_axes.py`, gmsh/OCC) **+0.0368 robot-hazır** ·
normal-kovaryans eksen +0.0227 · 2 turlu yineleme +0.0051 · yarıçap 12→8 mm ·
`size_mm` gerçek ölçüm (25 mm saçmalığı → 4.6 mm) · denetim artık **her çağrı yolunda** çalışıyor.

---

## 2. HEMEN KOŞULACAK: H ölçümü *(çıkarım BİTTİ, sadece süpürme kaldı)*

```bash
cd /c/Users/DE00024082/Desktop/code
export PYTHONWARNINGS=ignore PYTHONPATH=_diffusion_net_repo/src PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe h_sinif_onculu.py
```

* `results/_h_probs.pkl` (**51 MB**) diskte → **çıkarım tekrar koşmaz**, doğrudan alpha süpürmesi.
* Süre: birkaç dakika.
* **Betiği akşam düzelttim:** ilk sürümde yönlendirici `is_hi` hesaplanıp **kullanılmıyordu**,
  o yüzden `alpha=0` tabanı 0.7518 çıkmıştı (ürünün gerçek tabanı 0.7784). Artık çok-CP'de
  `conn_promote=0.25` uygulanıyor, sayılar **ürünle karşılaştırılabilir**.
* Ölçülen sınıf öncülleri: `[Housing 0.723, Contact 0.126, SnapPoint 0.028, CableEntry 0.029,
  LabelSurface 0.095]` → **bağlantı sınıfları toplam %15.4**, CableEntry yalnız **%2.9**.
  Dengesizlik hipotezi doğrulandı; argmax bu yüzden Housing lehine yanlı.
* **KILL (önceden yazılı):** tespit F1 gerilerse düşer. Recall artıp precision'ı gate geri
  alamıyorsa düşer.

---

## 3. SONRA: C eğitimi *(düzeltilmiş, sıfırdan başlayacak)*

```bash
export PYTHONWARNINGS=ignore PYTHONPATH=_diffusion_net_repo/src PYTHONIOENCODING=utf-8
nohup .venv/Scripts/python.exe train_axis_head.py \
      --epochs 60 --split mfg --val-mfg WEI --resume > results/axis_train.log 2>&1 &
```

* **H bittikten SONRA başlat** — bugün üç sürecin GPU/CPU için boğuştuğunu ve her şeyi
  yavaşlattığını ölçtük.
* Eski checkpoint'ler **silindi** (girdi temsili değişti, eski ağırlıklar anlamsız).
* `--resume` artık **gerçek devam**: model + optimizer (Adam momentleri) + epoch + en iyi skor
  `results/axis_net/last.pt`'ye her epoch yazılıyor. Yarıda kesilirse aynı komut devam ettirir.
* Mesh önbelleği **1542/1542 hazır** → epoch **~98 saniye** (düzeltmeden önce 1756 s, **18 kat**).
  60 epoch ≈ 1.6 saat.

### C'ye akşam giren üç düzeltme *(hepsi bu deponun kendi seg eğiticisinden)*

| kusur | kanıt | düzeltme |
|---|---|---|
| girdi normalize edilmiyordu | `_model_input("xyz")` **ham mm** döndürüyor; parçalar 40–250 mm | merkezle + köşegene böl |
| gradyan biriktirme yoktu | seg eğiticisinde var (`unscaled_loss / accum`) | `--accum 8` |
| LR azaltma yoktu | seg eğiticisinde `--lr-decay-every/rate` var | 15 epoch'ta bir ×0.5 |

*(İlk denemede bunlardan ikisi sessizce uygulanmamıştı; satır bazlı yeniden uygulandı ve
tek tek sayılarak doğrulandı.)*

### C'nin karar eşikleri *(ölçülmüş)*

| | >15° hata |
|---|---|
| sabit +Z (öğrenmeyen taban) | **%43.2** ← bunun altına inmezse ağ hiçbir şey öğrenmiyor |
| düzeltmelerden önceki en iyi | %47.3 (epoch 9) |
| **kill eşiği** (ürünü geçmeli) | **%15.8** |
| ulaşılabilir tavan (kopya taban) | %10.2 |

**C'nin tavanı ölçüldü: en fazla ~5.6 puan = F1'de +0.022.** Küçük kalem. Düzeltmelerden sonra
%43.2'nin altına inmezse **öldür ve G'ye geç** — asıl büyük kalem orada.

---

## 4. SIRADAKİ İŞLER — `YOL_075_ROBOT.md`

| madde | ne | durum |
|---|---|---|
| **G** | üretici CP'lerini **tespit** denetimi yap (8534 CP, eğitimde hiç kullanılmadı) | **en büyük açık kapı**, kurulmadı |
| **H** | sınıf-öncülü karar | çıkarım hazır, süpürme kaldı |
| **I** | yuva ekseni: **düzlem çiftlerinden** analitik eksen (A'nın reddettiği 300 aday) | kurulmadı |
| **D** | agresif çok-çözünürlüklü havuz (kayıtlı tavan %96.5) | kurulmadı |
| **E** | gate'e `eksen_güveni` özelliği | kurulmadı |
| **F** | **`geometry_key` ile manşeti yeniden ölç** | kurulmadı |

**F'yi unutma:** `family_key(pid)` bu korpusta parça numarasının kendisini döndürüyor
(1542 parça → 1542 "aile"). Bu bir hata değil, **belgelenmiş bir sınır** — docstring `geometry_key`
kullanılmasını söylüyor. Sonuç: "aile-dışı" dediğimiz her şey fiilen **parça-dışı**, ve kardeş
varyantların geometrisi **birebir aynı** (eksen farkı 0.000°). Yani manşet sayılar iddia ettiğim
kadar korumalı değil. `geometry_key` zaten var (`json_dataset.py:247`).
**Beklenti: sayı düşebilir. Düşerse doğru sayı odur.**

---

## 5. DİSKTE KALICI OLANLAR *(yeniden üretme)*

| yol | ne | boyut |
|---|---|---|
| `results/_h_probs.pkl` | H'nin çıkarımı, 100 parça | 51 MB |
| `results/mesh_cache/` | 1542 parçanın mesh'i | epoch'u 18× hızlandırır |
| `results/axis_dataset/` | 1542 parça / 387.106 hedef köşe | C'nin verisi |
| `results/brep_cache/` | B-rep silindirleri (101 parça, büyüdükçe dolar) | |
| `results/_tolerans_cache6.pkl` | tam boru hattı doğrulaması | manşetin makbuzu |
| `results/axis_dataset/STUCK_PARTS.txt` | `WEI 2502870000` — gmsh asıyor | |

**Bilinen boşluk:** `build_axis_dataset.py`'de parça-başı zaman aşımı yok; `WEI 2502870000`'de
asıldı ve ~370 parça toplanamadı. Kalanı istersen betiğe timeout ekleyip topla.

---

## 6. YARIN İÇİN DÜRÜST TAHMİN *(bugün 4 projeksiyonum ölçümde çöktü, bu yüzden aralık)*

| | şimdi | hepsi bitince |
|---|---|---|
| tespit F1 | 0.778 | **0.80 – 0.86** *(neredeyse tamamı G+D'ye bağlı; F bunu −0.01…−0.05 çeker)* |
| robot-hazır F1 | 0.527 | **0.57 – 0.62** |

**0.75 robot-hazır bu listeyle çıkmıyor** — eksen kolunda kalan bütçe ölçüldü (+0.022) ve
gerisi recall'a bağlı. Dürüst hedef: **tespit 0.82-0.85, robot-hazır 0.58-0.60.**
