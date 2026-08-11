# DEVAM NOTU — üretici-dışı transfer düzeltmesi
**Bırakılan an:** 2026-07-31 akşamı · **Ürün durumu: DEĞİŞTİRİLMEDİ, güvenli**

## Ürün şu an

| | |
|---|---|
| gate | 18 sütun (`gate_fiz_feats=true`) · üretim de 18 sütun üretiyor ✔ |
| `gate_ek_feats` | `false` (EK bloğu ölçüldü, dağıtılmadı) |
| **göreli eşik** | **KAPALI** — karar bekliyor, aşağıya bak |
| manşet | tespit **0.7437** / robot **0.4594** |
| testler | **107/107 yeşil** |

Hiçbir yarım dağıtım yok. Geri alma gerekmez.

---

## Bulunan sorun (acil olan bu)

**Gate, hiç görmediği bir üreticide çöküyor.** Ölçüldü (aday düzeyi, 18 sütun):

| bölme | F1 |
|---|---|
| geometri-dışı (manşetin protokolü) | 0.7422 |
| üretici 0 dışarıda | 0.6399 |
| **üretici 1 dışarıda** | **0.2799** |

Manşetimiz geometri-ayrık ama **üretici-karışık** veriyle ölçülüyor. Yeni bir üretici gelirse
beklenen seviye 0.28–0.64 bandı.

**Ayrıştırma:** kaybın ~%42'si **kalibrasyon** (sabit 0.40 eşiği kaymış dağılımda yanlış yerde;
üretici 1'de model adayların %10.3'üne pozitif diyor, gerçek %24.1), ~%58'i **sıralama**.

**Yan bulgu:** B-rep fiziksel sütunları transferi **+0.0469** iyileştiriyor — fiziksel büyüklükler
(mm) üreticiden üreticiye değişmiyor, korpus istatistikleri değişiyor.

---

## Yapılan tedavi (kalibrasyon tarafı) — karar aşamasında

**Kural:** sabit eşik yerine → aday, kendi **parçasındaki** en yüksek skorun **0.5 katını** geçmeli
**VE** mutlak **0.25** tabanını aşmalı. Taban şart: onsuz kural, hiç gerçek CP olmayan parçada bile
"en yükseğin yarısı"nı kabul edip garanti yanlış üretir.

Ölçülenler:

| ölçüm | sonuç |
|---|---|
| aday düzeyi, üretici-dışı en kötü | 0.2799 → **0.4222** (+0.142) |
| uçtan uca tanıdık veri, DEV | −0.0140 |
| uçtan uca tanıdık veri, VAL | −0.0022 |
| ortalama bedel | **−0.0081** (bütçe 0.02) |

Kod bağlandı: `wire_gate.GORELI_ESIK / GORELI_ORAN / GORELI_TABAN`, bayrak `cp_config.gate_goreli_esik`,
**varsayılan kapalı**.

---

## SIRADAKİ ADIM (yarım kalan tek iş)

`t8_uretici_disi_uctan_uca.py` **arka planda koşuyordu** → sonucu `results/t8_uretici_disi_uctan_uca.json`
ve `results/t8.log`.

Bu, **kararı verecek ölçüm**: üretici-dışı kazanç *uçtan uca* doğrulanıyor mu. Bugün aday düzeyi
iki kez yanılttı (EK bloğu; göreli eşiğin bedeli aday düzeyinde −0.0074 görünüp uçtan uca −0.0313 çıktı),
o yüzden karar bu betikle veriliyor.

**Önceden yazılı kill:** göreli eşik, üretici-dışı **uçtan uca** tespit F1'de sabiti **her iki
üreticide de** geçmezse **dağıtılmaz**.

Geçerse dağıtım tek satır: `cp_config.gate_goreli_esik = true` (oran 0.5, taban 0.25).
Geçmezse bayrak kapalı kalır ve bulgu "kalibrasyon aday düzeyinde çalışıyor, uçtan uca çalışmıyor"
diye kayda geçer.

---

## Sonra sırada bekleyenler

1. **Sıralama tedavisi** (kaybın %58'i): hangi sütunlar üretici imzası taşıyor — ölçek-bağımlı
   olanları (nn_dist, nverts, size) parça çapına bölüp yeniden ölç.
2. **İçbükey kenar topolojisi** → gate (`YOL_075_ROBOT.md` madde 6, gerekçesi ve kill'i yazılı).
3. **LOCKED** hâlâ harcanmadı.
