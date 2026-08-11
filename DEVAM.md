# EVE GELINCE BURADAN DEVAM — 2026-07-28

## Tek satırda durum

Ürünün korpus-temsili CP-F1'i **~0.79** (düşük-CP 0.820 · çok-CP 0.551).
Sıradaki iş **TEL-B'yi gerçek korpusa koşmak** — hazır ve doğrulanmış, sadece koşulmadı.

---

## 1. Yarım kalan çıkarımı bitir (ilk iş, ~10 dk)

PC kapanınca duracak. Kaldığı yerden devam eder, iş kaybolmaz:

```powershell
$env:PYTHONPATH="_diffusion_net_repo/src"
$env:CP_MV=10; $env:CP_VC=0.30; $env:CP_CL=3.0; $env:CP_DD=6.0
.venv\Scripts\python.exe build_rich_feats.py --list results\_todo_mv10.txt --out results\rich_mv10_all.npz
```

Kapanırken **863/1341**'deydi. Bittiğinde `results/rich_mv10_all.npz` hazır olur.

> Takılırsa: `results\_extract_current.txt` hangi parçada olduğunu yazar.
> O parçayı `_skip_parts.txt`'e ekleyip yeniden başlat.

---

## 2. Sıradaki iş: TEL-B'yi korpusa koş

`tel_b_kanal_profili.py` yazıldı ve **sentetik geometride 3/3 doğrulandı** (tel girişi / alet ağzı / tünel).
Henüz gerçek parçalara koşulmadı. Yapılacak:

- Düşük-CP altkümesinde her aday için 10 kanal özelliğini çıkar
- Hayatta kalan FP'ler ile TP'lerin dağılımını karşılaştır
- **Kill kriteri: mevcut gate üzerine ek AUC katkısı < 0.02 ise aile ÖLÜ ilan edilir**

Neden bu ilk: korpusun **%89.5'i düşük-CP** ve orada tavan 0.894 → potansiyel **+0.054 genel**.
Çok-CP dalı (min_v10, adaptif çözünürlük) daha kırık görünüyor ama korpusun %10.5'i → sadece +0.016.

---

## 3. Bugün kapanan üç kök hata (hepsi aynı aileden: sessizce başarısız olan kod)

| hata | etki |
|---|---|
| `rtree` yok → `trimesh.contains()/ray` her çağrıda patlıyor, `except` yutuyor | Görselleştirme aylardır yalan söylüyordu; **çalışan ışın motoru** bugün kazanıldı |
| `pymeshlab` remesh süreçler arası oynuyor | 8'de 1 parça farklı sonuç; disk önbelleğiyle çözüldü (20/20 aynı) |
| `gmsh.finalize()` `finally`'de değil | 4 geçmiş çıkarım ~225'te ölmüş, **~415 parça hiç çıkarılmamış** |

Son maddenin bedeli ölçüldü: kayıp parçalar geri gelince korpus 914→1753 ve **CP-F1 +0.0107**.

---

## 4. Çizim sözleşmesi: %100 sertifikalı

- `CIZIM_SOZLESMESI.md` — 9 madde
- `glb_audit.py` — GLB'yi **diskten geri okuyup** doğrular; mutasyon testinde **6/6** bozmayı yakaladı
- **70/70 parça sertifikalı** (`results/viz_certification.json`), 567/567 iğne doğru
- `viz.ps1` her üretimde denetçiyi otomatik koşar, ihlalli GLB'yi **siler** + `exit 1`

Kullanım: `.\viz.ps1 -Random 10` · `-Worst 10` · `-Best 10` · `.\viz.ps1 3002613 3012300`

---

## 5. Bugün öğrenilen iki metodoloji kuralı

**Süreç sınırını geçmeyen determinizm testi hiçbir şey test etmez.**
Remesh'i tek süreçte 3 kez koşup "deterministik" demiştim; ayrı süreçlerde oynuyordu.

**Çarpık örneklemde düz ortalama alma.** "F1 0.691" dedim; skor kümem çok-CP'yi %18.6 ile
temsil ediyordu, gerçek korpus %10.5. Doğru sayı ~0.79. Bundan sonra her F1 **rejim ayrımlı**
ve **korpus oranıyla ağırlıklı** raporlanacak.

---

## 6. Dokunulmayacak

**Kilitli holdout ikinci kez kullanılmayacak** — kararı `cp_config.json > locked_holdout_policy_2026_07_28`.
Ara kararlar WORK setinde aile-dışı + iç-içe eşikle ölçülür ve **"WORK OOF"** diye etiketlenir.
Holdout, tüm ürün değişiklikleri donduktan sonra **tek seferde** harcanır.
