# GECE RAPORU — otonom Faz A (2026-07-27 → 28)
> Hedef: ML CP-F1'i 0.85'e taşımak. Kural: leakage-free OOF, insan etiketi YOK, tezden sapma YOK,
> her iddia önce ölçüm. Bu rapor gecenin TAM dökümü — kazananlar VE öldürülenler.

---

## ⭐ ANA SONUÇ: ürün ilk kez materyal olarak ilerledi

> ⚠️ **DÜZELTME (Faz 2/P0 sonrası, 2026-07-28):** aşağıdaki ilk tablo **ŞİŞKİNDİ** — (a) ölçüm seti
> `BA_ALLOW_SEEN=1` ile 153 sızıntılı parça içeriyordu (etiketleri segmentasyon eğitiminde görülmüş),
> (b) aile anahtarı bbox-proxy idi, (c) ensemble sınıflandırıcı kullanılmıştı. **Geçerli sayılar
> ikinci tablodadır.** Canonical 0.750 de aynı sızıntıyı taşıyor (bedeli ölçüldü: −0.027).

**TEMİZ VE KİLİTLİ ÖLÇÜM (WORK seti: 541 parça, sızıntısız, kilitli holdout hariç; RF; nested-CV eşik):**

| Split | 13 feature | **zengin gate** | kazanç |
|---|---|---|---|
| part-out | 0.7264 | **0.7643** | +0.038 |
| **geometry-out (en katı)** | 0.6820 | **0.7273** | **+0.045** |
| family-out (STEP PRODUCT ailesi) | 0.7220 | **0.7560** | +0.034 |

Kazanç **üç split'te de tutarlı**, en katı split'te en büyük → sinyal gerçek, ezber değil.
Aday-recall 0.800 → **oracle-F1 tavanı 0.889** (mükemmel gate bile bunu aşamaz).

**(şişkin, tarihsel) ilk ölçüm:** canonical-823 + ensemble ile ALL 0.7886 part-out / 0.7693 aile-out.
Receipt: `results/rich_gate_receipt.json` · doğrulanmış tablo: `results/p0_verified_baselines.json`.

### Nasıl: TEMSİL zenginleştirmesi (gate'in gördüğü bilgi), model/etiket değil
13 el-yapımı feature'ın kör noktası: **adayın parçada NEREDE olduğu yok** (`outward` sadece kendi
ekseni boyunca ölçüyor). Vida üstte, tel önde — bu, kullanıcının haftalar önce yakaladığı
"vidadan ok çıkıyor" kusurunun sayısal karşılığı. Eklenen 41 feature:
- **A konum (9)** — mesh bbox içinde normalize konum + 6 yüze uzaklık → **en büyük katkı**
- **B çok-yarıçap (24)** — 3/6/10/15mm'de sınıf-olasılık profili + yoğunluk → **ikinci**
- C taper (5) → **ÖLÜ (negatif)** · D eğrilik (3) → ~0

---

## 🔪 DÜRÜSTÇE ÖLDÜRÜLENLER (hepsi ölçümle, iddia edilmeden)
1. **Eksen-kümeleme:** hipotez DOĞRU (wire-CP'ler boyut-eşleşmiş null'dan +0.058 fazla ortak eksen
   paylaşıyor, parçaların %78'i) **ama 13 feature zaten taşıyor** (+0.0036 AUC). FBI teşhisimin
   "gate'te eksen bilgisi yok" kısmı etkide yanlıştı.
2. **Skor havuzlama (√N argümanı):** +0.011 AUC ama top-N F1 **+0.002** → eşik altı.
3. **"FP'lerin %60'ı yapısal olarak öldürülebilir":** null-düzeltince fazla sadece **+0.031** =
   **taban-oranı artefaktı.** Kendi bulgumu iddia etmeden yakaladım.
4. **Huni/taper (Einführtrichter):** B-rep koni testi AUC 0.504, mesh taper profili negatif → çift
   bağımsız ölüm.
5. **Bağlam feature'ları** (PCA, rank, yoğunluk, ayna-eş): +0.003, marjinal.

---

## 📌 FAZ B İÇİN HAZIR (insan işi — henüz YAPILMADI, kullanıcı kararı)
**P1-a ölçümü GEÇTİ:** surviving-FP kütlesinin **%86'sı en büyük 40 ailede**; çok-parçalı ailelerde
FP-oranı std **0.001** (aile-içi davranış birebir sabit) → aile başına **tek** karar tüm aileyi
temizler. Yani gerekirse **40 hedefli evet/hayır** yeter; kör boyama değil.

---

## 🔒 FAZ 2 / P0 — ÖLÇÜM KİLİTLENDİ (kullanıcı listesi)
- **Sızıntı bulundu ve ölçüldü:** `BA_ALLOW_SEEN=1` (canonical `f1_sweep.py` de kullanıyor) segmentasyon
  eğitiminde görülmüş 153 parçayı skorlamaya sokuyor. Bedeli: ALL 0.7529 → **0.7264** (−0.027).
- **Aileler STEP `PRODUCT`'tan:** 748 aile (541 parça gerçek ürün-adı, gerisi parça-no fallback).
- **Geometry-hash:** 603 geometri, **345 duplike parça** → geometry-out split şart (en katı sonuç budur).
- **KİLİTLİ HOLDOUT ayrıldı:** 129 parça / 116 aile (`sha256(family)%100<20`). **P7'ye kadar hiçbir
  eğitim/eşik/seçim kararında kullanılmaz.** Çalışma seti 541 parça.
- **Frame denetimi:** `rich_feats` içi tutarlı (hepsi mesh frame) ✓. **BULGU:** `build_wei_aggr` /
  `build_aggr_rich` havuzlarında `P`=JSON frame ama `dir`=MESH frame → ikisini BİRLİKTE kullanan
  feature yazılmamalı (P0-a analizim yalnız `dir` kullandığı için geçerliydi, yine de belgelendi).
- **candidate-recall vs oracle-F1 ayrıldı:** WORK ALL recall 0.800 → oracle 0.889 (WEI 0.852→0.920).

## 🔓 GECE 2. HAMLE: P2/P3 test edildi ve KURALA GÖRE İPTAL
Agresif çok-çözünürlüklü havuz (WEI, aday-tavan recall **0.965**, mükemmel-gate tavanı 0.982) +
zengin gate = top-N **0.697** / aile-out 0.688 — **WEI 6k bazının (0.7296) ve kullanıcı eşiğinin
(0.723+0.020=0.743) ALTINDA** → *"End-to-end kazanım en az +0.020 değilse ALL multires koşusunu iptal et"*
kuralı gereği **ALL multires İPTAL**. Bulgu: **aday çoğaltmak yol değil** — recall tavanı açılıyor ama
precision seyrelmesi kazançtan hızlı büyüyor.

## (tarihsel) gece 2. hamle gerekçesi
F1'i şu an sınırlayan artık gate değil, **aday recall tavanı 0.808** (mükemmel gate bile max 0.894).
P2 (agresif çok-çözünürlüklü aday, recall tavanı 0.969) daha önce **zayıf gate** yüzünden askıya
alınmıştı (WEI top-N 0.662 < baz 0.723). **Bu gece gate materyal güçlendi (AUC 0.88 → 0.955)** →
P2'nin ön-koşulu karşılandı, yeniden test ediliyor: `build_aggr_rich.py WEI`
(smoke: 7.57x aday, aday-tavan recall **1.000**).
Tutarsa F1 tavanı 0.894 → ~0.98 açılır ve 0.85 aritmetik olarak mümkün hale gelir.

---

## 🎯 0.85'E KALAN YOL (dürüst)
- Şu an: **0.789** (parça-out) / 0.769 (aile-out). Kalan: **~0.06**.
- Gate ayrımı artık AUC 0.955 — asıl kısıt **aday tavanı**. P2 tutarsa tavan açılır.
- P2 + zengin gate yetmezse kalan meşru kaldıraç: **P1-c 40-soru adjudikasyon** (Faz B, insan 1 gün).
- Değişmeyen dürüstlük: 0.85 garanti değil; her adım ölçülüp raporlanıyor.

## 📁 Üretilen dosyalar
`build_rich_feats.py` · `analyze_rich.py` · `eval_rich_deploy.py` · `push_085.py` ·
`build_aggr_rich.py` · `p0a*/p1x*/p1a_*.py` (ölçüm scriptleri) ·
`results/rich_feats.npz` · `results/rich_gate_receipt.json` · `ROAD_TO_085_FBI.md`

> **Deploy NOT edildi:** `cp_config.json` hâlâ donmuş 13-feature ürünü tanımlıyor. Zengin gate'i
> ürüne almak senin kararın (receipt hazır).
