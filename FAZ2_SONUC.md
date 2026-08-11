# FAZ 2 SONUÇ — CP F1 → 0.85 saldırısı (2026-07-28, otonom gece)
> Kullanıcı listesi `TODO_085_PHASE2.md` birebir yürütüldü. **Her kol kendi GO/NO-GO kapısıyla karara
> bağlandı.** Tüm sayılar leakage-free OOF, KİLİTLİ holdout hariç, nested-CV eşikli.

---

## 0. ÖLÇÜM TEMELİ (P0) — kilitlendi
| | |
|---|---|
| Çalışma seti (WORK) | **541 parça** (sızıntısız ∧ kilitli-değil), 3222 aday |
| Kilitli holdout | **129 parça / 116 aile** — `sha256(family)%100<20`, P7'ye kadar dokunulmadı |
| Aile kaynağı | STEP `PRODUCT` adı (748 aile; %81 tek-parçalı) |
| Geometri | 603 geometri, **345 duplike parça** → geometry-out split şart |
| Aday-recall | ALL **0.800** → **oracle-F1 tavanı 0.889** (mükemmel gate bile aşamaz) |

**Bulunan sızıntı:** `BA_ALLOW_SEEN=1` (canonical `f1_sweep.py` de kullanıyor) segmentasyon eğitiminde
görülmüş 153 parçayı skora sokuyor. **Bedeli ölçüldü: ALL 0.7529 → 0.7264 (−0.027).** Canonical 0.750
bu şişkinliği taşıyor. Gecenin ilk raporundaki **0.789 geçersiz** (sızıntılı set + kaba aile + ensemble).

---

## 1. KOL KOL SONUÇLAR (hepsi kapıya bağlandı)

| Kol | Kapı | Sonuç | Karar |
|---|---|---|---|
| **Zengin temsil** (konum + çok-yarıçap) | — | part-out 0.7264→**0.7643**, geom-out 0.6820→**0.7273**, family-out 0.7220→**0.7560** | ✅ **TUTULDU** (3 split'te tutarlı, en katıda en büyük) |
| **P1 embedding** (DiffusionNet 128-d) | ALL+0.015 / WEI+0.020 | GO teknik olarak geçti (MLP); **mutlak** kazanç 0.7560→**0.7641** (+0.008), WEI +0.026 | ⚠️ **AÇIK, mütevazı** |
| **P2 set/graph + listwise** | family-out +0.010, 3 seed | BCE −0.0055, listwise −0.0096 → **ALL'da kaybediyor**; ama **WEI 0.6813→0.729 (+0.048)** | ❌ ALL kapısı KALDI → yönlendirme testine dönüştü |
| **P3 agresif aday havuzu** | WEI 0.723 üstüne +0.020 | recall tavanı 0.965'e çıktı ama F1 **0.697** (< 0.743) | ❌ **KAPANDI** → ALL multires iptal |
| **P4 aile transferi** | ağırlıklı +0.010 | gerçek kardeşlerde **0.696** (< ML 0.756), ağırlıklı **−0.009** | ❌ **KAPANDI** |
| **P5 metadata** | bağımsız kaynak şartı | STEP PRODUCT CP sayısını **vermiyor** (%20 tam-eşitlik, korelasyon ~0) | ⚠️ **VARSAYIMSAL** — birincil sayı = BAZ; yön-prior **yasak** (kaynak yok) |
| **P6 insan adjudication** | FP'nin %60'ı ≤40 tipte | gerçek ailelerle **%19** (216 FP = 216 tip); aile-bağımsız arketip kümelemede **0 saf-FP kümesi** | ❌ **KAPANDI** |

---

## 2. EN ÖNEMLİ BULGU (bilimsel sonuç)

**Kalan hatalar bir modelleme eksikliği değil — geometride var olmayan bir bilgi.**
Bu gece üç bağımsız seviyede kanıtlandı:
1. **Bireysel:** surviving-FP'lerin feature dağılımı TP'lerle çakışık (Faz A).
2. **Ailesel:** FP'ler aile-slotlarında tekrar etmiyor (%19 ≪ %60; 216 FP = 216 tip).
3. **Arketipsel:** feature uzayında FP-ağırlıklı **hiç saf küme yok** (K=20 ve K=40'ta sıfır).

Yani "hangi açıklık tel girişi, hangisi alet/test deliği" sorusu, elimizdeki 3D temsilde
**ayrıştırılamaz**. Bu, hedefli insan etiketlemesini de ekonomik olarak imkânsız kılıyor
(50–100 soru değil, aday-aday binlerce soru gerekirdi).

**Ayrıca kanıtlandı:** aday çoğaltmak (P3) ve aile transferi (P4) **negatif** — yani sorun
"yeterince aday bulamamak" ya da "benzer parçadan kopyalayamamak" değil.

---

## 3. DÜRÜST DURUM

| Metrik (WORK, family-out, nested-CV eşik) | Değer |
|---|---|
| 13-feature taban | 0.7220 |
| **En iyi model (rf-rich + mlp-emb ensemble)** | **0.7641** |
| Oracle-gate tavanı (mükemmel ayırıcı) | 0.889 |
| Hedef | 0.85 |

0.85'e kalan **~0.086**; oracle tavanına kalan 0.125. 0.85 için gate'in neredeyse-oracle olması
(P≈0.95 @ R≈0.78) gerekirdi — yukarıdaki kanıt bunun elimizdeki bilgiyle mümkün olmadığını gösteriyor.

---

## 4. P7 — KİLİTLİ HOLDOUT, TEK KEZ AÇILDI ✅

**Kilitlenen konfigürasyon** (yalnız WORK'te seçildi): `prob: RF(rich) + SetNet(3-seed)`, eşik **0.34**.
Deploy-edilebilirlik kuralı uygulandı: **rank-ensemble ELENDİ** (rank tüm korpustan hesaplanır → eşik
yüzdeliğe döner → robot tek parça işlerken tanımsız). Bedeli yok: 0.7745 → 0.7739.

**129 parça / 478 manufacturer CP — hiçbir kararda kullanılmadı:**

| Alt küme | parça | GT | P | R | **F1 (3 seed)** |
|---|---|---|---|---|---|
| **ALL** | 129 | 478 | 0.832 | 0.679 | **0.7476** ±0.0008 |
| WEI | 26 | 89 | 0.804 | 0.629 | 0.7059 |
| PXC | 103 | 389 | 0.839 | 0.690 | 0.7569 |
| high-CP (≥11) | 3 | 56 | 0.844 | **0.125** | **0.2172** ⚠️ |

**Bootstrap %95 GA (ALL, 2000 tekrar, parça-bazlı): [0.693, 0.818]**
**AUTO precision 0.826** (n=403) · AUTO+REVIEW F1 0.756 · *REVIEW kümesi boş çıktı — mevcut eşikte
tier ayrımı çalışmıyor, düzeltilmeli.*

### Referans (aynı holdout, karar alınmadı — sadece kıyas)
| | holdout ALL F1 |
|---|---|
| 13-feature taban | 0.7271 |
| **zengin temsil (tek RF)** | **0.7475** |
| kilitli final (prob: RF+SetNet) | 0.7476 |

→ **Gecenin işi transfer etti: +0.020.** Ancak kazancın **tamamı zengin temsilden** geliyor;
SetNet'in WORK üstünlüğü (0.7685 vs 0.7527) holdout'a **taşınmadı**. Embedding ve ensemble katkıları
da gerçekten görülmemiş veride gürültü içinde. Kilitli holdout disiplini tam da bunu ayırt etti.

### 🚦 BAŞARI KAPISI: **GEÇMEDİ** — locked holdout ALL 0.7476 (hedef ≥ 0.85)

---

## 4.5 DÜRÜST KAPANIŞ — 0.85 neden gelmedi (kanıt zinciri)

**Gelinen nokta:** kilitli, hiç dokunulmamış holdout'ta **0.748** (taban 0.727). Hedefe **0.10** var.

**Neden gelmediği artık spekülasyon değil, ölçüm:**
1. **Aday tavanı:** union recall 0.800 → mükemmel gate bile **0.889**. 0.85 için gate'in
   neredeyse-oracle olması gerekirdi (P≈0.95 @ R≈0.78).
2. **Aday çoğaltmak çözmüyor:** agresif havuz recall'ı 0.965'e çıkardı ama F1 **düştü** (0.697).
   Precision seyrelmesi kazançtan hızlı büyüyor.
3. **Kalan FP'ler TP'lerden ayrılamıyor — üç bağımsız seviyede kanıtlandı:**
   bireysel (feature çakışması) · ailesel (216 FP = 216 tip, %19 ≪ %60) ·
   arketipsel (K=20 ve K=40'ta **sıfır** saf-FP kümesi).
4. **Bu yüzden insan etiketlemesi de ekonomik değil:** 50–100 hedefli soru kütlenin ancak %19'una
   dokunurdu; aday-aday binlerce karar gerekirdi.
5. **Aile transferi negatif** (0.696 < 0.756) — "benzer parçadan kopyala" da çözüm değil.

**Yorum:** kalan hata bir modelleme eksikliği değil, **elimizdeki 3D temsilde bulunmayan bir bilgi**.
"Bu açıklık tel girişi mi, alet/test deliği mi?" sorusunun cevabı geometride yok — tezin `Contact`
sınıfının tanımı gereği ikisini birleştirmesiyle de tutarlı.

**Ek bulgular (ürün için eyleme dönük):**
- **high-CP parçalarda çöküş** (recall 0.125): ayrı ele alınmalı, mevcut ürün bu sınıfta çalışmıyor.
- **REVIEW kümesi boş**: tier eşiği anlamsız kalmış, kalibre edilmeli.
- **Canonical 0.750 şişkin** (sızıntı −0.027) — tüm raporlamada düzeltilmeli.

## 5. ÜRETİLEN DOSYALAR
`results/split_lock.json` (tek doğruluk kaynağı) · `results/p0_verified_baselines.json` ·
`results/rich_feats.npz` · `results/embed_feats.npz` · `results/oof_scores.npz` ·
`results/p1_embed_eval.json` · `results/p2_setnet_eval.json` · `results/p4_transfer.json` ·
`results/rich_gate_receipt.json` · scriptler `p2_p0_*.py`, `p2_p1*.py`, `p2_p2*.py`, `p2_p4_*.py`, `p2_p5_*.py`, `p2_p6_*.py`
