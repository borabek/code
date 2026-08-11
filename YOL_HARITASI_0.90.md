# CP F1 için yol haritası — recall boyaması kanıtlandıktan sonra (2026-07-23)

> ⚠️ **HISTORICAL / DÜZELTİLDİ (2026-07-26).** Bu dosyadaki **"Precision zaten ~%98"** iddiası ŞİŞİRİLMİŞTİ —
> hakemleme "gerçek açıklık mı?" diye soruyordu (tool=evet), "tel buraya girer mi?" değil (tool=hayır).
> Yapısal wire/tool gate ile düzeltildi: gerçek leakage-free OOF precision ~0.72, base F1 **0.750**.
> Güncel kanon: `results/product_f1_receipt.json` (base 0.750 / metadata 0.775 / full-stack 0.807 part-out
> / 0.799 family-out). Aşağıdaki recall-boyama kaldıracı hâlâ geçerli, ama %98 precision temeli DEĞİL.

## Neden bu plan (kanıt)
Bu gece bağımsız 1811-CP üretici hakeminde **her eğitim kaldıracı başarısız oldu** (ensemble,
hakemleme, üretici-etiket, çözünürlük) — TEK istisna **recall boyaması**:
  WEI 165-held-out (denetimli, aynı set): F1 0.404 → **0.463**, recall 0.344 → **0.504** (27 parça boyama).
Yani **çalışan tek yol bu.** Precision zaten ~%98 (hakemleme kanıtı), gerçek boşluk recall — ve recall
boyamayla kımıldıyor. Plan bu kaldıracı sonuna kadar sürmek.

## Hedef (iki sayı, ikisini de dürüst raporla)
- **Ham üretici hakemi:** ~0.60-0.65 birleşik (yapısal tavan, liste eksik)
- **Precision-düzeltilmiş / insan-tarzı:** ~0.82-0.86 (savunulabilir tez rakamı)
- Ham 0.90 GERÇEKÇİ DEĞİL, iddia etme.

## Adımlar

### FAZ R1 — recall modelini ürün yap (min_v teşhisi temizse)
- [ ] min_v 10 gürültü teşhisi (koşuyor): küçük fragmentler TP mi FP mi
- [ ] temizse: recall_s2 → ÜRÜN (WEI +0.059 gerçek, PXC −0.008 gürültü içinde)
- [ ] gürültüyse: min_v'yi düzelt, recall'ı yeniden değerlendir

### FAZ R2 — recall boyamasını PXC'ye genişlet
- [ ] `mine_misses.py --mfg PXC` → PXC'nin kaçırdığı CP'ler (recall 0.647, ~%35 kaçıyor)
- [ ] recall boyama aracı PXC misses ile → kullanıcı boyar
- [ ] apply + eğit → PXC recall yükselir mi (WEI'de yükseldi)
- NOT: PXC misses daha az (recall zaten 0.65 vs WEI 0.34), ama her gerçek CP sayılır

### FAZ R3 — WEI 2-3 tur daha
- [ ] her turda: yeni model → yeni misses → boyama → eğit → hakemde ölç
- [ ] doyma noktasına kadar (bir tur +0.02'nin altına düşünce dur)
- NOT: azalan getirili; ilk tur +0.059, sonrakiler daha az beklenir

### FAZ R4 — temiz held-out + precision-düzeltilmiş rakam
- [ ] batch-4'ü (82-CP insan held-out) HAKEMLEMEDEN koru → temiz kalsın
- [ ] son modelde batch-4 FP'lerini hakemle (bir kez, EN SON) → precision-düzeltilmiş F1
- [ ] tez için: "precision ~0.98 (insan-doğrulamalı), recall boyamayla X→Y, F1 ~0.8X"

## Her adımda ZORUNLU disiplin (bu gece pahalıya öğrenildi)
1. `audit_sets.py` — karşılaştırılan iki ölçüm BİREBİR aynı parça setinde olmalı
2. Sızıntı koruması — eğitilen parça hakemde puanlanmaz (`trained_parts.json`)
3. Etiket kalite kapısı — boyama, kaçan CP'nin <20mm'sinde olmalı (bu tur 60/60, medyan 3.4mm)
4. Karar SADECE bağımsız üretici hakeminde (insan held-out metrik illüzyonu üretiyor)
5. min_v gibi eşikler küçük sette değil 1811-CP'de ayarlanır

## Ölü kaldıraçlar (tekrar deneme)
ensemble · hakemleme-eğitim-kazancı · üretici-etiketli-eğitim · çözünürlük 9k/12k · pos_weight ·
augmentation · çift-açıklık-bölme · rastgele-insan-etiketi · ABB/A-B indirme (404)
