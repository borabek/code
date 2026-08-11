# RECALL LABELING HAND-OFF (P1.1) — hazır, boyama insana kalıyor (2026-07-25)

## Ne bu
Ürünün (keig96_s2) manufacturer ConnectionPoint'lerini KAÇIRDIĞI (segmentasyon-kör) açıklıklar mining'le
bulundu. Bunları boyamak recall tavanını açar — F1'in en büyük kalan kaldıracı. Certain-source: **konum
üreticiden (ground truth), şekil insandan** — random wscad DEĞİL, kısıta uyar.

## Rakamlar
- Mining: 150 WEI parçada 351 kaçan CP (`results/misses_wei.json`).
- **Arbiter-güvenli 102 parça / 245 kaçan CP** ayrıldı (48 parça 145-parça arbiter'la örtüşüyordu → ölçümü
  bozmamak için çıkarıldı; `results/misses_wei_safe.json`).

## Boyama nasıl (senin/uzman)
1. Aç: **`results/recall/paint_wei_night.html`** (Edge'de çift tık; 83MB, açılması biraz sürer).
2. Her parçada: kırmızı küre = üretici CP (parçanın içinde, kontakt); sarı ok = içeri yön. **Okun çıktığı
   yüzeydeki tel-girişi ağzını boya.** ⚠️ Tornavida/mandal (tool) ağzını DEĞİL — sadece telin girdiği ağzı.
3. Space = sonraki. Tarayıcı localStorage'a kaydeder (key `cp_recall_night`).
4. Bitince JSON'u indir.

## Boyama sonrası (ben)
`apply_recall.py` ile partial training label'a çevir → certain-source olarak eğitime ekle → sızıntı-korumalı
(trained_parts.json) → genişletilmiş hakemde OOF ölç. Beklenen: WEI recall +0.02-0.04 (recall boyama
geçmişi; plato riski var ama bu tur DOLU + arbiter-güvenli parçalarda).

## Disiplin
- Boyanan 102 parça EĞİTİM olur → hakemde puanlanmaz (145-parça arbiter korunur).
- tool/actuator ağzını boyama (wire-gate zaten onları eliyor; eğitime tool sokmak precision'ı bozar).

## ⭐ EN YUKSEK-ROI HEDEF GUNCELLEMESI (2026-07-25 analiz, P2.17)
Recall kaybi YUKSEK-CP terminallerinde yogun (results/recall_by_cpcount_2026_07_25.json):
- PXC 11-20 CP: recall 0.201 (262 kayip) · PXC 21+ CP: recall 0.047 (122 kayip) -- 24 parca, 384 CP kayip.
- WEI daha uniform (1-10 CP, 147 kayip); mevcut paket bunu kapsiyor.
ONERI: bir sonraki mining turu PXC 11+ CP terminallerini de hedeflesin (buyuk marshalling/cok-kutuplu, OOD).
Bu paket (WEI 102 parca) ilk tur; ikinci tur = high-CP PXC. En agir recall kaybi orada.

## 2. PAKET HAZIR: PXC high-CP (P2.17 en-agir-kayip hedefi) -- 2026-07-25
`results/recall/paint_pxc_highcp.html` (build ediliyor): 126 arbiter-guvenli PXC parca / 310 kacan CP,
--hard sirali (yuksek-CP once; 18 parca 11+ CP). Bunlar buyuk marshalling/cok-kutuplu terminaller,
recall 0.05-0.20 -- en agir kayip. ONCELIK: ilk siradaki yuksek-CP parcalar (en cok CP kazandirir).
Kaynak: results/misses_pxc_safe.json | key: cp_recall_pxc.
IKI PAKET birlikte ~+recall bekleniyor; boyama sonrasi apply_recall -> retrain -> OOF (823-parca) skor.
