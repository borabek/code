# GECE TUR-3 — hedef CP-F1 ≥ 0.82

**Başlangıç: 0.7817** (K2c, sızıntısız, gerçek boru hattı). **Gereken: +0.039.**

## Listenin dayandığı tek gözlem

Bu gece gate **tamamen değişti** (823 parça/min_v=30 → 1903 parça/min_v=4, +0.1273).
Ama boru hattındaki **diğer bütün ayarlar hâlâ ESKİ gate'e göre kalibre edilmiş durumda.**

Bu, bu geceki +0.1273'ü üreten mantığın aynısı — sadece bir bileşen ileri kaydırılmış hali.
Hiçbiri "yeni bilgi" değil, hepsi "bayatlamış kalibrasyon". Bu gece bu sınıf **3/3 ödedi**,
"yeni bilgi" sınıfı **3/3 öldü**.

## Ve bir de hiç bakılmamış yüzey

`vertex_conf`, `cluster_mm`, `ct_depth_min_mm` **yalnızca çok-CP'de** tarandı (K4b) ve orada
ölü knob çıktılar. **Düşük-CP'de hiç taranmadılar** — oysa düşük-CP korpusun **%89.5'i**.
Üç knob × %89.5 ağırlık = listenin en büyük tek yüzeyi.

---

## M1 — Düşük-CP post-işleme knobları  *(en büyük yüzey, %89.5)*

- [x] `vertex_conf`: **düşük-CP'de de ölü** (4 değerde de birebir aynı) → çok-CP bulgusu genelleşti
- [x] `ct_depth_min_mm` 2.0 düşük-CP'ye +0.009 ama çok-CP'yi yıkıyor (0.5355→0.3869) → 1.0'da kalır
- [x] `cluster_mm` 5.0 tek türetmede düşük-CP'ye +0.019 → rejim-koşullu aday
- [x] **M1c GERÇEK boru hattı: ÖLDÜ, −0.0254** (düşük-CP 0.9070 → 0.8786)

**M1 SONUÇ: ÖLÜ.** Asıl bulgu kaldıraç değil, **neden yanıldığım**: tek türetmede +0.019,
gerçek hatta −0.028 — **işaret ters**. Mekanizma: `cluster_mm` bir modelin İÇİNDE birleştiriyor,
union ise modeller ARASI çeşitlilikten değer üretiyor; içeride birleştirmek union'ın beslendiği
çeşitliliği yok ediyor ve tek türetmede bu görünmüyor.
Kural hafızaya yazıldı: `single-derivation-proxy-invalid`.

## M2 — `conn_promote` yeni gate altında

0.25 eski gate'le kalibre edildi. K5c tek türetmede 0.10'un çok-CP'de +0.05 verdiğini gösterdi.

- [ ] Yeni gate ile 0.10/0.15/0.20/0.25, rejim ayrımlı
- [ ] Kazanan gerçek boru hattında doğrulanır

**Neden ölmez:** ya kazanç ya K5c ölçümünün çürütülmesi.
**Beklenen:** 0 … +0.01

## M3 — Topluluk bileşimi yeni gate altında

`best_full` **eski gate ile** test edildi ve −0.047 kaybetti. Gate değişti; o test artık geçersiz.
Aynı mantık: bileşim, kendisinden sonraki bileşene göre seçilir.

- [ ] ürün-4 · +best_full(5) · çeşitli-4 — **yeni gate ile**, gerçek boru hattı

**Neden ölmez:** ya `best_full` artık kazanıyor ya eski sonuç yeni gate altında da doğrulanıyor.
**Beklenen:** −0.01 … +0.03

## M4 — Eşik, DAĞITILAN gate ile

Eşik 0.45/0.25, **OOF** gate skorlarında seçildi. Üründeki gate **tam-fit** (1903 parçanın
hepsiyle eğitilmiş) ve skor dağılımı biraz farklı.

- [ ] İnce ızgara (0.35–0.60, adım 0.025) dağıtılan gate ile, sızıntısız parçalarda

**Neden ölmez:** ya eşik kayıyor ya doğrulanıyor. Bu gece aynı iş +0.003 ve bir yanlış
dağıtım yakalaması getirdi.
**Beklenen:** 0 … +0.01

## M5 — Gate'i ÇOK-CP için ayrı eğit  *(rejimler zıt, tek model ikisine birden hizmet ediyor)*

Ölçülmüş gerçek: düşük-CP precision-sınırlı (0.956 P / 0.878 R), çok-CP recall-sınırlı
(0.690 P / 0.707 R). Tek gate iki zıt problemi çözmeye çalışıyor.
`min_v` ve eşik zaten rejime göre ayrıldı ve **ikisi de ödedi** — gate'in kendisi ayrılmadı.

- [x] Çok-CP'ye özel gate eğitildi (3522 aday), aile-dışı OOF
- [x] Çok-CP F1 0.6840 → **0.6940** (+0.010), ama **ağırlıklı yalnız +0.0010**

**M5 SONUÇ: ÖLÜ** — ve bir YAPISAL SINIR ortaya çıkardı:
**Çok-CP artık iğneyi oynatamaz.** %10.5 ağırlıkla, MÜKEMMEL bir çok-CP gate'i (F1 1.0) bile
ağırlıklı en fazla **+0.033** verir. Yani 0.78 → 0.82 için gereken +0.039'un **tamamı
düşük-CP'den gelmek zorunda.**

Düşük-CP'de kalan boşluk: 0.7677 → aday tavanı 0.946 = **+0.178**. Anatomisi (K2+K3):
gate'in ATTIĞI gerçekler tek-oy alanlar (`votes` AUC 0.86), TUTTUĞU yanlışlar ayırt edilemez
(AUC 0.62). Tek-oy sorununun çözümü gate'i zorlamak değil, **modellerin daha çok uzlaşması**
→ M3 (topluluk bileşimi) artık listenin tek gerçek adayı.

---

## Toplam ve dürüst hesap

| madde | beklenen |
|---|---|
| M1 düşük-CP knoblar | 0 … +0.03 |
| M2 promote | 0 … +0.01 |
| M3 topluluk | −0.01 … +0.03 |
| M4 eşik | 0 … +0.01 |
| M5 rejim-gate | 0 … +0.02 |

**Üst uç toplamı +0.10 → 0.88. Alt uç +0.00 → 0.78.**
0.82 için gereken **+0.039**, yani beş maddeden ikisinin orta bandı yeterli.

**Kill kriteri (hepsi, ölçümden önce):** korpus-ağırlıklı katkı < +0.01 → ürüne girmez
(bu turda bar 0.02 değil 0.01 — çünkü maddeler bağımsız ve toplanmaları hedefleniyor).
Her kazanan **gerçek boru hattında** ayrıca doğrulanır; tek-türetme ölçümü bu gece iki kez
yanılttı.

## Sıra

**M1 → M5 → M2 → M3 → M4**

M1 önce: en büyük yüzey (%89.5) ve hiç bakılmadı. M5 ikinci: en zayıf halka (çok-CP 0.70) ve
rejim ayrımı bu projede iki kez ödedi.
