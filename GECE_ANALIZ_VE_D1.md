# Gece analizi: hatalarım, ölümlerin nedenleri, ve bunlardan çıkan D1 listesi

## Bölüm 1 — Kendi hatalarım (altısı da ölçüm kodundan, veriden değil)

| # | hata | sonucu | yakalandı mı |
|---|---|---|---|
| 1 | Tek-türetmeli önbelleği **aday-oluşturma** kararlarında kullandım | 3 yanlış cevap, **2'sinde işaret ters** | evet, doğrulamayla |
| 2 | Kısa-yol AUC formülü (dengesiz sınıflarda şişiyor) | K7'yi 0.765 sandım, gerçeği **0.648** | evet, ama **sana yanlış rapor edilmişti** |
| 3 | İki gate'i **farklı aday havuzlarında** kıyasladım | canlı kaldıracı **ölü ilan ettim** (sonra +0.1273 çıktı) | evet, geç |
| 4 | Bir düzenekteki **farkı** başka düzenekteki **seviyeye** ekledim | sana 0.83–0.86 dedim, gerçek **0.78** | evet |
| 5 | Zehirli-parça koruması "asıldı" ile "elektrik kesildi"yi ayırmıyordu | masum parça kara listeye girecekti | evet |
| 6 | Hafızaya nedensellik iddiasını **abartarak** yazdım | `min_v` = +0.1273 sandım, payı **+0.021** | evet, atfetme ölçümüyle |

**Ortak imza:** altısı da *veriyi* değil *ölçme biçimini* yanlış kurmaktan doğdu. İkisi kalıcı
kurala dönüştü (`single-derivation-proxy-invalid`, `auc-formula-bug`).

## Bölüm 2 — Ölü kaldıraçlar, ÖLÜM NEDENİNE göre gruplanmış

### A. Parametre zaten optimumundaydı  *(8 kalem)*
`conn_promote` 0.25 · union yarıçapı 5.0 · `min_votes` 1 · gate eşikleri 0.45/0.25 ·
`vertex_conf` (tamamen ölü knob) · `ct_depth` 1.0 · `cluster_mm` 3.0 · eşik ince ızgarası

→ **Sonuç: knob yüzeyi TARANDI. Yeni bir sayı denemek artık bilgi üretmiyor.**

### B. Değişiklik birleşimin UZLAŞMASINI bozdu  *(3 kalem)*
`cluster_mm` 5.0 (model **içinde** birleştirme, modeller **arası** çeşitliliği yiyor) ·
`best_full` eklemek (−0.0995) · `keig128` ile 7 model (aday 371→472 ama GT kapsaması 85→83,
tek-oy %88→%96)

→ **Sonuç: topluluğun değeri UZLAŞMA. Aday sayısını artırıp uzlaşmayı düşüren her şey zarar
verir — daha çok aday, daha iyi değil.**

### C. Rejimin ağırlığı kalmadı  *(çok-CP tarafının tamamı)*
Rejime özel gate çok-CP'de +0.010 verdi → ağırlıklı **+0.0010**.

→ **Sonuç: çok-CP'nin TÜM kalan bütçesi +0.033. Mükemmel bir çok-CP gate'i bile gereken
+0.04'ü veremez. Oraya harcanan emek matematiksel olarak boşa gidiyor.**

### D. Kaldıracın TAVANI ölçülmemişti  *(2 kalem)*
CP sayısı öncülü: gerçek sayı bilinse bile tavan yalnız **+0.0089** ·
gate'i "daha çok parçayla eğit": ölçüm artefaktı (Bölüm 1 hata 3)

→ **Sonuç: bir kaldıraca emek vermeden ÖNCE tavanı ölç. Tavan barajın altındaysa kusursuz
uygulama bile yetmez.**

### E. Bilgi gerçekten özelliklerde yok  *(düşük-CP precision)*
K3: hayatta kalan 925 yanlış pozitifi en iyi ayıran özellik **AUC 0.62** · 568 aileye dağılmış ·
üretici kırılımı birebir aynı (%20/%20) · sızıntılı üst sınır yalnız +0.003

→ **Sonuç: düşük-CP precision mesh geometrisiyle kapanmıyor. Kanıtlanmış, tahmin değil.**

## Bölüm 3 — YAŞAYAN her şeyin ortak yanı

| değişiklik | gerçekte ne düzeltti |
|---|---|
| gate yeniden uydurma **+0.1273** | bileşen güncel boru hattına göre **bayattı** |
| `min_v` 10→4 **+0.0224** | filtre 30 vertex istiyordu, açıklıkların %82.7'sinde yoktu |
| AUTO katmanı **P 0.86→1.00** | karar verilmiş, config'e yazılmış, **koda bağlanmamış** |
| gate eşiği **+0.003** | dağıtılan değer **doğrulanmamış** |
| `channel_axis` **+0.0027** | algoritma yanlış eksene yuvarlıyordu |

**Beşinin de tek ortak yanı: yeni bir şey EKLEMEDİLER. Bozuk ya da bayat bir şeyi düzelttiler.**

---

# D1 LİSTESİ — analizle çelişmeyen, ölmeyecek kaldıraçlar

Her madde şu üç filtreyi geçiyor: **(A)** knob taraması değil · **(B)** uzlaşmayı düşürmüyor ·
**(C)** çok-CP'ye yatırım değil · **(D)** tavanı bilinen ya da tavan ölçümüyle başlıyor.

## D1.1 — `_snap_axis` hâlâ ÇÜRÜTÜLMÜŞ yöntemi kullanıyor  *(Yaşayan-sınıf, en somut)*

`cp_openings._snap_axis(d, p - bc)` ekseni **bbox merkezine** göre yönlendiriyor. Aynı depodaki
`cp_geometry.outward_along_axis`'in docstring'i bu yöntemi **açıkça çürütüyor**:
*"NEDEN BOUNDING-BOX DEGIL: kutu testi, noktanin parcanin neresinde durduguna gore isareti ters
cevirir; ayni yuzeydeki CP'ler zit yonlere savrulur (2026-07-28 gorsel hatasi)."*

Yani ürün, kendi kod tabanının yanlış ilan ettiği yöntemi çalıştırıyor. Yön, gate'in `outward`
özelliğini ve `outward_min` kapısını besliyor → **F1'i etkiler.**

- [ ] `_snap_axis`'in yönlendirmesini `outward_along_axis` ile değiştir
- [ ] Gerçek boru hattında ölç (gate'in `outward` özelliği değişeceği için gate de yeniden uydurulmalı)

**Neden ölmez:** ya düzeltir ya "bbox yönlendirmesi bu noktalarda zararsız" kanıtlanır.
Beş yaşayan değişikliğin dördü tam bu imzadan çıktı.

## D1.2 — Eğitim-zamanı birleşimi ile çalışma-zamanı birleşimi AYNI mı?  *(en ucuz, en yüksek sınıf)*

Kod tabanında **iki** birleşim uygulaması var: gate `f1_sweep.union_all` ile üretilmiş adaylarla
eğitildi, ürün `robot_cp._vote2` kullanıyor. Bir fark varsa gate **uyumsuz** demektir — ki bu,
bu gece +0.1273 ödeyen hatanın ta kendisi.

- [x] Null test (24 parça, 4 farklı türetme): **BİREBİR AYNI** — aday sayısı farkı 0,
      konum sapması **0.000000 mm**, oy dağılımları özdeş.

**D1.2 SONUÇ: doğrulandı, uyumsuzluk yok.** Varsayım kanıta dönüştü.

## D1.3 — Eğitim ile çalışma arasında sabit-değer denetimi

`dedupe_mm=10.0`, `ct_depth_min_mm=1.0` gibi değerler **hem** gate çıkarımında **hem** üründe
kodda sabit. Aynı olduklarını **varsaydım**, doğrulamadım.

- [x] Karşılaştırıldı. `dedupe_mm=10.0`, `ct_depth_min_mm=1.0` ve config'den gelen üç değer
      **eşleşiyor**. Ama biri eşleşmiyor:

| | `conn_promote` |
|---|---|
| gate eğitimi (`gate_regrow.py:144`) | **hiç geçilmiyor** → 0.0 |
| ürün çalışması (`robot_cp.py:146`) | **0.25** (çok-CP'de, yönlendirici kararıyla) |

- [x] Büyüklüğü ölçüldü (24 çok-CP parça): promote kapalı **94** aday → açık **350** aday.
      **256'sı (çok-CP adaylarının %73'ü) promote'tan geliyor ve gate onları EĞİTİMDE HİÇ GÖRMEDİ.**

**D1.3 SONUÇ: GERÇEK ve BÜYÜK uyumsuzluk.** Gate, çalışma anında çok-CP adaylarının %73'ünü
kör skorluyor. Çok-CP F1'inin 0.70'te çakılı kalması (düşük-CP 0.79–0.91 iken) bununla
tutarlı. Bu, bu gece +0.1273 ödeyen hatanın aynı sınıfı.

- [x] `gate_regrow.py` çalışma mantığını birebir taklit edecek şekilde düzeltildi
      (önce promote'suz türet → yönlendirici → çok-CP ise promote'lu yeniden türet)
- [ ] Yeniden çıkarım koşuyor; bitince gate uyumlu dağılımla eğitilip ölçülecek

## D1.4 — K7 renk kanalı, TAM ölçüm  *(analizin ZORUNLU kıldığı tek "yeni bilgi")*

Bölüm 2/E kanıtladı: düşük-CP precision mesh geometrisiyle kapanmıyor. Bölüm 2/C kanıtladı:
gereken +0.04 yalnız düşük-CP'den gelebilir. İkisi birlikte **yeni bilgiyi zorunlu kılıyor** —
yani K7 analizle çelişmiyor, analizin **sonucu**.

Durum: K7.1 çözüldü (renk 108/108, çerçeve 0.000mm) · K7.3 AUC kapısı **geçilmedi (0.648)** ama
ürün metriği erken sinyalde **+0.0195** · tam korpus çıkarımı koşuyor.

- [ ] Tam korpus (1903 parça) aile-dışı OOF, rejim ayrımlı
- [ ] Kapsamayı artır: veriye sorarak hangi rengin kontak olduğunu belirle (tek gümüş tonu
      sabit yazılmış, en az iki metalik gri var)
- [ ] **KILL: ürün metriğinde < +0.02**

**Neden ölebilir:** listedeki tek ölebilen madde — ve bunu açıkça yazıyorum. Ama Bölüm 2/E
gereği başka aday yok.

## Sıra ve gerekçe

**D1.2 → D1.3 → D1.1 → D1.4**

D1.2/D1.3 önce: dakikalar sürüyor ve "eğitim ≠ çalışma" sınıfı bu gece iki kez ödedi.
D1.1 üçüncü: somut, çürütülmüş bir yöntemin ürün yolunda durması.
D1.4 son: çıkarımı zaten koşuyor, bittiğinde ölçülecek.

## Dürüst beklenti

D1.1–D1.3 toplamı **+0.00 … +0.02** (uyumsuzluk bulunursa daha fazla). D1.4 **bilinmiyor**.
**0.82 yalnız D1.4 öderse gelir** — Bölüm 2/C ve 2/E bunu matematiksel olarak zorunlu kılıyor.
