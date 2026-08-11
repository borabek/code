# GECE TUR-2 — ölmeyecek kaldıraçlar (2026-07-29/30)

## Nasıl bulundum: bu gecenin +0.1273'ünü üreten mantık

Kazanç şuradan geldi: **`min_v` 10 → 4 yapınca aday havuzu 4 katına çıktı, ama gate hâlâ
eski havuzla eğitilmiş modeldi.** Yani kendi iyileştirmem, kendisinden sonra gelen bileşeni
uyumsuz bıraktı.

**Genelleştirilmiş soru — ve bu turun tamamı bunun üzerine kurulu:**

> Boru hattında, **bugün değişen bir şeyin AŞAĞISINDA** kalan ve hâlâ **eski dağılıma göre
> kalibre edilmiş** başka ne var?

Bunlar "yeni bilgi" değil, "bayatlamış kalibrasyon". Bu gece ölen 3 kaldıracın hepsi yeni
bilgi ekliyordu; ödeyen 3'ünün hepsi bayat kalibrasyon düzeltiyordu.

## Boru hattı ve bugün ne değişti

```
STEP → mesh → segmentasyon → cp_openings(min_v 10→4 ★DEĞİŞTİ) → union(min_votes=1)
     → wire_gate(★YENİDEN UYDURULDU) → eşik(★0.45/0.25) → AUTO/REVIEW katmanı(0.66 ?)
                                    ↑
                          rejim router (aday sayısına bakar) → conn_promote(0.25 ?)
```

★ = bugün değişti. Soru işaretli olanlar **eski dağılıma göre kalibre**.

---

## N1 — AUTO katman eşiği 0.66  *(en acil, güvenlik ilgilendiriyor)*

`robot_auto_gate_threshold = 0.66` **eski gate'in skor dağılımında** ölçüldü (%95 precision
hedefi için). Bugün gate'i değiştirdim → skor dağılımı başka. 0.66 artık hiçbir şeyi garanti
etmiyor.

- [ ] Yeni gate'in skorlarında %95 AUTO precision veren eşiği bul (aile-dışı OOF)
- [ ] Kapsama ne oldu, raporla

**Neden ölmez:** ya eşik kayar (düzeltilir) ya aynı çıkar (doğrulanmış olur). CP-F1'i
değiştirmez ama **robotun neye otonom davrandığını** belirler.

## N2 — Rejim router  *(aday sayısı 4 katına çıktı)*

`results/highcp_router.pkl` girdisinde **birleşim aday sayısı** var (AUC 0.9945 ile eğitilmişti).
`min_v` 10→4 ile aday sayısı 4 katına çıktı → router'ın en güçlü girdisi kaydı.
Router `conn_promote`'un çalışıp çalışmayacağına karar veriyor.

- [ ] Router'ın yeni dağılımdaki AUC'si (aile-dışı OOF)
- [ ] Düştüyse yeni dağılımla yeniden eğit ve gerçek hatta ölç

**Neden ölmez:** router bozuksa `conn_promote` yanlış parçalarda çalışıyor demektir
(bu tam olarak 2026-07-29'da bir kez yaşandı, PXC.3273112 F1 0.273 → 0.710).

## N3 — `min_votes` (union) *(aday arzı 4 katına çıktı)*

`robot_min_votes = 1` (union) kararı, **eski aday arzıyla** verilmişti: union recall'u
maksimize ediyordu, precision'ı gate topluyordu. Arz 4 katına çıkınca denge değişmiş olabilir.

- [ ] `min_votes` 1/2/3 × yeni gate eşiği ortak taraması, rejim ayrımlı
- [ ] Gerçek boru hattında doğrula

**Neden ölmez:** ya union hâlâ doğru (doğrulanır) ya 2-oy daha iyi (kazanç).

## N4 — `conn_promote` 0.25

K5c tek türetmede **0.10'un +0.0505** (çok-CP) verdiğini gösterdi ama doğrulanmadı.
Ayrıca eşik eski aday dağılımında seçilmişti.

- [ ] Gerçek boru hattında 0.10 vs 0.25 (K4c düzeneği)

**Neden ölmez:** tek türetme ölçümü bu gece iki kez yanılttı; bu doğrulama ya kazancı onaylar
ya da o ölçümü de çürütür — ikisi de bilgidir.

---

## Sıra ve gerekçe

**N1 → N2 → N3 → N4**

N1 önce çünkü bugün gate'i değiştirerek onu **kendi elimle geçersiz kıldım** ve robotun
otonomi kararını etkiliyor. N2 ikinci çünkü bozuksa `conn_promote` yanlış yerde çalışıyor.
N3/N4 arz değişiminin doğrudan sonuçları.

## Beklenti — dürüst

Bunların hiçbiri +0.13 gibi değil. Gate uyumsuzluğu en büyük kalemdi ve kapandı.
Gerçekçi toplam: **+0.00 … +0.04**. Ama dördü de "ya düzeltir ya doğrular" sınıfında —
"ölçtük, hiçbir şey çıkmadı" ile bitmezler.

**Kill kriteri (hepsi için, ölçümden önce):** korpus-ağırlıklı CP-F1 katkısı < +0.02 →
ürüne girmez; N1 için ölçüt CP-F1 değil **AUTO precision ≥ 0.95**.
