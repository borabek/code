# TO-DO — STEP yüz rengi: 0.85'e giden tek ölçülmemiş yol

> Eklendi 2026-07-29. Bu liste, gece koşan dört kaldıraçtan **ayrı**. Onlar 0.77–0.80
> aralığına götürür; bu, duvarın kendisine saldıran tek fikir.

## Neden bu, neden tek

Bilgi duvarı bulgusu şunu ölçtü: modele **skorlandığı parçaları göstermek** yalnız **+0.003**
kazandırıyor, oysa aday tavanına **+0.151** var. Yani kalan hata veri, kapasite ya da
genelleştirme sorunu değil — **cevap modele gösterilen şeyde yok.**

Bugüne kadar denenen her şey mesh geometrisiydi ve hepsi öldü:

| sonda | CP-F1 katkısı |
|---|---|
| kanal profili (TEL-B) | +0.003 |
| B-rep yüzey bilgisi (TEL-G) | +0.005 |
| saf geometrik dedektör | en iyi 0.400 (ML 0.82) |
| sızıntılı üst sınır | +0.003 |

Ayırım **fonksiyonel**: tel girişi bir **kontakta** biter, alet ağzı bir kola/yaya. Geometri bunu
göremiyor (recon AUC 0.61). Ama STEP dosyası bunu **yazıyor**: AP214 `OVER_RIDING_STYLED_ITEM`
ile yüz başına malzeme rengi okunabiliyor — 108/108 yüzde doğrulandı, gümüş
`rgb(0.824, 0.824, 0.784)` her parçada metal.

Dolaylı kanıt da var: kanal dibindeki Contact olasılığı (`dip_CT`) gerçek girişlerde medyan
**0.43**, yanlış pozitiflerde **0.22** — hipotez yönü doğrulanmış. O sonda ürüne yansımadı çünkü
girdisi segmentasyon **tahmini**. Renk tahmin değil, **CAD'in kendi verisi**.

## Engel (dürüst)

| deneme | sonuç |
|---|---|
| gmsh ile taşımak | gmsh rengi **atıyor** |
| yüz sırasıyla eşleştirmek | null testinde **2/5** — kabul edilemez |
| OCC XCAF | renk tablosu yükleniyor ama **0 alt-şekil etiketi** |

Eksik parça: **geometri tabanlı yüz eşleştirici**. Bir saatlik iş değil.

---

## Adımlar

- [ ] **R1 — Yüz eşleştirici (asıl iş).** STEP'in B-rep yüzlerini mesh üçgenlerine eşle.
      Sıra/indis kullanma (denendi, 2/5). Yöntem: her B-rep yüzünün ağırlık merkezi + normali +
      alanı ile mesh üçgen kümelerini eşle; belirsiz kalanı **eşleşmedi** işaretle, uydurma.
      **KAPI:** null testi ≥ 4/5 (bilinen renkli parçada doğru yüze doğru renk).

- [ ] **R2 — Kanal dibi rengi özelliği.** Her aday CP için: ağızdan eksen boyunca ışın at, ilk
      çarptığı yüzün rengini oku. `dip_metal` = o yüz metal mi (gümüş paletine yakın mı).
      Ek: `metal_frac_6mm` (6 mm içindeki metal yüz oranı).

- [ ] **R3 — Ayırt ediciliği ölç (ürüne dokunmadan).** TP vs FP dağılımı + AUC.
      **KAPI:** `dip_metal` AUC ≥ 0.70 değilse dur — `dip_CT` zaten 0.61 veriyordu, renk bunu
      belirgin şekilde geçmiyorsa yeni bilgi getirmiyor demektir.

- [ ] **R4 — Gate'e ekle, sızıntısız ölç.** Aileye göre GroupKFold OOF, rejim ayrımlı,
      korpus ağırlıklı. **KILL:** korpus-ağırlıklı CP-F1 katkısı < **+0.02** → ölü.

- [ ] **R5 — Üretici-dışı sonda.** Kolay bölünmede kazanan, dağılım kayınca kaybedebiliyor
      (ExtraTrees dersi: aile-dışı +0.012, üretici-dışı −0.061). **Holdout harcamadan önce.**

- [ ] **R6 — Kapsam denetimi.** Kaç STEP'te renk gerçekten var? Renksiz parçada özellik
      `NaN` olmalı ve gate bunu tolere etmeli — sessizce 0 yazmak sahte sinyal üretir.

## Beklenti — dürüst

**Bilinmiyor.** Bu listedeki tek "bilinmiyor" ve değeri de o. Diğer dört kaldıracın tavanı
ölçüldü ve toplamı 0.80'i zor geçiyor. Renk kanalı ya duvarı kırar ya da R3'te ölür — ikisi de
öğrenmedir. R3 kapısı, R1'in emeğini boşa harcamadan önce karar verdirir.

## Nereden başlanır

**R1.** Öncesinde hiçbir şey ölçülemez. `step_face_colors.py` renk okumayı zaten yapıyor;
eksik olan tek şey yüzü mesh'e bağlamak.
