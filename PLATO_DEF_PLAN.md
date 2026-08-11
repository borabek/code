# PLATOYU DEF ETME PLANI — FBI sorgusu + tez analizi + evde koşulacak kesin leverlar
*(2026-07-23, analiz modu — komutlar EVDE koşulacak)*

## 0. Durum tespiti (bugün ölçülen, sağlam)
- Ürün = `recall_hard_s2.pt` + post-proc (cluster 5 / min_v 30 / vconf 0.5).
- Sızıntısız 145-parça WEI hakemi (`_hw_r3.txt`): **F1 0.571 / P 0.597 / R 0.547**.
- Ölü kaldıraçlar (ölçüldü): R3 recall boyama (gerileme), UNION ensemble, çok-ölçek 9k, vconf düşürme.
- Bariz teşhis: WEI FN'lerinin %89'u "segmentasyon körü" (ama → **§1'de bu sorgulanıyor**).
- **F1 platosu = RECALL** (WEI 0.55). Precision hakemlikle ~0.98 (robot-etkin).

---

## 1. FBI SORGUSU — kendi sonucumun DELİKLERİ (önce bunları kapat, sonra lever)
Askeri kural: hedefi vurmadan önce istihbaratın doğru mu? "89% kör" sonucu 3 varsayıma dayanıyor.

### D1 — Teşhis YANLIŞ YERE baktı olabilir (en kritik)
Üretici CP'si = KONTAKT, açıklığın ~**15mm İÇİNDE**. Model açıklığın AĞZINDA ateşler (v_o, ~15mm
DIŞARIDA). Eski teşhis "G'nin 8mm çevresi"ne baktı → model ağızda ateşlese bile **KÖR sayılmış olabilir**.
→ Eğer düzeltilmiş eksen-boyu arama "model ağızda ateşliyor" derse, bu **segmentasyon körü DEĞİL,
türetme/şekil hatası** = POST-PROC ile ucuz düzeltilir. **KOŞ:** `fbi_recall.py` (H1 satırı).
KARAR: H1 > %30 ise → derivation leverı gerçek, plato o kadar sert değil.

### D2 — WEI frame/eksen convention'ı
WEI yön hatası 24-90° (kararsız), PXC 0°. Eğer WEI `InsertDirection` işareti/frame'i bizimkinden
farklıysa, axis-aware eşleme SİSTEMATİK kaçırıyor → FN yapay şişiyor.
**KOŞ (evde):** her WEI parçada `align_frames` residual'ını yazdır (>2mm = şüpheli) + 10 WEI parçada
üretici InsertDirection ile bizim v_o-v_s yönünü elle karşılaştır.
KARAR: residual yüksek/işaret ters ise → eşleme düzeltmesi bedava recall verir (phantom FN).

### D3 — WEI FN'leri GERÇEK açıklık mı?
Üretici bazen tel girmeyen noktaları (köprü, test, dahili) CP listeler. Bunları model doğru
görmezse biz haksız cezalanıyoruz.
**KOŞ (evde):** 15-20 WEI FN'i render et (`infer_step_cp.py` + üretici CP işaretli), göz + domain
uzmanı: bu gerçekten tel girişi mi? KARAR: %X'i gerçek değilse recall tavanı o kadar da düşük değil.

---

## 2. TEZ ANALİZİ — Scheffler'in KULLANDIĞI ama bizim KULLANMADIĞIMIZ silahlar
Tezi tekrar taradım. Kritik: **tez tek-kol değil, ÜÇ-KOL pipeline (Abbildung 45).** Biz sadece 1 kol
kullanıyoruz. Kaçırdığımız kollar = bağımsız sinyal = domain-gap'i paylaşmayan recall kaynağı.

### T1 — 2D GÖRÜNTÜ KOLU (YOLOv6, tezin 3. kolu) ★ EN GÜÇLÜ FARKLI SİNYAL
Tez 6 kanonik görünümden (alt/ön/sol/üst/arka/sağ) 512×512 render alıp **YOLOv6 ile açıklık
bounding-box tespiti** yapıp 3D'ye projekte ediyor. Biz HİÇ kullanmadık.
- **Neden plato-kırıcı:** görüntü kolu tamamen FARKLI modalite — gölgeli render'da kavite/delik
  net görünür, mesh-segmentasyon domain-gap'inden BAĞIMSIZ. Model WEI mesh'ine kör ama render'da
  delik görünür. Union → körlüğü bypass eder.
- **Maliyet:** büyük (YOLOv6 eğit + 2D kutu etiketi). AMA tez-sadık (tezin ana katkısı bu füzyon).
- **Ucuz ön-test (evde):** klasik CV ile başla — 6 render'da koyu dairesel/dikdörtgen bölge tespiti
  (Hough/blob), 3D'ye projekte, model CP'leriyle union. İşe yararsa YOLOv6'ya yatır.

### T2 — CAD-GEOMETRİK KOL (step_openings) ★ ZATEN ELİMİZDE, BAĞIMSIZ, ÖĞRENİLMEMİŞ
`step_openings.py` = CAD'den doğrudan açıklık (silindir/slot/dikdörtgen), F1 ~73%, **öğrenilmemiş →
WEI domain-gap'i YOK.** Model WEI'ye kör ama açıklık geometride varsa CAD yakalar.
- **KOŞ:** `fbi_recall.py` (H2 + MODEL+CAD union satırı) — CAD, model-FN'lerinin %kaçını kapsıyor +
  union WEI recall'i ne yapıyor. KARAR: union R belirgin artarsa (>+0.10) → **plato KIRILDI**, üstelik
  etiketsiz + tez-sadık. Bu benim en yüksek-beklenti bahsim.
- Bilinen kör nokta: rect/spring-clamp (rect_clusters=True aç). Frame: STEP→JSON `align_frames`.

### T3 — MANIFOLD/MANIFOLDPLUS REMESH (tezin GERÇEK remesh'i)
Tez Manifold+ManifoldPlus kullanıyor; biz pymeshlab. Hafıza: "remesh tel tünellerini mühürlüyor."
Farklı remesh WEI açıklıklarını KORUYABİLİR (uniform 9k mühürü açmadı ama Manifold farklı algoritma).
- **KOŞ (evde):** 20 WEI FN-ağırlıklı parçada Manifold remesh → aynı model → recall. KARAR: açılırsa
  remesh leverı gerçek (eğitimsiz).

### T4 — Reframe: tez 0.90 CP F1 İDDİA ETMEDİ
Tez segmentasyon tezi: CableEntry IoU **0.472** (biz 0.705 — zaten GEÇTİK). Tez 0.90 CP F1 hiç
demedi. Yani **0.90 tezin ÜSTÜNDE bir hedef** — jüri için bunu böyle çerçevele (biz tezi geçtik,
0.90 stretch hedef). Bu bir "lever" değil ama savunma stratejisi.

---

## 3. KESİN LEVERLAR — öncelik sırası (evde koş, karar kriteriyle)
| # | Lever | Tip | Maliyet | Beklenti | Karar kriteri |
|---|---|---|---|---|---|
| **1** | `fbi_recall.py` (D1+T2 birden) | teşhis+CAD | ~15dk GPU | **YÜKSEK** | H1>%30 → post-proc açık; union R>+0.10 → CAD plato-kırıcı |
| **2** | CAD-union üründe kalıcı yap | ürün | küçük | orta-yüksek | fbi_recall olumluysa cp_openings'e CAD-union ekle, 145'te doğrula |
| **3** | D2/D3 ölçüm bütünlüğü | audit | ~1sa | orta | phantom FN varsa recall tavanı yükselir |
| **4** | T3 Manifold remesh | remesh | ~2sa | orta | 20 parçada recall açılırsa yay |
| **5** | T1 2D-CV ön-test (klasik) | yeni kol | ~yarım gün | orta-yüksek | render'da açıklık tespiti union'a katkı verirse YOLOv6'ya geç |
| **6** | Focal loss retrain | eğitim | ~1gün | düşük-orta | sadece etiket gelince; hard-vertex recall |
| **7** | WEI tam etiket kampanyası | insan | 4-8sa uzman | **kesin ama pahalı** | 0.90'ın garantili yolu, elektrik mühendisi şart |

## 4. Askeri mantık: hangi sırayla saldır
1. **İSTİHBARAT ÖNCE (Lever 1+3):** "89% kör" doğru mu? fbi_recall + frame audit. Belki plato
   sandığımız kadar sert değil (D1/D2 phantom FN'i açarsa).
2. **BEDAVA BAĞIMSIZ SİNYAL (Lever 2, T2):** CAD-union — etiketsiz, tez-sadık, domain-gap'siz.
   Bu tek başına recall'ı kımıldatabilir. EN İYİ risk/getiri.
3. **YENİ MODALİTE (Lever 5→T1):** 2D görüntü kolu — tezin asıl füzyonu, en bağımsız sinyal,
   ama en pahalı. Sadece 1-2 tıkanırsa.
4. **SON ÇARE (Lever 7):** tam etiket — kesin ama uzman + haftalar.

## 4.5 FBI SADAKAT DENETİMİ — her lever kod+tez bağlılığına vuruldu
Kriter: (a) tez BUNU yapıyor/öneriyor mu, yoksa scope-creep mi? (b) certain-source kısıtı / kod
tabanı ihlali var mı?

| Lever | Tez sadakati | Kod/kısıt | VERDICT |
|---|---|---|---|
| **fbi_recall.py teşhis (D1)** | ölçüm — nötr | mevcut araç, eğitim yok | ✅ TUT (saf ölçüm) |
| **Ölçüm bütünlüğü (D2/D3)** | ölçüm — nötr | align_frames/render | ✅ TUT (saf audit) |
| **Manifold/ManifoldPlus remesh (T3)** | ✅✅ tezin GERÇEK remesh'i (biz pymeshlab'e SAPTIK) | Manifold derlemek gerek | ✅ TUT — sadakati ARTIRIR |
| **WEI tam 5-sınıf etiket (T7)** | ✅✅✅ tezin ÖZ yöntemi + feedback-loop önerisi | tam destekli, certain-source | ✅ TUT — en sadık |
| **YOLOv6 2D görüntü kolu (T1)** | ✅✅ tezin 3. kolu (Abbildung 45) | büyük iş + 2D kutu etiketi | ✅ TUT ama pahalı |
| **CAD-union üründe (T2)** | ⚠️ TENSION: tez CP'yi SEGMENTASYONDAN türetir; step_openings klasik CAD = tezin YERİNE koyduğu eski yöntem. "Çok-kaynak füzyon" ruhu var ama CP-türetme yöntemi DEĞİL | inference-time union, kısıt ihlali YOK; ama step_openings kendisi ~73% + rect kör | ⚠️ ŞARTLI — robot-ürün leverı olarak TUT, TEZ başlığı yapma; fbi_recall olumluysa "çok-kaynak füzyon" diye çerçevele |
| **2D-CV kısayolu (Hough/blob)** | ❌ tez YOLOv6 (öğrenilmiş) kullanır; klasik CV = hack, hikayeyi bulandırır | güvenilmez | ❌ **ELE** — görüntü kolu yapılacaksa YOLOv6 düzgün yapılır |
| **Focal loss (T6)** | ⚠️ tezin HPO/loss kapsamında ama spesifik önerilmez | kolay, certain-source | 🔸 TUT ama DÜŞÜK öncelik (89% körse reweight çözmez) |

**ELENEN:** 2D-CV klasik kısayolu (tez-dışı hack).
**DEMOTE:** CAD-union → tez başlığı değil, robot-ürün/füzyon leverı (tez-narratif tension'ı açıkça yaz).
**SADAKAT SIRASI (yüksek→düşük):** tam etiket > Manifold remesh > YOLOv6 kolu > (ölçümler nötr) > focal > CAD-union(şartlı).

## 4.6 EVDE KOŞMA SIRASI (sadakat + plato-getiri + maliyet dengesi)
1. **İSTİHBARAT (bedava, nötr-sadık):** `fbi_recall.py` + D2 frame audit → plato gerçek mi/phantom mu.
2. **Manifold remesh (yüksek sadakat, orta maliyet):** tezin gerçek remesh'i, tünel-mühürünü açabilir.
3. **CAD-union (ŞARTLI, sadece #1 olumluysa):** bedava recall ama tez-narratif tension'ı ile.
4. **YOLOv6 kolu (yüksek sadakat, pahalı):** #1-3 yetmezse tezin 3. kolu.
5. **Tam etiket (en sadık, kesin, uzman gerek):** 0.90 garantili yol.

## 5. Bu turda YAPILMAYANLAR (disiplin)
- Random wscad / pseudo-label eğitimi ASLA (certain-source kısıtı).
- locked-11 test setine dokunma.
- Karar sadece sızıntısız 145-parça hakemde (insan-val metrik illüzyonu üretir — R3 kanıtı).
- Tek seed'e ürün deme; 3 seed + audit_sets.
