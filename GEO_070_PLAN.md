# GEO F1 ≥ 0.70 HAREKÂT PLANI — yalnız plan (2026-07-29)

## Neden bu hedef zor ve neden yine de denenebilir

Üç ölçüm saf geometriye karşı: `geo_cp` 0.400, B-rep **özellikleri** +0.005, GEO-BREP 0.332.
Ama üçünün de ortak zaafı var: **tek açıklığa tek başına bakıyorlar.** Hiçbiri şunları kullanmadı:
parçanın bütünsel düzeni (DIN-ray çerçevesi, simetri, adım düzeni), eksen istikameti uzlaşması,
kanalın **neye bağlandığı**, ve STEP'in montaj/katı yapısı. Bu plan tam o el değmemiş bilgiye gidiyor.

Hedef aritmetiği: F1 0.70 ≈ P 0.65 × R 0.75. Yani:
- **Recall tavanı 0.58 → ≥0.75** (aday üretimi: yarık/push-in girişleri hiç aranmıyor)
- **Precision 0.26 → ~0.65** (289 FP'nin kimliği bilinmiyor — önce otopsi)

## İlkeler (geceden çıkan dersler, pazarlıksız)

1. **Önce teşhis, sonra icat.** FP otopsisi yapılmadan tek kaldıraç yazılmaz (TEL dersi).
2. **Kill kriteri F1 üzerinden, baştan yazılı** (AUC değil — TEL-B dersi).
3. **Dev/eval ayrımı**: tarama-ayar dev alt-kümesinde; eval alt-kümesine faz sonunda TEK bakış.
4. **Üretici-dışı sağlamlık her fazda** — geo'nun varlık sebebi bu; aile-dışıyla yetinme (ET dersi).
5. Her betik makbuz (JSON) yazar; görsel doğrulama sertifikalı GLB araçlarıyla.

---

## G0 — Ölçüm protokolünü kilitle (30 dk)

- [ ] Dev: 120 düşük-CP + 30 çok-CP parça (kilitli holdout HARİÇ). Eval: ayrık 120+30, dokunulmaz.
- [ ] Eşleşme: mevcut eksen-duyarlı greedy (perp tol + ±40mm), `big_arbiter` ile aynı — değişmez.
- [ ] Makbuz şablonu: her faz `results/geo070_<faz>.json` yazar (P/R/F1 + parça sayısı + tarih).
- **Kapı yok** — altyapı.

## G1 — FP/FN OTOPSİSİ (2-3 saat) ← her şey buna bağlı

- [ ] 289 FP'yi karakterize et: yarıçap, derinlik, **eksen yönü** (bbox eksenlerine göre),
      hangi yüzeyde (6 bbox yüzünden hangisine yakın/paralel), delik mi kör mü (öbür uç açık mı),
      en yakın TP'ye uzaklık, parça içi tekrar sayısı.
- [ ] Kümele (basit kural-tabanlı yeter): "üstten vida", "alttan montaj", "boydan boya geçen",
      "perçin/pim", "diğer". Her kümenin FP payını yaz.
- [ ] 50 FN'yi de karakterize et: kaçan GT'ler silindir mi değil mi (yarık payı = G2'nin gerekçesi).
- [ ] 10 parçayı sertifikalı GLB ile görselleştir (robot_viz'e geo-aday modu — sadece görsel, 30 dk).
- **KAPI 1:** FP'lerin ≥%60'ı ≤4 açıklanabilir geometrik imzalı kümeye düşmeli.
  **Düşmüyorsa (dağınıksa) → 0.70 ulaşılamaz, plan burada kapanır** ve kapanış makbuzu yazılır.

## G2 — RECALL TAVANI: aday üretimini tamamla (3-4 saat)

- [ ] **Yarık adayları**: B-rep'te paralel düzlem çiftleri (aralık 1-6mm, derinlik ≥2mm, yüzeye
      açılan) = push-in/yay-kelepçe girişleri. Bilinen rect-slot kör noktası tam burası.
- [ ] **Huni ağızlar**: silindire bitişik Cone/Torus yüzeyi = pahlı giriş; ağzı koninin dış ucuna taşı.
- [ ] Çift-ağız mantığını koru (her kanalın iki ucu ayrı aday, ray testiyle elenir).
- [ ] Aday-recall ölç (gate yok, sadece üretim): dev + eval ayrı raporla.
- **KAPI 2:** aday-recall ≥ **0.75** (dev'de) ve aday/GT ≤ 10.
  **Altında kalırsa** yarık parametreleriyle bir tur daha; yine olmuyorsa hedef 0.70'ten
  0.55'e resmen indirilir ya da kapanır — sessizce devam edilmez.

## G3 — UCUZ SONDALAR: hangi bilgi gerçekten mevcut? (her biri 15 dk, toplam 1 saat)

Kaldıraç yazmadan ÖNCE varlık kontrolü — yoksa o kaldıraç listeden düşer:

- [ ] **S1 çoklu-katı**: `gmsh.model.getEntities(3)` — STEP'lerde gövde + metal iletken AYRI katı mı?
      20 örnek parçada say. (Varsa C3 açılır: "kanal ikinci katıda bitmeli" çok güçlü bir süzgeç.)
- [ ] **S2 renk/malzeme**: STEP AP214 renk kaydı var mı (plastik gövde vs metal)? 20 parçada kontrol.
- [ ] **S3 DIN-ray izi**: arka yüzde 35mm standart ray oluğu B-rep'ten bulunabiliyor mu
      (paralel düzlem çifti ~35mm + kanca profili)? 20 parçada dene.
- [ ] **S4 GT eksen istatistiği**: üretici InsertDirection'ları parça başına kaç yöne dağılıyor?
      (Hipotez: 1-2 baskın yön; vida/montaj eksenleri dik.) 100 parçada histogram.
- Her sonda makbuza `mevcut/yok + kapsam %` yazar.

## G4 — PRECISION KALDIRAÇLARI (her biri ayrı ölçülür; 4-6 saat)

Sıra, G1 otopsisinin FP kümelerine göre yeniden dizilir. Her kaldıraç dev'de tek başına ölçülür:
**kabul = FP'yi ≥%15 azalt, recall kaybı ≤%3.** Geçmeyen anında düşer, ısrar yok.

- [ ] **C1 kanonik çerçeve + yüzey önseli** (S3 geçerse): ray oluğu = arka; tel girişleri kablolama
      yüzlerinde, montaj delikleri ray ekseninde → yanlış yüzdeki adayları bastır.
- [ ] **C2 eksen uzlaşması** (S4 geçerse): gerçek girişlerin eksenleri parça içinde 1-2 baskın yönde
      kümelenir; aykırı eksenli adaylar (dik vida delikleri) cezalandırılır. Öğrenme yok — çoğunluk oyu.
- [ ] **C3 ikinci-katı testi** (S1 geçerse): kanal, ekseni boyunca N mm içinde İKİNCİ katıya
      (metal iletken/kelepçe) ulaşmalı; plastikte kör biten veya boydan boya geçen delikler elenir.
- [ ] **C4 boydan-boya-geçme süzgeci**: iki ucu da serbest alana açılan delik = montaj deliği → ele.
      (ray_hits ile iki yönlü test; şimdiden yazılabilir, S gerektirmez.)
- [ ] **C5 tekrar/simetri önseli**: aynı yarıçap+eksenle adım düzenli tekrar eden adaylar ödüllendir
      (kutup dizisi); ayna-simetri düzleminde tek başına duran büyük delik cezalandır.
      DİKKAT: 2-CP parçalarda zayıf — rejim-ayrımlı ölç, düşük-CP'ye zarar veriyorsa yalnız
      yönlendirici çok-CP dediğinde uygula (CC-E yönlendirici hazır).
- [ ] **C6 huni/pah bitişikliği**: ağızda Cone/Torus komşuluğu zayıf sinyal (AUC 0.35 tek başına) —
      yalnız skor bileşeni olarak, tek başına süzgeç değil.
- [ ] (S2 geçerse) **C7 renk önseli**: giriş plastik gövdede açılır, metalde biter. Not: geometri
      dışına ilk adım — makbuzda ayrı işaretlenir, tez-sadakati tartışması kullanıcıya bırakılır.

## G5 — BİRLEŞİM + EŞİK (2 saat)

- [ ] Geçen kaldıraçları iki modda birleştir: (a) sert süzgeç zinciri, (b) ağırlıksız basit skor
      (her önsel ±1 oy — öğrenilmiş ağırlık YOK, yoksa "eğitimsiz" iddiası düşer).
- [ ] Dev'de tara; **eval'e TEK bakış**.
- **KAPI 3 (dev):** F1 ≥ 0.55 → devam. 0.45-0.55 → hedef resmen 0.55-0.60'a revize edilir.
  **< 0.45 → kol kalıcı kapanır**, kapanış makbuzu + hafıza kaydı.
- **KAPI 4 (eval, tek bakış):** F1 ≥ **0.70** hedef; ≥0.60 bile ML-yedek rolü için kayda değer.

## G6 — SAĞLAMLIK (1 saat) — geo'nun varlık sebebinin ispatı

- [ ] **Üretici-dışı**: PXC'de ayarla → WEI'de ölç ve tersi. Beklenti: fark ≈ 0 (eğitim yok).
      Fark büyükse süzgeçler gizlice üreticiye uymuş demektir — ihlal, geri dön.
- [ ] Determinizm: aynı parça 3 ayrı süreçte bit-aynı aday listesi (remesh önbelleği zaten var).
- [ ] Hız: ≤ 5 sn/parça.

## G7 — SONUÇ YOLLARI

- [ ] **Başarı (≥0.70 eval)**: ML ile füzyon ölç — (a) görülmemiş-üretici yedeği, (b) oy birliği
      modu (geo+ML aynı yerde → AUTO güveni artır), (c) ML adayı olmayan bölgede geo-recall katkısı.
- [ ] **Kısmi (0.55-0.70)**: yalnız yedek-rol makbuzu; ürün değişmez.
- [ ] **Başarısızlık**: kapanış kaydı — hangi kapıda, hangi sayıyla; hafızaya "üçüncü ve son kapanış"
      notu. Bir daha açılma şartı: G3 sondalarından YENİ bir bilgi kaynağının çıkması.

---

## Dürüst risk tablosu

| risk | erken sinyal | nerede yakalanır |
|---|---|---|
| FP'ler açıklanamaz/dağınık | otopsi kümeleri %60'ı örtmez | KAPI 1 (2-3 saatte belli olur) |
| Yarık adayları recall'u taşımaz | aday-recall < 0.75 | KAPI 2 |
| STEP'ler tek-katı / renksiz | S1-S2 boş döner | G3 (1 saatte belli olur) |
| Önseller üreticiye özgü çıkar | üretici-dışı fark büyük | G6 |
| Her şey çalışır ama 0.55'te doyar | dev F1 0.45-0.55 bandı | KAPI 3 → hedef revizyonu |

**Toplam tahmin: 14-18 saat**, ama KAPI 1 ve G3 ilk ~4 saatte "ulaşılamaz"ı gösterebilir —
plan en pahalı işi en sona koyacak şekilde dizildi.
