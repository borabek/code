# TODO: Robot F1 0.75 roadmap (thesis-aligned)

Bu repo zaten `MASTER_TEZI_7_GUN_F1_ACCURACY_90_PLANI.md` ve `TANINMAYAN_ROBOT_F1_075_TODO.md` içinde detaylı bir tez-sadık yol haritası içeriyor. Aşağıdaki liste, mevcut kod dosyalarına bakılarak robot F1 tavanını 0.75'e çekmek için en yüksek kaldıraçlı görevleri ve öncelikleri özetler.

## Öncelikli gerçeklikler

- `robot_cp.py` ürün hattının gerçek yolu; final ölçüm `robot_cp.extract` ile yapılmalı.
- Tez odaklı ana model: **semantic DiffusionNet** (`diffusionnet.py`, `train_cp.py` ve `metrics.py`).
- `cp_regressor.py` tipi doğrudan heat/offset/direction modeli ana tez yolu değil; ancak pose-refiner olarak ablation olabilir.
- Ölçüm sözleşmesi değişmez: **lateral ≤2 mm**, **workpiece işaretli yön ≤10°**, **axial ≤40 mm**.
- Değerlendirme one-to-one olmalı: greedy değil, `metrics.py` içindeki Hungarian eşlemesiyle.
- Mevcut tavanlar `TANINMAYAN_ROBOT_F1_075_TODO.md` ve `YOL_075_ROBOT.md` dosyalarında ölçülmüş; düzeltmeden önce bu tavanlar referans alınmalı.

## P0 — Ölçüm hattını kilitle

- [ ] `robot_cp.py` ve `metrics.py` kullanılarak gerçek ürün hattını çağıran tek canonical evaluator yaz.
- [ ] `sina_kume.py` / `robot_cp.adaylari_uret` ile `metrics.keypoint_report` arasında parity testi ekle.
- [ ] `metrics.match_predictions` ile one-to-one Hungarian değerlendirmesi zorunlu olsun.
- [ ] `D7` final veri sözleşmesini kilitle; bu veri finalden önce kullanılmaz.
- [ ] Artık `D6`/DEV sonuçlarını final kabul etme; sadece `DEV` olarak etiketle.
- [ ] Sonuçları `results/` içinde hash, config ve seed ile sakla.

## P1 — Aday havuzu / merge / dedupe düzeltmesi

- [ ] `robot_cp._vote2` ve `robot_cp.adaylari_uret` içindeki `vote_pool_mm`, `dedupe_mm` ve `cluster_mm` parametrelerini kontrol edilmiş ablasyonla seç.
- [ ] Aday havuzunu daraltmayıp geniş tut; gate sonrası pose seçimini de topluca ölç.
- [ ] `connector3d.py` tabanlı boundary peeling / component clustering mantığını doğrula.
- [ ] `cp_openings.connection_points` ve `robot_cp._vote2` tarafından aynı CP'nin birden fazla GT'ye düşmesini engelle.
- [ ] Yalnız `votes` gücünü değil, one-to-one candidate oracle’ı da ölç: genel `≥0.96`, high-CP `≥0.85` hedef.
- [ ] High-CP parçaların oracle'ı `TANINMAYAN_ROBOT_F1_075_TODO.md`'deki `0.4685` eşiğinden `≥0.85`'e taşımak için komşu CP yutma/merge davranışını düzelt.
- [ ] Yarıçapları mesh çözünürlüğü/yerel açıklığa göre ölçekleyerek sabit mm değil dinamik davranış dener.

## P2 — Semantic label üretimi ve temiz QA

- [ ] `thesis_remesh.py` ile 6k/12k remesh bantlarını gerçekçi şekilde sabitle; `5.500-6.500` ve `11.000-13.000` bandı sağlanmayan parçaları yeniden üret veya quarantine et.
- [ ] `cp_semantic_labels.py` benzeri bir kod ile JSON CP’den güvenilir semantik yüzey maskesi çıkar.
- [ ] `cp_semantic_labels.py` içinde yüzey/surface maskesi, connected component, local normal filtresi ve boundary ring ignore mantığı uygulansın.
- [ ] En az 100 parça için insan QA yap; üretici/morfoloji/CP yoğunluğu tabakalı olsun.
- [ ] `train_cp.py` ya da yeni `train_cp_semantic.py` wrapper ile tez-best DiffusionNet konfigürasyonunu kullan.
- [ ] Sınıf ağırlıkları sadece frozen train split ve remesh çözünürlüğüne göre yeniden hesaplansın.
- [ ] Çalışma sırasında `WSCAD-dev` CP Jaccard ve `source` macro class Jaccard raporları üret.

## P3 — Hedef-Domain eksiklerini kapatmak

- [ ] `diffusionnet.py` ile öğrenme hızı, dropout, block sayısı, eigenvector sayısı ve weighted NLL ayarlarını tez referansına göre sabitle.
- [ ] Eksik hedef-domain parçaları için `pseudo-label` değil, **hedefli aktif etiketleme** yap.
- [ ] FN/yanlış pose parça kümelerini DiffusionNet embedding ve geometri özellikleriyle ayır.
- [ ] Her parti için insan etiketleme saatine karşı oracle kazancını ölç.
- [ ] Rastgele etiketleme değil, `NIT/SE` ve yüksek-CP, square/spring-cage gibi zayıf genelleme kümelerini seç.

## P4 — Pose seçenek sözlüğü ve oracle

- [ ] `robot_cp.py` hattını, tek eksen yerine pose seçenek bankasıyla genişlet.
- [ ] Pose bankası kaynakları: tez için `v_o-v_s`, analitik silindir ekseni, mouth-boundary normal, planar yüz/slot ekseni, free-space ray ve mevcut metadata adayları.
- [ ] `p3c`/silindir seçici mantığını yeniden ölç; önce oracle ile `≥0.82` genel, high-CP `≥0.35` hedeflesin.
- [ ] Gerçek robot pose transformasyonuna `≥0.92` oracle hedefle, idealde `≥0.84` gerçek transform.
- [ ] Pose seçeneklerini oluştururken `point` ve `direction` Cartesian product'ını takip et; doğru yönü yanlış konuma kilitleme.
- [ ] Her seçenek için confidence ve abstention/fallback ekle.

## P5 — Ortak robot-hazır ranker ve assignment

- [ ] Aday ve pose seçeneklerini birleştirip tek model/score hattında değerlendir.
- [ ] Aday/pose üzerinde `manufacturer` bilgisi vermekten kaçın; geometri, segmentasyon güveni, component, B-rep ve pose tutarlılığı kullan.
- [ ] Conflict graph / bipartite assignment ile tek GT'ye birden fazla tahminin gitmesini engelle.
- [ ] Gate’i final eşikten ayrı tut; geniş proposal pool önce korunup sonra ortak ranker ile seçilsin.
- [ ] `D7`/outer fold üreticide worst-manufacturer ve high-CP performansına bak.
- [ ] `P5` için ara kapılar: `tespit ≥0.70 robot ≥0.35`, `tespit ≥0.80 robot ≥0.55`, `tespit ≥0.90 dönüşüm ≥0.84 robot ≥0.75`.

## P6 — Kanıtlanmamış limitleri zorlamadan denetle

- [ ] `MM_MAX`, B-rep aday sayısı ve top-k sınırlarını trace et; doğru seçenek sınırı bilen varsa artır.
- [ ] `EKSEN_ONCE`, `EKSEN_SONRA`, radius payı sınırlarını yalnız train/inner-DEV’de kalibre et.
- [ ] High-CP candidate oracle hâlâ düşükse 6k/12k uniform mesh pilotu yap; teze uygun, ancak compute bedelini raporla.
- [ ] Ensemble yalnız uçtan uca şimdilik son çare olsun; en az üç seed ile kişi seç.

## P7 — 0.75 gelmese bile ürün yolunu koru

- [ ] AUTO/REVIEW çalışma kipini ayır; yüksek güvenli CP'leri otomatik, düşük güvenli olanları operatör review’e gönder.
- [ ] AUTO coverage, FP maliyeti, operatör kazanımı ve latency ayrı KPI olsun.
- [ ] Fiziksel çakışma / erişilebilirlik güvenlik metriğini F1’den ayrı tut.
- [ ] WSCAD/catalog expected-count ancak metadata-assisted ürün kipinde kullanılmalı; STEP-only final ölçüm karıştırılmamalı.

## Kodu ve tezi bağlayan önemli hatırlatmalar

- Tez sadakati: `diffusionnet.py` tabanlı **semantic 5 sınıf** çözüm.
- `cp_regressor.py` ve `train_cp.py --backbone diffusionnet` yerine, tez metodunu doğrudan `train_cp_semantic.py` / semantic wrapper üzerinden çalıştır.
- Yalnızca `weighted NLL`/CE, HKS/XYZ ve `k_eig=64` ya da `128` gibi sınırlar üzerinde çalış.
- `thesis_remesh.py` ve split mirası `raw_parent_id + raw geometry hash + catalog family` ile korunmalı.
- `connector3d.py` tabanlı instance/patch pipeline robot CP'nin fiziksel doğruluğunu sağlamak için kullanılmalı.

## Mevcut kod dosyalarında öncelikli inceleme alanları

- `robot_cp.py` — product candidate generation + robot-ready geometry
- `connector3d.py` — instance segmentation, boundary peeling, cluster/normal hesapları
- `metrics.py` — one-to-one keypoint matching, F1/Jaccard/accuracy
- `diffusionnet.py` — thesis-model builder ve HPO space
- `train_cp.py` — train/dev split, corpus split, model wrapper
- `thesis_remesh.py` — remesh sabitliği ve split mirası
- `cp_semantic_labels.py` — semantic surface mask generator (yeni veya eksik)
- `infer_pipeline.py` — semantic label -> instance -> connector graph hattı

---

Bu dosya repo içinde kalıcı yol haritası olarak kullanılabilir. `TANINMAYAN_ROBOT_F1_075_TODO.md` ve `MASTER_TEZI_7_GUN_F1_ACCURACY_90_PLANI.md` en güncel detaylı referanslarınızdır; bu yeni dosya onların özünü kod bazında, robot F1 0.75 hedefine göre toparlar.