# MÜFETTİŞ RAPORU — v2 (DÜZELTİLMİŞ)
**Tarih:** 2026-07-31 · **Kapsam:** (A) tezin kendi seçenekleri, (B) tezin yolunu sürdüren dış işler

> ## ⚠️ v1 GERİ ÇEKİLDİ
> İlk raporda üç madde önerdim (M1 yönelim, M2 spektral taban, M3 ön-eğitim). Kullanıcı
> "bunlar tezde geçiyor mu, emin ol; M3 zaten 3 hafta önce baştan başlama sebebimiz değil miydi"
> diye sorunca **tez metnini ve kendi makbuzlarımızı açtım: üçün ikisi ölü, biri de yanlış
> etiketlenmişti.** Aşağıda düzeltilmiş hâli.
>
> **Kök neden:** listeyi *"tezin hangi seçeneklerini denemedik"* diye kurdum. Doğru soru
> *"kanıt neyi destekliyor"* imiş. Bu, tam olarak "listelerin yarısı ölü çıkıyor" şikâyetinin
> mekanizması.

---

## 0. SONUÇ — tek madde de ÖLÇÜLDÜ ve DÜŞTÜ (2026-07-31 akşamı)

`n_eig` 128 eğitildi (tohum 0, `recall_hard_keig128_s0.pt`) ve **alınmadı**. Üç belirti aynı yöne:

| ölçüm | keig96_s0 (mevcut) | keig128_s0 | not |
|---|---|---|---|
| val Conn-IoU | **0.6378** | 0.6237 | altında (tohum yayılımı 0.025 içinde) |
| tepe→son val düşüşü | 0.015 | **0.042** | **tezin uyardığı aşırı öğrenme belirtisi** |
| VAL uçtan uca **tespit** | **0.7273** | 0.7242 | **−0.0034, gürültü** |
| VAL uçtan uca robot | 0.3853 | 0.4086 | +0.0236, gürültü (ikincil metrik) |

Önceden yazılı kill: tespit F1 +0.01 → gelen **−0.0031** → **ALINMAZ**.
Makbuz: `results/s1_keig128_val.json`, `results/keig128.log`.

**Kazanılan bilgi (koşu boşa gitmedi):** tez §1945 ">64 özvektör aşırı öğrenmeye yol açar" diyordu.
Biz **96'da bunun tersini** görmüştük (F1 0.383→0.477, 3/3 tohum). **128'de tezin uyarısı tuttu.**
Yani tez yanlış değil, **sınırı yanlış yerde çizmiş**: tavan 64'te değil, **96 ile 128 arasında**.
96'da durmak artık kanıtlı bir karar.

---

## 1. (tarihçe) AYAKTA KALAN TEK MADDE — artık düştü

### S1 — Spektral tabanı 96'nın ötesine taşı (`n_eig` 128)

**Kendi kanıtımız:** `n_eig` 64 → 96 bizim **tek plato kıran** hamlemizdi: F1 **0.383 → 0.477**,
üç tohumda üçü de kazandı. Sonra orada durduk.

**Mekanizma:** DiffusionNet spektral alanda çalışır; temsil edebileceği en ince yapıyı köşe
sayısı değil **özvektör sayısı** belirler. Bizim kaçırdığımız şey küçük açıklıklar — yani ince yapı.

**Tez ne diyor (dürüstçe):** `n_eig` uzayı `32/64/128/256/512/1024` (Tablo 3) — yani 128 uzayda.
**AMA** tez §1945 açıkça şunu ölçmüş: *"64'ten fazla özvektör... aşırı öğrenmeye yol açmıştır"*
ve en iyi modelini 64'te bırakmış. Yani bu madde **teze sadık değil, tezle bilinçli ayrışma.**
Meşruiyeti tezden değil, **kendi 3/3 tohumlu ölçümümüzden** geliyor (biz 96'da tezin bulgusunun
tersini gördük). v1'de buna "tez-sadık" demem yanlıştı.

**Maliyet:** 1 eğitim + yeni operatör önbelleği (`ops_k128`). **Kill:** VAL'de tespit F1 +0.01 yoksa düşer.

---

## 2. DENETİMDE ÖLENLER — hesap yapılmadan, makbuzla

| aday | neden öldü | kaynak |
|---|---|---|
| **Döndürme artırımı** | **Ölçülmüş, iki kez zarar vermiş.** `sw_v1_base (augment yok) 0.639` → `sw_v2_mildaug (0.35) 0.589`; not aynen: *"augment hurts quality even when mild"*. Agresif (1.05) ayar da zarar vermiş. | `PRODUCT_MODEL.md:389,393` |
| **HKS girdisi** | Tez §1502: HKS **katı olmayan** değişmezlik için önerilir ve *"bu çalışmada daha az öneme sahiptir"*. Bizim parçalarımız **katı** — yanlış alet. | Masterarbeit §1502 |
| **Etiketsiz ön-eğitim (v1'deki M3)** | v1'de dayanak yaptığım "+8.8 puan" **hiç ayrıştırılmamış**: kaydın kendisi *"'pretrain helps' is NOT yet isolated (scope and init both changed vs v22)"* diyor. Üstelik o dönemin tüm ölçümleri **CAD sözde-etiketlerine** karşıydı; onlar insan GT'siyle **F1 ~0.46** uyuşuyor. **3 hafta önce baştan başlama sebebimiz buydu.** | `hp-v25-record`, `ground-truth-is-the-ceiling` |
| **Tezin kayıp ağırlıkları (Tablo 4)** | Gerçek bir sapma (tez §1929: *"denenen DNN'den bağımsız olarak Tablo 4 ağırlıkları kullanılmıştır"*, biz `inv-freq` kullanıyoruz) **ama yönü aleyhimize**: normalize edilince tez CableEntry'yi 11.77, biz 13.14 veriyoruz — yani geçersek hedef sınıfı **düşürmüş** oluruz; LabelSurface ise 6.67'den 11.07'ye fırlar. | Masterarbeit Tablo 4 vs `train_seg_extra.py:185` |
| `c_width` 128–1024, `n_blocks` 4–5 | Tez §1945 aynı cümlede bunların da aşırı öğrenmeye yol açtığını ölçmüş. Bizim kaydımız da *"darboğaz kapasite değil"* diyor (M0: JSON'da F1 1.0, STEP'te 0.046). | Masterarbeit §1945 + `domain-gap-is-the-blocker` |
| `loss: ce` | Saf yeniden ağırlıklandırma; bizde bu sınıftaki tüm kollar öldü. | — |
| CageNet / PoissonNet / MeT | Yeni **mimari**, yeni **bilgi** yok: aynı mesh, aynı etiket. Yüksek maliyet. | — |
| B-rep graf hattı (UV-Net/AAGNet/BRepNet) | **Tezden sapma.** Fikirleri kayda değer ama bu rapor tez hattını inceliyor. | — |
| MFInstSeg (62.495 etiketli STEP) | Alan farkı: işlenmiş metal blok vs plastik klemens. **En pahalı dersimiz** alan farkı. | `domain-gap-is-the-blocker` |
| Robotik kablo/konnektör literatürü | Farklı problem: çalışma anında kamera+dokunma hizalaması. | — |

---

## 3. MÜFETTİŞ NOTU

Bu denetimin asıl çıktısı liste değil, **listenin kısalması**. Üç maddelik öneri, tez metni ve
kendi makbuzlarımız açılınca **bir maddeye** düştü — ve hiçbir hesap harcanmadan düştü.

Bu, bu oturumun genel desenini doğruluyor: **bizde ölen kollar, denenmemiş oldukları için değil,
zaten ölçülmüş oldukları için ölüyor.** Yeni bir kol önerilmeden önce sorulacak iki soru:

1. Bu daha önce ölçüldü mü? *(makbuzlara ve `PRODUCT_MODEL.md`'ye bak, hafızaya değil)*
2. Modele **sahip olmadığı bir bilgi** mi getiriyor, yoksa var olanı yeniden mi ayarlıyor?

---

## 4. Kaynaklar
- Masterarbeit_Scheffler.md — §1502 (HKS/rotasyon), §1676 & Tablo 3 (HPO uzayı), §1929 & Tablo 4
  (kayıp ağırlıkları), §1945 (>64 özvektör = aşırı öğrenme), Tablo 5 (en iyi hiperparametreler)
- `PRODUCT_MODEL.md:389,393` — augment A/B makbuzu
- hafıza: `hp-v25-record`, `ground-truth-is-the-ceiling`, `domain-gap-is-the-blocker`
- DiffusionNet — https://arxiv.org/pdf/2012.00888
- (tez-dışı, kayda geçsin) UV-Net — https://arxiv.org/pdf/2006.10211 · AAGNet —
  https://github.com/whjdark/AAGNet · CageNet — https://arxiv.org/html/2505.18772
