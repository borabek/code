# ÇİZİM SÖZLEŞMESİ — robotun ürettiği each GLB'nin uymak zorunda olduğu 9 madde

**Amaç:** "çizim %100 doğru" iddiasını **denetlenebilir** hale getirmek. Ölçülemeyen a hedef
savunulamaz. Aşağıdaki 9 madde `glb_audit.py` tarafından **GLB dosyası geri okunarak** sınanır.

**Kapsam:** this sözleşme ÇİZİM SADAKATİ hakkındadır — i.e. GLB'de görünen each şeyin gerçeği
göstermesi. Modelin CP'leri doğru bulup bulmaması (F1) AYRI a konudur ve this sözleşmenin konusu
değildir. Yanlış bulunmuş a CP de **doğru çizilmek** zorundadır.

---

## Neden bağımsız denetçi

2026-07-28'de görselleştirme üç kez "düzeltildi" ve üçünde de kendi kendini "geçti" ilan etti.
Kök why: doğrulama, çizen kodun **içindeydi** ve aynı bozuk çağrıyı (`trimesh.contains`, rtree
yokluğunda each seferinde `ModuleNotFoundError`) kullanıyordu. `except` bunu yutunca doğrulama
"8/8 geçti" yazdı — hiçbir şey sınamadan.

**Kural: çizen kod kendini doğrulayamaz.** Denetçi, GLB'yi diskten geri okur; çizim sırasında
kullanılan hiçbir ara değişkene erişmez. Tek girdisi: `.glb` dosyası + `.receipt.json`.

---

## MADDELER

| # | Madde | Sınama |
|---|---|---|
| **M1** | Her üretici CP'si için **tam as a** işaretçi çizilir | GLB'deki yeşil+sarı küre sayısı = `n_gt` |
| **M2** | Her robot tahmini için **tam as a** işaretçi çizilir | GLB'deki kırmızı+mor küre sayısı = `n_pred` |
| **M3** | Her işaretçinin **ucu gövdenin dışındadır** | Denetçi, GLB'den gövdeyi ayırır ve each uç için kendi ışın-parite testini koşar |
| **M4** | Her işaretçinin **tabanı** ilgili CP noktasındadır | Küre merkezi − iğne yönü × boy = makbuzdaki nokta (tolerans 0.05 mm) |
| **M5** | İşaretçi yönü **fizikseldir**: gövdeden dışarı bakar | Uçtan geriye ışın gövdeyi keser; ileriye kesmez |
| **M6** | **Renk = eşleşme durumu** (yeşil/sarı/kırmızı/mor tanımına birebir uyar) | Renk sayımları makbuzdaki TP/FP/FN with tutarlı: yeşil=TP, sarı=FN, kırmızı=TP, mor=FP |
| **M7** | **Mavi çizgi sadece eşleşen çiftler** için vardır | Mavi bileşen sayısı = `tp` |
| **M8** | **Dosya adındaki F1 = rapordaki F1** | Dosya adı ayrıştırılır, makbuzla karşılaştırılır (tolerans 0.005) |
| **M9** | **Aynı girdi = aynı GLB** | Aynı parça iki kez üretilir, geometri hash'i birebir aynı olmalı |
| **M10** | **Fiziksel kusur işaretçileri AYRI kanaldadır** | `fiz_*` düğümlerinin sayısı makbuzdaki bayrak sayısıyla birebir; this düğümler M6 renk sayımına GİRMEZ |

---

## M10 — FİZİKSEL DEFECT KANALI (2026-08-05)

Ölçüldü (A4): yanlış pozitifler fiziksel kusurda **zengin** — gövde içi 3.16×, dar ağız
2.18×, önü kapalı 1.85×. Bu kusurlar F1'de görünmez but GLB'ye bakan insan için
"this nokta why yanlış" sorusunun doğrudan yanıtıdır.

**Neden ayrı kanal, why yeni renk değil:** M6 "renk = eşleşme durumu" diyor ve denetçi
renk sayımlarını TP/FP/FN with karşılaştırıyor. Kusurları yeni renklerle göstermek M6'yı
**bozardı**. Bunun yerine aynı noktaya yarı boyutta ikinci a küre konur, `fiz_<bayrak>`
adıyla. M6 aynen çalışır, M10 ayrı sayılır.

| bayrak | renk | anlamı |
|---|---|---|
| `govde_ici` | 🔴 kırmızı `(255,0,0)` | nokta gövdenin İÇİNDE — robot oraya giremez |
| `onu_kapali` | 🟠 turuncu `(255,140,0)` | ileride <5mm'de engel — tel giremez |
| `dar_agiz` | 🟡 sarı `(255,255,0)` | ağız iç çapı telden dar |
| `ters_yon` | 🔵 camgöbeği `(0,204,255)` | yaklaşma vektörü gövdeye bakıyor |

Bayrak otherwise hiçbir şey çizilmez — bayraksız GLB'ler **bit-özdeş** kalır.

## İHLAL POLİTİKASI

**Bir madde bile geçmezse GLB yazılmaz / silinir ve süreç error koduyla biter.**

Sessiz geçiş imkânsızdır:
- Denetçi hiçbir yerde `except: pass` kullanmaz.
- Denetçinin kullandığı geometri primitiflerinin (`cp_geometry.ray_hits` vb.) doğruluğu, analitik
  as bilinen sentetik şekillerle (`tests/test_cp_geometry.py`) ayrıca sabitlenmiştir — i.e.
  denetçinin kendisi de a yerden doğrulanır.
- Doğrulama yapamayan a denetçi **"geçti" diyemez**; "DENETLENEMEDİ" der ve this da ihlaldir.

---

## RENK TANIMI (M6'nın referansı)

| Renk | RGB | Kaynak | Anlam |
|---|---|---|---|
| 🟢 yeşil | `(0,210,0)` | üretici | CP bulundu (TP tarafı) |
| 🟡 sarı | `(255,210,0)` | üretici | CP kaçırıldı (FN) |
| 🔴 kırmızı | `(220,0,0)` | robot | prediction doğru (TP) |
| 🟣 mor | `(255,0,200)` | robot | prediction fazladan (FP) |
| 🔵 mavi | `(0,120,255)` | eşleşme | ağız↔yuva bağı |
| ⬜ gri | `(185,185,190)` | gövde | parçanın kendisi |

Renkler `robot_viz.py` ve `glb_audit.py` arasında **paylaşılan sabittir**; ikisi de `VIZ_COLORS`
sözlüğünden okur, elle yazılmaz.
