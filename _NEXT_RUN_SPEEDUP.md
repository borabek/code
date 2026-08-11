# Sonraki kosu icin GUVENLI hizlandirma (val'i BIT BAZINDA etkilemez)

## fwd_verts: 28000 -> 48000
- GPU su an sadece 1.6/4.3GB kullaniyor -- yari yariya bos.
- cp_regressor: "chunked==fused gradient identity holds" -> ileri-gecis batch boyutu
  gradyanlari DEGISTIRMEZ, val bit-ozdes. Sadece daha az kernel launch + dolu GPU.
- run_hp_v31_ftc6.yaml icinde `fwd_verts: 28000` -> `48000` yap.
- Bu kosuya uygulanmadi cunku config launch'ta okunur; degistirmek restart = 35dk
  prep kaybi, 10-epoch kosuda basabas. Sonraki TAM kosuda (v31->60 / v32) uygula.

## YAPMA (val'i etkiler veya bellegi patlatir):
- AMP/fp16: numerik degisir -> val etkilenir. KULLANMA.
- CP_PREP_WORKERS > 6: ucus-ustu bellek 2x -> commit patlar (2026-07-13 kanit).
- CP_LAZY_HIER kapatmak: 12GB hiyerarsi RAM'de -> commit patlar.

## Gercek darbogaz
Kismen lazy-hierarchy disk okumalari (bellek-guvenligi takasi) + CPU feature build.
Ikisini de degistirmek bellek/disk riski -> guvenli degil.
