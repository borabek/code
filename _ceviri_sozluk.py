# -*- coding: utf-8 -*-
"""TR -> EN sozlugu. YALNIZ ayirt edici, alana ozgu terimler.

Kural: tek harfli / kisa / belirsiz adlar (f, oku, yap, sec, kur, sha, rec)
DISARIDA -- kelime siniriyla bile carpisma riski var.
Uzun olan ONCE gelmeli (esle_macar, esle_detay'dan once).
"""
SOZLUK = {
    # --- cekirdek fonksiyonlar (cok cagrilan) ---
    "esle_macar": "match_hungarian", "esle_detay": "match_greedy",
    "esle_isaretli": "match_signed", "esle3": "match3",
    "karar_skoru": "decision_score", "karar_maskesi": "decision_mask",
    "adaylari_uret": "derive_candidates", "urun_cikti": "product_output",
    "kanonik_zincir": "canonical_chain", "kanonik_bloku": "canonical_block",
    "halka_normalleri": "ring_normals", "halka_isaret": "ring_sign",
    "kalabalik_bastir": "suppress_crowd", "kalabalik_maskesi": "crowd_mask",
    "pose_duzelt": "pose_correct", "aci_duzelt": "angle_correct",
    "uye_yonu_sec": "pick_member_direction",
    "yon_sozluk_sec": "pick_direction_from_dictionary",
    "adaylar": "candidates", "tepe_normalleri": "vertex_normals_at",
    "step_haritasi": "step_map", "parca_ici": "within_part",
    "tier_ata": "assign_tier", "kapi_sonrasi_zincir": "post_gate_chain",
    "yerel_cerceve": "local_frame", "birlesik_havuz": "merged_pool",
    "havuz_seyrelt": "thin_pool", "aday_etiketi": "candidate_label",
    "proje_etiketi": "project_label", "eksen_ornek": "axis_sample",
    "konum_uyar": "warn_position", "gate_sec": "pick_gate",
    "tahmin_et": "estimate", "kendini_dogrula": "self_check",
    # --- sik gecen isimler ---
    "olcum": "measurement", "olculdu": "measured", "olcut": "criterion",
    "parca": "part", "parcalar": "parts", "aday": "candidate",
    "havuz": "pool", "yon": "direction", "eksen": "axis",
    "isaret": "sign", "isaretli": "signed", "isaretsiz": "unsigned",
    "yanal": "lateral", "eksenel": "axial", "derinlik": "depth",
    "agiz": "mouth", "aciklik": "opening", "govde": "body",
    "kume": "cluster", "kumeleme": "clustering", "tohum": "seed",
    "kat": "fold", "bolme": "split", "sinav": "exam", "kunye": "metadata",
    "makbuz": "receipt", "gerekce": "rationale", "hukum": "verdict",
    "kanit": "evidence", "tavan": "ceiling", "taban": "baseline",
    "kapi": "gate", "esik": "threshold", "oran": "ratio",
    "guven": "confidence", "kesinlik": "precision", "gurultu": "noise",
    "sapma": "deviation", "hata": "error", "kayip": "loss",
    "egitim": "training", "cikarim": "inference", "topluluk": "ensemble",
    "yonlendirici": "router", "secici": "selector", "bekci": "guard",
    "sizinti": "leakage", "onbellek": "cache", "yedek": "backup",
    "surum": "version", "dagitildi": "deployed", "kapandi": "closed",
    "duman_testi": "smoke_test", "geri_al": "rollback",
    "manset": "headline", "sonda": "probe", "kol": "arm",
    "yogun": "dense", "seyrek": "sparse", "rejim": "regime",
    "uretici": "manufacturer", "marka": "brand", "korpus": "corpus",
}
