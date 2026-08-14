# -*- coding: utf-8 -*-
"""DOSYA ADI TR -> EN. Parca parca cevirir (kos_ -> run_, sonda_ -> probe_).

GUVENLIK:
  * yeni ad MEVCUT bir dosyayla ya da Python stdlib moduluyle CAKISAMAZ
  * `git mv` kullanilir -- gecmis korunur
  * ad degisimi TUM metin dosyalarinda kelime siniriyla uygulanir
    (import satirlari, "dosya.py" seklindeki string referanslar, .sh, .md)
"""
PARCA = {
 "kos": "run", "sonda": "probe", "duman_testi": "smoke_test",
 "geri_al": "rollback", "manset": "headline", "kanonik": "canonical",
 "zincir": "chain", "kume": "cluster", "kumesi": "set", "olcum": "measure",
 "adet": "count", "tahmini": "estimate", "tahmin": "estimate",
 "halka": "ring", "isaret": "sign", "havuz": "pool", "seyrelt": "thin",
 "esik": "threshold", "kayit": "record", "urun": "product",
 "yon": "direction", "bankasi": "bank", "secici": "selector",
 "birlesik": "merged", "agiz": "mouth", "tanimlayici": "descriptor",
 "tanim": "definition", "derin": "deep", "kafes": "lattice",
 "sira": "order", "damgala": "stamp", "protokol": "protocol",
 "uye": "member", "isin": "ray", "ekseni": "axis", "eksen": "axis",
 "konum": "position", "ortalama": "mean", "tezgah": "bench",
 "yeni": "new", "karar": "decision", "sinav": "exam", "parca": "part",
 "duzeyi": "level", "dagit": "distribute", "hizalama": "alignment",
 "etiket": "label", "kalitesi": "quality", "gelistirme": "improvement",
 "taramasi": "sweep", "tarama": "sweep", "kismi": "partial",
 "guven": "confidence", "kapisi": "gate", "hibrit": "hybrid",
 "ikinci": "second", "kademe": "stage", "onbellek": "cache",
 "recete": "recipe", "tam": "full", "yapisal": "structural",
 "kaynak": "source", "ek": "extra", "oznitelik": "feature",
 "regresyon": "regression", "uret": "build", "normal": "normal",
 "ozet": "summary", "duzelt": "fix", "kalibrasyon": "calibration",
 "siralama": "ranking", "yogun": "dense", "seyrek": "sparse",
 "kirpma": "clip", "toplulugu": "ensemble", "topluluk": "ensemble",
 "kiyas": "compare", "esli": "paired", "fark": "diff", "otopsi": "autopsy",
 "hata": "error", "bankasi": "bank", "teshis": "diagnosis",
 "yeniden": "re", "bas": "head", "offset": "offset", "sozluk": "dict",
 "varyant": "variant", "remesh": "remesh", "aralik": "interval",
 "araligi": "interval", "calisma": "operating", "noktasi": "point",
 "bimodal": "bimodal", "aday": "candidate", "auc": "auc",
 "cikti": "output", "vs": "vs", "son": "final", "islem": "post",
 "yerel": "local", "karsitlik": "contrast", "segmentasyon": "segmentation",
 "d7": "d7", "d6": "d6", "p6": "p6", "brep": "brep", "gate": "gate",
 "glb": "glb", "gt": "gt", "seg": "seg", "cp": "cp", "auc": "auc",
}
