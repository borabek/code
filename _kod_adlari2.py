# -*- coding: utf-8 -*-
"""Kod icindeki TURKCE degisken adlari -> EN, CAKISMA KORUMALI.

ONCEKI DENEMEDE NE OLDU: iki farkli Turkce ad ayni Ingilizce ada
esleniyordu (`puan`->score_, `skor`->score_) ve dosyada zaten var olan
adla cakisma kontrolu YOKTU. Iki ayri degisken tek isimde birlesti ve
URUN CIKTISI DEGISTI (duman testi "gecti" ama koordinatlar farkliydi).

BU SURUMDE:
  1. esleme BIREBIR (ayni hedefe iki kaynak YOK) -- basta dogrulanir
  2. DOSYA BAZINDA: hedef ad o dosyada ZATEN varsa o dosyada DEGISTIRILMEZ
  3. her dosya ayristirilir
  4. cagiran taraf duman testiyle koordinat karsilastirmasi yapar
"""
import ast, builtins, io, keyword, subprocess, sys, tokenize, collections

AD = {
 "kaynak": "src_", "kural": "rule_", "deger": "val_", "sayi": "cnt_",
 "ayni": "same_", "farkli": "diff_", "once": "pre_", "sonra": "post_",
 "bos": "empty_", "eski": "old_", "yeni": "new_", "ilk": "first_",
 "son": "last_", "toplam": "total_", "yuzde": "pct_", "oran": "ratio_",
 "genislik": "width_", "yukseklik": "height_", "uzunluk": "len_",
 "mesafe": "dist_", "merkez": "center_", "sinir": "bound_",
 "dosya": "file_", "dizin": "dir_", "satir": "line_", "sutun": "col_",
 "liste": "lst_", "anahtar": "key_", "girdi": "inp_", "cikti": "out_",
 "sonuc": "res_", "hata": "err_", "mesaj": "msg_", "durum": "state_",
 "adim": "step_", "kosul": "cond_", "secim": "sel_", "kayit": "rec_",
 "veri": "data_", "etiket": "label_", "tahmin": "pred_", "kahin": "oracle_",
 "esik": "thr_", "agirlik": "wgt_", "puan": "score_", "sira": "rank_",
 "kume": "clust_", "havuz": "pool_", "aday": "cand_", "nokta": "pt_",
 "tepe": "vtx_", "yuz": "face_", "kenar": "edge_", "yon": "dirn_",
 "eksen": "axis_", "isaret": "sign_", "parca": "part_", "kol": "arm_",
 "olcum": "meas_", "kat": "fold_", "tohum": "seed_", "bolme": "split_",
 "gerekce": "why_", "makbuz": "receipt_", "sebep": "cause_",
}
YASAK = set(keyword.kwlist) | set(dir(builtins))
AD = {k: v for k, v in AD.items() if v not in YASAK and k not in YASAK}
# 1) BIREBIRLIK
ters = collections.Counter(AD.values())
cak = [v for v, n in ters.items() if n > 1]
assert not cak, f"esleme birebir DEGIL: {cak}"


def isle(yol):
    try:
        src = io.open(yol, encoding="utf-8").read()
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return None, 0
    adlar = {t.string for t in toks if t.type == tokenize.NAME}
    # 2) DOSYA BAZINDA CAKISMA: hedef zaten varsa o adi ATLA
    yerel = {k: v for k, v in AD.items() if k in adlar and v not in adlar}
    if not yerel:
        return None, 0
    yeni, n = [], 0
    for t in toks:
        v = t.string
        if t.type == tokenize.NAME and v in yerel:
            v = yerel[v]; n += 1
        yeni.append((t.type, v))
    try:
        out = tokenize.untokenize(yeni)
        ast.parse(out)
    except Exception:
        return None, 0
    return out, n


def main():
    uygula = "--uygula" in sys.argv
    d = t = 0
    for f in subprocess.run(["git", "ls-files", "*.py"], capture_output=True,
                            text=True).stdout.split():
        out, n = isle(f)
        if out is None:
            continue
        d += 1; t += n
        if uygula:
            io.open(f, "w", encoding="utf-8").write(out)
    print(f"{d} dosya | {t} ad cevrildi (cakisma korumali)")
    if not uygula:
        print("(kuru kosum)")


main()
