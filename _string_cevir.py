# -*- coding: utf-8 -*-
"""STRING icindeki Turkce DUZYAZI -> EN. Yalniz MESAJ gorunumlu olanlar.

TEHLIKE: string'ler yalnizca ekrana basilmaz -- config anahtari, sozluk
anahtari, dosya yolu da olabilir. Birini cevirmek urunu SESSIZCE bozar
(`cfg.get("robot_pose_head")` gibi).

BU YUZDEN SIKI HEURISTIK -- bir string ancak SU SARTLARIN HEPSINI
saglarsa cevrilir:
  * en az 3 KELIME (bosluklu)  -> anahtarlar tek token olur
  * en az 18 karakter
  * yol/uzanti izi YOK  (/ \ .py .json .pt .stp)
  * f-string kalibi disinda { } YOK
  * en az 2 Turkce kelime iceriyor
"""
import ast, io, re, subprocess, sys, tokenize
import importlib.util as _u
_s = _u.spec_from_file_location("s2", "_ceviri_sozluk2.py")
_m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
SZ = _m.SOZLUK2
D = re.compile(r"\b(" + "|".join(re.escape(k) for k in
               sorted(SZ, key=len, reverse=True)) + r")\b")
TR = re.compile(r"\b(icin|ile|degil|yok|var|bir|bu|ama|cunku|yani|olan|"
                r"olarak|gore|kadar|daha|cok|her|hic|ayni|farkli|once|"
                r"sonra|kural|sayi|yalniz|zaten|neden|sebep|kaynak|deger|"
                r"olcum|parca|aday|havuz|kapi|esik|hata|sonuc|dosya)\b")
YASAK = re.compile(r"[/\\]|[.](py|json|pt|stp|stl|npz|pkl|md)")


def uygun(s):
    ic = s.strip("\"'")
    if len(ic) < 18 or ic.count(" ") < 2:
        return False
    if YASAK.search(ic):
        return False
    return len(TR.findall(ic)) >= 2


def isle(yol):
    try:
        src = io.open(yol, encoding="utf-8").read()
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return None, 0
    yeni, n = [], 0
    for t in toks:
        v = t.string
        if t.type == tokenize.STRING and uygun(v):
            y = D.sub(lambda m: SZ[m.group(1)], v)
            if y != v:
                v = y; n += 1
        yeni.append((t.type, v))
    if not n:
        return None, 0
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
    print(f"{d} dosya | {t} mesaj string'i cevrildi")
    if not uygula:
        print("(kuru kosum)")


main()
