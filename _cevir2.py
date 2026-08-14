# -*- coding: utf-8 -*-
"""TR -> EN 2. tur: YALNIZ yorum ve docstring metni.

NEDEN TOKENIZER: 2. tur sozlukte `ve->and`, `veya->or`, `degil->not`,
`eger->if` gibi PYTHON ANAHTAR KELIMELERI var. Bunlari koda uygulamak
`ve` adli bir degiskeni `and` yapar ve dosyayi tamamen bozar. Bu yuzden
donusum Python'un kendi `tokenize` modulu ile YALNIZ COMMENT ve
docstring (ifade olarak duran STRING) belirteclerine uygulanir.
"""
import io, re, subprocess, sys, tokenize
import importlib.util as _u
_s = _u.spec_from_file_location("s2", "_ceviri_sozluk2.py")
_m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
SZ = _m.SOZLUK2
DESEN = re.compile(r"\b(" + "|".join(re.escape(k) for k in
                   sorted(SZ, key=len, reverse=True)) + r")\b")


def cevir_metin(t):
    return DESEN.sub(lambda m: SZ[m.group(1)], t)


def dosya_isle(yol):
    try:
        src = io.open(yol, encoding="utf-8").read()
    except Exception:
        return None, 0
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return None, 0
    yeni, n = [], 0
    onceki = tokenize.INDENT
    for t in toks:
        tip, val = t.type, t.string
        if tip == tokenize.COMMENT:
            y = cevir_metin(val)
            n += (y != val); val = y
        elif tip == tokenize.STRING and onceki in (tokenize.INDENT,
                tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL,
                tokenize.ENCODING):
            # ifade konumundaki string = docstring
            y = cevir_metin(val)
            n += (y != val); val = y
        yeni.append((tip, val))
        if tip not in (tokenize.NL, tokenize.COMMENT):
            onceki = tip
    if not n:
        return None, 0
    try:
        out = tokenize.untokenize(yeni)
    except Exception:
        return None, 0
    return out, n


def main():
    uygula = "--uygula" in sys.argv
    fs = [f for f in subprocess.run(["git", "ls-files", "*.py"],
          capture_output=True, text=True).stdout.split() ]
    import ast
    d = t = 0
    for f in fs:
        out, n = dosya_isle(f)
        if out is None:
            continue
        try:
            ast.parse(out)
        except SyntaxError:
            continue
        d += 1; t += n
        if uygula:
            io.open(f, "w", encoding="utf-8").write(out)
    print(f"{d} dosya | {t} yorum/docstring belirteci cevrildi")
    if not uygula:
        print("(kuru kosum -- --uygula ile yazar)")


if __name__ == "__main__":
    main()
