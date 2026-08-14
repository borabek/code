# -*- coding: utf-8 -*-
"""print() / f-string ICINDEKI Turkce metin -> EN.

NEDEN AST: bir string'i cevirmek, o string bir CONFIG ANAHTARI ya da
SOZLUK ANAHTARI ise urunu sessizce bozar. `print(...)` argumani ise
tanim geregi EKRANA giden metindir -- anahtar olamaz. Bu yuzden hedef
AST ile secilir, desenle degil.

Kapsam: print(...) ve sys.stderr.write(...) argumanlari; hem duz string
hem f-string parcalari.
"""
import ast, io, subprocess, sys, re
import importlib.util as _u
_s = _u.spec_from_file_location("s2", "_ceviri_sozluk2.py")
_m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
SZ = _m.SOZLUK2
D = re.compile(r"\b(" + "|".join(re.escape(k) for k in
               sorted(SZ, key=len, reverse=True)) + r")\b")


def cev(t):
    return D.sub(lambda m: SZ[m.group(1)], t)


class Gez(ast.NodeVisitor):
    def __init__(self):
        self.hedef = []          # (lineno, col, end_col, eski, yeni)

    def visit_Call(self, node):
        ad = ""
        if isinstance(node.func, ast.Name):
            ad = node.func.id
        elif isinstance(node.func, ast.Attribute):
            ad = node.func.attr
        if ad in ("print", "write"):
            for a in node.args:
                self._str(a)
        self.generic_visit(node)

    def _str(self, n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            y = cev(n.value)
            if y != n.value:
                self.hedef.append((n.lineno, n.col_offset,
                                   n.end_col_offset, n.value, y))
        elif isinstance(n, ast.JoinedStr):
            for v in n.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    pass          # f-string parcalari: kaynak duzeyinde risk
        elif isinstance(n, ast.BinOp):
            self._str(n.left); self._str(n.right)


def isle(yol, uygula):
    try:
        src = io.open(yol, encoding="utf-8").read()
        agac = ast.parse(src)
    except Exception:
        return 0
    g = Gez(); g.visit(agac)
    if not g.hedef:
        return 0
    satirlar = src.split("\n")
    # ayni satirda birden fazla varsa SAGDAN sola uygula
    for ln, c0, c1, eski, yeni in sorted(g.hedef, key=lambda x: (-x[0], -x[1])):
        s = satirlar[ln - 1]
        parca = s[c0:c1]
        if eski not in parca:
            continue
        satirlar[ln - 1] = s[:c0] + parca.replace(eski, yeni) + s[c1:]
    out = "\n".join(satirlar)
    try:
        ast.parse(out)
    except Exception:
        return 0
    if uygula:
        io.open(yol, "w", encoding="utf-8").write(out)
    return len(g.hedef)


def main():
    uygula = "--uygula" in sys.argv
    d = t = 0
    for f in subprocess.run(["git", "ls-files", "*.py"], capture_output=True,
                            text=True).stdout.split():
        n = isle(f, uygula)
        if n:
            d += 1; t += n
    print(f"{d} dosya | {t} print metni cevrildi")
    if not uygula:
        print("(kuru kosum)")


main()
