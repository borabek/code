"""Pull real vendor article numbers out of a manufacturer range-overview PDF.

Used to build the download list for the next corpus batch: we need MORE screw-type
terminal blocks (UK/UT/UKH -- the model's measured blind spot, per-family F1 59%),
and guessing catalog numbers is not an option. These PDFs are the vendor's own
range overviews, so every number in them is a real, orderable part.

Emits "article  type" lines, skipping anything already in the local STEP pool.
"""
import argparse
import os
import re

import pypdf

# Phoenix Contact order numbers are 7 digits starting with 3 (terminal blocks).
ART = re.compile(r"\b(3[0-9]{6})\b")
# Type designations we care about: screw-connection terminal-block families.
TYPE = re.compile(
    r"\b(UT|UK|UKH|USLKG|UKK|UDK|URTK|UKM|MBK|UTI|UTN|UKN|ATP|UIK|USST|UTTB)"
    r"[\s\-]?[0-9][0-9,\.\-/A-Z ]{0,14}", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--have", default="_mevcut_katalog_numaralari.txt",
                    help="catalog numbers already in the STEP pool (skip these)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    have = set()
    if os.path.exists(args.have):
        have = {l.strip() for l in open(args.have, encoding="utf-8") if l.strip()}

    reader = pypdf.PdfReader(args.pdf)
    pairs, seen = [], set()
    for page in reader.pages:
        txt = page.extract_text() or ""
        # keep line structure: a range-overview row usually holds the type on the
        # left and the order number on the right of the SAME line
        for line in txt.splitlines():
            arts = ART.findall(line)
            if not arts:
                continue
            tm = TYPE.search(line)
            typ = tm.group(0).strip() if tm else ""
            for a in arts:
                if a in seen:
                    continue
                seen.add(a)
                pairs.append((a, typ))

    new = [(a, t) for a, t in pairs if a not in have]
    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as fh:
        for a, t in new:
            fh.write(f"{a}\t{t}\n")

    print(f"{os.path.basename(args.pdf)}: {len(pairs)} parca no bulundu, "
          f"{len(pairs) - len(new)} zaten elimizde, {len(new)} YENI -> {args.out}")


if __name__ == "__main__":
    main()
