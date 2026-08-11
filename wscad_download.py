"""Bulk-download STEP (.stp) models from WSCAD Universe for a list of article numbers.

Reuses the session the user logged into (_wscad_profile) -- no credentials here.
Flow per part (reverse-engineered from the live site):
    /part?manufacturerId=<mid>&partNumber=<art>&format=WSCAD&norm=IEC
    -> click the 'download' toolbar button
    -> click 'Download STEP'

Parts with no STEP model, or that don't exist, are skipped and logged; the run
continues. Already-downloaded articles are skipped, so the script is resumable.

Usage:
  python wscad_download.py --list WSCAD_liste_phoenix.txt --mid 63 --out all_wscad_stp
"""
import argparse
import os
import re
import shutil
import sys
import time

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

PROFILE = os.path.abspath(os.environ.get("WSCAD_PROFILE", "_wscad_profile"))
PART = ("https://www.wscaduniverse.com/part?manufacturerId={mid}&partNumber={art}"
        "&format=WSCAD&norm=IEC")


def already_have(out_dir):
    have = set()
    for f in os.listdir(out_dir) if os.path.isdir(out_dir) else []:
        m = re.search(r"wscaduniverse_([0-9A-Za-z-]+)_", f)
        if m:
            have.add(m.group(1))
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--mid", default="63", help="WSCAD manufacturerId (63 = Phoenix Contact)")
    ap.add_argument("--out", default="all_wscad_stp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=45000)
    args = ap.parse_args()

    # list may be "article" or "article	manufacturerId	description"
    arts, mids = [], {}
    for line in open(args.list, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        art = cols[0].strip()
        arts.append(art)
        if len(cols) > 1 and cols[1].strip().isdigit():
            mids[art] = cols[1].strip()
    os.makedirs(args.out, exist_ok=True)
    have = already_have(args.out)
    todo = [a for a in arts if a not in have]
    if args.limit:
        todo = todo[:args.limit]
    print(f"liste: {len(arts)} | zaten var: {len(arts) - len(todo)} | indirilecek: {len(todo)}",
          flush=True)

    tmp = os.path.abspath(os.environ.get("WSCAD_TMP", "_wscad_dl"))
    os.makedirs(tmp, exist_ok=True)
    ok = skip = fail = 0
    t0 = time.time()

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            PROFILE, headless=True, accept_downloads=True, downloads_path=tmp)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        for i, art in enumerate(todo, 1):
            try:
                page.goto(PART.format(mid=mids.get(art, args.mid), art=art),
                          wait_until="domcontentloaded", timeout=args.timeout)
                page.wait_for_timeout(1800)

                btn = page.locator("button:has-text('download')").first
                if not btn.count():
                    skip += 1
                    print(f"[{i}/{len(todo)}] {art}: parca/indirme yok", flush=True)
                    continue
                btn.click()
                page.wait_for_timeout(900)

                step = page.locator("button:has-text('Download STEP')").first
                if not step.count():
                    skip += 1
                    print(f"[{i}/{len(todo)}] {art}: STEP formati YOK", flush=True)
                    continue

                with page.expect_download(timeout=args.timeout) as dl_info:
                    step.click()
                dl = dl_info.value
                name = dl.suggested_filename or f"wscaduniverse_{art}.stp"
                if not name.lower().endswith((".stp", ".step")):
                    name += ".stp"
                # keep the pipeline's naming convention: wscaduniverse_<art>_<...>.stp
                if not name.startswith("wscaduniverse_"):
                    name = f"wscaduniverse_{art}_{time.strftime('%Y-%m-%d-%H-%M-%S')}.stp"
                dst = os.path.join(args.out, name)
                dl.save_as(dst)
                ok += 1
                if ok % 10 == 0 or i <= 3:
                    rate = i / max(time.time() - t0, 1e-6) * 60
                    left = (len(todo) - i) / max(rate, 1e-6)
                    print(f"[{i}/{len(todo)}] {art}: OK ({ok} indi, {rate:.0f}/dk, "
                          f"~{left:.0f} dk kaldi)", flush=True)
            except PWTimeout:
                fail += 1
                print(f"[{i}/{len(todo)}] {art}: ZAMAN ASIMI", flush=True)
            except Exception as exc:                      # noqa: BLE001
                fail += 1
                print(f"[{i}/{len(todo)}] {art}: HATA {str(exc)[:60]}", flush=True)

        ctx.close()

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nBITTI: {ok} indi, {skip} atlandi (STEP yok), {fail} hata "
          f"-> {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
