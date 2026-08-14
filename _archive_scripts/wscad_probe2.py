"""Open a WSCAD part page DIRECTLY by article number and find the 3D/STEP download."""
import os
import sys

from playwright.sync_api import sync_playwright

PROFILE = os.path.abspath("_wscad_profile")
PART = ("https://www.wscaduniverse.com/part?manufacturerId={mid}&partNumber={art}"
        "&format=WSCAD&norm=IEC")


def main():
    art = sys.argv[1] if len(sys.argv) > 1 else "3001035"
    mid = sys.argv[2] if len(sys.argv) > 2 else "63"

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(PROFILE, headless=True,
                                                   accept_downloads=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(PART.format(mid=mid, art=art), wait_until="networkidle",
                  timeout=90000)
        page.wait_for_timeout(5000)
        print("URL:", page.url)
        print("baslik:", page.title())
        body = page.inner_text("body")
        print("metin (first 500):", body[:500].replace("\n", " | "))
        print("\n--- 3D / STEP izleri ---")
        for kw in ("3D", "STEP", "stp", "Download", "CAD", "Geometry", "Macro"):
            if kw.lower() in body.lower():
                print(f"  '{kw}' metinde VAR")
        print("\n--- tiklanabilir kontroller ---")
        for sel in ("button", "a[href]", "[role=button]", "select", "mat-select",
                    "[class*=download i]", "[class*=3d i]"):
            n = page.locator(sel).count()
            if not n:
                continue
            print(f"  {sel}: {n}")
            for i in range(min(n, 14)):
                try:
                    el = page.locator(sel).nth(i)
                    t = (el.inner_text() or "").strip()[:45].replace("\n", " ")
                    h = el.get_attribute("href") or ""
                    if t or h:
                        print(f"     [{i}] {t!r} {h[:70]}")
                except Exception:
                    pass
        ctx.close()


if __name__ == "__main__":
    main()
