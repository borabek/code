"""Open WSCAD Universe in a PERSISTENT browser profile so the user can log in once.

The session cookies live in ./_wscad_profile, so the download script
(wscad_download.py) can reuse the logged-in session later -- including after the
user has left. No password ever passes through this script; the human types it
into the real browser window.

Run:  .venv/Scripts/python.exe wscad_login.py
Then: log in, and leave the window open until it says LOGIN OK.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

PROFILE = os.path.abspath("_wscad_profile")
URL = "https://www.wscaduniverse.com/en/"


def main():
    os.makedirs(PROFILE, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            PROFILE, headless=False, accept_downloads=True,
            args=["--start-maximized"], no_viewport=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=90000)
        print("TARAYICI ACIK -- WSCAD Universe'e GIRIS YAP.")
        print("Giris yaptiktan sonra bu pencereyi KAPATMA; 10 dk boyunca bekliyorum.\n")

        # poll for a logged-in signal: the page stops showing a login control
        deadline = time.time() + 600
        while time.time() < deadline:
            time.sleep(5)
            try:
                body = page.content().lower()
            except Exception:                       # navigation in flight
                continue
            logged = ("logout" in body or "abmelden" in body
                      or "my account" in body or "mein konto" in body)
            if logged:
                print("LOGIN OK -- oturum kaydedildi:", PROFILE)
                print("Artik indirmeyi ben yurutebilirim (wscad_download.py).")
                ctx.close()
                return 0
        print("zaman asimi -- giris tespit edilemedi; pencereyi acik birak ve tekrar dene")
        ctx.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
