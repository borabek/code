# -*- coding: utf-8 -*-
"""WSCAD availability census -- how many of the missing parts REALLY exist there.

Why this exists: the SIE bulk download returned 10/10 HTTP 404 with a 31-byte body that was
just the filename it would have produced -- i.e. "no STEP for this part", not a bad id and not
a dead session. Before spending hours downloading, measure what is actually obtainable.

Two guards against fooling ourselves:
  1. NULL TEST -- parts whose STEP is already on disk are searched first. If the search cannot
     find those, the search is broken and the census means nothing, so we abort.
  2. EXACT match only -- the search is fuzzy: "304589" comes back as Phoenix "3045897"
     (one extra digit). Counting any hit would massively overstate availability. A part counts
     only when the returned partNumber equals ours, case-insensitively.

WHAT THIS DOES AND DOES NOT MEASURE: found_exact means the part is SEARCHABLE under our exact
number. It does NOT mean a STEP exists for it -- that is a separate question answered only by the
download endpoint (partType 2). Measured 2026-07-30: all 186 Allen Bradley 1492 terminals are
searchable and none is downloadable as STEP; their own metadata reports formats:[0,1], i.e. WSCAD
data + EPLAN .edz, no STEP. Treat this census as a candidate list plus a verified manufacturer id,
never as an availability figure.

Output: results/wscad_availability.csv (part, our_mfg, found, wscad_mfg_id, wscad_mfg_name)
"""
import os, sys, csv, json, glob, collections
from playwright.sync_api import sync_playwright

PROFILE = os.path.abspath("_wsprofile")
HOME = "https://www.wscaduniverse.com/en/"
OUT = "results/wscad_availability.csv"
DELAY_MS = 900

SEARCH = """async (q) => {
  let b=null; for (const k of Object.keys(localStorage)){const v=localStorage.getItem(k)||"";
    const m=v.match(/eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/); if(m){b=m[0];break;}}
  if(!b) return {err:"no session token"};
  const r = await fetch("https://bff.wscaduniverse.com/api/search?searchText="+encodeURIComponent(q)+"&norm=0&language=en",
    {headers:{accept:"application/json",authorization:"Bearer "+b}});
  if(r.status!==200) return {err:"status "+r.status};
  const j = JSON.parse(await r.text());
  return {n: j.partsFoundTotal,
          parts: (j.parts||[]).map(p=>({pn:p.partNumber, id:p.manufacturerId, m:p.manufacturerName}))};
}"""


def exact(res, pn):
    """Return the hit whose partNumber equals ours exactly (case-insensitive), else None."""
    for p in res.get("parts") or []:
        if str(p.get("pn", "")).strip().upper() == pn.strip().upper():
            return p
    return None


def main():
    have = sorted({os.path.basename(s).split("_")[1] for s in glob.glob("all_wscad_stp/*.stp")})
    missing = []
    with open("results/eksik_step_listesi.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            missing.append((r["uretici"], r["parca_no"]))
    print(f"null test: {min(6,len(have))} parts already on disk | census: {len(missing)} missing parts",
          flush=True)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(PROFILE, channel="msedge", headless=True)
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        pg.goto(HOME, wait_until="domcontentloaded"); pg.wait_for_timeout(2500)

        print("\n=== NULL TEST (parts we already downloaded must be findable) ===", flush=True)
        ok = 0
        for pn in have[:6]:
            r = pg.evaluate(SEARCH, pn)
            e = exact(r, pn)
            print(f"  {pn:<20} n={r.get('n')} exact={'YES ' + e['m'] if e else 'no'}", flush=True)
            ok += bool(e)
            pg.wait_for_timeout(DELAY_MS)
        if ok < 4:
            print(f"\nNULL TEST FAILED ({ok}/6 findable) -> the search is unreliable, aborting "
                  f"census. No conclusion may be drawn about availability.", flush=True)
            ctx.close(); sys.exit(2)
        print(f"NULL TEST PASSED ({ok}/6 findable) -> census is meaningful\n", flush=True)

        rows = []
        stat = collections.Counter()
        with open(OUT, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["part", "our_mfg", "found_exact", "wscad_mfg_id", "wscad_mfg_name", "n_fuzzy"])
            for i, (mfg, pn) in enumerate(missing, 1):
                try:
                    r = pg.evaluate(SEARCH, pn)
                except Exception as ex:
                    r = {"err": type(ex).__name__}
                e = exact(r, pn) if "err" not in r else None
                w.writerow([pn, mfg, int(bool(e)), (e or {}).get("id", ""),
                            (e or {}).get("m", ""), r.get("n", "")])
                stat[(mfg, bool(e))] += 1
                if e: rows.append((mfg, pn, e["id"], e["m"]))
                if i % 50 == 0:
                    got = sum(v for (m_, f), v in stat.items() if f)
                    print(f"  {i}/{len(missing)}  found so far: {got}", flush=True)
                    fh.flush()
                pg.wait_for_timeout(DELAY_MS)
        ctx.close()

    print("\n=== AVAILABILITY BY MANUFACTURER ===", flush=True)
    mfgs = sorted({m for m, _ in missing})
    tot_f = 0
    for m in mfgs:
        f = stat[(m, True)]; n = f + stat[(m, False)]
        tot_f += f
        if n: print(f"  {m:<8} {f:>4}/{n:<4} obtainable ({100*f/n:5.1f}%)")
    print(f"\nTOTAL obtainable: {tot_f}/{len(missing)} ({100*tot_f/max(len(missing),1):.1f}%)")
    print(f"receipt -> {OUT}")


if __name__ == "__main__":
    main()
