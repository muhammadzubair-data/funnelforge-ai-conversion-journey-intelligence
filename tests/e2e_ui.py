"""Browser end-to-end check of the running FunnelForge dashboard (Playwright + Chromium).

Not collected by pytest (file name). Usage:
    streamlit run app.py --server.port 8501 &
    python tests/e2e_ui.py http://localhost:8501 [--chrome /path/to/chrome]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_messy_sample import build  # noqa: E402

PAGES = [("", "Funnel Overview", 2), ("segments", "Channel & Device Comparison", 4), ("daily", "Daily Trends", 3),
         ("scenarios", "Scenario Planner", 2), ("data", "Data Upload & Exports", 0),
         ("method", "Methodology & Limitations", 0)]


def idle(page, extra=2.0):
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=180_000)
    time.sleep(1.5)
    prev = -1
    for _ in range(150):
        busy = page.locator("[data-testid='stSpinner']").count() + page.locator("[data-testid='stStatusWidget']").count()
        n = page.locator(".js-plotly-plot").count()
        if busy == 0 and n == prev:
            break
        prev = n
        time.sleep(1)
    time.sleep(extra)


def download(page, label) -> bytes:
    with page.expect_download() as dl:
        page.get_by_role("button", name=label).first.click()
    return Path(dl.value.path()).read_bytes()


def sidebar_go(page, title):
    page.locator("[data-testid='stSidebarNav'] a", has_text=title).first.click()
    idle(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://localhost:8501")
    ap.add_argument("--chrome", default=os.environ.get("CHROME_PATH"))
    a = ap.parse_args()
    base = a.url.rstrip("/")
    fails: list = []
    with sync_playwright() as p:
        launch = {"args": ["--no-sandbox"]}
        if a.chrome:
            launch["executable_path"] = a.chrome
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        print("Pages:")
        for path, title, min_charts in PAGES:
            page.goto(f"{base}/{path}", wait_until="networkidle")
            idle(page)
            exc = page.locator("[data-testid='stException']").count()
            errs = page.locator("[data-testid='stAlertContentError']").count()
            charts = page.locator(".js-plotly-plot").count()
            ok_title = title in page.locator(".ff-hero h1").first.inner_text()
            print(f"  {path or 'funnel':<10} exceptions={exc} errors={errs} charts={charts} title={'ok' if ok_title else 'MISSING'}")
            if exc or errs or charts < min_charts or not ok_title:
                fails.append((path, exc, errs, charts, ok_title))

        print("Downloads:")
        page.goto(f"{base}/", wait_until="networkidle")
        idle(page)
        f = pd.read_csv(io.BytesIO(download(page, "Download funnel (CSV)")))
        ok = f.sessions.tolist() == [4000, 2836, 1137, 678, 444]
        print(f"  funnel.csv    sessions={f.sessions.tolist()} ok={ok}")
        if not ok:
            fails.append(("funnel", f.sessions.tolist()))
        page.goto(f"{base}/segments", wait_until="networkidle")
        idle(page)
        s = pd.read_csv(io.BytesIO(download(page, "Download segment table (CSV)")))
        ok = s.sessions.sum() == 4000 and abs(s.revenue.sum() - 46872.1) < 1
        print(f"  segments.csv  sessions={s.sessions.sum()} revenue={s.revenue.sum():.2f} ok={ok}")
        if not ok:
            fails.append(("segments", s))
        page.goto(f"{base}/scenarios", wait_until="networkidle")
        idle(page)
        sc = pd.read_csv(io.BytesIO(download(page, "Download scenario (CSV)")))
        val = dict(zip(sc.metric.dropna(), sc.value.dropna()))
        ok = abs(float(val["scenario_purchases"]) - 481.23) < .05
        print(f"  scenario.csv  scenario_purchases={float(val['scenario_purchases']):.2f} ok={ok}")
        if not ok:
            fails.append(("scenario", val))

        print("Filters:")
        page.goto(f"{base}/", wait_until="networkidle")
        idle(page)
        page.locator("[data-testid='stSidebar'] button[aria-label='Remove Mobile']").first.click()
        idle(page)
        body = page.locator("[data-testid='stMain']").inner_text()
        ok = "Filtered: 1,623 sessions" in body
        print(f"  device filter without Mobile -> {('1,623 sessions' if ok else body[:120])}")
        if not ok:
            fails.append(("filter", body[:200]))

        print("Upload:")
        page.goto(f"{base}/data", wait_until="networkidle")
        idle(page)
        tmp = Path(tempfile.mkdtemp()) / "ga_export.csv"
        build().to_csv(tmp, index=False)
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(str(tmp))
        idle(page, 4)
        body = page.locator("[data-testid='stMain']").inner_text()
        ok = "ga_export.csv" in body and "view_item → product_view" in body and "1493" in body.replace(",", "")
        print(f"  upload        accepted, validation report shown={ok}")
        if not ok:
            fails.append(("upload", body[:300]))
        rep = pd.read_csv(io.BytesIO(download(page, "Validation report")))
        if "Sessions kept" not in rep.check.tolist():
            fails.append(("validation csv",))
        for title in ["Funnel Overview", "Channel & Device Comparison", "Scenario Planner"]:
            sidebar_go(page, title)
            used = page.locator(".pill.ok").count() > 0
            exc = page.locator("[data-testid='stException']").count()
            print(f"  session       {title:<28} uses upload={used} exceptions={exc}")
            if not used or exc:
                fails.append(("session", title))
        other = browser.new_page(viewport={"width": 1440, "height": 900})
        other.goto(f"{base}/", wait_until="networkidle")
        idle(other)
        iso = other.locator(".pill.warn").count() > 0 and other.locator(".pill.ok").count() == 0
        print(f"  isolation     second browser session sees only the demo={iso}")
        if not iso:
            fails.append(("isolation",))
        other.close()
        sidebar_go(page, "Data Upload & Exports")
        page.get_by_role("button", name="Remove upload").click()
        idle(page)
        back = page.locator(".pill.warn").count() > 0
        print(f"  upload        removed, back to synthetic={back}")
        if not back:
            fails.append(("remove",))
        bad = tmp.with_name("bad.csv")
        pd.DataFrame({"when": ["2026-01-01"], "value": [1]}).to_csv(bad, index=False)
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(str(bad))
        idle(page, 3)
        err = page.locator("[data-testid='stAlertContentError']").all_inner_texts()
        print(f"  upload        bad file error shown={bool(err)}")
        if not err or "Missing required column" not in err[0]:
            fails.append(("bad upload", err))
        browser.close()
    print("FAILURES:" if fails else "All browser checks passed.", *fails, sep="\n  ")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
