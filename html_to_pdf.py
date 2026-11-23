"""
Generate ECG_Conference_Paper.pdf from ECG_Conference_Paper.html
using Playwright headless Chromium — preserves all content and embedded images.
"""
import os
import pathlib
from playwright.sync_api import sync_playwright

HTML_FILE = pathlib.Path(r"d:/Projects/ECG/ECG_Conference_Paper.html").resolve()
PDF_FILE  = pathlib.Path(r"d:/Projects/ECG/ECG_Conference_Paper.pdf").resolve()

assert HTML_FILE.exists(), f"HTML not found: {HTML_FILE}"

print(f"Source : {HTML_FILE}  ({HTML_FILE.stat().st_size:,} bytes)")
print(f"Target : {PDF_FILE}")

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    # Load the local file — file:// URI so base64 images resolve instantly
    page.goto(HTML_FILE.as_uri(), wait_until="networkidle", timeout=60_000)

    # Wait for fonts to load (Google Fonts)
    page.wait_for_timeout(2500)

    # Print to PDF
    # A4, clean margins, background graphics ON (for colour blocks)
    page.pdf(
        path=str(PDF_FILE),
        format="A4",
        print_background=True,
        margin={
            "top":    "18mm",
            "bottom": "18mm",
            "left":   "14mm",
            "right":  "14mm",
        },
    )

    browser.close()

size = PDF_FILE.stat().st_size
print(f"\nPDF written: {PDF_FILE}")
print(f"Size       : {size:,} bytes  ({size // 1024:,} KB)")
print("Done.")
