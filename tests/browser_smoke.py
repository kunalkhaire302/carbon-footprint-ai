from __future__ import annotations

import threading
from pathlib import Path

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

from backend.app import create_app
from backend.app.config import TestSettings

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "test-artifacts"


def run() -> None:
    SCREENSHOTS.mkdir(exist_ok=True)
    server = make_server("127.0.0.1", 5001, create_app(TestSettings.from_env()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            for name, viewport in (
                ("desktop", {"width": 1440, "height": 1000}),
                ("mobile", {"width": 390, "height": 844}),
            ):
                page = browser.new_page(viewport=viewport)
                errors: list[str] = []
                page.on("pageerror", lambda error, sink=errors: sink.append(str(error)))
                page.goto("http://127.0.0.1:5001", wait_until="networkidle")
                assert page.get_by_role("heading", name="Carbon Footprint AI").is_visible()
                assert page.get_by_role("button", name="Calculate Footprint").is_visible()
                page.get_by_role("button", name="Calculate Footprint").click()
                page.locator("#resultsSection.results-visible").wait_for()
                page.wait_for_function("document.querySelector('#categorySummary').textContent.length > 0")
                assert page.locator("#totalScore").inner_text() != "0.00"
                assert page.locator("#categorySummary").text_content()
                assert page.locator("#comparisonSummary").text_content()
                assert "browser only" in page.locator(".privacy-note").inner_text()
                assert not errors, errors
                page.screenshot(path=SCREENSHOTS / f"{name}.png", full_page=True)
                page.close()
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)


if __name__ == "__main__":
    run()
    print("Browser smoke test passed")
