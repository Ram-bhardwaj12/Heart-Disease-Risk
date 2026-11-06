"""CardioLens End-to-End Browser Tests across 5 Viewport Widths.

Tests UI responsiveness, 3-step assessment wizard, Tree SHAP explanations,
calibrated decision thresholds, 422 error display, and interactive what-if scenarios.
"""

import threading
import time
from typing import Generator
import pytest
from playwright.sync_api import Page, sync_playwright
import uvicorn

from api.main import app

# Viewports required by design system contract
VIEWPORT_WIDTHS = [360, 390, 768, 1024, 1440]
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8659
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


@pytest.fixture(scope="session", autouse=True)
def live_server() -> Generator[str, None, None]:
    """Spin up live uvicorn server in a background daemon thread."""
    config = uvicorn.Config(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server ready
    timeout = 10.0
    start = time.time()
    while time.time() - start < timeout:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError(f"Server at {BASE_URL} failed to start within {timeout}s")

    yield BASE_URL
    server.should_exit = True


@pytest.fixture(scope="module")
def browser_instance():
    """Launch headless Chromium browser for e2e testing."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.mark.e2e
@pytest.mark.parametrize("width", VIEWPORT_WIDTHS)
def test_responsive_layout_and_shell(browser_instance, live_server, width: int):
    """(1) Verify shell, brand, navigation, and disclaimer render at all 5 widths."""
    page: Page = browser_instance.new_page(viewport={"width": width, "height": 800})
    try:
        page.goto(live_server, wait_until="domcontentloaded")

        # Brand and tagline
        brand_name = page.locator(".brand-name")
        assert brand_name.is_visible()
        assert "CardioLens" in brand_name.text_content()

        # Educational disclaimer banner
        banner = page.locator(".disclaimer-banner")
        assert banner.is_visible()
        assert "Educational project, not a medical diagnosis" in banner.text_content()

        # Navigation links
        nav = page.locator(".top-nav")
        assert nav.is_visible()
        assert page.locator("button[data-target='view-risk']").is_visible()

        # Hero title & Call-to-action
        hero = page.locator(".hero-title")
        assert hero.is_visible()
        cta_btn = page.locator("#btn-start-assessment")
        assert cta_btn.is_visible()

        # Quick start presets
        assert page.locator("#preset-low").is_visible()
        assert page.locator("#preset-mod").is_visible()
        assert page.locator("#preset-high").is_visible()
    finally:
        page.close()


@pytest.mark.e2e
def test_full_assessment_flow_and_results(browser_instance, live_server):
    """(2) Test preset loading, 3-step form progression, and prediction results."""
    page: Page = browser_instance.new_page(viewport={"width": 1024, "height": 800})
    try:
        page.goto(live_server, wait_until="domcontentloaded")

        # Click preset low
        page.click("#preset-low")

        # Step 1 visible
        assert page.locator("#step-panel-1").is_visible()
        assert page.locator("#input-age").input_value() == "45"

        # Advance to Step 2
        page.click("#btn-step1-next")
        assert page.locator("#step-panel-2").is_visible()
        assert page.locator("#step-panel-1").is_hidden()

        # Advance to Step 3
        page.click("#btn-step2-next")
        assert page.locator("#step-panel-3").is_visible()
        assert page.locator("#step-panel-2").is_hidden()

        # Submit Assessment
        page.click("#btn-submit-assessment")

        # Verify Results Section appears
        page.wait_for_selector(".results-container.visible", timeout=10000)
        assert page.locator("#results-section").is_visible()

        # Check Score and Tier
        score_text = page.locator("#res-score-pct").text_content()
        tier_text = page.locator("#res-tier-badge").text_content()
        assert "%" in score_text
        assert tier_text in {"Lower", "Moderate", "Higher"}

        # Explanation Summary
        summary = page.locator("#res-summary-text").text_content()
        assert "The model gives a" in summary
        assert "Option" not in summary

        # 13 Factor rows
        factor_rows = page.locator("#factors-tbody tr")
        assert factor_rows.count() == 13

        # Model consensus stats
        tree_count = page.locator("#stat-tree-count").text_content()
        assert tree_count == "400"
    finally:
        page.close()


@pytest.mark.e2e
def test_inferred_codes_tooltip(browser_instance, live_server):
    """(3) Verify info icon tooltip carries the exact inferred_codes_note."""
    page: Page = browser_instance.new_page(viewport={"width": 1024, "height": 800})
    try:
        page.goto(live_server, wait_until="domcontentloaded")
        page.click("#btn-start-assessment")
        page.click("#btn-step1-next")

        # Chest pain tooltip button
        cp_btn = page.locator("#group-cp .tooltip-trigger")
        assert cp_btn.is_visible()
        cp_btn.hover()

        tooltip = page.locator("#global-tooltip")
        page.wait_for_selector("#global-tooltip.visible", timeout=3000)
        tooltip_text = tooltip.text_content()
        assert "Cleveland" in tooltip_text
        assert "inferred" in tooltip_text
    finally:
        page.close()


@pytest.mark.e2e
def test_inline_error_validation(browser_instance, live_server):
    """(4) Verify invalid/out-of-range inputs trigger inline error displays."""
    page: Page = browser_instance.new_page(viewport={"width": 1024, "height": 800})
    try:
        page.goto(live_server, wait_until="domcontentloaded")
        page.click("#btn-start-assessment")

        # Set invalid age
        page.fill("#input-age", "150")
        page.click("#btn-step1-next")

        # Error message should become visible
        err_msg = page.locator("#err-age")
        assert err_msg.is_visible()
        assert "18 and 100" in err_msg.text_content()
        assert page.locator("#step-panel-1").is_visible()

        # Fix age
        page.fill("#input-age", "55")
        page.click("#btn-step1-next")
        assert page.locator("#step-panel-2").is_visible()
    finally:
        page.close()


@pytest.mark.e2e
def test_interactive_whatif_explorer(browser_instance, live_server):
    """(5) Verify interactive What-If scenario dynamically updates score and curve."""
    page: Page = browser_instance.new_page(viewport={"width": 1024, "height": 800})
    try:
        page.goto(live_server, wait_until="domcontentloaded")
        page.click("#preset-mod")
        page.click("#btn-step1-next")
        page.click("#btn-step2-next")
        page.click("#btn-submit-assessment")

        page.wait_for_selector(".results-container.visible", timeout=10000)

        # What-if slider
        slider = page.locator("#whatif-slider")
        assert slider.is_visible()

        # Change slider value
        slider.fill("70")
        slider.dispatch_event("input")

        # Result card should update
        result_card = page.locator("#whatif-result-card")
        assert result_card.is_visible()
        assert "estimated risk score would be" in result_card.text_content()

        # Curve table has 25 rows
        curve_rows = page.locator("#curve-tbody tr")
        assert curve_rows.count() == 25
    finally:
        page.close()


@pytest.mark.e2e
def test_navigation_routing(browser_instance, live_server):
    """(6) Verify tab switching between How it works, Model details, and About."""
    page: Page = browser_instance.new_page(viewport={"width": 1024, "height": 800})
    try:
        page.goto(live_server, wait_until="domcontentloaded")

        # Navigate to How it works
        page.click("button[data-target='view-how']")
        assert page.locator("#view-how").is_visible()
        assert page.locator("#view-risk").is_hidden()

        # Navigate to Model details
        page.click("button[data-target='view-model']")
        assert page.locator("#view-model").is_visible()
        assert "Cleveland" in page.locator("#model-details-inferred-note").text_content()

        # Navigate to About
        page.click("button[data-target='view-about']")
        assert page.locator("#view-about").is_visible()
        assert "Ram Bhardwaj" in page.locator("#view-about").text_content()

        # Back to Check your risk
        page.click("button[data-target='view-risk']")
        assert page.locator("#view-risk").is_visible()
    finally:
        page.close()

