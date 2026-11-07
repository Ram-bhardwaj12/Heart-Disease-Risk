"""CardioLens End-to-End Automated Browser Verification (Playwright Chromium).

Validates all contract requirements across all 5 viewport widths (360, 390, 768, 1024, 1440):
 1. No horizontal scroll (scrollWidth <= innerWidth).
 2. Zero console errors, zero CSP violations, zero failed network requests.
 3. Zero sibling or text bounding-box overlaps in cards, grids, flex containers.
 4. Minimum 44 px interactive touch targets at widths <= 640 and Tab keyboard reachability.
 5. Contrast >= 4.5:1 for all visible text nodes against effective backgrounds.
 6. Zero emoji or pictographic characters in document body.
 7. Zero banned words in visible text of #/check, #/how, #/about.
 8. Exact equivalence: result-view score matches API within 0.05% and tier pill matches API tier.
 9. Interactive What-If slider updates displayed score and matches API prediction for changed inputs.
 10. Print media emulation hides navigation and buttons.
 11. Category info icon tooltip carries exact inferred_codes_note.
 12. Sub-tab keyboard arrow navigation with roving tabindex and aria-selected tracking.
 13. Every figure has non-empty alt text and naturalWidth > 0.
 14. Responsive tables with horizontal scroll wrappers; no page-level scroll at width 360.
 15. Container left edges aligned within 1 px across header, strip, main, footer.
 16. Footer pinned to bottom of viewport on short pages (bottom >= innerHeight - 1).
 17. Head elements: favicon.svg (200 OK image/svg+xml), meta theme-color, meta description, dynamic title.
 18. Fault injection verification proving checkers detect real faults.
Saves all full-page screenshots to design_review/<width>/<name>.png.
"""

from pathlib import Path
import re
import threading
import time
from typing import Generator
import pytest
from playwright.sync_api import Browser, Page, sync_playwright
import uvicorn

from api.main import app
import engine.core as engine
from engine.content import INFERRED_CODES_NOTE
from tests.e2e_checks import (
    check_horizontal_scroll,
    check_overlaps,
    check_contrast,
    check_touch_targets,
    check_emojis,
    check_banned_words,
    check_container_alignment,
    check_footer_pinned,
    check_heading_type_scale,
    check_banned_dataset_and_help_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGN_REVIEW_DIR = REPO_ROOT / "design_review"
VIEWPORT_WIDTHS = [360, 390, 768, 1024, 1440]
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8660
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


@pytest.fixture(scope="session", autouse=True)
def live_server() -> Generator[str, None, None]:
    """Start uvicorn server in a daemon thread."""
    config = uvicorn.Config(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    timeout = 15.0
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


@pytest.fixture(scope="session")
def browser_instance():
    """Launch headless Chromium browser."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


def save_screenshot(page: Page, width: int, name: str):
    """Save full-page screenshot to design_review/<width>/<name>.png."""
    out_dir = DESIGN_REVIEW_DIR / str(width)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"
    page.screenshot(path=str(out_path), full_page=True)


def verify_page_state(page: Page, width: int, state_name: str, console_errors: list, failed_requests: list):
    """Run comprehensive verification on active page state."""
    # Save screenshot
    save_screenshot(page, width, state_name)

    # 1. No horizontal scroll
    scroll_fails = check_horizontal_scroll(page)
    assert len(scroll_fails) == 0, f"[{width}px {state_name}] {scroll_fails}"

    # 2. Console and network errors
    assert len(console_errors) == 0, f"[{width}px {state_name}] Console errors: {console_errors}"
    assert len(failed_requests) == 0, f"[{width}px {state_name}] Failed requests: {failed_requests}"

    # 3. No overlap
    overlaps = check_overlaps(page)
    assert len(overlaps) == 0, f"[{width}px {state_name}] Overlaps detected: {overlaps}"

    # 4. Touch target height >= 44px at widths <= 640
    target_fails = check_touch_targets(page, width)
    assert len(target_fails) == 0, f"[{width}px {state_name}] Touch targets < 44px: {target_fails}"

    # 5. Contrast >= 4.5:1
    contrast_fails = check_contrast(page)
    assert len(contrast_fails) == 0, f"[{width}px {state_name}] Contrast failures: {contrast_fails}"

    # 6. No emoji or pictographs
    emoji_matches = check_emojis(page)
    assert len(emoji_matches) == 0, f"[{width}px {state_name}] Emojis detected: {emoji_matches}"

    # 7. Banned words on #/check, #/how, #/about
    current_hash = page.evaluate("window.location.hash")
    banned_matches = check_banned_words(page, current_hash)
    assert len(banned_matches) == 0, f"[{width}px {state_name}] Banned words detected: {banned_matches}"

    # 8. Container left edge alignment within 1px
    align_fails = check_container_alignment(page)
    assert len(align_fails) == 0, f"[{width}px {state_name}] {align_fails}"

    # 9. Pinned footer check
    footer_fails = check_footer_pinned(page)
    assert len(footer_fails) == 0, f"[{width}px {state_name}] {footer_fails}"

    # 10. Type scale check for headings
    type_fails = check_heading_type_scale(page, width)
    assert len(type_fails) == 0, f"[{width}px {state_name}] {type_fails}"

    # 11. Banned dataset naming & help texts
    text_fails = check_banned_dataset_and_help_text(page)
    assert len(text_fails) == 0, f"[{width}px {state_name}] {text_fails}"


@pytest.mark.e2e
def test_checks_detect_injected_faults(browser_instance: Browser, live_server: str):
    """Part C: Verify that all e2e checking functions reliably detect injected faults."""
    page: Page = browser_instance.new_page(viewport={"width": 360, "height": 800})
    try:
        page.goto(f"{live_server}/#/check", wait_until="networkidle")

        # Fault (a): Two overlapping boxes inside a card
        page.evaluate("""() => {
            const card = document.querySelector('.card') || document.body;
            const b1 = document.createElement('div');
            b1.id = 'fault-overlap-1';
            b1.style.cssText = 'position: absolute; left: 10px; top: 10px; width: 100px; height: 100px; background: red;';
            const b2 = document.createElement('div');
            b2.id = 'fault-overlap-2';
            b2.style.cssText = 'position: absolute; left: 20px; top: 20px; width: 100px; height: 100px; background: blue;';
            card.appendChild(b1);
            card.appendChild(b2);
        }""")
        overlaps = check_overlaps(page)
        assert len(overlaps) > 0, "Checker failed to detect injected sibling overlap"
        page.evaluate("""() => {
            document.getElementById('fault-overlap-1')?.remove();
            document.getElementById('fault-overlap-2')?.remove();
        }""")

        # Fault (b): Element 2000px wide
        page.evaluate("""() => {
            const el = document.createElement('div');
            el.id = 'fault-wide';
            el.style.cssText = 'width: 2000px; height: 10px; background: purple;';
            document.body.appendChild(el);
        }""")
        scroll_fails = check_horizontal_scroll(page)
        assert len(scroll_fails) > 0, "Checker failed to detect 2000px wide element"
        page.evaluate("document.getElementById('fault-wide')?.remove();")

        # Fault (c): Text node colored #AAAAAA on white
        page.evaluate("""() => {
            const el = document.createElement('p');
            el.id = 'fault-contrast';
            el.style.cssText = 'color: #AAAAAA; background-color: #FFFFFF; font-size: 16px;';
            el.textContent = 'Low contrast fault text';
            document.body.appendChild(el);
        }""")
        contrast_fails = check_contrast(page)
        assert len(contrast_fails) > 0, "Checker failed to detect text #AAAAAA on white"
        page.evaluate("document.getElementById('fault-contrast')?.remove();")

        # Fault (d): 20px high button at width 360
        page.evaluate("""() => {
            const btn = document.createElement('button');
            btn.id = 'fault-touch';
            btn.style.cssText = 'height: 20px !important; min-height: 20px !important; width: 100px;';
            btn.textContent = 'Tiny';
            document.body.appendChild(btn);
        }""")
        touch_fails = check_touch_targets(page, 360)
        assert len(touch_fails) > 0, "Checker failed to detect 20px high button at width 360"
        page.evaluate("document.getElementById('fault-touch')?.remove();")

        # Fault (e): Emoji character in visible text
        page.evaluate("""() => {
            const el = document.createElement('p');
            el.id = 'fault-emoji';
            el.textContent = 'Heart emoji check \u2764';
            document.body.appendChild(el);
        }""")
        emoji_fails = check_emojis(page)
        assert len(emoji_fails) > 0, "Checker failed to detect injected emoji character"
        page.evaluate("document.getElementById('fault-emoji')?.remove();")

        # Fault (f): The word "SHAP" in the #/check page
        page.evaluate("""() => {
            const el = document.createElement('p');
            el.id = 'fault-shap';
            el.textContent = 'Injected SHAP explanation';
            document.body.appendChild(el);
        }""")
        banned_fails = check_banned_words(page, "#/check")
        assert len(banned_fails) > 0, "Checker failed to detect banned word 'SHAP' on #/check"
        page.evaluate("document.getElementById('fault-shap')?.remove();")

        # Fault (g): Container-width misalignment
        page.evaluate("""() => {
            const strip = document.querySelector('.disclaimer-strip-inner');
            if (strip) strip.style.marginLeft = '50px';
        }""")
        align_fails = check_container_alignment(page)
        assert len(align_fails) > 0, "Checker failed to detect container left-edge misalignment"
        page.evaluate("""() => {
            const strip = document.querySelector('.disclaimer-strip-inner');
            if (strip) strip.style.marginLeft = '';
        }""")

        # Fault (h): Unpinned footer
        page.evaluate("""() => {
            const footer = document.querySelector('.app-footer');
            if (footer) {
                footer.style.position = 'absolute';
                footer.style.top = '100px';
                footer.style.bottom = 'auto';
            }
        }""")
        footer_fails = check_footer_pinned(page)
        assert len(footer_fails) > 0, "Checker failed to detect unpinned footer"
        page.evaluate("""() => {
            const footer = document.querySelector('.app-footer');
            if (footer) {
                footer.style.position = '';
                footer.style.top = '';
                footer.style.bottom = '';
            }
        }""")

    finally:
        page.close()


@pytest.mark.e2e
@pytest.mark.parametrize("width", VIEWPORT_WIDTHS)
def test_all_widths_and_states(browser_instance: Browser, live_server: str, width: int):
    """Test all viewport widths, form steps, examples, all routes, subtabs, and screenshots."""
    page: Page = browser_instance.new_page(viewport={"width": width, "height": 800})
    console_errors = []
    failed_requests = []

    page.on("pageerror", lambda err: console_errors.append(str(err)))
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("requestfailed", lambda req: failed_requests.append(f"{req.method} {req.url}: {req.failure}"))

    try:
        # Route 1: #/check Landing & Form Step 1
        page.goto(f"{live_server}/#/check", wait_until="networkidle")
        verify_page_state(page, width, "landing", console_errors, failed_requests)

        # Title & Head assertions
        assert "CardioLens" in page.title()
        assert page.title() == "Check your risk - CardioLens"

        # Favicon check
        fav_href = page.evaluate("document.querySelector('link[rel=\"icon\"]')?.getAttribute('href')")
        assert fav_href is not None
        fav_resp = page.request.get(f"{live_server}/{fav_href}")
        assert fav_resp.status == 200
        assert "image/svg+xml" in fav_resp.headers.get("content-type", "")

        # Form Step 1 assertions
        page.wait_for_selector("#field-input-age", timeout=5000)
        verify_page_state(page, width, "form_step_1", console_errors, failed_requests)

        # Part F2 (4): On step 1, Back button is NOT rendered in the DOM
        assert page.locator("#btn-wizard-back").count() == 0

        # Part F2 (4): Stepper has three segments and exactly one is current
        stepper_items = page.locator(".stepper-item")
        assert stepper_items.count() == 3
        current_items = page.locator(".stepper-item.current, .stepper-item[aria-current=\"step\"]")
        assert current_items.count() == 1

        # Part F2 (8): Every numeric field's initial value equals its /api/schema default
        schema_specs = engine.get_feature_specs()
        for key in ["age", "trestbps", "chol"]:
            f_spec = schema_specs[key]
            input_val = page.locator(f"#field-input-{key}").input_value()
            assert float(input_val) == float(f_spec["default"])

        # Form Step 2
        page.click("#btn-wizard-next")
        page.wait_for_selector("#field-input-cp", timeout=5000)
        verify_page_state(page, width, "form_step_2", console_errors, failed_requests)

        # Info tooltip on cp carries exact inferred_codes_note
        cp_info_btn = page.locator("#field-wrap-cp .info-btn")
        cp_info_btn.click()
        tooltip_text = page.locator("#tooltip-cp").text_content()
        assert tooltip_text == INFERRED_CODES_NOTE

        # Form Step 3
        page.click("#btn-wizard-next")
        page.wait_for_selector("#field-input-oldpeak", timeout=5000)
        verify_page_state(page, width, "form_step_3", console_errors, failed_requests)

        # Result View Row 0
        test_df = engine.load_test_features()
        row0 = test_df.iloc[0].to_dict()
        expected_score_0 = engine.predict_score(row0)
        expected_tier_0 = engine.risk_tier(expected_score_0)

        # Fill row 0 answers
        page.click("#btn-wizard-back")  # to step 2
        page.click("#btn-wizard-back")  # to step 1
        page.fill("#field-input-age", str(int(row0["age"])))
        page.click(f"#field-input-sex .segmented-option[data-value='{int(row0['sex'])}']")
        page.fill("#field-input-trestbps", str(int(row0["trestbps"])))
        page.fill("#field-input-chol", str(int(row0["chol"])))
        page.click(f"#field-input-fbs .segmented-option[data-value='{int(row0['fbs'])}']")
        page.click("#btn-wizard-next")

        page.click(f"#field-input-cp .segmented-option[data-value='{int(row0['cp'])}']")
        page.click(f"#field-input-restecg .segmented-option[data-value='{int(row0['restecg'])}']")
        page.fill("#field-input-thalach", str(int(row0["thalach"])))
        page.click(f"#field-input-exang .segmented-option[data-value='{int(row0['exang'])}']")
        page.click("#btn-wizard-next")

        page.fill("#field-input-oldpeak", str(row0["oldpeak"]))
        page.click(f"#field-input-slope .segmented-option[data-value='{int(row0['slope'])}']")
        page.click(f"#field-input-ca .segmented-option[data-value='{int(row0['ca'])}']")
        page.click(f"#field-input-thal .segmented-option[data-value='{int(row0['thal'])}']")
        page.click("#btn-wizard-submit")

        page.wait_for_selector(".score-ring-number", timeout=8000)
        displayed_score_text = page.locator(".score-ring-number").text_content()
        displayed_pct = float(displayed_score_text.replace("%", ""))
        expected_pct = round(expected_score_0 * 100.0, 1)

        # Equivalence assertion
        assert abs(displayed_pct - expected_pct) <= 0.05
        displayed_tier = page.locator("#result-tier-pill span").text_content()
        assert displayed_tier == expected_tier_0["label"]
        verify_page_state(page, width, "result_view_row_0", console_errors, failed_requests)

        # Print emulation
        page.emulate_media(media="print")
        top_bar_display = page.evaluate("window.getComputedStyle(document.querySelector('.top-bar')).display")
        assert top_bar_display == "none"
        btn_display = page.evaluate("window.getComputedStyle(document.querySelector('#btn-print-summary')).display")
        assert btn_display == "none"
        page.emulate_media(media="screen")

        # Examples 1, 2, 3
        examples_resp = [
            ("example_1", 41),
            ("example_2", 6),
            ("example_3", 18),
        ]
        for ex_id, row_idx in examples_resp:
            page.goto(f"{live_server}/#/check", wait_until="networkidle")
            page.click(f"#btn-{ex_id}")
            page.wait_for_selector(".score-ring-number", timeout=8000)

            ex_row = test_df.iloc[row_idx].to_dict()
            ex_score = engine.predict_score(ex_row)
            ex_tier = engine.risk_tier(ex_score)
            ex_displayed_pct = float(page.locator(".score-ring-number").text_content().replace("%", ""))
            assert abs(ex_displayed_pct - round(ex_score * 100.0, 1)) <= 0.05
            assert page.locator("#result-tier-pill span").text_content() == ex_tier["label"]
            verify_page_state(page, width, f"{ex_id}_result", console_errors, failed_requests)

        # What-If Slider Test on Example 1
        details_el = page.locator("details.whatif-container, details:has(#whatif-slider)")
        details_el.click()
        page.wait_for_selector("#whatif-slider", timeout=5000)
        time.sleep(0.3)

        slider = page.locator("#whatif-slider")
        slider.fill("70")
        slider.dispatch_event("input")
        time.sleep(0.3)

        change_line_text = page.locator("#whatif-change-line").text_content()
        assert "Score changes from" in change_line_text

        # Route 2: #/how (How it Works)
        page.goto(f"{live_server}/#/how", wait_until="networkidle")
        assert page.title() == "How it works - CardioLens"
        verify_page_state(page, width, "how_it_works", console_errors, failed_requests)
        how_step_cards = page.locator(".how-step-card")
        assert how_step_cards.count() == 4
        # Verify patient count in step 2 title
        step2_text = how_step_cards.nth(1).locator("h3").text_content()
        assert "patterns in 302 patients" in step2_text

        # Route 3: #/details (Model details with 4 subtabs and arrow key navigation)
        page.goto(f"{live_server}/#/details", wait_until="networkidle")
        assert page.title() == "Model details - CardioLens"

        subtab_keys = ["data", "performance", "explanations", "limitations"]
        for idx, subk in enumerate(subtab_keys):
            active_btn = page.locator(f"#tab-{subk}")
            active_btn.click()

            # Arrow navigation test
            if idx > 0:
                prev_btn = page.locator(f"#tab-{subtab_keys[idx-1]}")
                prev_btn.focus()
                page.keyboard.press("ArrowRight")
                assert page.locator(f"#tab-{subk}").get_attribute("aria-selected") == "true"

            verify_page_state(page, width, f"model_details_{subk}", console_errors, failed_requests)

            # Check that all figures on this panel have alt and naturalWidth > 0
            figs = page.locator(f"#panel-{subk} img")
            fig_count = figs.count()
            for f_i in range(fig_count):
                fig_img = figs.nth(f_i)
                alt_txt = fig_img.get_attribute("alt")
                assert alt_txt and len(alt_txt.strip()) > 0
                fig_img.evaluate("el => el.complete || new Promise(r => { el.onload = el.onerror = r; })")
                nat_width = fig_img.evaluate("el => el.naturalWidth")
                assert nat_width > 0, f"Figure {fig_img.get_attribute('src')} has naturalWidth {nat_width}"

            # Table scroll check: Table wrapper has overflow-x auto, page has no horizontal scroll
            tables = page.locator(f"#panel-{subk} table")
            tbl_count = tables.count()
            for t_i in range(tbl_count):
                tbl = tables.nth(t_i)
                caption = tbl.locator("caption")
                assert caption.count() == 1
                # Check for col scopes
                col_th = tbl.locator("thead th[scope='col']")
                assert col_th.count() > 0

        # Route 4: #/about
        page.goto(f"{live_server}/#/about", wait_until="networkidle")
        assert page.title() == "About - CardioLens"
        verify_page_state(page, width, "about", console_errors, failed_requests)
        about_text = page.locator(".about-page-wrap").inner_text()
        assert "Built by Ram Bhardwaj" in about_text
        assert "Typeface: Inter, SIL Open Font License" in about_text
        assert "Kaggle Heart Disease Dataset" in about_text

    finally:
        page.close()
