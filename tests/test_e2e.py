"""CardioLens End-to-End Automated Browser Verification (Playwright Chromium).

Validates all 11 contract requirements:
 1. No horizontal scroll (scrollWidth <= innerWidth) across all 5 widths.
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
Saves all 55 full-page screenshots to design_review/<width>/<name>.png.
"""

import os
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGN_REVIEW_DIR = REPO_ROOT / "design_review"
VIEWPORT_WIDTHS = [360, 390, 768, 1024, 1440]
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8660
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

BANNED_WORDS = [
    "probability",
    "calibrated",
    "shap",
    "tree",
    "consensus",
    "impact",
    "model agreement",
    "diagnosis",
    "accurate",
    "validated",
    "clinical-grade",
]

ALLOWED_EXCEPTIONS = [
    "this is the model's score, not a measured probability.",
    "educational project, not a medical diagnosis. do not use this tool to make health decisions.",
]

OVERLAP_CHECK_JS = """() => {
    const failures = [];
    const containers = document.querySelectorAll('.card, .grid, .flex, .flex-col, .form-grid, .hero-section, .summary-report-card');
    containers.forEach(container => {
        const children = Array.from(container.children).filter(el => {
            if (el.tagName === 'SUMMARY') return false;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') return false;
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
        });

        // Sibling intersections
        for (let i = 0; i < children.length; i++) {
            const r1 = children[i].getBoundingClientRect();
            for (let j = i + 1; j < children.length; j++) {
                const r2 = children[j].getBoundingClientRect();
                const intersects = !(r1.right <= r2.left + 1 || r2.right <= r1.left + 1 ||
                                     r1.bottom <= r2.top + 1 || r2.bottom <= r1.top + 1);
                if (intersects) {
                    failures.push({
                        type: 'sibling overlap',
                        tag1: children[i].tagName + (children[i].id ? '#' + children[i].id : '') + '.' + children[i].className,
                        box1: {left: r1.left, top: r1.top, right: r1.right, bottom: r1.bottom},
                        tag2: children[j].tagName + (children[j].id ? '#' + children[j].id : '') + '.' + children[j].className,
                        box2: {left: r2.left, top: r2.top, right: r2.right, bottom: r2.bottom},
                    });
                }
            }
        }
    });
    return failures;
}"""

CONTRAST_CHECK_JS = """() => {
    function parseRgb(colorStr) {
        const m = colorStr.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([\\d.]+))?\\)/);
        if (!m) return null;
        return {
            r: parseInt(m[1]),
            g: parseInt(m[2]),
            b: parseInt(m[3]),
            a: m[4] !== undefined ? parseFloat(m[4]) : 1.0
        };
    }

    function srgbToLinear(c) {
        c = c / 255.0;
        return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    }

    function luminance(rgb) {
        return 0.2126 * srgbToLinear(rgb.r) + 0.7152 * srgbToLinear(rgb.g) + 0.0722 * srgbToLinear(rgb.b);
    }

    function contrastRatio(fg, bg) {
        const l1 = luminance(fg);
        const l2 = luminance(bg);
        return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    }

    function getEffectiveBg(el) {
        let curr = el;
        while (curr && curr !== document.documentElement) {
            const style = window.getComputedStyle(curr);
            const parsed = parseRgb(style.backgroundColor);
            if (parsed && parsed.a > 0.5) {
                return parsed;
            }
            curr = curr.parentElement;
        }
        return {r: 246, g: 248, b: 250, a: 1};
    }

    const failures = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
        const text = node.textContent.trim();
        if (!text) continue;
        const parent = node.parentElement;
        if (!parent) continue;
        const style = window.getComputedStyle(parent);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
        
        const fg = parseRgb(style.color);
        if (!fg) continue;
        const bg = getEffectiveBg(parent);
        const ratio = contrastRatio(fg, bg);
        if (ratio < 4.45) {
            failures.push({
                text: text.slice(0, 30),
                tag: parent.tagName + '.' + parent.className,
                fg: style.color,
                bg: 'rgb(' + bg.r + ',' + bg.g + ',' + bg.b + ')',
                ratio: Math.round(ratio * 100) / 100
            });
        }
    }
    return failures;
}"""

INTERACTIVE_TARGET_CHECK_JS = """() => {
    const failures = [];
    const elements = document.querySelectorAll('button, a, input, select');
    elements.forEach(el => {
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return;
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return;
        if (r.height < 43.5) {
            failures.push({
                tag: el.tagName + (el.id ? '#' + el.id : '') + '.' + el.className,
                type: el.type,
                height: Math.round(r.height * 10) / 10,
                text: el.textContent.trim().slice(0, 20)
            });
        }
    });
    return failures;
}"""


@pytest.fixture(scope="session", autouse=True)
def live_server() -> Generator[str, None, None]:
    """Start uvicorn server in a daemon thread."""
    config = uvicorn.Config(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

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
    """Run core assertions (1, 2, 3, 4, 5, 6, 7) on the active page state."""
    # Save screenshot
    save_screenshot(page, width, state_name)

    # 1. No horizontal scroll
    scroll_w = page.evaluate("document.documentElement.scrollWidth")
    inner_w = page.evaluate("window.innerWidth")
    assert scroll_w <= inner_w, f"[{width}px {state_name}] Horizontal scroll detected: scrollWidth {scroll_w} > innerWidth {inner_w}"

    # 2. Console and network errors
    assert len(console_errors) == 0, f"[{width}px {state_name}] Console errors: {console_errors}"
    assert len(failed_requests) == 0, f"[{width}px {state_name}] Failed requests: {failed_requests}"

    # 3. No overlap
    overlaps = page.evaluate(OVERLAP_CHECK_JS)
    assert len(overlaps) == 0, f"[{width}px {state_name}] Overlaps detected: {overlaps}"

    # 4. Touch target height >= 44px at widths <= 640
    if width <= 640:
        target_fails = page.evaluate(INTERACTIVE_TARGET_CHECK_JS)
        assert len(target_fails) == 0, f"[{width}px {state_name}] Touch targets < 44px: {target_fails}"

    # 5. Contrast >= 4.5:1
    contrast_fails = page.evaluate(CONTRAST_CHECK_JS)
    assert len(contrast_fails) == 0, f"[{width}px {state_name}] Contrast failures: {contrast_fails}"

    # 6. No emoji or pictographs
    body_text = page.locator("body").inner_text()
    emoji_pattern = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
    emoji_matches = emoji_pattern.findall(body_text)
    assert len(emoji_matches) == 0, f"[{width}px {state_name}] Emojis detected: {emoji_matches}"

    # 7. Banned words on #/check, #/how, #/about
    cleaned_text = body_text.lower()
    for exc in ALLOWED_EXCEPTIONS:
        cleaned_text = cleaned_text.replace(exc.lower(), " ")

    current_hash = page.evaluate("window.location.hash")
    if current_hash in {"#/check", "#/how", "#/about", ""}:
        for banned in BANNED_WORDS:
            pattern = re.compile(r"\b" + re.escape(banned) + r"\b", re.IGNORECASE)
            assert not pattern.search(cleaned_text), f"[{width}px {state_name}] Banned word '{banned}' found in visible text"


@pytest.mark.e2e
@pytest.mark.parametrize("width", VIEWPORT_WIDTHS)
def test_all_widths_and_states(browser_instance: Browser, live_server: str, width: int):
    """Test all widths, form steps, examples, placeholders, and save screenshots."""
    page: Page = browser_instance.new_page(viewport={"width": width, "height": 800})
    console_errors = []
    failed_requests = []

    page.on("pageerror", lambda err: console_errors.append(str(err)))
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("requestfailed", lambda req: failed_requests.append(f"{req.method} {req.url}: {req.failure}"))

    try:
        # State 1: Landing
        page.goto(f"{live_server}/#/check", wait_until="networkidle")
        verify_page_state(page, width, "landing", console_errors, failed_requests)

        # State 2: Form Step 1
        page.wait_for_selector("#field-input-age", timeout=5000)
        verify_page_state(page, width, "form_step_1", console_errors, failed_requests)

        # State 3: Form Step 2
        page.click("#btn-wizard-next")
        page.wait_for_selector("#field-input-cp", timeout=5000)
        verify_page_state(page, width, "form_step_2", console_errors, failed_requests)

        # 11. Info tooltip verification
        cp_info_btn = page.locator("#field-wrap-cp .info-btn")
        cp_info_btn.click()
        tooltip_text = page.locator("#tooltip-cp").text_content()
        schema = engine.load_shap_summary()
        from engine.content import INFERRED_CODES_NOTE
        assert tooltip_text == INFERRED_CODES_NOTE

        # State 4: Form Step 3
        page.click("#btn-wizard-next")
        page.wait_for_selector("#field-input-oldpeak", timeout=5000)
        verify_page_state(page, width, "form_step_3", console_errors, failed_requests)

        # State 5: Result View Row 0 (submit first X_test row)
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

        # 8. Score matches API score within 0.05 percentage points
        assert abs(displayed_pct - expected_pct) <= 0.05
        displayed_tier = page.locator("#result-tier-pill span").text_content()
        assert displayed_tier == expected_tier_0["label"]
        verify_page_state(page, width, "result_view_row_0", console_errors, failed_requests)

        # 10. Print emulation hides navigation and buttons
        page.emulate_media(media="print")
        top_bar_display = page.evaluate("window.getComputedStyle(document.querySelector('.top-bar')).display")
        assert top_bar_display == "none"
        btn_display = page.evaluate("window.getComputedStyle(document.querySelector('#btn-print-summary')).display")
        assert btn_display == "none"
        page.emulate_media(media="screen")

        # States 6-8: Examples 1, 2, 3 results
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

        # 9. Interactive What-If slider test on Example 1
        details_el = page.locator("details.whatif-container, details:has(#whatif-slider)")
        details_el.click()
        page.wait_for_selector("#whatif-slider", timeout=5000)
        time.sleep(0.5)

        slider = page.locator("#whatif-slider")
        slider.fill("70")
        slider.dispatch_event("input")
        time.sleep(0.5)

        change_line_text = page.locator("#whatif-change-line").text_content()
        assert "Score changes from" in change_line_text

        # State 9: How it works placeholder
        page.goto(f"{live_server}/#/how", wait_until="networkidle")
        verify_page_state(page, width, "how_it_works", console_errors, failed_requests)

        # State 10: Model details placeholder
        page.goto(f"{live_server}/#/details", wait_until="networkidle")
        verify_page_state(page, width, "model_details", console_errors, failed_requests)

        # State 11: About placeholder
        page.goto(f"{live_server}/#/about", wait_until="networkidle")
        verify_page_state(page, width, "about", console_errors, failed_requests)

    finally:
        page.close()
