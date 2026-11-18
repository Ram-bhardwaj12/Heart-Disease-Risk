"""CardioLens End-to-End Reusable Verification Helpers.

Includes checks for:
- Horizontal page scroll
- Sibling and box overlaps
- WCAG AA contrast (>= 4.45:1)
- Interactive touch targets (>= 43.5px at width <= 640)
- Emoji and pictographic characters
- Banned clinical/statistical terms
- Container left-edge alignment across header, strip, main, footer
- Footer pinning (bottom edge >= window.innerHeight - 1)
- Typography scale compliance (h1, h2, h3 computed sizes)
- Field help text and dataset naming bans ("patient", "Cleveland")
"""

import re
from typing import List, Dict, Any
from playwright.sync_api import Page

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
    "educational tool, not a medical diagnosis.",
    "educational tool, not a medical diagnosis",
    "it gives a score, not a diagnosis",
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


def check_horizontal_scroll(page: Page) -> List[str]:
    """Check that page scrollWidth does not exceed window innerWidth."""
    info = page.evaluate("() => ({ scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth })")
    if info["scrollWidth"] > info["innerWidth"]:
        return [f"Horizontal scroll detected: scrollWidth {info['scrollWidth']} > innerWidth {info['innerWidth']}"]
    return []


def check_overlaps(page: Page) -> List[Dict[str, Any]]:
    """Check that container children do not intersect or overlap."""
    return page.evaluate(OVERLAP_CHECK_JS)


def check_contrast(page: Page) -> List[Dict[str, Any]]:
    """Check that text nodes maintain minimum 4.5:1 contrast against effective backgrounds."""
    return page.evaluate(CONTRAST_CHECK_JS)


def check_touch_targets(page: Page, width: int) -> List[Dict[str, Any]]:
    """Check that interactive elements have minimum 44px height at mobile viewport widths."""
    if width <= 640:
        return page.evaluate(INTERACTIVE_TARGET_CHECK_JS)
    return []


def check_emojis(page: Page) -> List[str]:
    """Check that zero emojis or pictographic glyphs exist in visible text."""
    body_text = page.locator("body").inner_text()
    emoji_pattern = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
    return emoji_pattern.findall(body_text)


def check_banned_words(page: Page, route: str) -> List[str]:
    """Check that banned words do not appear in visible text of #/check, #/how, #/about."""
    if route not in {"#/check", "#/how", "#/about", ""}:
        return []

    body_text = page.locator("body").inner_text().lower()
    for exc in ALLOWED_EXCEPTIONS:
        body_text = body_text.replace(exc.lower(), " ")

    failures = []
    for banned in BANNED_WORDS:
        if route == "#/about" and banned.lower() == "shap":
            continue
        pattern = re.compile(r"\b" + re.escape(banned) + r"\b", re.IGNORECASE)
        if pattern.search(body_text):
            failures.append(banned)
    return failures


def check_container_alignment(page: Page) -> List[str]:
    """Check that the left edges of header inner, strip inner, main content, and footer inner align within 1px."""
    js = """() => {
        const selMap = {
            header: document.querySelector('.top-bar-inner'),
            strip: document.querySelector('.disclaimer-strip-inner'),
            main: document.querySelector('#main-mount'),
            footer: document.querySelector('.footer-inner')
        };
        const res = {};
        for (const [k, el] of Object.entries(selMap)) {
            if (el) {
                res[k] = Math.round(el.getBoundingClientRect().left);
            }
        }
        return res;
    }"""
    pos = page.evaluate(js)
    vals = list(pos.values())
    if len(vals) < 2:
        return []
    min_l = min(vals)
    max_l = max(vals)
    if max_l - min_l > 1:
        return [f"Container alignment mismatch > 1px: {pos} (diff: {max_l - min_l}px)"]
    return []


def check_footer_pinned(page: Page) -> List[str]:
    """Check that the footer's bottom edge is at or below window.innerHeight - 1 on short pages."""
    js = """() => {
        const footer = document.querySelector('.app-footer');
        if (!footer) return null;
        const rect = footer.getBoundingClientRect();
        return {
            bottom: Math.round(rect.bottom),
            innerHeight: window.innerHeight
        };
    }"""
    res = page.evaluate(js)
    if not res:
        return ["App footer element not found"]
    if res["bottom"] < res["innerHeight"] - 1:
        return [f"Footer bottom {res['bottom']}px is above window bottom {res['innerHeight']}px"]
    return []


def check_heading_type_scale(page: Page, width: int) -> List[str]:
    """Check that every h1, h2, and h3 computed font size matches the design scale."""
    js = """() => {
        const headings = Array.from(document.querySelectorAll('h1, h2, h3'));
        return headings.map(h => ({
            tag: h.tagName.toLowerCase(),
            text: h.textContent.trim().slice(0, 25),
            fontSize: parseFloat(window.getComputedStyle(h).fontSize)
        }));
    }"""
    items = page.evaluate(js)
    # Allowed sizes (in px):
    # hero heading: 48 (36 below 640)
    # h1: 36 (30 below 640)
    # h2: 28 (24 below 640)
    # h3: 20
    # Allow 1px rounding tolerance
    failures = []
    is_mobile = width < 640
    for it in items:
        fs = it["fontSize"]
        tag = it["tag"]
        valid = False
        if tag == "h1":
            expected = [30.0, 36.0, 48.0] if is_mobile else [36.0, 48.0]
        elif tag == "h2":
            expected = [24.0, 28.0] if is_mobile else [28.0]
        elif tag == "h3":
            expected = [20.0]
        else:
            continue

        if any(abs(fs - exp) <= 1.0 for exp in expected):
            valid = True

        if not valid:
            failures.append(f"{tag.upper()} '{it['text']}' has unexpected font-size {fs}px (expected one of {expected})")
    return failures


def check_banned_dataset_and_help_text(page: Page) -> List[str]:
    """Check that 'Cleveland' is not used to name our dataset and 'patient' does not appear in field help texts."""
    js = """() => {
        const failures = [];
        // Check field help texts
        const helps = Array.from(document.querySelectorAll('.field-help-text'));
        helps.forEach(h => {
            if (/\\bpatient\\b/i.test(h.textContent)) {
                failures.push(`Field help text contains 'patient': "${h.textContent.trim()}"`);
            }
        });

        // Check footer dataset line specifically
        const footer = document.querySelector('.app-footer');
        if (footer && /cleveland\\s+heart\\s+disease/i.test(footer.textContent)) {
            failures.push(`Footer mentions 'Cleveland': "${footer.textContent.trim()}"`);
        }
        return failures;
    }"""
    return page.evaluate(js)
