/**
 * CardioLens Navigation Bar & Footer Component
 */

import { COPY } from "../copy.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export function createTopBar(disclaimerText = "") {
  const wrapper = document.createElement("div");

  // Sticky Bar
  const topBar = document.createElement("header");
  topBar.className = "top-bar";

  const container = document.createElement("div");
  container.className = "container top-bar-inner";

  // Brand Logo and Name
  const brandLink = document.createElement("a");
  brandLink.className = "brand-link";
  brandLink.setAttribute("href", "#/check");
  brandLink.setAttribute("aria-label", "CardioLens Home");

  const logoSvg = createLogoSvg();
  brandLink.appendChild(logoSvg);

  const brandName = document.createElement("span");
  brandName.textContent = COPY.brand.name;
  brandLink.appendChild(brandName);

  container.appendChild(brandLink);

  // Desktop Navigation
  const desktopNav = document.createElement("nav");
  desktopNav.setAttribute("aria-label", "Main Desktop Navigation");

  const desktopUl = document.createElement("ul");
  desktopUl.className = "nav-links-desktop";

  const navItems = [
    { hash: "#/check", label: COPY.nav.check },
    { hash: "#/how", label: COPY.nav.how },
    { hash: "#/details", label: COPY.nav.details },
    { hash: "#/about", label: COPY.nav.about },
  ];

  navItems.forEach((item) => {
    const li = document.createElement("li");
    const link = document.createElement("a");
    link.className = "nav-item-link";
    link.setAttribute("href", item.hash);
    link.textContent = item.label;
    li.appendChild(link);
    desktopUl.appendChild(li);
  });
  desktopNav.appendChild(desktopUl);

  // Nav Actions container (Theme toggle & Mobile menu)
  const navActions = document.createElement("div");
  navActions.className = "nav-actions";
  navActions.appendChild(desktopNav);

  // Theme Toggle Button
  const themeToggle = createThemeToggle();
  navActions.appendChild(themeToggle);

  // Mobile Menu Toggle Button (< 900 px)
  const menuBtn = document.createElement("button");
  menuBtn.type = "button";
  menuBtn.className = "menu-toggle-btn";
  menuBtn.setAttribute("aria-expanded", "false");
  menuBtn.setAttribute("aria-controls", "mobile-menu-panel");
  menuBtn.textContent = COPY.nav.menu;
  navActions.appendChild(menuBtn);

  container.appendChild(navActions);

  topBar.appendChild(container);

  // Mobile Dropdown Panel
  const mobilePanel = document.createElement("div");
  mobilePanel.id = "mobile-menu-panel";
  mobilePanel.className = "menu-panel";

  const mobileUl = document.createElement("ul");
  mobileUl.className = "nav-links-mobile";

  navItems.forEach((item) => {
    const li = document.createElement("li");
    const link = document.createElement("a");
    link.className = "nav-item-link";
    link.setAttribute("href", item.hash);
    link.textContent = item.label;
    link.addEventListener("click", () => {
      mobilePanel.classList.remove("open");
      menuBtn.setAttribute("aria-expanded", "false");
    });
    li.appendChild(link);
    mobileUl.appendChild(li);
  });

  mobilePanel.appendChild(mobileUl);
  topBar.appendChild(mobilePanel);

  // Toggle button handler
  menuBtn.addEventListener("click", () => {
    const isOpen = mobilePanel.classList.contains("open");
    mobilePanel.classList.toggle("open", !isOpen);
    menuBtn.setAttribute("aria-expanded", !isOpen ? "true" : "false");
  });

  wrapper.appendChild(topBar);

  // Slim Disclaimer Strip below bar
  if (disclaimerText) {
    const strip = document.createElement("aside");
    strip.className = "disclaimer-strip";
    strip.setAttribute("role", "note");

    const stripContainer = document.createElement("div");
    stripContainer.className = "container disclaimer-strip-inner";
    stripContainer.textContent = disclaimerText;

    strip.appendChild(stripContainer);
    wrapper.appendChild(strip);
  }

  return wrapper;
}

export function createFooter(modelInfo = null, disclaimerText = "") {
  const footer = document.createElement("footer");
  footer.className = "app-footer";

  const container = document.createElement("div");
  container.className = "container footer-inner";

  // Col 1: Product name and tagline
  const nameP = document.createElement("p");
  nameP.className = "footer-col footer-col-brand";
  nameP.textContent = `${COPY.brand.name} - ${COPY.brand.tagline}`;
  container.appendChild(nameP);

  // Col 2: Dataset line
  const metaP = document.createElement("p");
  metaP.className = "footer-col footer-col-dataset";
  const n = (modelInfo && modelInfo.dataset_facts && modelInfo.dataset_facts.unique_rows) ? modelInfo.dataset_facts.unique_rows : 302;
  metaP.textContent = `Kaggle Heart Disease Dataset, ${n} unique patients`;
  container.appendChild(metaP);

  // Col 3: Disclaimer sentence
  const discP = document.createElement("p");
  discP.className = "footer-col footer-col-disclaimer text-muted";
  discP.textContent = disclaimerText || COPY.disclaimer;
  container.appendChild(discP);

  footer.appendChild(container);
  return footer;
}

export function createLogoSvg() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "28");
  svg.setAttribute("height", "28");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "var(--color-brand)");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("class", "brand-logo");
  svg.setAttribute("aria-hidden", "true");

  // Heart Path
  const heart = document.createElementNS(SVG_NS, "path");
  heart.setAttribute(
    "d",
    "M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"
  );
  svg.appendChild(heart);

  // Lens Circle Ring
  const circle = document.createElementNS(SVG_NS, "circle");
  circle.setAttribute("cx", "12");
  circle.setAttribute("cy", "11");
  circle.setAttribute("r", "4.5");
  circle.setAttribute("stroke-width", "2");
  svg.appendChild(circle);

  return svg;
}

function createThemeToggle() {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "theme-toggle-btn";
  btn.id = "theme-toggle-btn";

  function updateIcon() {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const label = isDark ? "Switch to light theme" : "Switch to dark theme";
    btn.setAttribute("aria-label", label);
    btn.setAttribute("title", label);
    btn.replaceChildren(isDark ? createSunSvg() : createMoonSvg());
  }

  btn.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", nextTheme);
    try {
      localStorage.setItem("cardiolens_theme", nextTheme);
    } catch (e) {}
    updateIcon();
  });

  updateIcon();
  return btn;
}

function createSunSvg() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "20");
  svg.setAttribute("height", "20");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");

  const circle = document.createElementNS(SVG_NS, "circle");
  circle.setAttribute("cx", "12");
  circle.setAttribute("cy", "12");
  circle.setAttribute("r", "4");
  svg.appendChild(circle);

  const rays = [
    "M12 2v2",
    "M12 20v2",
    "M4.93 4.93l1.41 1.41",
    "M17.66 17.66l1.41 1.41",
    "M2 12h2",
    "M20 12h2",
    "M4.93 19.07l1.41-1.41",
    "M17.66 6.34l1.41-1.41",
  ];
  rays.forEach((d) => {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    svg.appendChild(path);
  });
  return svg;
}

function createMoonSvg() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "20");
  svg.setAttribute("height", "20");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");

  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z");
  svg.appendChild(path);
  return svg;
}

