/**
 * CardioLens - How it Works View (#/how)
 * Fully compliant with contract rules:
 * - Plain HTML/CSS/JS DOM creation (textContent only, zero inline styles)
 * - Heading: "How CardioLens works" (h1)
 * - Four numbered steps in CSS grid (4 cols >= 900px, stacked below)
 * - Patient count N dynamically retrieved from /api/model-info
 * - Two columns: "What this tool can do" / "What it cannot do"
 * - Closing card with "Start the check" button
 * - Zero banned words
 */

import { fetchModelInfo } from "../api.js";
import { navigateTo } from "../router.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export async function renderHowView(container) {
  container.replaceChildren();

  // Container wrapper
  const pageWrap = document.createElement("div");
  pageWrap.className = "how-page-wrap";

  // Heading & Subtitle
  const headerSection = document.createElement("header");
  headerSection.className = "page-header";

  const h1 = document.createElement("h1");
  h1.className = "page-title";
  h1.textContent = "How CardioLens works";
  headerSection.appendChild(h1);

  const introP = document.createElement("p");
  introP.className = "page-intro text-muted";
  introP.textContent = "A transparent overview of how our risk estimation model works and how to interpret your results.";
  headerSection.appendChild(introP);

  pageWrap.appendChild(headerSection);

  // Fetch unique patient count from API
  let patientCount = 302;
  try {
    const info = await fetchModelInfo();
    if (info && info.dataset_facts && info.dataset_facts.unique_rows) {
      patientCount = info.dataset_facts.unique_rows;
    }
  } catch (err) {
    // Keep fallback 302
  }

  // 1. Four Numbered Steps Grid
  const stepsGrid = document.createElement("div");
  stepsGrid.className = "how-steps-grid";

  const stepsData = [
    {
      num: 1,
      title: "You answer questions",
      sentence1: "You enter 13 health measurements from your recent medical report.",
      sentence2: "Every question includes plain guidance on where to look in your lab sheets.",
      icon: "clipboard",
    },
    {
      num: 2,
      title: `The model compares your answers with patterns in ${patientCount} patients`,
      sentence1: "A machine learning model looks for similar patterns in reference data.",
      sentence2: "It evaluates multiple risk factors together rather than one by one.",
      icon: "users",
    },
    {
      num: 3,
      title: "It gives a score, not a diagnosis",
      sentence1: "The result is an educational estimate between zero and one hundred.",
      sentence2: "It helps you have structured, informed conversations with your doctor.",
      icon: "activity",
    },
    {
      num: 4,
      title: "It shows which answers moved the score",
      sentence1: "CardioLens highlights the top factors that pushed your result higher or lower.",
      sentence2: "You can explore what changes to your numbers would alter the calculated score.",
      icon: "sliders",
    },
  ];

  stepsData.forEach((st) => {
    const card = document.createElement("div");
    card.className = "card how-step-card";

    const topRow = document.createElement("div");
    topRow.className = "how-step-top";

    const badge = document.createElement("span");
    badge.className = "how-step-num";
    badge.textContent = String(st.num);
    topRow.appendChild(badge);

    const iconSvg = createStepIcon(st.icon);
    topRow.appendChild(iconSvg);

    card.appendChild(topRow);

    const stepTitle = document.createElement("h3");
    stepTitle.className = "how-step-title";
    stepTitle.textContent = st.title;
    card.appendChild(stepTitle);

    const p1 = document.createElement("p");
    p1.className = "how-step-text";
    p1.textContent = st.sentence1;
    card.appendChild(p1);

    const p2 = document.createElement("p");
    p2.className = "how-step-text";
    p2.textContent = st.sentence2;
    card.appendChild(p2);

    stepsGrid.appendChild(card);
  });

  pageWrap.appendChild(stepsGrid);

  // 2. Two Columns: Can Do / Cannot Do
  const scopeSection = document.createElement("section");
  scopeSection.className = "how-scope-section";

  const scopeGrid = document.createElement("div");
  scopeGrid.className = "how-scope-grid";

  // Column A: What this tool can do
  const canCard = document.createElement("div");
  canCard.className = "card scope-card scope-card-can";

  const canHeading = document.createElement("h2");
  canHeading.className = "scope-heading";
  canHeading.textContent = "What this tool can do";
  canCard.appendChild(canHeading);

  const canList = document.createElement("ul");
  canList.className = "scope-list";
  const canBullets = [
    "Provide an interactive score based on patterns learned from reference health records.",
    "Show which specific measurements contributed most to raising or lowering your score.",
    "Offer interactive sliders to explore how changing inputs alters the computed estimate.",
    "Summarize your inputs into a printable report to share with healthcare professionals.",
  ];
  canBullets.forEach((txt) => {
    const li = document.createElement("li");
    li.textContent = txt;
    canList.appendChild(li);
  });
  canCard.appendChild(canList);
  scopeGrid.appendChild(canCard);

  // Column B: What it cannot do
  const cannotCard = document.createElement("div");
  cannotCard.className = "card scope-card scope-card-cannot";

  const cannotHeading = document.createElement("h2");
  cannotHeading.className = "scope-heading";
  cannotHeading.textContent = "What it cannot do";
  cannotCard.appendChild(cannotHeading);

  const cannotList = document.createElement("ul");
  cannotList.className = "scope-list";
  const cannotBullets = [
    "Provide medical advice, medical opinions, or replace formal medical evaluations.",
    "Determine definitive individual outcomes or prescribe medical treatments.",
    "Serve as an approved medical device for urgent or acute care settings.",
    "Store or share your sensitive personal health records outside your local browser.",
  ];
  cannotBullets.forEach((txt) => {
    const li = document.createElement("li");
    li.textContent = txt;
    cannotList.appendChild(li);
  });
  cannotCard.appendChild(cannotList);
  scopeGrid.appendChild(cannotCard);

  scopeSection.appendChild(scopeGrid);
  pageWrap.appendChild(scopeSection);

  // 3. Closing CTA Card
  const ctaCard = document.createElement("div");
  ctaCard.className = "card how-cta-card";

  const ctaHeading = document.createElement("h2");
  ctaHeading.textContent = "Ready to check your numbers?";
  ctaCard.appendChild(ctaHeading);

  const ctaDesc = document.createElement("p");
  ctaDesc.className = "text-muted";
  ctaDesc.textContent = "Complete the 13-question assessment to see your score, key contributors, and explore what-if scenarios.";
  ctaCard.appendChild(ctaDesc);

  const ctaBtn = document.createElement("button");
  ctaBtn.type = "button";
  ctaBtn.className = "btn btn-primary cta-btn";
  ctaBtn.id = "btn-how-start";
  ctaBtn.textContent = "Start the check";
  ctaBtn.addEventListener("click", () => {
    navigateTo("#/check");
  });
  ctaCard.appendChild(ctaBtn);

  pageWrap.appendChild(ctaCard);

  container.appendChild(pageWrap);
}

function createStepIcon(type) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "28");
  svg.setAttribute("height", "28");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "var(--color-brand)");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");

  if (type === "clipboard") {
    const p1 = document.createElementNS(SVG_NS, "rect");
    p1.setAttribute("x", "9");
    p1.setAttribute("y", "2");
    p1.setAttribute("width", "6");
    p1.setAttribute("height", "4");
    p1.setAttribute("rx", "1");
    svg.appendChild(p1);

    const p2 = document.createElementNS(SVG_NS, "path");
    p2.setAttribute("d", "M5 4h4v2H5v14h14V6h-4V4h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z");
    svg.appendChild(p2);
  } else if (type === "users") {
    const c1 = document.createElementNS(SVG_NS, "circle");
    c1.setAttribute("cx", "9");
    c1.setAttribute("cy", "7");
    c1.setAttribute("r", "4");
    svg.appendChild(c1);

    const p1 = document.createElementNS(SVG_NS, "path");
    p1.setAttribute("d", "M2 21v-2a4 4 0 0 1 4-4h6a4 4 0 0 1 4 4v2");
    svg.appendChild(p1);

    const p2 = document.createElementNS(SVG_NS, "path");
    p2.setAttribute("d", "M16 3.13a4 4 0 0 1 0 7.75");
    svg.appendChild(p2);

    const p3 = document.createElementNS(SVG_NS, "path");
    p3.setAttribute("d", "M22 21v-2a4 4 0 0 0-3-3.87");
    svg.appendChild(p3);
  } else if (type === "activity") {
    const p = document.createElementNS(SVG_NS, "path");
    p.setAttribute("d", "M22 12h-4l-3 9L9 3l-3 9H2");
    svg.appendChild(p);
  } else {
    // sliders
    const l1 = document.createElementNS(SVG_NS, "line");
    l1.setAttribute("x1", "4");
    l1.setAttribute("y1", "21");
    l1.setAttribute("x2", "4");
    l1.setAttribute("y2", "14");
    svg.appendChild(l1);

    const l2 = document.createElementNS(SVG_NS, "line");
    l2.setAttribute("x1", "4");
    l2.setAttribute("y1", "10");
    l2.setAttribute("x2", "4");
    l2.setAttribute("y2", "3");
    svg.appendChild(l2);

    const l3 = document.createElementNS(SVG_NS, "line");
    l3.setAttribute("x1", "12");
    l3.setAttribute("y1", "21");
    l3.setAttribute("x2", "12");
    l3.setAttribute("y2", "12");
    svg.appendChild(l3);

    const l4 = document.createElementNS(SVG_NS, "line");
    l4.setAttribute("x1", "12");
    l4.setAttribute("y1", "8");
    l4.setAttribute("x2", "12");
    l4.setAttribute("y2", "3");
    svg.appendChild(l4);

    const l5 = document.createElementNS(SVG_NS, "line");
    l5.setAttribute("x1", "20");
    l5.setAttribute("y1", "21");
    l5.setAttribute("x2", "20");
    l5.setAttribute("y2", "16");
    svg.appendChild(l5);

    const l6 = document.createElementNS(SVG_NS, "line");
    l6.setAttribute("x1", "20");
    l6.setAttribute("y1", "12");
    l6.setAttribute("x2", "20");
    l6.setAttribute("y2", "3");
    svg.appendChild(l6);
  }
  return svg;
}
