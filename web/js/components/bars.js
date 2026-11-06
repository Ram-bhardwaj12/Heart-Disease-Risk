/**
 * CardioLens Factor Impact Bars Component
 * Horizontal bars displaying factors that raised or lowered the score.
 * Never overlaps labels and bars; stacked layout.
 */

import { formatAmountPoints } from "../format.js";
import { COPY } from "../copy.js";

export function createFactorBars(topRaising = [], topLowering = [], allContributions = []) {
  const container = document.createElement("div");
  container.className = "factors-section";

  // Section Heading
  const heading = document.createElement("h3");
  heading.textContent = COPY.results.factorsHeading;
  container.appendChild(heading);

  // Combine top factors (up to 3 raising, up to 3 lowering)
  const topFactors = [...(topRaising || []).slice(0, 3), ...(topLowering || []).slice(0, 3)];

  // Determine max absolute amount for relative scaling
  let maxAmt = 0.01;
  topFactors.forEach((c) => {
    const abs = Math.abs(Number(c.amount) || 0);
    if (abs > maxAmt) maxAmt = abs;
  });

  if (topFactors.length === 0) {
    const emptyNotice = document.createElement("p");
    emptyNotice.className = "text-muted";
    emptyNotice.textContent = "No single answer moved the score by more than one percentage point.";
    container.appendChild(emptyNotice);
  } else {
    topFactors.forEach((factor) => {
      container.appendChild(renderFactorItem(factor, maxAmt));
    });
  }

  // Mandatory Caption
  const caption = document.createElement("p");
  caption.className = "text-muted";
  caption.style.setProperty("margin-top", "var(--space-1)");
  caption.textContent = COPY.results.factorsCaption;
  container.appendChild(caption);

  // Closed <details> for all 13 answers
  if (allContributions && allContributions.length > 0) {
    const details = document.createElement("details");
    details.style.setProperty("margin-top", "var(--space-2)");

    const summary = document.createElement("summary");
    summary.textContent = COPY.results.showAllAnswers;
    details.appendChild(summary);

    const listWrap = document.createElement("div");
    listWrap.className = "flex-col";
    listWrap.style.setProperty("margin-top", "var(--space-2)");

    allContributions.forEach((factor) => {
      listWrap.appendChild(renderFactorItem(factor, maxAmt));
    });

    details.appendChild(listWrap);
    container.appendChild(details);
  }

  return container;
}

function renderFactorItem(factor, maxAmt) {
  const item = document.createElement("div");
  item.className = "factor-item";

  const isRaised = factor.direction === "raised" || (Number(factor.amount) || 0) >= 0;
  const dirClass = isRaised ? "raised" : "lowered";
  const dirWord = isRaised ? COPY.results.raised : COPY.results.lowered;
  const pts = formatAmountPoints(factor.amount);

  // Label Row: Short label & value on left, direction & points on right
  const labelRow = document.createElement("div");
  labelRow.className = "factor-label-row";

  const nameSpan = document.createElement("span");
  nameSpan.textContent = `${factor.short_label || factor.label} (${factor.value_display})`;
  labelRow.appendChild(nameSpan);

  const tagSpan = document.createElement("span");
  tagSpan.className = `factor-change-tag ${dirClass}`;
  tagSpan.textContent = `${dirWord} ${pts} points`;
  labelRow.appendChild(tagSpan);

  item.appendChild(labelRow);

  // Bar Track and Fill
  const track = document.createElement("div");
  track.className = "factor-bar-track";

  const absAmt = Math.abs(Number(factor.amount) || 0);
  const barWidthPct = Math.min(100, Math.max(5, (absAmt / maxAmt) * 100));

  const fill = document.createElement("div");
  fill.className = `factor-bar-fill ${dirClass}`;
  fill.style.setProperty("width", `${barWidthPct}%`);
  track.appendChild(fill);

  item.appendChild(track);

  return item;
}

