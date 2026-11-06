/**
 * CardioLens Score Ring Component
 * Inline SVG circle ring with animated progress stroke.
 * Zero inline style attributes (uses SVG attributes and CSS classes).
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const RADIUS = 40;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS; // ~251.327

export function createScoreRing(scorePercent, tierKey = "lower") {
  const container = document.createElement("div");
  container.className = "score-ring-wrap";

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "180");
  svg.setAttribute("height", "180");
  svg.setAttribute("viewBox", "0 0 100 100");
  svg.setAttribute("class", "score-ring-svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Score ring showing ${scorePercent}%`);

  // Background Track Circle
  const track = document.createElementNS(SVG_NS, "circle");
  track.setAttribute("cx", "50");
  track.setAttribute("cy", "50");
  track.setAttribute("r", String(RADIUS));
  track.setAttribute("fill", "none");
  track.setAttribute("stroke", "var(--color-surface-subtle)");
  track.setAttribute("stroke-width", "8");
  svg.appendChild(track);

  // Animated Foreground Circle
  const pct = Math.min(100, Math.max(0, Number(scorePercent) || 0));
  const offset = CIRCUMFERENCE * (1 - pct / 100);

  let strokeColor = "var(--color-tier-lower-text)";
  if (tierKey === "moderate") strokeColor = "var(--color-tier-moderate-text)";
  else if (tierKey === "higher") strokeColor = "var(--color-tier-higher-text)";

  const progress = document.createElementNS(SVG_NS, "circle");
  progress.setAttribute("cx", "50");
  progress.setAttribute("cy", "50");
  progress.setAttribute("r", String(RADIUS));
  progress.setAttribute("fill", "none");
  progress.setAttribute("stroke", strokeColor);
  progress.setAttribute("stroke-width", "8");
  progress.setAttribute("stroke-linecap", "round");
  progress.setAttribute("stroke-dasharray", String(CIRCUMFERENCE));
  progress.setAttribute("stroke-dashoffset", String(offset));
  svg.appendChild(progress);

  container.appendChild(svg);

  // Center Score Label
  const numberOverlay = document.createElement("div");
  numberOverlay.className = "score-ring-number";
  numberOverlay.textContent = `${pct.toFixed(1)}%`;
  container.appendChild(numberOverlay);

  return container;
}

