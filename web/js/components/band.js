/**
 * CardioLens Decision Cutoff Band Component
 * Horizontal bar showing three calibrated decision zones and a marker for the current score.
 * Zero inline style attributes (uses CSSOM setProperty).
 */

export function createCutoffBand(scorePercent, cutoffs = { low: 0.3, high: 0.5 }) {
  const container = document.createElement("div");
  container.className = "cutoff-band-wrap";

  const bar = document.createElement("div");
  bar.className = "cutoff-band-bar";

  const lowPct = (cutoffs.low || 0.3) * 100;
  const highPct = (cutoffs.high || 0.5) * 100;
  const modWidth = highPct - lowPct;
  const highWidth = 100 - highPct;

  const segLower = document.createElement("div");
  segLower.className = "cutoff-segment-lower";
  segLower.style.setProperty("width", `${lowPct}%`);
  bar.appendChild(segLower);

  const segModerate = document.createElement("div");
  segModerate.className = "cutoff-segment-moderate";
  segModerate.style.setProperty("width", `${modWidth}%`);
  bar.appendChild(segModerate);

  const segHigher = document.createElement("div");
  segHigher.className = "cutoff-segment-higher";
  segHigher.style.setProperty("width", `${highWidth}%`);
  bar.appendChild(segHigher);

  // Score Marker
  const clampedScore = Math.min(100, Math.max(0, Number(scorePercent) || 0));
  const marker = document.createElement("div");
  marker.className = "cutoff-marker";
  marker.style.setProperty("left", `${clampedScore}%`);
  bar.appendChild(marker);

  container.appendChild(bar);

  // Labels below the bar
  const labels = document.createElement("div");
  labels.className = "cutoff-labels";

  const lblLower = document.createElement("span");
  lblLower.textContent = `Lower (< ${Math.round(lowPct)}%)`;
  labels.appendChild(lblLower);

  const lblMod = document.createElement("span");
  lblMod.textContent = `Moderate (${Math.round(lowPct)}-${Math.round(highPct)}%)`;
  labels.appendChild(lblMod);

  const lblHigher = document.createElement("span");
  lblHigher.textContent = `Higher (>= ${Math.round(highPct)}%)`;
  labels.appendChild(lblHigher);

  container.appendChild(labels);

  // Mandatory Caption
  const caption = document.createElement("p");
  caption.className = "text-muted";
  caption.style.setProperty("margin-top", "var(--space-1)");
  caption.style.setProperty("text-align", "center");
  caption.textContent = "This is the model's score, not a measured probability.";
  container.appendChild(caption);

  return container;
}

