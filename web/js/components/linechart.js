/**
 * CardioLens What-If SVG Line Chart Component
 * Responsive SVG chart with 3 shaded tier bands, data curve, current value marker,
 * labelled axes, and accessible tooltips.
 * Zero inline style attributes.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

export function createWhatIfChart(curve = [], currentVal = null, cutoffs = { low: 0.3, high: 0.5 }, unit = "") {
  const container = document.createElement("div");
  container.className = "whatif-chart-wrap";

  if (!curve || curve.length === 0) {
    const empty = document.createElement("p");
    empty.className = "text-muted";
    empty.textContent = "No simulation curve data available.";
    container.appendChild(empty);
    return container;
  }

  const svgWidth = 560;
  const svgHeight = 240;
  const margin = { top: 20, right: 25, bottom: 40, left: 50 };
  const plotWidth = svgWidth - margin.left - margin.right;
  const plotHeight = svgHeight - margin.top - margin.bottom;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");
  svg.setAttribute("viewBox", `0 0 ${svgWidth} ${svgHeight}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Line chart showing simulated score changes across feature range");

  // X range
  const xValues = curve.map((p) => Number(p.value));
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const rangeX = maxX - minX || 1;

  // Y range: 0.0 to 1.0 (0% to 100%)
  const minY = 0.0;
  const maxY = 1.0;

  const scaleX = (val) => margin.left + ((val - minX) / rangeX) * plotWidth;
  const scaleY = (val) => margin.top + plotHeight - (val / maxY) * plotHeight;

  // 1. Shaded Tier Bands
  const lowC = cutoffs.low || 0.3;
  const highC = cutoffs.high || 0.5;

  // Lower Zone Band (0 to lowC)
  const bandLower = document.createElementNS(SVG_NS, "rect");
  bandLower.setAttribute("x", String(margin.left));
  bandLower.setAttribute("y", String(scaleY(lowC)));
  bandLower.setAttribute("width", String(plotWidth));
  bandLower.setAttribute("height", String(scaleY(0) - scaleY(lowC)));
  bandLower.setAttribute("fill", "#EBF7F0");
  bandLower.setAttribute("opacity", "0.7");
  svg.appendChild(bandLower);

  // Moderate Zone Band (lowC to highC)
  const bandMod = document.createElementNS(SVG_NS, "rect");
  bandMod.setAttribute("x", String(margin.left));
  bandMod.setAttribute("y", String(scaleY(highC)));
  bandMod.setAttribute("width", String(plotWidth));
  bandMod.setAttribute("height", String(scaleY(lowC) - scaleY(highC)));
  bandMod.setAttribute("fill", "#FEF7EC");
  bandMod.setAttribute("opacity", "0.7");
  svg.appendChild(bandMod);

  // Higher Zone Band (highC to 1.0)
  const bandHigher = document.createElementNS(SVG_NS, "rect");
  bandHigher.setAttribute("x", String(margin.left));
  bandHigher.setAttribute("y", String(scaleY(1.0)));
  bandHigher.setAttribute("width", String(plotWidth));
  bandHigher.setAttribute("height", String(scaleY(highC) - scaleY(1.0)));
  bandHigher.setAttribute("fill", "#FDF2F1");
  bandHigher.setAttribute("opacity", "0.7");
  svg.appendChild(bandHigher);

  // 2. Cutoff Guide Lines
  [lowC, highC].forEach((cutoffVal) => {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", String(margin.left));
    line.setAttribute("x2", String(margin.left + plotWidth));
    line.setAttribute("y1", String(scaleY(cutoffVal)));
    line.setAttribute("y2", String(scaleY(cutoffVal)));
    line.setAttribute("stroke", "#CBD5E1");
    line.setAttribute("stroke-width", "1");
    line.setAttribute("stroke-dasharray", "4 4");
    svg.appendChild(line);
  });

  // 3. Grid Axes
  const axisX = document.createElementNS(SVG_NS, "line");
  axisX.setAttribute("x1", String(margin.left));
  axisX.setAttribute("x2", String(margin.left + plotWidth));
  axisX.setAttribute("y1", String(margin.top + plotHeight));
  axisX.setAttribute("y2", String(margin.top + plotHeight));
  axisX.setAttribute("stroke", "var(--color-border)");
  axisX.setAttribute("stroke-width", "1");
  svg.appendChild(axisX);

  const axisY = document.createElementNS(SVG_NS, "line");
  axisY.setAttribute("x1", String(margin.left));
  axisY.setAttribute("x2", String(margin.left));
  axisY.setAttribute("y1", String(margin.top));
  axisY.setAttribute("y2", String(margin.top + plotHeight));
  axisY.setAttribute("stroke", "var(--color-border)");
  axisY.setAttribute("stroke-width", "1");
  svg.appendChild(axisY);

  // 4. Y Axis Labels
  [0.0, lowC, highC, 1.0].forEach((val) => {
    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", String(margin.left - 8));
    text.setAttribute("y", String(scaleY(val) + 4));
    text.setAttribute("text-anchor", "end");
    text.setAttribute("fill", "var(--color-muted)");
    text.setAttribute("font-size", "11");
    text.textContent = `${Math.round(val * 100)}%`;
    svg.appendChild(text);
  });

  // 5. X Axis Labels (min, mid, max)
  const midX = (minX + maxX) / 2;
  [minX, midX, maxX].forEach((val, i) => {
    const text = document.createElementNS(SVG_NS, "text");
    const anchor = i === 0 ? "start" : i === 2 ? "end" : "middle";
    text.setAttribute("x", String(scaleX(val)));
    text.setAttribute("y", String(margin.top + plotHeight + 22));
    text.setAttribute("text-anchor", anchor);
    text.setAttribute("fill", "var(--color-muted)");
    text.setAttribute("font-size", "11");
    const unitSuffix = unit ? ` ${unit}` : "";
    text.textContent = `${Number.isInteger(val) ? val : val.toFixed(1)}${unitSuffix}`;
    svg.appendChild(text);
  });

  // 6. Data Curve Path
  let dStr = "";
  curve.forEach((pt, idx) => {
    const x = scaleX(Number(pt.value));
    const y = scaleY(Number(pt.score));
    if (idx === 0) {
      dStr += `M ${x} ${y}`;
    } else {
      dStr += ` L ${x} ${y}`;
    }
  });

  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", dStr);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "var(--color-brand)");
  path.setAttribute("stroke-width", "3");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.appendChild(path);

  // 7. Active Point Marker
  if (currentVal !== null && currentVal !== undefined) {
    const numVal = Number(currentVal);
    // Find closest curve point
    let closest = curve[0];
    let minDiff = Math.abs(Number(closest.value) - numVal);
    curve.forEach((pt) => {
      const diff = Math.abs(Number(pt.value) - numVal);
      if (diff < minDiff) {
        minDiff = diff;
        closest = pt;
      }
    });

    const cx = scaleX(Number(closest.value));
    const cy = scaleY(Number(closest.score));

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", String(cx));
    circle.setAttribute("cy", String(cy));
    circle.setAttribute("r", "6");
    circle.setAttribute("fill", "var(--color-brand)");
    circle.setAttribute("stroke", "#FFFFFF");
    circle.setAttribute("stroke-width", "2");
    svg.appendChild(circle);

    // Callout badge above marker
    const tooltipText = document.createElementNS(SVG_NS, "text");
    tooltipText.setAttribute("x", String(cx));
    tooltipText.setAttribute("y", String(Math.max(15, cy - 12)));
    tooltipText.setAttribute("text-anchor", "middle");
    tooltipText.setAttribute("fill", "var(--color-ink)");
    tooltipText.setAttribute("font-size", "12");
    tooltipText.setAttribute("font-weight", "600");
    tooltipText.textContent = `${(Number(closest.score) * 100).toFixed(1)}%`;
    svg.appendChild(tooltipText);
  }

  container.appendChild(svg);
  return container;
}

