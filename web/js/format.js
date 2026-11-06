/**
 * CardioLens Formatting Utilities
 */

export function formatPercent(scoreOrPercent) {
  if (scoreOrPercent === null || scoreOrPercent === undefined) return "--%";
  const num = Number(scoreOrPercent);
  if (isNaN(num)) return "--%";
  // If in [0, 1] range, convert to [0, 100]
  const pct = num <= 1.0 && num >= 0.0 ? num * 100.0 : num;
  return `${pct.toFixed(1)}%`;
}

export function formatAmountPoints(amount) {
  if (amount === null || amount === undefined) return "0.0";
  const num = Number(amount);
  if (isNaN(num)) return "0.0";
  // amount x 100, one decimal
  const pts = Math.abs(num * 100.0);
  return pts.toFixed(1);
}

export function formatDate(date = new Date()) {
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

