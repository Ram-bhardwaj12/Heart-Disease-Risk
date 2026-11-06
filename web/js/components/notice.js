/**
 * CardioLens Notice Card Component
 * Displays physiological warning notices or missing value reliability warnings.
 */

import { COPY } from "../copy.js";

export function createNoticeCard(warnings = [], missingUsed = []) {
  if ((!warnings || warnings.length === 0) && (!missingUsed || missingUsed.length === 0)) {
    return null;
  }

  const card = document.createElement("div");
  card.className = "notice-box";
  card.setAttribute("role", "alert");

  if (warnings && warnings.length > 0) {
    const title = document.createElement("div");
    title.className = "notice-title";
    title.textContent = COPY.results.noticeWarning;
    card.appendChild(title);

    const list = document.createElement("ul");
    list.className = "notice-list";
    warnings.forEach((w) => {
      const li = document.createElement("li");
      li.textContent = `${w.field}: ${w.message}`;
      list.appendChild(li);
    });
    card.appendChild(list);
  }

  if (missingUsed && missingUsed.length > 0) {
    const title = document.createElement("div");
    title.className = "notice-title";
    title.style.setProperty("margin-top", warnings && warnings.length > 0 ? "var(--space-2)" : "0");
    title.textContent = COPY.results.noticeMissing;
    card.appendChild(title);

    const p = document.createElement("p");
    p.className = "text-muted";
    p.textContent = `Values missing or not performed: ${missingUsed.join(", ")}. Standard median or mode values were used.`;
    card.appendChild(p);
  }

  return card;
}

