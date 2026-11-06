/**
 * CardioLens - About View (#/about)
 * Finished placeholder card stating in one sentence what the page will contain.
 */

import { COPY } from "../copy.js";

export function renderAboutView(container) {
  container.replaceChildren();

  const card = document.createElement("div");
  card.className = "card";
  card.style.setProperty("max-width", "800px");
  card.style.setProperty("margin-inline", "auto");
  card.style.setProperty("margin-top", "var(--space-4)");

  const heading = document.createElement("h2");
  heading.textContent = COPY.nav.about;
  heading.style.setProperty("margin-bottom", "var(--space-2)");
  card.appendChild(heading);

  const text = document.createElement("p");
  text.textContent = COPY.placeholders.about;
  card.appendChild(text);

  container.appendChild(card);
}

