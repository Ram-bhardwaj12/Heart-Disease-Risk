/**
 * CardioLens - About View (#/about)
 * Compliant with all Part D3 & D4 specifications:
 * - Short introduction
 * - "Built by Ram Bhardwaj"
 * - B.Tech CSE (Machine Learning and Data Science), Amity University Madhya Pradesh
 * - "Data Science working-model project"
 * - Kaggle Heart Disease Dataset facts from /api/model-info
 * - Technology chips: Python, scikit-learn, SHAP, FastAPI, HTML, CSS, JavaScript
 * - Font licence line: "Typeface: Inter, SIL Open Font License"
 * - Disclaimer card
 * - Zero enrollment number, zero teammate line, zero URLs, zero emoji
 */

import { fetchModelInfo } from "../api.js";
import { COPY } from "../copy.js";

export async function renderAboutView(container) {
  container.replaceChildren();

  const wrap = document.createElement("div");
  wrap.className = "about-page-wrap";

  // Page Header
  const header = document.createElement("header");
  header.className = "page-header";

  const h1 = document.createElement("h1");
  h1.className = "page-title";
  h1.textContent = "About CardioLens";
  header.appendChild(h1);

  const introP = document.createElement("p");
  introP.className = "page-intro text-muted";
  introP.textContent = "CardioLens is an interactive machine learning application designed to explore risk scoring patterns and explainable factors behind clinical measurements.";
  header.appendChild(introP);

  wrap.appendChild(header);

  // Fetch model info for dataset counts
  let rawRows = 1025;
  let uniqueRows = 302;
  try {
    const info = await fetchModelInfo();
    if (info && info.dataset_facts) {
      if (info.dataset_facts.raw_rows) rawRows = info.dataset_facts.raw_rows;
      if (info.dataset_facts.unique_rows) uniqueRows = info.dataset_facts.unique_rows;
    }
  } catch (err) {
    // Keep defaults
  }

  // 1. Author and Academic Project Card
  const authorCard = document.createElement("div");
  authorCard.className = "card about-card";

  const authorH2 = document.createElement("h2");
  authorH2.textContent = "Built by Ram Bhardwaj";
  authorCard.appendChild(authorH2);

  const degreeP = document.createElement("p");
  degreeP.className = "about-degree";
  degreeP.textContent = "B.Tech CSE (Machine Learning and Data Science), Amity University Madhya Pradesh";
  authorCard.appendChild(degreeP);

  const projectTag = document.createElement("p");
  projectTag.className = "about-project-type text-muted";
  projectTag.textContent = "Data Science working-model project";
  authorCard.appendChild(projectTag);

  wrap.appendChild(authorCard);

  // 2. Dataset Attribution Card
  const dataCard = document.createElement("div");
  dataCard.className = "card about-card";

  const dataH2 = document.createElement("h2");
  dataH2.textContent = "Dataset";
  dataCard.appendChild(dataH2);

  const dataP = document.createElement("p");
  dataP.textContent = `Kaggle Heart Disease Dataset (${rawRows} raw records, ${uniqueRows} unique patients after deduplication).`;
  dataCard.appendChild(dataP);

  wrap.appendChild(dataCard);

  // 3. Technologies Used
  const techCard = document.createElement("div");
  techCard.className = "card about-card";

  const techH2 = document.createElement("h2");
  techH2.textContent = "Technologies";
  techCard.appendChild(techH2);

  const chipsContainer = document.createElement("div");
  chipsContainer.className = "tech-chips-container";

  const techs = ["Python", "scikit-learn", "SHAP", "FastAPI", "HTML", "CSS", "JavaScript"];
  techs.forEach((t) => {
    const chip = document.createElement("span");
    chip.className = "tech-chip";
    chip.textContent = t;
    chipsContainer.appendChild(chip);
  });
  techCard.appendChild(chipsContainer);

  const fontLine = document.createElement("p");
  fontLine.className = "font-license-line text-muted";
  fontLine.textContent = "Typeface: Inter, SIL Open Font License";
  techCard.appendChild(fontLine);

  wrap.appendChild(techCard);

  // 4. Disclaimer Card
  const discCard = document.createElement("div");
  discCard.className = "card about-disclaimer-card";

  const discH2 = document.createElement("h2");
  discH2.textContent = "Disclaimer";
  discCard.appendChild(discH2);

  const discP = document.createElement("p");
  discP.className = "text-muted";
  discP.textContent = COPY.disclaimer;
  discCard.appendChild(discP);

  wrap.appendChild(discCard);

  container.appendChild(wrap);
}
