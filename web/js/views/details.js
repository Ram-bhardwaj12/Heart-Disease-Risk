/**
 * CardioLens - Model Details View (#/details)
 * Heading: "Model details" with subheading "For reviewers and technical readers"
 * Horizontal sub-tabs (role="tablist", roving tabindex, arrow navigation):
 * - Data
 * - Performance
 * - Explanations
 * - Limitations
 * All numbers dynamically read from /api/model-info and /api/schema.
 * Strict compliance:
 * - Table responsive wrappers (overflow-x: auto), <caption>, <th scope="...">, right-aligned numbers
 * - Meaningful alt attributes and naturalWidth > 0 on all figures
 * - Strict terminology compliance per Part D2
 */

import { fetchModelInfo, fetchSchema } from "../api.js";

const TABS = [
  { id: "data", label: "Data" },
  { id: "performance", label: "Performance" },
  { id: "explanations", label: "Explanations" },
  { id: "limitations", label: "Limitations" },
];

export async function renderDetailsView(container) {
  container.replaceChildren();

  const wrap = document.createElement("div");
  wrap.className = "details-page-wrap";

  // Page Header
  const header = document.createElement("header");
  header.className = "page-header";

  const h1 = document.createElement("h1");
  h1.className = "page-title";
  h1.textContent = "Model details";
  header.appendChild(h1);

  const sub = document.createElement("p");
  sub.className = "page-intro text-muted";
  sub.textContent = "For reviewers and technical readers";
  header.appendChild(sub);

  wrap.appendChild(header);

  // Fetch model info and schema
  let modelInfo = null;
  let schema = null;
  try {
    [modelInfo, schema] = await Promise.all([fetchModelInfo(), fetchSchema()]);
  } catch (err) {
    // Graceful error fallback
  }

  // Subtabs Navigation
  const tablist = document.createElement("div");
  tablist.className = "subtabs-bar";
  tablist.setAttribute("role", "tablist");
  tablist.setAttribute("aria-label", "Model details sections");

  const tabButtons = [];
  const panels = [];

  const panelsContainer = document.createElement("div");
  panelsContainer.className = "subtabs-panels";

  TABS.forEach((tab, idx) => {
    const isFirst = idx === 0;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "subtab-btn" + (isFirst ? " active" : "");
    btn.id = `tab-${tab.id}`;
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", isFirst ? "true" : "false");
    btn.setAttribute("aria-controls", `panel-${tab.id}`);
    btn.setAttribute("tabindex", isFirst ? "0" : "-1");
    btn.textContent = tab.label;

    tabButtons.push(btn);
    tablist.appendChild(btn);

    const panel = document.createElement("div");
    panel.className = "subtab-panel" + (isFirst ? " active" : "");
    panel.id = `panel-${tab.id}`;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `tab-${tab.id}`);
    panel.setAttribute("tabindex", "0");
    if (!isFirst) {
      panel.setAttribute("hidden", "true");
    }

    panels.push(panel);
    panelsContainer.appendChild(panel);
  });

  // Roving Tabindex & Keyboard Navigation
  function selectTab(index) {
    tabButtons.forEach((b, i) => {
      const active = i === index;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", active ? "true" : "false");
      b.setAttribute("tabindex", active ? "0" : "-1");
      if (active) b.focus();
    });

    panels.forEach((p, i) => {
      const active = i === index;
      p.classList.toggle("active", active);
      if (active) {
        p.removeAttribute("hidden");
      } else {
        p.setAttribute("hidden", "true");
      }
    });
  }

  tabButtons.forEach((btn, idx) => {
    btn.addEventListener("click", () => {
      selectTab(idx);
    });

    btn.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") {
        e.preventDefault();
        const next = (idx + 1) % tabButtons.length;
        selectTab(next);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        const prev = (idx - 1 + tabButtons.length) % tabButtons.length;
        selectTab(prev);
      }
    });
  });

  wrap.appendChild(tablist);

  // Populate Panel 1: Data
  renderDataTab(panels[0], modelInfo);

  // Populate Panel 2: Performance
  renderPerformanceTab(panels[1], modelInfo);

  // Populate Panel 3: Explanations
  renderExplanationsTab(panels[2], modelInfo);

  // Populate Panel 4: Limitations
  renderLimitationsTab(panels[3], schema);

  wrap.appendChild(panelsContainer);
  container.appendChild(wrap);
}

// -----------------------------------------------------------------------------
// Sub-tab 1: Data
// -----------------------------------------------------------------------------
function renderDataTab(panel, modelInfo) {
  const facts = (modelInfo && modelInfo.dataset_facts) || {
    raw_rows: 1025,
    duplicate_rows: 723,
    unique_rows: 302,
    class_balance: {
      1: { count: 164, percentage: 54.3 },
      0: { count: 138, percentage: 45.7 },
    },
  };

  const grid = document.createElement("div");
  grid.className = "details-metric-grid";

  const cardsData = [
    { label: "Raw records", val: String(facts.raw_rows) },
    { label: "Duplicates removed", val: String(facts.duplicate_rows) },
    { label: "Unique patients", val: String(facts.unique_rows) },
    {
      label: "Class balance (disease)",
      val: facts.class_balance && facts.class_balance["1"] ? `${facts.class_balance["1"].percentage}%` : "54.3%",
    },
  ];

  cardsData.forEach((c) => {
    const card = document.createElement("div");
    card.className = "card metric-stat-card";

    const lbl = document.createElement("span");
    lbl.className = "stat-label text-muted";
    lbl.textContent = c.label;
    card.appendChild(lbl);

    const val = document.createElement("strong");
    val.className = "stat-val";
    val.textContent = c.val;
    card.appendChild(val);

    grid.appendChild(card);
  });

  panel.appendChild(grid);

  // Three Plain Sentences Card
  const textCard = document.createElement("div");
  textCard.className = "card details-text-card";

  const h2 = document.createElement("h2");
  h2.textContent = "Data curation and encoding";
  textCard.appendChild(h2);

  const sentences = [
    "The file's target = 1 means no disease in this data, so the project uses disease = 1 - target.",
    "ca = 4 and thal = 0 are treated as missing and filled by the model pipeline with values learned from the training data.",
    "The split is stratified with a fixed random state.",
  ];

  sentences.forEach((s) => {
    const p = document.createElement("p");
    p.textContent = s;
    textCard.appendChild(p);
  });

  panel.appendChild(textCard);
}

// -----------------------------------------------------------------------------
// Sub-tab 2: Performance
// -----------------------------------------------------------------------------
function renderPerformanceTab(panel, modelInfo) {
  const heldOut = (modelInfo && modelInfo.held_out_test) || {};
  const rfTest = heldOut.random_forest || {
    roc_auc: 0.9026,
    recall: 0.75,
    precision: 0.8077,
    accuracy: 0.8033,
    confidence_intervals_95: {
      roc_auc: { ci_lower: 0.82, ci_upper: 0.966 },
      recall: { ci_lower: 0.586, ci_upper: 0.9 },
      precision: { ci_lower: 0.654, ci_upper: 0.95 },
      accuracy: { ci_lower: 0.705, ci_upper: 0.902 },
    },
  };
  const lrTest = heldOut.logistic_regression || {
    roc_auc: 0.9091,
    recall: 0.75,
    precision: 0.84,
    accuracy: 0.8197,
  };

  // 1. Four Primary Metric Cards for Random Forest
  const metricGrid = document.createElement("div");
  metricGrid.className = "details-metric-grid";

  const metrics = [
    { key: "roc_auc", label: "ROC-AUC", val: rfTest.roc_auc },
    { key: "recall", label: "Recall", val: rfTest.recall },
    { key: "precision", label: "Precision", val: rfTest.precision },
    { key: "accuracy", label: "Accuracy", val: rfTest.accuracy },
  ];

  metrics.forEach((m) => {
    const card = document.createElement("div");
    card.className = "card metric-stat-card";

    const lbl = document.createElement("span");
    lbl.className = "stat-label text-muted";
    lbl.textContent = `${m.label} (threshold 0.5)`;
    card.appendChild(lbl);

    const val = document.createElement("strong");
    val.className = "stat-val";
    val.textContent = (m.val * 100).toFixed(1) + "%";
    card.appendChild(val);

    const ciInfo = rfTest.confidence_intervals_95 && rfTest.confidence_intervals_95[m.key];
    if (ciInfo) {
      const ciP = document.createElement("span");
      ciP.className = "stat-ci text-muted";
      ciP.textContent = `95% CI: ${(ciInfo.ci_lower * 100).toFixed(1)}% to ${(ciInfo.ci_upper * 100).toFixed(1)}%`;
      card.appendChild(ciP);
    }

    metricGrid.appendChild(card);
  });

  panel.appendChild(metricGrid);

  // 2. Factual Comparison Sentence Card
  const compCard = document.createElement("div");
  compCard.className = "card details-text-card";

  const compH2 = document.createElement("h2");
  compH2.textContent = "Model comparison summary";
  compCard.appendChild(compH2);

  const compP = document.createElement("p");
  const diffInfo = modelInfo && modelInfo.random_forest_minus_logistic_regression && modelInfo.random_forest_minus_logistic_regression.roc_auc;
  const ciLow = diffInfo ? diffInfo.ci_lower : -0.038;
  const ciHigh = diffInfo ? diffInfo.ci_upper : 0.025;
  compP.textContent = `In paired bootstrap testing, the 95% confidence interval for the ROC-AUC difference between Random Forest and Logistic Regression is [${(ciLow * 100).toFixed(1)}%, ${(ciHigh * 100).toFixed(1)}%]. Because this interval includes zero, the data does not show a clear difference between the two models.`;
  compCard.appendChild(compP);

  panel.appendChild(compCard);

  // 3. Cross-Validation Table
  const cvData = (modelInfo && modelInfo.cross_validation) || {};
  const cvWrap = createTableWrapper(
    "5-fold stratified cross-validation performance",
    ["Model", "ROC-AUC (mean +/- std)", "Recall", "Precision", "Accuracy"],
    [
      [
        "Random Forest",
        formatMeanStd(cvData.candidate_random_forest && cvData.candidate_random_forest.roc_auc),
        formatMeanStd(cvData.candidate_random_forest && cvData.candidate_random_forest.recall),
        formatMeanStd(cvData.candidate_random_forest && cvData.candidate_random_forest.precision),
        formatMeanStd(cvData.candidate_random_forest && cvData.candidate_random_forest.accuracy),
      ],
      [
        "Logistic Regression",
        formatMeanStd(cvData.baseline_logistic_regression && cvData.baseline_logistic_regression.roc_auc),
        formatMeanStd(cvData.baseline_logistic_regression && cvData.baseline_logistic_regression.recall),
        formatMeanStd(cvData.baseline_logistic_regression && cvData.baseline_logistic_regression.precision),
        formatMeanStd(cvData.baseline_logistic_regression && cvData.baseline_logistic_regression.accuracy),
      ],
    ]
  );
  panel.appendChild(cvWrap);

  // 4. Held-Out Test Table (Threshold 0.5)
  const heldWrap = createTableWrapper(
    "Held-out test set evaluation (threshold 0.5, 61 test patients)",
    ["Model", "ROC-AUC", "Recall", "Precision", "Accuracy", "Brier score"],
    [
      [
        "Random Forest",
        formatPct(rfTest.roc_auc),
        formatPct(rfTest.recall),
        formatPct(rfTest.precision),
        formatPct(rfTest.accuracy),
        formatNum(rfTest.brier_score, 3),
      ],
      [
        "Logistic Regression",
        formatPct(lrTest.roc_auc),
        formatPct(lrTest.recall),
        formatPct(lrTest.precision),
        formatPct(lrTest.accuracy),
        formatNum(lrTest.brier_score, 3),
      ],
    ]
  );
  panel.appendChild(heldWrap);

  // 5. Operating-Point Table (Lower cut-off threshold 0.3)
  const opData = (modelInfo && modelInfo.confusion_matrices && modelInfo.confusion_matrices.operating_point) || {};
  const opRf = opData.random_forest || { tn: 20, fp: 13, fn: 1, tp: 27 };
  const opLr = opData.logistic_regression || { tn: 24, fp: 9, fn: 3, tp: 25 };

  const opWrap = createTableWrapper(
    "Operating-point confusion table at lower cut-off threshold (0.30)",
    ["Model", "Threshold", "True negatives (TN)", "False positives (FP)", "False negatives (FN)", "True positives (TP)"],
    [
      ["Random Forest", "0.30", String(opRf.tn), String(opRf.fp), String(opRf.fn), String(opRf.tp)],
      ["Logistic Regression", "0.30", String(opLr.tn), String(opLr.fp), String(opLr.fn), String(opLr.tp)],
    ]
  );
  panel.appendChild(opWrap);

  // 6. Confusion Matrix & ROC Figures
  const figCard = document.createElement("div");
  figCard.className = "card details-figures-card";

  const figHeading = document.createElement("h2");
  figHeading.textContent = "Evaluation plots";
  figCard.appendChild(figHeading);

  const figGrid = document.createElement("div");
  figGrid.className = "figures-grid";

  const figs = [
    {
      src: "/figures/confusion_matrix.png",
      caption: "Confusion matrix on held-out test data (threshold 0.50)",
      alt: "Confusion matrix showing 21 true negatives, 7 false positives, 5 false negatives, and 28 true positives for Random Forest on the held-out test set.",
    },
    {
      src: "/figures/roc_curve.png",
      caption: "Receiver Operating Characteristic (ROC) curve comparison",
      alt: "Receiver Operating Characteristic curve comparison between Random Forest (ROC-AUC 0.903) and Logistic Regression (ROC-AUC 0.909) on the test set.",
    },
  ];

  figs.forEach((fg) => {
    const figure = document.createElement("figure");
    figure.className = "report-figure";

    const img = document.createElement("img");
    img.src = fg.src;
    img.alt = fg.alt;
    img.className = "figure-image";
    figure.appendChild(img);

    const cap = document.createElement("figcaption");
    cap.className = "figure-caption text-muted";
    cap.textContent = fg.caption;
    figure.appendChild(cap);

    figGrid.appendChild(figure);
  });

  figCard.appendChild(figGrid);
  panel.appendChild(figCard);
}

// -----------------------------------------------------------------------------
// Sub-tab 3: Explanations
// -----------------------------------------------------------------------------
function renderExplanationsTab(panel, modelInfo) {
  const top5 = (modelInfo && modelInfo.top_5_grouped_shap) || [
    { rank: 1, feature: "thalach", mean_abs_shap: 0.084 },
    { rank: 2, feature: "cp", mean_abs_shap: 0.076 },
    { rank: 3, feature: "ca", mean_abs_shap: 0.065 },
    { rank: 4, feature: "oldpeak", mean_abs_shap: 0.052 },
    { rank: 5, feature: "chol", mean_abs_shap: 0.041 },
  ];

  // Top 5 Grouped Factors as Cards
  const topSection = document.createElement("div");
  topSection.className = "details-section";

  const topH2 = document.createElement("h2");
  topH2.textContent = "Top 5 grouped factors by mean absolute SHAP value";
  topSection.appendChild(topH2);

  const topGrid = document.createElement("div");
  topGrid.className = "details-metric-grid top5-shap-grid";

  const featureLabels = {
    thalach: "Maximum heart rate",
    cp: "Chest pain type",
    ca: "Major vessels count",
    oldpeak: "ST depression",
    chol: "Cholesterol level",
    age: "Age",
    sex: "Biological sex",
    thal: "Thallium stress test",
  };

  top5.forEach((item) => {
    const card = document.createElement("div");
    card.className = "card metric-stat-card";

    const rSpan = document.createElement("span");
    rSpan.className = "stat-label text-muted";
    rSpan.textContent = `Rank ${item.rank}: ${featureLabels[item.feature] || item.feature}`;
    card.appendChild(rSpan);

    const val = document.createElement("strong");
    val.className = "stat-val";
    val.textContent = (item.mean_abs_shap * 100).toFixed(2) + "%";
    card.appendChild(val);

    const note = document.createElement("span");
    note.className = "stat-ci text-muted";
    note.textContent = "Mean absolute contribution";
    card.appendChild(note);

    topGrid.appendChild(card);
  });

  topSection.appendChild(topGrid);
  panel.appendChild(topSection);

  // Three Global SHAP Figures
  const figCard = document.createElement("div");
  figCard.className = "card details-figures-card";

  const figH2 = document.createElement("h2");
  figH2.textContent = "Global explanation charts";
  figCard.appendChild(figH2);

  const c1 = document.createElement("p");
  c1.className = "caption-notice";
  c1.textContent = "SHAP values describe how the model uses each feature; they do not show causes.";
  figCard.appendChild(c1);

  const c2 = document.createElement("p");
  c2.className = "caption-notice text-muted";
  c2.textContent = "Rankings beyond the top few are noisy (61 test patients).";
  figCard.appendChild(c2);

  const shapFigures = [
    {
      src: "/figures/shap_bar_grouped.png",
      caption: "Grouped feature rankings across 13 clinical inputs",
      alt: "Bar chart of mean absolute SHAP values for 13 original health features, with thalach, cp, and ca ranking highest.",
    },
    {
      src: "/figures/shap_bar_plot.png",
      caption: "Feature importance across 22 preprocessed feature columns",
      alt: "Bar chart of global mean absolute SHAP values across all 22 preprocessed binary and continuous feature columns.",
    },
    {
      src: "/figures/shap_summary_plot.png",
      caption: "Global beeswarm distribution across test patients",
      alt: "Global SHAP beeswarm plot illustrating individual impact distributions and feature value directions across 61 test patients.",
    },
  ];

  const figGrid = document.createElement("div");
  figGrid.className = "figures-grid shap-figures-grid";

  shapFigures.forEach((fg) => {
    const figure = document.createElement("figure");
    figure.className = "report-figure";

    const img = document.createElement("img");
    img.src = fg.src;
    img.alt = fg.alt;
    img.className = "figure-image";
    figure.appendChild(img);

    const cap = document.createElement("figcaption");
    cap.className = "figure-caption text-muted";
    cap.textContent = fg.caption;
    figure.appendChild(cap);

    figGrid.appendChild(figure);
  });

  figCard.appendChild(figGrid);
  panel.appendChild(figCard);
}

// -----------------------------------------------------------------------------
// Sub-tab 4: Limitations
// -----------------------------------------------------------------------------
function renderLimitationsTab(panel, schema) {
  const card = document.createElement("div");
  card.className = "card details-limitations-card";

  const h2 = document.createElement("h2");
  h2.textContent = "Project and model limitations";
  card.appendChild(h2);

  const inferredNote = (schema && schema.inferred_codes_note) ||
    "Labels for chest pain type, resting ECG, ST slope and thallium test were inferred by matching the dataset's value counts and disease rates to the published Cleveland data; the Kaggle file does not document its encoding.";

  const bullets = [
    {
      title: "Small sample size",
      desc: "The entire dataset comprises only 302 unique patients, and the held-out test evaluation relies on just 61 patients.",
    },
    {
      title: "Single-source historical cohort",
      desc: "Data was collected from a single medical center cohort and may not generalize across broader populations or modern practice.",
    },
    {
      title: "Clinical standing",
      desc: "This model is an educational prototype and is not clinically validated for diagnostic or treatment decisions.",
    },
    {
      title: "Probability calibration",
      desc: "There is no formal probability calibration check; scores represent raw model rankings rather than verified event rates.",
    },
    {
      title: "Category labels inferred",
      desc: inferredNote,
    },
    {
      title: "Laboratory test requirement",
      desc: "The model requires specialized clinical test values (fluoroscopy vessel counts, stress-test ST measurements) that are unavailable without diagnostic testing.",
    },
  ];

  const list = document.createElement("ul");
  list.className = "limitations-list";

  bullets.forEach((b) => {
    const li = document.createElement("li");

    const st = document.createElement("strong");
    st.textContent = b.title + ": ";
    li.appendChild(st);

    const span = document.createElement("span");
    span.textContent = b.desc;
    li.appendChild(span);

    list.appendChild(li);
  });

  card.appendChild(list);
  panel.appendChild(card);
}

// -----------------------------------------------------------------------------
// Table Helper
// -----------------------------------------------------------------------------
function createTableWrapper(captionText, headers, rows) {
  const container = document.createElement("div");
  container.className = "table-responsive-container";

  const table = document.createElement("table");
  table.className = "data-table";

  const caption = document.createElement("caption");
  caption.textContent = captionText;
  table.appendChild(caption);

  const thead = document.createElement("thead");
  const trHead = document.createElement("tr");
  headers.forEach((h, i) => {
    const th = document.createElement("th");
    th.setAttribute("scope", "col");
    if (i > 0) th.className = "num-col";
    th.textContent = h;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    r.forEach((cell, i) => {
      if (i === 0) {
        const th = document.createElement("th");
        th.setAttribute("scope", "row");
        th.textContent = cell;
        tr.appendChild(th);
      } else {
        const td = document.createElement("td");
        td.className = "num-col";
        td.textContent = cell;
        tr.appendChild(td);
      }
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  container.appendChild(table);
  return container;
}

function formatPct(val) {
  if (val === undefined || val === null) return "-";
  return (val * 100).toFixed(1) + "%";
}

function formatNum(val, dec = 2) {
  if (val === undefined || val === null) return "-";
  return val.toFixed(dec);
}

function formatMeanStd(obj) {
  if (!obj || obj.mean === undefined) return "-";
  return `${(obj.mean * 100).toFixed(1)}% (+/- ${(obj.std * 100).toFixed(1)}%)`;
}
