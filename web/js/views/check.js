/**
 * CardioLens - Check Your Risk View Controller (#/check)
 * Implements Hero, Try an example, 3-step wizard form, and Result view.
 * Zero inline style attributes (uses SVG attributes and CSS classes).
 * Strict wording compliance (no banned words).
 */

import { COPY } from "../copy.js";
import { formatPercent, formatAmountPoints, formatDate } from "../format.js";
import { predictScore, fetchWhatIf, pingHealth } from "../api.js";
import { getState, setInputs, resetInputs } from "../state.js";
import { createField, setFieldError, clearFieldError } from "../components/field.js";
import { createStepperHeader } from "../components/stepper.js";
import { createScoreRing } from "../components/ring.js";
import { createCutoffBand } from "../components/band.js";
import { createFactorBars } from "../components/bars.js";
import { createWhatIfChart } from "../components/linechart.js";
import { createNoticeCard } from "../components/notice.js";

const SVG_NS = "http://www.w3.org/2000/svg";

const STEP_FIELD_KEYS = {
  1: ["age", "sex", "trestbps", "chol", "fbs"],
  2: ["cp", "restecg", "thalach", "exang"],
  3: ["oldpeak", "slope", "ca", "thal"],
};

export function renderCheckView(container) {
  container.replaceChildren();

  const state = getState();
  const root = document.createElement("div");
  root.className = "check-view-root";

  // Section 1: Hero & Examples
  const heroSection = createHeroSection(state, (exampleInputs) => {
    // Fill all answers and submit directly
    setInputs(exampleInputs);
    executePrediction(root, exampleInputs);
  });
  root.appendChild(heroSection);

  // Section 2: Form Container Card
  const formCard = document.createElement("div");
  formCard.id = "assessment-form-card";
  formCard.className = "card form-card";
  renderFormStep(formCard, root, state.currentStep || 1);
  root.appendChild(formCard);

  // Section 3: Result Container Region (aria-live="polite")
  const resultRegion = document.createElement("div");
  resultRegion.id = "assessment-result-region";
  resultRegion.setAttribute("aria-live", "polite");
  resultRegion.style.setProperty("display", "none");
  root.appendChild(resultRegion);

  container.appendChild(root);
}

// -----------------------------------------------------------------------------
// Hero & Examples Section
// -----------------------------------------------------------------------------

function createHeroSection(state, onSelectExample) {
  const section = document.createElement("section");
  section.className = "hero-section";
  section.id = "hero-overview-section";

  const content = document.createElement("div");
  content.className = "hero-content";

  // Educational disclaimer badge
  const noteBadge = document.createElement("div");
  noteBadge.className = "hero-disclaimer-note";
  noteBadge.textContent = COPY.hero.disclaimerNote;
  content.appendChild(noteBadge);

  const headline = document.createElement("h1");
  headline.textContent = COPY.hero.headline;
  content.appendChild(headline);

  const quickLine = document.createElement("p");
  quickLine.className = "hero-quick-line";
  quickLine.textContent = COPY.hero.quickLine;
  content.appendChild(quickLine);

  // 3 Short Steps
  const stepsList = document.createElement("div");
  stepsList.className = "hero-steps-pills";
  COPY.hero.threeSteps.forEach((stepText) => {
    const pill = document.createElement("span");
    pill.className = "hero-step-pill";
    pill.textContent = stepText;
    stepsList.appendChild(pill);
  });
  content.appendChild(stepsList);

  const actions = document.createElement("div");
  actions.className = "hero-actions";

  const startBtn = document.createElement("button");
  startBtn.type = "button";
  startBtn.className = "btn btn-primary btn-large btn-hero-start";
  startBtn.id = "btn-start-check";
  startBtn.textContent = COPY.hero.startBtn;
  startBtn.addEventListener("click", () => {
    const formCard = document.getElementById("assessment-form-card");
    if (formCard) {
      const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      formCard.scrollIntoView({ behavior: prefersReduced ? "auto" : "smooth" });
      const firstInput = formCard.querySelector("input:not([type='checkbox']), select, button");
      if (firstInput) {
        firstInput.focus();
      }
    }
  });
  actions.appendChild(startBtn);

  const howLink = document.createElement("a");
  howLink.className = "btn btn-secondary btn-large";
  howLink.setAttribute("href", "#/how");
  howLink.textContent = COPY.hero.howLink;
  actions.appendChild(howLink);

  content.appendChild(actions);

  // Examples Sub-row
  const examplesDiv = document.createElement("div");
  examplesDiv.className = "examples-section";

  const exHeading = document.createElement("h3");
  exHeading.textContent = COPY.examples.heading;
  examplesDiv.appendChild(exHeading);

  const exCaption = document.createElement("p");
  exCaption.className = "text-muted";
  exCaption.textContent = COPY.examples.caption;
  examplesDiv.appendChild(exCaption);

  const buttonsRow = document.createElement("div");
  buttonsRow.className = "examples-buttons-row";

  const examplesList = state.examples || [];
  examplesList.forEach((ex) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-outline btn-example";
    btn.id = `btn-${ex.id}`;
    btn.textContent = ex.title;
    btn.addEventListener("click", () => {
      onSelectExample(ex.inputs);
    });
    buttonsRow.appendChild(btn);
  });

  examplesDiv.appendChild(buttonsRow);
  content.appendChild(examplesDiv);
  section.appendChild(content);

  // Abstract SVG Illustration on wide screens
  const illDiv = document.createElement("div");
  illDiv.className = "hero-illustration";
  illDiv.appendChild(createAbstractSvg());
  section.appendChild(illDiv);

  return section;
}

function createAbstractSvg() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "280");
  svg.setAttribute("height", "180");
  svg.setAttribute("viewBox", "0 0 280 180");
  svg.setAttribute("fill", "none");
  svg.setAttribute("role", "presentation");
  svg.setAttribute("aria-hidden", "true");

  // Soft Background Shield
  const bgRect = document.createElementNS(SVG_NS, "rect");
  bgRect.setAttribute("width", "280");
  bgRect.setAttribute("height", "180");
  bgRect.setAttribute("rx", "16");
  bgRect.setAttribute("fill", "var(--color-surface)");
  bgRect.setAttribute("stroke", "var(--color-border)");
  svg.appendChild(bgRect);

  // Gauge Arc Shape
  const arc = document.createElementNS(SVG_NS, "path");
  arc.setAttribute("d", "M 60 120 A 80 80 0 0 1 220 120");
  arc.setAttribute("stroke", "var(--color-brand-tint)");
  arc.setAttribute("stroke-width", "16");
  arc.setAttribute("stroke-linecap", "round");
  svg.appendChild(arc);

  const arcFilled = document.createElementNS(SVG_NS, "path");
  arcFilled.setAttribute("d", "M 60 120 A 80 80 0 0 1 150 44");
  arcFilled.setAttribute("stroke", "var(--color-brand)");
  arcFilled.setAttribute("stroke-width", "16");
  arcFilled.setAttribute("stroke-linecap", "round");
  svg.appendChild(arcFilled);

  // Heart-rate Wave Path
  const wave = document.createElementNS(SVG_NS, "path");
  wave.setAttribute("d", "M 40 120 L 90 120 L 105 85 L 120 145 L 135 105 L 150 120 L 240 120");
  wave.setAttribute("stroke", "var(--color-ink)");
  wave.setAttribute("stroke-width", "3");
  wave.setAttribute("stroke-linecap", "round");
  wave.setAttribute("stroke-linejoin", "round");
  svg.appendChild(wave);

  return svg;
}

// -----------------------------------------------------------------------------
// 3-Step Wizard Form
// -----------------------------------------------------------------------------

function renderFormStep(formCard, root, stepNumber) {
  // Fire-and-forget ping to wake up sleeping server when navigating form steps
  pingHealth();
  formCard.replaceChildren();
  const state = getState();
  const schema = state.schema;

  if (!schema || !schema.field_specs) {
    const loading = document.createElement("p");
    loading.className = "text-muted";
    loading.textContent = "Loading questions...";
    formCard.appendChild(loading);
    return;
  }

  // Stepper Header
  formCard.appendChild(createStepperHeader(stepNumber, 3));

  // Form Fields Container
  const formGrid = document.createElement("div");
  formGrid.className = "form-grid";

  const fieldKeys = STEP_FIELD_KEYS[stepNumber] || [];
  fieldKeys.forEach((key) => {
    const spec = schema.field_specs[key];
    if (spec) {
      const currentVal = state.inputs[key] !== undefined ? state.inputs[key] : spec.default;
      const fieldEl = createField(
        spec,
        currentVal,
        (changedKey, newVal) => {
          setInputs({ [changedKey]: newVal });
          clearFieldError(changedKey);
        },
        schema.inferred_codes_note
      );
      formGrid.appendChild(fieldEl);
    }
  });

  formCard.appendChild(formGrid);

  // Wizard Navigation Bar
  const navBar = document.createElement("div");
  navBar.className = "wizard-nav";

  if (stepNumber > 1) {
    const backBtn = document.createElement("button");
    backBtn.type = "button";
    backBtn.className = "btn btn-secondary btn-large";
    backBtn.textContent = COPY.stepper.back;
    backBtn.id = "btn-wizard-back";
    backBtn.addEventListener("click", () => {
      renderFormStep(formCard, root, stepNumber - 1);
    });
    navBar.appendChild(backBtn);
  }

  if (stepNumber < 3) {
    const nextBtn = document.createElement("button");
    nextBtn.type = "button";
    nextBtn.className = "btn btn-primary btn-large";
    nextBtn.textContent = COPY.stepper.next;
    nextBtn.id = "btn-wizard-next";
    nextBtn.addEventListener("click", () => {
      if (validateStep(stepNumber, schema)) {
        renderFormStep(formCard, root, stepNumber + 1);
      }
    });
    navBar.appendChild(nextBtn);
  } else {
    const submitBtn = document.createElement("button");
    submitBtn.type = "button";
    submitBtn.className = "btn btn-primary btn-large";
    submitBtn.textContent = COPY.stepper.submit;
    submitBtn.id = "btn-wizard-submit";
    submitBtn.addEventListener("click", () => {
      if (validateStep(3, schema)) {
        executePrediction(root, state.inputs);
      }
    });
    navBar.appendChild(submitBtn);
  }

  formCard.appendChild(navBar);
}

function validateStep(stepNumber, schema) {
  const state = getState();
  const keys = STEP_FIELD_KEYS[stepNumber] || [];
  let allValid = true;
  let firstInvalidKey = null;

  keys.forEach((key) => {
    clearFieldError(key);
    const spec = schema.field_specs[key];
    const val = state.inputs[key];

    // Optional field check
    if (spec.optional && (val === null || val === undefined)) {
      return;
    }

    if (val === null || val === undefined || val === "") {
      setFieldError(key, COPY.errors.required);
      if (!firstInvalidKey) firstInvalidKey = key;
      allValid = false;
      return;
    }

    const num = Number(val);
    if (isNaN(num)) {
      setFieldError(key, COPY.errors.invalidNumber);
      if (!firstInvalidKey) firstInvalidKey = key;
      allValid = false;
      return;
    }

    if (spec.min !== null && num < spec.min) {
      if (key === "age") {
        setFieldError(key, `Enter an age between ${Math.round(spec.min)} and ${Math.round(spec.max)}.`);
      } else {
        setFieldError(key, `Value must be at least ${spec.min}.`);
      }
      if (!firstInvalidKey) firstInvalidKey = key;
      allValid = false;
    } else if (spec.max !== null && num > spec.max) {
      if (key === "age") {
        setFieldError(key, `Enter an age between ${Math.round(spec.min)} and ${Math.round(spec.max)}.`);
      } else {
        setFieldError(key, `Value must be at most ${spec.max}.`);
      }
      if (!firstInvalidKey) firstInvalidKey = key;
      allValid = false;
    }
  });

  if (!allValid && firstInvalidKey) {
    const invalidInput = document.getElementById(`field-input-${firstInvalidKey}`);
    if (invalidInput) {
      invalidInput.scrollIntoView({ behavior: "smooth", block: "center" });
      invalidInput.focus();
    }
  }

  return allValid;
}

// -----------------------------------------------------------------------------
// Predict & Result View
// -----------------------------------------------------------------------------

async function executePrediction(root, inputs) {
  const formCard = document.getElementById("assessment-form-card");
  const resultRegion = document.getElementById("assessment-result-region");

  if (formCard) formCard.style.setProperty("display", "none");
  if (resultRegion) {
    resultRegion.style.setProperty("display", "block");
    renderLoadingSkeleton(resultRegion);
    resultRegion.scrollIntoView({ behavior: "smooth" });
  }

  try {
    const prediction = await predictScore(inputs);
    renderResultView(resultRegion, root, prediction, inputs);
  } catch (err) {
    if (err.status === 422 && err.data && err.data.errors) {
      // Return to form and display errors
      if (formCard) formCard.style.setProperty("display", "block");
      if (resultRegion) resultRegion.style.setProperty("display", "none");

      let firstErrStep = null;
      err.data.errors.forEach((e) => {
        setFieldError(e.field, e.message);
        if (!firstErrStep) {
          if (STEP_FIELD_KEYS[1].includes(e.field)) firstErrStep = 1;
          else if (STEP_FIELD_KEYS[2].includes(e.field)) firstErrStep = 2;
          else if (STEP_FIELD_KEYS[3].includes(e.field)) firstErrStep = 3;
        }
      });

      if (firstErrStep) {
        renderFormStep(formCard, root, firstErrStep);
      }
    } else {
      renderErrorRetryCard(resultRegion, () => executePrediction(root, inputs));
    }
  }
}

function renderLoadingSkeleton(container) {
  container.replaceChildren();

  const card = document.createElement("div");
  card.className = "card";
  card.style.setProperty("min-height", "300px");
  card.style.setProperty("display", "flex");
  card.style.setProperty("flex-direction", "column");
  card.style.setProperty("gap", "var(--space-3)");
  card.style.setProperty("align-items", "center");
  card.style.setProperty("justify-content", "center");

  const ringSkeleton = document.createElement("div");
  ringSkeleton.className = "skeleton-box";
  ringSkeleton.style.setProperty("width", "160px");
  ringSkeleton.style.setProperty("height", "160px");
  ringSkeleton.style.setProperty("border-radius", "50%");
  card.appendChild(ringSkeleton);

  const textSkeleton = document.createElement("div");
  textSkeleton.className = "skeleton-box";
  textSkeleton.style.setProperty("width", "240px");
  textSkeleton.style.setProperty("height", "24px");
  card.appendChild(textSkeleton);

  container.appendChild(card);
}

function renderErrorRetryCard(container, onRetry) {
  container.replaceChildren();

  const card = document.createElement("div");
  card.className = "card";
  card.style.setProperty("text-align", "center");
  card.style.setProperty("padding-block", "var(--space-4)");

  const p = document.createElement("p");
  p.textContent = COPY.errors.networkError;
  p.style.setProperty("margin-bottom", "var(--space-2)");
  card.appendChild(p);

  const retryBtn = document.createElement("button");
  retryBtn.type = "button";
  retryBtn.className = "btn btn-primary";
  retryBtn.textContent = COPY.errors.retryBtn;
  retryBtn.addEventListener("click", onRetry);
  card.appendChild(retryBtn);

  container.appendChild(card);
}

function renderResultView(container, root, data, originalInputs) {
  container.replaceChildren();
  const state = getState();
  const schema = state.schema;

  // Action Buttons Bar: Edit answers & Start over
  const topActions = document.createElement("div");
  topActions.className = "flex";
  topActions.style.setProperty("justify-content", "flex-end");
  topActions.style.setProperty("margin-bottom", "var(--space-2)");

  const editBtn = document.createElement("button");
  editBtn.type = "button";
  editBtn.className = "btn btn-secondary";
  editBtn.textContent = COPY.results.editAnswers;
  editBtn.id = "btn-edit-answers";
  editBtn.addEventListener("click", () => {
    container.style.setProperty("display", "none");
    const formCard = document.getElementById("assessment-form-card");
    if (formCard) {
      formCard.style.setProperty("display", "block");
      renderFormStep(formCard, root, 1);
      formCard.scrollIntoView({ behavior: "smooth" });
    }
  });
  topActions.appendChild(editBtn);

  const startOverBtn = document.createElement("button");
  startOverBtn.type = "button";
  startOverBtn.className = "btn btn-secondary";
  startOverBtn.textContent = COPY.results.startOver;
  startOverBtn.id = "btn-start-over";
  startOverBtn.addEventListener("click", () => {
    resetInputs();
    container.style.setProperty("display", "none");
    const formCard = document.getElementById("assessment-form-card");
    if (formCard) {
      formCard.style.setProperty("display", "block");
      renderFormStep(formCard, root, 1);
    }
    const hero = document.getElementById("hero-overview-section");
    if (hero) hero.scrollIntoView({ behavior: "smooth" });
  });
  topActions.appendChild(startOverBtn);
  container.appendChild(topActions);

  // A. Notice Card if warnings or missing_used present
  const noticeCard = createNoticeCard(data.warnings, data.missing_used);
  if (noticeCard) {
    container.appendChild(noticeCard);
  }

  // B. Score Card
  const scoreCard = document.createElement("div");
  scoreCard.className = "card score-card";

  const tierKey = (data.tier && data.tier.key) || "lower";
  const scoreRing = createScoreRing(data.score_percent, tierKey);
  scoreCard.appendChild(scoreRing);

  // Tier Pill with SVG icon
  const tierPill = document.createElement("div");
  tierPill.className = `tier-pill tier-${tierKey}`;
  tierPill.id = "result-tier-pill";

  const pillSvg = createTierIconSvg(tierKey);
  tierPill.appendChild(pillSvg);

  const pillLabel = document.createElement("span");
  pillLabel.textContent = (data.tier && data.tier.label) || "Lower";
  tierPill.appendChild(pillLabel);
  scoreCard.appendChild(tierPill);

  // Tier Text from API
  const tierTextP = document.createElement("p");
  tierTextP.className = "tier-text";
  tierTextP.textContent = (data.tier && data.tier.text) || "";
  scoreCard.appendChild(tierTextP);

  // Summary Sentence from API
  const summaryP = document.createElement("p");
  summaryP.className = "summary-sentence";
  summaryP.id = "result-summary-sentence";
  summaryP.textContent = data.summary || "";
  scoreCard.appendChild(summaryP);

  // Horizontal Cutoff Band
  const cutoffBand = createCutoffBand(data.score_percent, data.cutoffs);
  scoreCard.appendChild(cutoffBand);

  container.appendChild(scoreCard);

  // C. What Shaped This Score (Horizontal Factor Bars)
  const factorsCard = document.createElement("div");
  factorsCard.className = "card";
  factorsCard.style.setProperty("margin-top", "var(--space-3)");
  const factorBars = createFactorBars(data.top_raising, data.top_lowering, data.contributions);
  factorsCard.appendChild(factorBars);
  container.appendChild(factorsCard);

  // D. Try a Change (Closed <details> What-If Scenario)
  const whatIfCard = createWhatIfSection(originalInputs, data.score_percent, data.cutoffs, schema);
  container.appendChild(whatIfCard);

  // E. Your Summary (Printable Card)
  const summaryCard = createPrintableSummaryCard(data, originalInputs, schema);
  container.appendChild(summaryCard);

  // F. What to Do Next (3 Neutral Sentences)
  const nextCard = document.createElement("div");
  nextCard.className = "card";
  nextCard.style.setProperty("margin-top", "var(--space-3)");

  const nextHeading = document.createElement("h3");
  nextHeading.textContent = COPY.results.whatNextHeading;
  nextHeading.style.setProperty("margin-bottom", "var(--space-1)");
  nextCard.appendChild(nextHeading);

  COPY.results.whatNextSentences.forEach((sentence) => {
    const p = document.createElement("p");
    p.textContent = sentence;
    p.style.setProperty("margin-bottom", "var(--space-1)");
    nextCard.appendChild(p);
  });
  container.appendChild(nextCard);
}

function createTierIconSvg(tierKey) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "16");
  svg.setAttribute("height", "16");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2.5");
  svg.setAttribute("aria-hidden", "true");

  if (tierKey === "lower") {
    // Checkmark
    const poly = document.createElementNS(SVG_NS, "polyline");
    poly.setAttribute("points", "20 6 9 17 4 12");
    svg.appendChild(poly);
  } else if (tierKey === "moderate") {
    // Minus / Horizontal Line
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", "5");
    line.setAttribute("y1", "12");
    line.setAttribute("x2", "19");
    line.setAttribute("y2", "12");
    svg.appendChild(line);
  } else {
    // Arrow pointing up
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", "12");
    line.setAttribute("y1", "19");
    line.setAttribute("x2", "12");
    line.setAttribute("y2", "5");
    svg.appendChild(line);
    const poly = document.createElementNS(SVG_NS, "polyline");
    poly.setAttribute("points", "5 12 12 5 19 12");
    svg.appendChild(poly);
  }
  return svg;
}

// -----------------------------------------------------------------------------
// Interactive What-If Section
// -----------------------------------------------------------------------------

function createWhatIfSection(inputs, baselineScorePercent, cutoffs, schema) {
  const details = document.createElement("details");
  details.className = "card form-card";
  details.style.setProperty("margin-top", "var(--space-3)");

  const summary = document.createElement("summary");
  summary.textContent = COPY.results.whatIfHeading;
  details.appendChild(summary);

  const container = document.createElement("div");
  container.className = "whatif-container";

  // Select numeric feature
  const selectRow = document.createElement("div");
  selectRow.className = "field-wrapper";

  const lbl = document.createElement("label");
  lbl.className = "field-label";
  lbl.textContent = "Select answer to change:";
  selectRow.appendChild(lbl);

  const select = document.createElement("select");
  select.className = "field-select";
  select.id = "whatif-feature-select";

  const continuousKeys = ["age", "trestbps", "chol", "thalach", "oldpeak"];
  continuousKeys.forEach((key) => {
    const spec = schema && schema.field_specs ? schema.field_specs[key] : null;
    const option = document.createElement("option");
    option.value = key;
    option.textContent = spec ? spec.short_label : key;
    select.appendChild(option);
  });
  selectRow.appendChild(select);
  container.appendChild(selectRow);

  // Slider & Value Display
  const sliderRow = document.createElement("div");
  sliderRow.className = "number-slider-wrap";

  const sliderLabelRow = document.createElement("div");
  sliderLabelRow.className = "flex";
  sliderLabelRow.style.setProperty("justify-content", "space-between");

  const minSpan = document.createElement("span");
  minSpan.className = "text-muted";
  const valSpan = document.createElement("span");
  valSpan.className = "field-label";
  const maxSpan = document.createElement("span");
  maxSpan.className = "text-muted";

  sliderLabelRow.appendChild(minSpan);
  sliderLabelRow.appendChild(valSpan);
  sliderLabelRow.appendChild(maxSpan);
  sliderRow.appendChild(sliderLabelRow);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "field-range-slider";
  slider.id = "whatif-slider";
  sliderRow.appendChild(slider);
  container.appendChild(sliderRow);

  // Score Changes Line
  const changeLine = document.createElement("p");
  changeLine.className = "field-label";
  changeLine.id = "whatif-change-line";
  changeLine.textContent = COPY.results.whatIfScoreChange(baselineScorePercent, baselineScorePercent);
  container.appendChild(changeLine);

  // Chart Container Slot
  const chartSlot = document.createElement("div");
  chartSlot.id = "whatif-chart-slot";
  container.appendChild(chartSlot);

  // State & debouncing for What-If
  let activeFeature = "age";
  let currentVal = Number(inputs[activeFeature] || 50);
  let abortController = null;
  let debounceTimer = null;
  let cachedCurve = [];

  function updateSliderParams() {
    const spec = schema && schema.field_specs ? schema.field_specs[activeFeature] : null;
    const min = spec ? (activeFeature === "age" ? 18 : (spec.training_min !== undefined ? spec.training_min : spec.min)) : 20;
    const max = spec ? (activeFeature === "age" ? 100 : (spec.training_max !== undefined ? spec.training_max : spec.max)) : 80;
    const step = spec && spec.step ? spec.step : (activeFeature === "oldpeak" ? 0.1 : 1);

    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);

    currentVal = inputs[activeFeature] !== undefined && inputs[activeFeature] !== null ? Number(inputs[activeFeature]) : Number(spec.default || min);
    slider.value = String(currentVal);

    minSpan.textContent = `Min: ${min}`;
    maxSpan.textContent = `Max: ${max}`;
    valSpan.textContent = `Value: ${currentVal}`;
  }

  async function loadCurve() {
    if (abortController) abortController.abort();
    abortController = new AbortController();

    try {
      const resp = await fetchWhatIf(inputs, activeFeature, 25, abortController.signal);
      cachedCurve = resp.curve || [];
      renderChartAndLine();
    } catch (err) {
      if (err.name !== "AbortError") {
        console.warn("What-If fetch failed:", err);
      }
    }
  }

  function renderChartAndLine() {
    if (!cachedCurve || cachedCurve.length === 0) return;

    // Find nearest point
    let nearest = cachedCurve[0];
    let minDiff = Math.abs(Number(nearest.value) - currentVal);
    cachedCurve.forEach((pt) => {
      const diff = Math.abs(Number(pt.value) - currentVal);
      if (diff < minDiff) {
        minDiff = diff;
        nearest = pt;
      }
    });

    const newScorePct = (Number(nearest.score) * 100).toFixed(1);
    changeLine.textContent = COPY.results.whatIfScoreChange(baselineScorePercent, newScorePct);

    // Render SVG Line Chart
    const spec = schema && schema.field_specs ? schema.field_specs[activeFeature] : null;
    const chart = createWhatIfChart(cachedCurve, currentVal, cutoffs, spec ? spec.unit : "");
    chartSlot.replaceChildren(chart);
  }

  select.addEventListener("change", () => {
    activeFeature = select.value;
    updateSliderParams();
    loadCurve();
  });

  slider.addEventListener("input", () => {
    currentVal = Number(slider.value);
    valSpan.textContent = `Value: ${currentVal}`;

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      renderChartAndLine();
    }, 250);
  });

  // Load initial params on details opening
  details.addEventListener("toggle", () => {
    if (details.open && cachedCurve.length === 0) {
      updateSliderParams();
      loadCurve();
    }
  });

  details.appendChild(container);
  return details;
}

// -----------------------------------------------------------------------------
// Printable Summary Card
// -----------------------------------------------------------------------------

function createPrintableSummaryCard(data, inputs, schema) {
  const card = document.createElement("div");
  card.className = "summary-report-card";
  card.style.setProperty("margin-top", "var(--space-3)");

  const heading = document.createElement("h3");
  heading.textContent = COPY.results.summaryHeading;
  card.appendChild(heading);

  const metaP = document.createElement("p");
  metaP.className = "text-muted";
  metaP.textContent = `${COPY.brand.name} Risk Summary - ${formatDate(new Date())}`;
  card.appendChild(metaP);

  // Score & Tier summary row
  const scoreRow = document.createElement("p");
  scoreRow.className = "field-label";
  scoreRow.textContent = `Score: ${data.score_percent}% | Category: ${data.tier ? data.tier.label : "Lower"}`;
  card.appendChild(scoreRow);

  // Two-column list of answers
  const answersGrid = document.createElement("div");
  answersGrid.className = "answers-two-col";

  const allSpecs = (schema && schema.field_specs) || {};
  Object.keys(allSpecs).forEach((key) => {
    const spec = allSpecs[key];
    const val = inputs[key];

    const row = document.createElement("div");
    row.className = "answers-row";

    const labelSpan = document.createElement("span");
    labelSpan.textContent = spec.short_label || spec.label;
    row.appendChild(labelSpan);

    const valSpan = document.createElement("span");
    valSpan.style.setProperty("font-weight", "600");

    if (val === null || val === undefined) {
      valSpan.textContent = "Not performed / skipped";
    } else if (spec.type === "choice" && spec.options) {
      const matchOpt = spec.options.find((o) => o.value === val);
      valSpan.textContent = matchOpt ? matchOpt.label : String(val);
    } else {
      const unitStr = spec.unit ? ` ${spec.unit}` : "";
      valSpan.textContent = `${val}${unitStr}`;
    }

    row.appendChild(valSpan);
    answersGrid.appendChild(row);
  });
  card.appendChild(answersGrid);

  // Disclaimer sentence
  const discP = document.createElement("p");
  discP.className = "text-muted";
  discP.style.setProperty("margin-top", "var(--space-2)");
  discP.textContent = data.disclaimer || (schema ? schema.disclaimer : "");
  card.appendChild(discP);

  // Print & Copy Buttons
  const btnRow = document.createElement("div");
  btnRow.className = "flex";
  btnRow.style.setProperty("margin-top", "var(--space-2)");

  const printBtn = document.createElement("button");
  printBtn.type = "button";
  printBtn.className = "btn btn-primary";
  printBtn.id = "btn-print-summary";
  printBtn.textContent = COPY.results.printBtn;
  printBtn.addEventListener("click", () => {
    window.print();
  });
  btnRow.appendChild(printBtn);

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "btn btn-secondary";
  copyBtn.id = "btn-copy-summary";
  copyBtn.textContent = COPY.results.copyBtn;
  copyBtn.addEventListener("click", () => {
    const summaryText = buildPlainTextSummary(data, inputs, schema);
    navigator.clipboard.writeText(summaryText).then(() => {
      copyBtn.textContent = COPY.results.copiedFeedback;
      setTimeout(() => {
        copyBtn.textContent = COPY.results.copyBtn;
      }, 2500);
    });
  });
  btnRow.appendChild(copyBtn);

  card.appendChild(btnRow);
  return card;
}

function buildPlainTextSummary(data, inputs, schema) {
  const lines = [
    `CardioLens Heart Risk Summary`,
    `Date: ${formatDate(new Date())}`,
    `Score: ${data.score_percent}%`,
    `Tier: ${data.tier ? data.tier.label : "Lower"}`,
    `Summary: ${data.summary || ""}`,
    ``,
    `Entered Answers:`,
  ];

  const allSpecs = (schema && schema.field_specs) || {};
  Object.keys(allSpecs).forEach((key) => {
    const spec = allSpecs[key];
    const val = inputs[key];
    let valStr = "";
    if (val === null || val === undefined) {
      valStr = "Not performed";
    } else if (spec.type === "choice" && spec.options) {
      const matchOpt = spec.options.find((o) => o.value === val);
      valStr = matchOpt ? matchOpt.label : String(val);
    } else {
      valStr = `${val}${spec.unit ? " " + spec.unit : ""}`;
    }
    lines.push(`- ${spec.short_label || spec.label}: ${valStr}`);
  });

  lines.push(``);
  lines.push(`Disclaimer: ${data.disclaimer || (schema ? schema.disclaimer : "")}`);
  return lines.join("\n");
}

