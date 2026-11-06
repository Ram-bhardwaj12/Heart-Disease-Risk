/**
 * CardioLens - Web Application Controller
 * Strict compliance: Zero emojis, explicit error displays, full API integration.
 */

(function () {
  'use strict';

  // Application State
  const state = {
    schema: null,
    currentStep: 1,
    inputs: {},
    lastPrediction: null,
    selectedWhatIfFeature: 'age',
  };

  // Preset Patient Profiles
  const PRESETS = {
    low: {
      age: 45,
      sex: 0,
      trestbps: 120,
      chol: 195,
      fbs: 0,
      cp: 1,
      restecg: 1,
      thalach: 168,
      exang: 0,
      oldpeak: 0.2,
      slope: 2,
      ca: 0,
      thal: 2,
    },
    mod: {
      age: 55,
      sex: 1,
      trestbps: 135,
      chol: 250,
      fbs: 0,
      cp: 2,
      restecg: 1,
      thalach: 145,
      exang: 0,
      oldpeak: 1.0,
      slope: 1,
      ca: 0,
      thal: 2,
    },
    high: {
      age: 63,
      sex: 1,
      trestbps: 152,
      chol: 286,
      fbs: 1,
      cp: 0,
      restecg: 2,
      thalach: 122,
      exang: 1,
      oldpeak: 2.6,
      slope: 1,
      ca: 2,
      thal: 3,
    },
  };

  // DOM Elements
  const elements = {
    navLinks: document.querySelectorAll('.nav-link'),
    pageViews: document.querySelectorAll('.page-view'),
    landingSection: document.getElementById('landing-section'),
    formSection: document.getElementById('form-section'),
    resultsSection: document.getElementById('results-section'),
    assessmentForm: document.getElementById('assessment-form'),
    btnStart: document.getElementById('btn-start-assessment'),
    presetLow: document.getElementById('preset-low'),
    presetMod: document.getElementById('preset-mod'),
    presetHigh: document.getElementById('preset-high'),
    stepIndicators: document.querySelectorAll('.step-indicator'),
    stepPanels: [
      document.getElementById('step-panel-1'),
      document.getElementById('step-panel-2'),
      document.getElementById('step-panel-3'),
    ],
    btnCancelForm: document.getElementById('btn-cancel-form'),
    btnStep1Next: document.getElementById('btn-step1-next'),
    btnStep2Prev: document.getElementById('btn-step2-prev'),
    btnStep2Next: document.getElementById('btn-step2-next'),
    btnStep3Prev: document.getElementById('btn-step3-prev'),
    chkCaAvailable: document.getElementById('chk-ca-available'),
    chkThalAvailable: document.getElementById('chk-thal-available'),
    inputCa: document.getElementById('input-ca'),
    inputThal: document.getElementById('input-thal'),
    resScorePct: document.getElementById('res-score-pct'),
    resTierBadge: document.getElementById('res-tier-badge'),
    resTierText: document.getElementById('res-tier-text'),
    meterMarker: document.getElementById('meter-marker'),
    noticeCard: document.getElementById('notice-card'),
    noticeList: document.getElementById('notice-list'),
    resSummaryText: document.getElementById('res-summary-text'),
    factorsTbody: document.getElementById('factors-tbody'),
    statTreeCount: document.getElementById('stat-tree-count'),
    statShareGe05: document.getElementById('stat-share-ge-05'),
    statMeanScore: document.getElementById('stat-mean-score'),
    whatifSelect: document.getElementById('whatif-feature-select'),
    whatifSlider: document.getElementById('whatif-slider'),
    whatifMinLbl: document.getElementById('whatif-min-lbl'),
    whatifMaxLbl: document.getElementById('whatif-max-lbl'),
    whatifCurrentValLbl: document.getElementById('whatif-current-val-lbl'),
    whatifResultCard: document.getElementById('whatif-result-card'),
    curveThVal: document.getElementById('curve-th-val'),
    curveTbody: document.getElementById('curve-tbody'),
    btnPrint: document.getElementById('btn-print-summary'),
    btnEditInputs: document.getElementById('btn-edit-inputs'),
    btnReset: document.getElementById('btn-reset-assessment'),
    printDate: document.getElementById('print-date'),
    tooltip: document.getElementById('global-tooltip'),
    modelDetailsInferredNote: document.getElementById('model-details-inferred-note'),
  };

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------

  async function init() {
    setupNavigation();
    setupTooltips();
    setupPresets();
    setupWizard();
    setupWhatIf();
    setupOptionalFields();

    try {
      const resp = await fetch('/api/schema');
      if (resp.ok) {
        state.schema = await resp.json();
        populateSchemaDefaults();
        if (elements.modelDetailsInferredNote && state.schema.inferred_codes_note) {
          elements.modelDetailsInferredNote.textContent = state.schema.inferred_codes_note;
        }
      }
    } catch (err) {
      console.warn('Failed to prefetch /api/schema:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // Top Navigation & Page Views
  // ---------------------------------------------------------------------------

  function setupNavigation() {
    elements.navLinks.forEach((link) => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('data-target');
        switchView(targetId);
      });
    });

    const brandLink = document.getElementById('nav-brand');
    if (brandLink) {
      brandLink.addEventListener('click', (e) => {
        e.preventDefault();
        switchView('view-risk');
      });
    }
  }

  function switchView(targetId) {
    elements.navLinks.forEach((l) => {
      l.classList.toggle('active', l.getAttribute('data-target') === targetId);
    });

    elements.pageViews.forEach((view) => {
      if (view.id === targetId) {
        view.style.display = 'block';
      } else {
        view.style.display = 'none';
      }
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // ---------------------------------------------------------------------------
  // Tooltip Support
  // ---------------------------------------------------------------------------

  function setupTooltips() {
    document.querySelectorAll('.tooltip-trigger').forEach((btn) => {
      btn.addEventListener('mouseenter', (e) => {
        showTooltip(e.target);
      });
      btn.addEventListener('mouseleave', () => {
        hideTooltip();
      });
      btn.addEventListener('focus', (e) => {
        showTooltip(e.target);
      });
      btn.addEventListener('blur', () => {
        hideTooltip();
      });
    });
  }

  function showTooltip(target) {
    if (!elements.tooltip) return;
    const note =
      (state.schema && state.schema.inferred_codes_note) ||
      "Labels for chest pain type, resting ECG, ST slope and thallium test were inferred by matching the dataset's value counts and disease rates to the published Cleveland data; the Kaggle file does not document its encoding.";

    elements.tooltip.textContent = note;
    elements.tooltip.classList.add('visible');

    const rect = target.getBoundingClientRect();
    const top = rect.bottom + window.scrollY + 6;
    const left = Math.min(rect.left + window.scrollX - 40, window.innerWidth - 300);

    elements.tooltip.style.top = `${top}px`;
    elements.tooltip.style.left = `${Math.max(10, left)}px`;
  }

  function hideTooltip() {
    if (!elements.tooltip) return;
    elements.tooltip.classList.remove('visible');
  }

  // ---------------------------------------------------------------------------
  // Optional Fields Toggles (ca & thal)
  // ---------------------------------------------------------------------------

  function setupOptionalFields() {
    if (elements.chkCaAvailable && elements.inputCa) {
      elements.chkCaAvailable.addEventListener('change', () => {
        elements.inputCa.disabled = !elements.chkCaAvailable.checked;
        if (!elements.chkCaAvailable.checked) {
          clearFieldError('ca');
        }
      });
    }

    if (elements.chkThalAvailable && elements.inputThal) {
      elements.chkThalAvailable.addEventListener('change', () => {
        elements.inputThal.disabled = !elements.chkThalAvailable.checked;
        if (!elements.chkThalAvailable.checked) {
          clearFieldError('thal');
        }
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Presets & Defaults
  // ---------------------------------------------------------------------------

  function setupPresets() {
    if (elements.btnStart) {
      elements.btnStart.addEventListener('click', () => {
        openForm();
      });
    }

    if (elements.presetLow) {
      elements.presetLow.addEventListener('click', () => {
        loadPreset(PRESETS.low);
      });
    }

    if (elements.presetMod) {
      elements.presetMod.addEventListener('click', () => {
        loadPreset(PRESETS.mod);
      });
    }

    if (elements.presetHigh) {
      elements.presetHigh.addEventListener('click', () => {
        loadPreset(PRESETS.high);
      });
    }

    if (elements.btnCancelForm) {
      elements.btnCancelForm.addEventListener('click', () => {
        elements.formSection.style.display = 'none';
        elements.landingSection.style.display = 'block';
        elements.resultsSection.classList.remove('visible');
      });
    }
  }

  function openForm() {
    elements.landingSection.style.display = 'none';
    elements.resultsSection.classList.remove('visible');
    elements.formSection.style.display = 'block';
    goToStep(1);
  }

  function loadPreset(profile) {
    populateForm(profile);
    openForm();
  }

  function populateSchemaDefaults() {
    if (!state.schema || !state.schema.fields) return;
    state.schema.fields.forEach((f) => {
      const input = document.getElementById(`input-${f.key}`);
      if (input && !input.value) {
        input.value = f.default !== null && f.default !== undefined ? f.default : '';
      }
    });
  }

  function populateForm(data) {
    Object.keys(data).forEach((key) => {
      const input = document.getElementById(`input-${key}`);
      if (!input) return;
      const val = data[key];
      if (key === 'ca') {
        if (val === null || val === undefined) {
          elements.chkCaAvailable.checked = false;
          input.disabled = true;
        } else {
          elements.chkCaAvailable.checked = true;
          input.disabled = false;
          input.value = String(val);
        }
      } else if (key === 'thal') {
        if (val === null || val === undefined) {
          elements.chkThalAvailable.checked = false;
          input.disabled = true;
        } else {
          elements.chkThalAvailable.checked = true;
          input.disabled = false;
          input.value = String(val);
        }
      } else {
        input.value = String(val);
      }
      clearFieldError(key);
    });
  }

  // ---------------------------------------------------------------------------
  // Wizard Navigation & Step Logic
  // ---------------------------------------------------------------------------

  function setupWizard() {
    if (elements.btnStep1Next) {
      elements.btnStep1Next.addEventListener('click', () => {
        if (validateStep(1)) {
          goToStep(2);
        }
      });
    }

    if (elements.btnStep2Prev) {
      elements.btnStep2Prev.addEventListener('click', () => {
        goToStep(1);
      });
    }

    if (elements.btnStep2Next) {
      elements.btnStep2Next.addEventListener('click', () => {
        if (validateStep(2)) {
          goToStep(3);
        }
      });
    }

    if (elements.btnStep3Prev) {
      elements.btnStep3Prev.addEventListener('click', () => {
        goToStep(2);
      });
    }

    elements.stepIndicators.forEach((ind) => {
      ind.addEventListener('click', () => {
        const targetStep = parseInt(ind.getAttribute('data-step'), 10);
        if (targetStep < state.currentStep) {
          goToStep(targetStep);
        } else if (targetStep > state.currentStep) {
          let canAdvance = true;
          for (let s = state.currentStep; s < targetStep; s++) {
            if (!validateStep(s)) {
              canAdvance = false;
              break;
            }
          }
          if (canAdvance) {
            goToStep(targetStep);
          }
        }
      });
    });

    if (elements.assessmentForm) {
      elements.assessmentForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (validateStep(3)) {
          await submitAssessment();
        }
      });
    }
  }

  function goToStep(stepNumber) {
    state.currentStep = stepNumber;
    elements.stepPanels.forEach((panel, idx) => {
      panel.style.display = idx + 1 === stepNumber ? 'block' : 'none';
    });

    elements.stepIndicators.forEach((ind) => {
      const step = parseInt(ind.getAttribute('data-step'), 10);
      ind.classList.toggle('active', step === stepNumber);
      ind.classList.toggle('completed', step < stepNumber);
    });

    window.scrollTo({ top: elements.formSection.offsetTop - 80, behavior: 'smooth' });
  }

  // ---------------------------------------------------------------------------
  // Validation
  // ---------------------------------------------------------------------------

  const STEP_FIELDS = {
    1: ['age', 'sex', 'trestbps', 'chol', 'fbs'],
    2: ['cp', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope'],
    3: ['ca', 'thal'],
  };

  function validateStep(stepNumber) {
    const fields = STEP_FIELDS[stepNumber] || [];
    let isValid = true;

    fields.forEach((fKey) => {
      const err = checkFieldValidity(fKey);
      if (err) {
        showFieldError(fKey, err);
        isValid = false;
      } else {
        clearFieldError(fKey);
      }
    });

    return isValid;
  }

  function checkFieldValidity(fKey) {
    const input = document.getElementById(`input-${fKey}`);
    if (!input) return null;

    if (fKey === 'ca' && elements.chkCaAvailable && !elements.chkCaAvailable.checked) {
      return null;
    }
    if (fKey === 'thal' && elements.chkThalAvailable && !elements.chkThalAvailable.checked) {
      return null;
    }

    const valStr = input.value.trim();
    if (!valStr && valStr !== '0') {
      return 'This field is required.';
    }

    const num = Number(valStr);
    if (isNaN(num)) {
      return 'Value must be a valid number.';
    }

    // Physiological plausible checks
    if (fKey === 'age' && (num < 18 || num > 100)) {
      return 'Age must be between 18 and 100 years.';
    }
    if (fKey === 'trestbps' && (num < 60 || num > 260)) {
      return 'Resting BP must be between 60 and 260 mm Hg.';
    }
    if (fKey === 'chol' && (num < 80 || num > 700)) {
      return 'Cholesterol must be between 80 and 700 mg/dl.';
    }
    if (fKey === 'thalach' && (num < 40 || num > 230)) {
      return 'Max heart rate must be between 40 and 230 bpm.';
    }
    if (fKey === 'oldpeak' && (num < 0 || num > 10)) {
      return 'ST depression must be between 0.0 and 10.0 mm.';
    }

    return null;
  }

  function showFieldError(fKey, message) {
    const input = document.getElementById(`input-${fKey}`);
    const errBox = document.getElementById(`err-${fKey}`);
    if (input) input.classList.add('has-error');
    if (errBox) {
      errBox.textContent = message;
      errBox.classList.add('visible');
    }
  }

  function clearFieldError(fKey) {
    const input = document.getElementById(`input-${fKey}`);
    const errBox = document.getElementById(`err-${fKey}`);
    if (input) input.classList.remove('has-error');
    if (errBox) {
      errBox.textContent = '';
      errBox.classList.remove('visible');
    }
  }

  function clearAllErrors() {
    document.querySelectorAll('.field-error-msg').forEach((el) => {
      el.textContent = '';
      el.classList.remove('visible');
    });
    document.querySelectorAll('.form-input, .form-select').forEach((el) => {
      el.classList.remove('has-error');
    });
  }

  // ---------------------------------------------------------------------------
  // Collect Form Inputs
  // ---------------------------------------------------------------------------

  function collectInputs() {
    const inputs = {};
    const keys = [
      'age',
      'sex',
      'cp',
      'trestbps',
      'chol',
      'fbs',
      'restecg',
      'thalach',
      'exang',
      'oldpeak',
      'slope',
      'ca',
      'thal',
    ];

    keys.forEach((k) => {
      if (k === 'ca' && elements.chkCaAvailable && !elements.chkCaAvailable.checked) {
        inputs[k] = null;
        return;
      }
      if (k === 'thal' && elements.chkThalAvailable && !elements.chkThalAvailable.checked) {
        inputs[k] = null;
        return;
      }

      const input = document.getElementById(`input-${k}`);
      if (!input) return;

      const val = input.value.trim();
      if (val === '') {
        inputs[k] = null;
      } else {
        inputs[k] = k === 'oldpeak' ? parseFloat(val) : Number(val);
      }
    });

    return inputs;
  }

  // ---------------------------------------------------------------------------
  // Submit Assessment to /api/predict
  // ---------------------------------------------------------------------------

  async function submitAssessment() {
    clearAllErrors();
    const payloadInputs = collectInputs();
    state.inputs = payloadInputs;

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: payloadInputs }),
      });

      if (resp.status === 422) {
        const errorData = await resp.json();
        handleValidationErrors(errorData.errors || []);
        return;
      }

      if (!resp.ok) {
        alert('Server returned an unexpected error (' + resp.status + '). Please check inputs.');
        return;
      }

      const data = await resp.json();
      state.lastPrediction = data;
      renderResults(data);
    } catch (err) {
      console.error('API call failed:', err);
      alert('Unable to reach CardioLens API server. Please ensure the backend is running.');
    }
  }

  function handleValidationErrors(errors) {
    if (!errors || errors.length === 0) return;

    let firstErrorStep = null;

    errors.forEach((err) => {
      const fKey = err.field;
      showFieldError(fKey, err.message);

      if (!firstErrorStep) {
        if (STEP_FIELDS[1].includes(fKey)) firstErrorStep = 1;
        else if (STEP_FIELDS[2].includes(fKey)) firstErrorStep = 2;
        else if (STEP_FIELDS[3].includes(fKey)) firstErrorStep = 3;
      }
    });

    if (firstErrorStep && firstErrorStep !== state.currentStep) {
      goToStep(firstErrorStep);
    }
  }

  // ---------------------------------------------------------------------------
  // Render Results View
  // ---------------------------------------------------------------------------

  function renderResults(data) {
    elements.formSection.style.display = 'none';
    elements.landingSection.style.display = 'none';
    elements.resultsSection.classList.add('visible');

    // Score KPI & Tier
    elements.resScorePct.textContent = `${data.score_percent}%`;
    const tierKey = data.tier.key.toLowerCase();
    elements.resTierBadge.className = `tier-badge tier-${tierKey}`;
    elements.resTierBadge.textContent = data.tier.label;
    elements.resTierText.textContent = data.tier.text;

    // Meter Marker (clamped to 0-100%)
    const clampedPct = Math.min(100, Math.max(0, data.score_percent));
    elements.meterMarker.style.left = `${clampedPct}%`;

    // Warnings Notice Card
    if (data.warnings && data.warnings.length > 0) {
      elements.noticeCard.classList.add('visible');
      elements.noticeList.innerHTML = '';
      data.warnings.forEach((w) => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${escapeHtml(w.field)}:</strong> ${escapeHtml(w.message)}`;
        elements.noticeList.appendChild(li);
      });
    } else {
      elements.noticeCard.classList.remove('visible');
    }

    // Explanation Summary Sentence
    elements.resSummaryText.textContent = data.summary;

    // Factor Contributions Table
    renderFactorTable(data.contributions || []);

    // Consensus Stats
    if (data.model_agreement) {
      elements.statTreeCount.textContent = data.model_agreement.tree_count;
      elements.statShareGe05.textContent = `${Math.round(data.model_agreement.share_at_or_above_half * 1000) / 10}%`;
      elements.statMeanScore.textContent = `${Math.round(data.model_agreement.mean_score * 1000) / 10}%`;
    }

    // Initialize What-If Section
    initWhatIfForCurrentPatient();

    // Print Date
    if (elements.printDate) {
      elements.printDate.textContent = new Date().toLocaleString();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function renderFactorTable(contributions) {
    elements.factorsTbody.innerHTML = '';

    contributions.forEach((c) => {
      const tr = document.createElement('tr');

      const isRaised = c.direction === 'raised';
      const badgeClass = isRaised ? 'impact-raised' : 'impact-lowered';
      const dirText = isRaised ? 'Raised score' : 'Lowered score';
      const amtSign = c.amount >= 0 ? `+${c.amount.toFixed(4)}` : c.amount.toFixed(4);

      tr.innerHTML = `
        <td><strong>${escapeHtml(c.short_label)}</strong></td>
        <td>${escapeHtml(c.value_display)}</td>
        <td><span class="factor-impact-badge ${badgeClass}">${dirText}</span></td>
        <td><code>${amtSign}</code></td>
      `;
      elements.factorsTbody.appendChild(tr);
    });
  }

  // ---------------------------------------------------------------------------
  // What-If Interactive Scenario
  // ---------------------------------------------------------------------------

  function setupWhatIf() {
    if (elements.whatifSelect) {
      elements.whatifSelect.addEventListener('change', () => {
        state.selectedWhatIfFeature = elements.whatifSelect.value;
        updateWhatIfSliderBounds();
        fetchWhatIfCurve();
      });
    }

    if (elements.whatifSlider) {
      elements.whatifSlider.addEventListener('input', () => {
        const val = parseFloat(elements.whatifSlider.value);
        elements.whatifCurrentValLbl.textContent = `Value: ${val}`;
        updateWhatIfFromSlider(val);
      });
    }

    if (elements.btnPrint) {
      elements.btnPrint.addEventListener('click', () => {
        window.print();
      });
    }

    if (elements.btnEditInputs) {
      elements.btnEditInputs.addEventListener('click', () => {
        elements.resultsSection.classList.remove('visible');
        elements.formSection.style.display = 'block';
        goToStep(1);
      });
    }

    if (elements.btnReset) {
      elements.btnReset.addEventListener('click', () => {
        elements.resultsSection.classList.remove('visible');
        elements.formSection.style.display = 'none';
        elements.landingSection.style.display = 'block';
        elements.assessmentForm.reset();
        populateSchemaDefaults();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }
  }

  function initWhatIfForCurrentPatient() {
    if (!elements.whatifSelect) return;
    state.selectedWhatIfFeature = elements.whatifSelect.value || 'age';
    updateWhatIfSliderBounds();
    fetchWhatIfCurve();
  }

  function updateWhatIfSliderBounds() {
    const fKey = state.selectedWhatIfFeature;
    const spec = (state.schema && state.schema.field_specs && state.schema.field_specs[fKey]) || null;
    const currVal = state.inputs[fKey] !== undefined && state.inputs[fKey] !== null ? state.inputs[fKey] : (spec ? spec.default : 50);

    const min = spec && spec.training_min !== undefined ? spec.training_min : (spec ? spec.min : 20);
    const max = spec && spec.training_max !== undefined ? spec.training_max : (spec ? spec.max : 80);
    const step = spec && spec.step ? spec.step : (fKey === 'oldpeak' ? 0.1 : 1);

    elements.whatifSlider.min = min;
    elements.whatifSlider.max = max;
    elements.whatifSlider.step = step;
    elements.whatifSlider.value = currVal;

    elements.whatifMinLbl.textContent = `Min: ${min}`;
    elements.whatifMaxLbl.textContent = `Max: ${max}`;
    elements.whatifCurrentValLbl.textContent = `Value: ${currVal}`;
    if (elements.curveThVal) {
      elements.curveThVal.textContent = `Simulated ${spec ? spec.short_label : fKey}`;
    }
  }

  let whatIfCurveData = [];

  async function fetchWhatIfCurve() {
    const fKey = state.selectedWhatIfFeature;
    try {
      const resp = await fetch('/api/whatif', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputs: state.inputs,
          feature: fKey,
          n_points: 25,
        }),
      });

      if (!resp.ok) return;

      const data = await resp.json();
      whatIfCurveData = data.curve || [];
      renderCurveTable(whatIfCurveData);

      const currentVal = parseFloat(elements.whatifSlider.value);
      updateWhatIfFromSlider(currentVal);
    } catch (err) {
      console.warn('What-If query failed:', err);
    }
  }

  function updateWhatIfFromSlider(targetVal) {
    if (!whatIfCurveData || whatIfCurveData.length === 0) return;

    // Find nearest curve point
    let nearest = whatIfCurveData[0];
    let minDiff = Math.abs(nearest.value - targetVal);

    whatIfCurveData.forEach((pt) => {
      const diff = Math.abs(pt.value - targetVal);
      if (diff < minDiff) {
        minDiff = diff;
        nearest = pt;
      }
    });

    const baselineScore = state.lastPrediction ? state.lastPrediction.score_percent : 0;
    const simScorePct = Math.round(nearest.score * 1000) / 10;
    const diffPct = Math.round((simScorePct - baselineScore) * 10) / 10;
    const diffStr = diffPct >= 0 ? `+${diffPct}%` : `${diffPct}%`;

    const fSpec = state.schema && state.schema.field_specs && state.schema.field_specs[state.selectedWhatIfFeature];
    const fLabel = fSpec ? fSpec.short_label : state.selectedWhatIfFeature;

    elements.whatifResultCard.innerHTML = `
      If <strong>${escapeHtml(fLabel)}</strong> were changed to <strong>${nearest.value}</strong>,
      the estimated risk score would be <strong>${simScorePct}%</strong>
      (change of <strong>${diffStr}</strong> from current baseline).
    `;
  }

  function renderCurveTable(curve) {
    elements.curveTbody.innerHTML = '';
    const lowC = (state.schema && state.schema.cutoffs && state.schema.cutoffs.low) || 0.3;
    const highC = (state.schema && state.schema.cutoffs && state.schema.cutoffs.high) || 0.5;

    curve.forEach((pt) => {
      const tr = document.createElement('tr');
      const scorePct = `${Math.round(pt.score * 1000) / 10}%`;

      let tierName = 'Lower';
      if (pt.score >= highC) tierName = 'Higher';
      else if (pt.score >= lowC) tierName = 'Moderate';

      tr.innerHTML = `
        <td><strong>${pt.value}</strong></td>
        <td>${scorePct}</td>
        <td>${tierName}</td>
      `;
      elements.curveTbody.appendChild(tr);
    });
  }

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Kickoff on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

