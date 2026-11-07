/**
 * CardioLens Form Field Component
 * Supports numeric input with range slider, segmented buttons, select dropdowns,
 * optional skip checkboxes, and info tooltips with inferred_codes_note.
 * Zero inline styles (uses SVG attributes and CSS classes).
 */

import { COPY } from "../copy.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export function createField(spec, initialValue, onChange, inferredNote = "") {
  const wrapper = document.createElement("div");
  wrapper.className = "field-wrapper";
  wrapper.id = `field-wrap-${spec.key}`;

  // 1. Label Row
  const labelRow = document.createElement("div");
  labelRow.className = "field-label-row";

  const label = document.createElement("label");
  label.className = "field-label";
  label.setAttribute("for", `field-input-${spec.key}`);

  const labelText = document.createElement("span");
  labelText.textContent = spec.label;
  label.appendChild(labelText);

  // Inferred note info icon if applicable
  const inferredKeys = ["cp", "restecg", "slope", "thal"];
  if (inferredKeys.includes(spec.key) && inferredNote) {
    const infoBtn = createInfoButton(spec.key, inferredNote);
    label.appendChild(infoBtn);
  }

  labelRow.appendChild(label);
  wrapper.appendChild(labelRow);

  // Tooltip Box (initially hidden)
  if (inferredKeys.includes(spec.key) && inferredNote) {
    const tooltipBox = document.createElement("div");
    tooltipBox.className = "info-tooltip-box";
    tooltipBox.id = `tooltip-${spec.key}`;
    tooltipBox.textContent = inferredNote;
    wrapper.appendChild(tooltipBox);
  }

  // 2. Help text
  if (spec.help) {
    const help = document.createElement("p");
    help.className = "field-help field-help-text";
    help.textContent = spec.help;
    wrapper.appendChild(help);
  }

  // 3. Control rendering
  if (spec.type === "choice") {
    // Segmented buttons for <= 4 options, select otherwise
    if (spec.options && spec.options.length <= 4) {
      const segCtrl = createSegmentedControl(spec, initialValue, onChange);
      wrapper.appendChild(segCtrl);
    } else {
      const select = createSelectDropdown(spec, initialValue, onChange);
      wrapper.appendChild(select);
    }
  } else {
    // Number input plus slider for numeric ranges from /api/schema
    const numWrap = createNumberWithSlider(spec, initialValue, onChange);
    wrapper.appendChild(numWrap);
  }

  // 4. Optional Skip Checkbox for ca and thal
  if (spec.optional) {
    const checkRow = document.createElement("div");
    checkRow.className = "checkbox-row";

    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.id = `skip-chk-${spec.key}`;
    chk.checked = initialValue === null || initialValue === undefined;

    const chkLabel = document.createElement("label");
    chkLabel.setAttribute("for", `skip-chk-${spec.key}`);
    chkLabel.textContent = COPY.stepper.skipCheckbox;

    chk.addEventListener("change", () => {
      const isSkipped = chk.checked;
      toggleFieldDisabled(wrapper, isSkipped);
      if (isSkipped) {
        onChange(spec.key, null);
      } else {
        const fallback = spec.default !== null && spec.default !== undefined ? spec.default : (spec.options ? spec.options[0].value : 0);
        onChange(spec.key, fallback);
      }
    });

    checkRow.appendChild(chk);
    checkRow.appendChild(chkLabel);
    wrapper.appendChild(checkRow);

    // Initial disabled state if skipped
    if (chk.checked) {
      toggleFieldDisabled(wrapper, true);
    }
  }

  // 5. Range hint (for numbers)
  if (spec.type !== "choice" && spec.min !== null && spec.max !== null) {
    const hint = document.createElement("div");
    hint.className = "field-hint";
    const unitStr = spec.unit ? ` ${spec.unit}` : "";
    hint.textContent = `Range: ${Math.round(spec.min)} to ${Math.round(spec.max)}${unitStr}`;
    wrapper.appendChild(hint);
  }

  // 6. Inline error placeholder
  const errorBox = document.createElement("div");
  errorBox.className = "field-error";
  errorBox.id = `field-error-${spec.key}`;
  errorBox.setAttribute("role", "alert");
  wrapper.appendChild(errorBox);

  return wrapper;
}

function createInfoButton(key, text) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "info-btn";
  btn.setAttribute("aria-label", "Information about category encoding");
  btn.setAttribute("aria-expanded", "false");

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "16");
  svg.setAttribute("height", "16");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");

  const circle = document.createElementNS(SVG_NS, "circle");
  circle.setAttribute("cx", "12");
  circle.setAttribute("cy", "12");
  circle.setAttribute("r", "10");
  svg.appendChild(circle);

  const line1 = document.createElementNS(SVG_NS, "line");
  line1.setAttribute("x1", "12");
  line1.setAttribute("y1", "16");
  line1.setAttribute("x2", "12");
  line1.setAttribute("y2", "12");
  svg.appendChild(line1);

  const line2 = document.createElementNS(SVG_NS, "line");
  line2.setAttribute("x1", "12");
  line2.setAttribute("y1", "8");
  line2.setAttribute("x2", "12.01");
  line2.setAttribute("y2", "8");
  svg.appendChild(line2);

  btn.appendChild(svg);

  btn.addEventListener("click", (e) => {
    e.preventDefault();
    const tooltip = document.getElementById(`tooltip-${key}`);
    if (tooltip) {
      const isVisible = tooltip.classList.contains("visible");
      tooltip.classList.toggle("visible", !isVisible);
      btn.setAttribute("aria-expanded", !isVisible ? "true" : "false");
    }
  });

  return btn;
}

function createNumberWithSlider(spec, initialVal, onChange) {
  const container = document.createElement("div");
  container.className = "number-slider-wrap";

  const numInput = document.createElement("input");
  numInput.type = "number";
  numInput.className = "field-input";
  numInput.id = `field-input-${spec.key}`;
  numInput.name = spec.key;
  numInput.min = String(spec.min !== null ? spec.min : 0);
  numInput.max = String(spec.max !== null ? spec.max : 1000);
  numInput.step = String(spec.step || 1);
  numInput.value = initialVal !== null && initialVal !== undefined ? String(initialVal) : String(spec.default || spec.min || 0);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "field-range-slider";
  slider.min = numInput.min;
  slider.max = numInput.max;
  slider.step = numInput.step;
  slider.value = numInput.value;

  // Two-way synchronization
  numInput.addEventListener("input", () => {
    slider.value = numInput.value;
    onChange(spec.key, numInput.value === "" ? null : Number(numInput.value));
  });

  slider.addEventListener("input", () => {
    numInput.value = slider.value;
    onChange(spec.key, Number(slider.value));
  });

  container.appendChild(numInput);
  container.appendChild(slider);
  return container;
}

function createSegmentedControl(spec, initialVal, onChange) {
  const container = document.createElement("div");
  container.className = "segmented-control";
  container.id = `field-input-${spec.key}`;
  container.setAttribute("role", "radiogroup");
  container.setAttribute("aria-label", spec.label);

  const currentVal = initialVal !== null && initialVal !== undefined ? initialVal : spec.default;

  spec.options.forEach((opt) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "segmented-option";
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(opt.value === currentVal));
    btn.textContent = opt.label;
    btn.dataset.value = String(opt.value);

    if (opt.value === currentVal) {
      btn.classList.add("selected");
    }

    btn.addEventListener("click", () => {
      container.querySelectorAll(".segmented-option").forEach((b) => {
        b.classList.remove("selected");
        b.setAttribute("aria-checked", "false");
      });
      btn.classList.add("selected");
      btn.setAttribute("aria-checked", "true");
      onChange(spec.key, opt.value);
    });

    container.appendChild(btn);
  });

  return container;
}

function createSelectDropdown(spec, initialVal, onChange) {
  const select = document.createElement("select");
  select.className = "field-select";
  select.id = `field-input-${spec.key}`;
  select.name = spec.key;

  const currentVal = initialVal !== null && initialVal !== undefined ? initialVal : spec.default;

  spec.options.forEach((opt) => {
    const option = document.createElement("option");
    option.value = String(opt.value);
    option.textContent = opt.label;
    if (opt.value === currentVal) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  select.addEventListener("change", () => {
    onChange(spec.key, Number(select.value));
  });

  return select;
}

function toggleFieldDisabled(wrapper, disabled) {
  const inputs = wrapper.querySelectorAll("input:not([type='checkbox']), select, .segmented-option");
  inputs.forEach((el) => {
    if (el.tagName === "BUTTON") {
      el.disabled = disabled;
    } else {
      el.disabled = disabled;
    }
  });
  if (disabled) {
    wrapper.classList.add("disabled");
  } else {
    wrapper.classList.remove("disabled");
  }
}

export function setFieldError(key, message) {
  const errEl = document.getElementById(`field-error-${key}`);
  const inputEl = document.getElementById(`field-input-${key}`);
  if (errEl) {
    errEl.textContent = message;
    errEl.classList.add("visible");
  }
  if (inputEl) {
    inputEl.classList.add("has-error");
  }
}

export function clearFieldError(key) {
  const errEl = document.getElementById(`field-error-${key}`);
  const inputEl = document.getElementById(`field-input-${key}`);
  if (errEl) {
    errEl.textContent = "";
    errEl.classList.remove("visible");
  }
  if (inputEl) {
    inputEl.classList.remove("has-error");
  }
}

