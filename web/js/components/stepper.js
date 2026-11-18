/**
 * CardioLens Wizard Stepper Component
 * Renders a segmented 3-step progress bar with:
 * - Labels: "About you", "Report, part 1", "Report, part 2"
 * - Step heading (h3, 20px)
 * - Current segment highlighted
 * - Completed segments marked with an SVG check
 * - Visually hidden screen reader announcement ("Step n of 3")
 */

const SVG_NS = "http://www.w3.org/2000/svg";

const STEP_LABELS = [
  { step: 1, label: "About you" },
  { step: 2, label: "Report, part 1" },
  { step: 3, label: "Report, part 2" },
];

export function createStepperHeader(currentStep = 1, totalSteps = 3) {
  const wrapper = document.createElement("div");
  wrapper.className = "stepper-container";

  // 1. Visually Hidden Screen Reader announcement
  const srText = document.createElement("span");
  srText.className = "sr-only";
  srText.textContent = `Step ${currentStep} of ${totalSteps}`;
  wrapper.appendChild(srText);

  // 2. Segmented Stepper Bar
  const stepperList = document.createElement("ol");
  stepperList.className = "stepper-segments";
  stepperList.setAttribute("role", "list");

  STEP_LABELS.forEach(({ step, label }) => {
    const item = document.createElement("li");
    item.className = "stepper-item";
    item.setAttribute("role", "listitem");

    const isCurrent = step === currentStep;
    const isCompleted = step < currentStep;

    if (isCurrent) {
      item.classList.add("current");
      item.setAttribute("aria-current", "step");
    } else if (isCompleted) {
      item.classList.add("completed");
    } else {
      item.classList.add("upcoming");
    }

    const itemContent = document.createElement("div");
    itemContent.className = "stepper-item-content";

    const badge = document.createElement("span");
    badge.className = "stepper-badge";

    if (isCompleted) {
      // SVG Check Mark
      const checkSvg = document.createElementNS(SVG_NS, "svg");
      checkSvg.setAttribute("width", "14");
      checkSvg.setAttribute("height", "14");
      checkSvg.setAttribute("viewBox", "0 0 24 24");
      checkSvg.setAttribute("fill", "none");
      checkSvg.setAttribute("stroke", "currentColor");
      checkSvg.setAttribute("stroke-width", "3");
      checkSvg.setAttribute("stroke-linecap", "round");
      checkSvg.setAttribute("stroke-linejoin", "round");
      checkSvg.setAttribute("aria-hidden", "true");

      const checkPath = document.createElementNS(SVG_NS, "path");
      checkPath.setAttribute("d", "M20 6 9 17l-5-5");
      checkSvg.appendChild(checkPath);
      badge.appendChild(checkSvg);
    } else {
      badge.textContent = String(step);
    }

    const labelSpan = document.createElement("span");
    labelSpan.className = "stepper-label";
    labelSpan.textContent = label;

    itemContent.appendChild(badge);
    itemContent.appendChild(labelSpan);
    item.appendChild(itemContent);
    stepperList.appendChild(item);
  });

  wrapper.appendChild(stepperList);

  // 3. Step heading (h3 20px)
  const currentObj = STEP_LABELS.find((s) => s.step === currentStep) || STEP_LABELS[0];
  const stepHeading = document.createElement("h3");
  stepHeading.className = "stepper-heading";
  stepHeading.textContent = `Step ${currentStep} of ${totalSteps}: ${currentObj.label}`;
  wrapper.appendChild(stepHeading);

  return wrapper;
}
