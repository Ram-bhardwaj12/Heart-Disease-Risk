/**
 * CardioLens Wizard Stepper Component
 */

import { COPY } from "../copy.js";

const STEP_TITLES = {
  1: COPY.stepper.step1,
  2: COPY.stepper.step2,
  3: COPY.stepper.step3,
};

export function createStepperHeader(currentStep = 1, totalSteps = 3) {
  const header = document.createElement("div");
  header.className = "stepper-header";

  const title = document.createElement("h2");
  title.textContent = STEP_TITLES[currentStep] || `Step ${currentStep}`;
  header.appendChild(title);

  const progress = document.createElement("span");
  progress.className = "stepper-progress";
  progress.textContent = COPY.stepper.stepOf(currentStep, totalSteps);
  header.appendChild(progress);

  return header;
}

