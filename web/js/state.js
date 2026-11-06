/**
 * CardioLens In-Memory Application State
 * Strict requirement: In-memory only. No localStorage, sessionStorage, cookies,
 * or URL query/hash storage of patient answers.
 */

const state = {
  schema: null,
  examples: [],
  modelInfo: null,
  currentStep: 1,
  inputs: {},
  prediction: null,
  whatIfFeature: "age",
  whatIfValue: null,
  whatIfCurve: [],
};

const listeners = new Set();

export function getState() {
  return state;
}

export function setState(updates) {
  Object.assign(state, updates);
  notify();
}

export function setInputs(newInputs) {
  state.inputs = { ...state.inputs, ...newInputs };
  notify();
}

export function resetInputs() {
  state.inputs = {};
  if (state.schema && state.schema.fields) {
    state.schema.fields.forEach((f) => {
      if (f.default !== null && f.default !== undefined) {
        state.inputs[f.key] = f.default;
      }
    });
  }
  state.prediction = null;
  state.currentStep = 1;
  notify();
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function notify() {
  for (const listener of listeners) {
    try {
      listener(state);
    } catch (err) {
      console.error("State listener error:", err);
    }
  }
}

