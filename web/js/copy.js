/**
 * CardioLens UI Copy & Interface Strings
 * Centralized strings strictly conforming to wording guidelines.
 * No banned words: probability (except in score caption), calibrated, SHAP, tree,
 * consensus, impact, model agreement, diagnosis (except in disclaimer), accurate,
 * validated, clinical-grade.
 */

export const COPY = {
  brand: {
    name: "CardioLens",
    tagline: "Understand what is behind your heart risk score.",
  },

  nav: {
    check: "Check your risk",
    how: "How it works",
    details: "Model details",
    about: "About",
    menu: "Menu",
  },

  hero: {
    headline: "Check your heart risk",
    sentence1: "Answer questions about your health and test results.",
    sentence2: "See which answers shaped your score.",
    startBtn: "Start the check",
    howLink: "How it works",
  },

  examples: {
    heading: "Try an example",
    caption: "Sample patients from the held-out test set of a public dataset, shown only to demonstrate the site.",
  },

  stepper: {
    step1: "About you",
    step2: "Your medical report, part 1",
    step3: "Your medical report, part 2",
    stepOf: (current, total) => `Step ${current} of ${total}`,
    back: "Back",
    next: "Next",
    submit: "See my result",
    skipCheckbox: "I don't have this value",
  },

  results: {
    editAnswers: "Edit my answers",
    startOver: "Start over",
    noticeWarning: "Some answers are outside the range the model learned from",
    noticeMissing: "You skipped some values, so the score is less reliable",
    scoreCaption: "This is the model's score, not a measured probability.",
    factorsHeading: "What shaped this score",
    factorsCaption: "This shows how the model used each answer. It does not show what causes disease.",
    showAllAnswers: "Show all 13 answers",
    raised: "Raised",
    lowered: "Lowered",
    whatIfHeading: "Try a change",
    whatIfScoreChange: (fromVal, toVal) => `Score changes from ${fromVal}% to ${toVal}%`,
    summaryHeading: "Your summary",
    printBtn: "Print or save as PDF",
    copyBtn: "Copy summary",
    copiedFeedback: "Copied to clipboard",
    whatNextHeading: "What to do next",
    whatNextSentences: [
      "Share this summary with a doctor.",
      "This tool cannot say whether you have a condition.",
      "Seek urgent care for emergency symptoms.",
    ],
  },

  placeholders: {
    how: "This page will explain how the model calculates scores from your answers.",
    details: "This page will share technical findings, cross-validation metrics, and dataset facts.",
    about: "This page will share background information about the CardioLens project and its creators.",
  },

  errors: {
    required: "This field is required.",
    invalidNumber: "Please enter a valid number.",
    networkError: "Unable to reach the server. Please check your connection and try again.",
    retryBtn: "Retry",
  },
};

