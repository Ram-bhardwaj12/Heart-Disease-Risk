/**
 * CardioLens - Main Application Entry Point
 * Strictly conforms to ES-module architecture. Zero inline scripts, zero inline styles.
 */

import { fetchSchema, fetchExamples, fetchModelInfo, pingHealth } from "./api.js";
import { setState, setInputs } from "./state.js";
import { registerRoute, initRouter } from "./router.js";
import { createTopBar, createFooter } from "./components/nav.js";
import { renderCheckView } from "./views/check.js";
import { renderHowView } from "./views/how.js";
import { renderDetailsView } from "./views/details.js";
import { renderAboutView } from "./views/about.js";

async function bootstrap() {
  // Fire-and-forget health ping on page load to wake up spinning server
  pingHealth();
  const headerMount = document.getElementById("header-mount");
  const mainMount = document.getElementById("main-mount");
  const footerMount = document.getElementById("footer-mount");

  // Fetch initial schema and data
  let schema = null;
  let examples = [];
  let modelInfo = null;

  try {
    const [s, ex, info] = await Promise.all([
      fetchSchema(),
      fetchExamples(),
      fetchModelInfo(),
    ]);
    schema = s;
    examples = ex;
    modelInfo = info;
  } catch (err) {
    console.error("Initialization fetch failed:", err);
  }

  // Update State
  setState({
    schema,
    examples,
    modelInfo,
  });

  // Populate default inputs from schema
  if (schema && schema.fields) {
    const defaults = {};
    schema.fields.forEach((f) => {
      if (f.default !== null && f.default !== undefined) {
        defaults[f.key] = f.default;
      }
    });
    setInputs(defaults);
  }

  // Mount Header with Disclaimer
  const disclaimerText = schema ? schema.disclaimer : "";
  if (headerMount) {
    headerMount.replaceChildren(createTopBar(disclaimerText));
  }

  // Mount Footer
  if (footerMount) {
    footerMount.replaceChildren(createFooter(modelInfo, disclaimerText));
  }

  // Register Hash Routes
  registerRoute("#/check", () => renderCheckView(mainMount));
  registerRoute("#/how", () => renderHowView(mainMount));
  registerRoute("#/details", () => renderDetailsView(mainMount));
  registerRoute("#/about", () => renderAboutView(mainMount));

  // Start Router
  initRouter();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}

