/**
 * CardioLens API Client
 * Fetch wrappers with adaptive timeout, waking-server detection, and automatic retry.
 */

import { COPY } from "./copy.js";

let isFirstModelRequest = true;

const RETRY_STATUS_CODES = [502, 503, 504];

function showWakingUpStatus() {
  const resultRegion = document.getElementById("assessment-result-region");
  if (resultRegion && !document.getElementById("waking-up-status-line")) {
    const statusLine = document.createElement("p");
    statusLine.id = "waking-up-status-line";
    statusLine.className = "waking-status-text text-muted";
    statusLine.setAttribute("aria-live", "polite");
    statusLine.textContent = COPY.results.wakingUp;
    resultRegion.appendChild(statusLine);
  }
}

function removeWakingUpStatus() {
  const statusLine = document.getElementById("waking-up-status-line");
  if (statusLine) {
    statusLine.remove();
  }
}

export function pingHealth() {
  try {
    fetch("/api/health")
      .then((r) => r.text())
      .catch(() => {});
  } catch (_) {
    // ignore
  }
}

async function request(url, options = {}, externalSignal = null, isModelEndpoint = false) {
  const timeoutMs = isModelEndpoint ? (isFirstModelRequest ? 75000 : 20000) : 20000;
  if (isModelEndpoint) {
    isFirstModelRequest = false;
  }

  let wakingTimer = null;
  if (isModelEndpoint) {
    wakingTimer = setTimeout(() => {
      showWakingUpStatus();
    }, 8000);
  }

  const cleanup = () => {
    if (wakingTimer) {
      clearTimeout(wakingTimer);
      wakingTimer = null;
    }
    if (isModelEndpoint) {
      removeWakingUpStatus();
    }
  };

  const executeOnce = async () => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    if (externalSignal) {
      externalSignal.addEventListener("abort", () => controller.abort());
    }

    const finalOptions = {
      ...options,
      signal: controller.signal,
      headers: {
        "Accept": "application/json",
        ...(options.headers || {}),
      },
    };

    try {
      const response = await fetch(url, finalOptions);
      clearTimeout(timeoutId);

      const isJson = (response.headers.get("content-type") || "").includes("application/json");
      const data = isJson ? await response.json() : null;

      if (!response.ok) {
        const error = new Error(`Request failed with status ${response.status}`);
        error.status = response.status;
        error.data = data;
        throw error;
      }

      return data;
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        const timeoutErr = new Error(`Request timed out after ${timeoutMs / 1000} seconds`);
        timeoutErr.isTimeout = true;
        throw timeoutErr;
      }
      throw err;
    }
  };

  try {
    const result = await executeOnce();
    cleanup();
    return result;
  } catch (err) {
    const isRetryable =
      isModelEndpoint &&
      (err.isTimeout === true ||
        err.name === "TypeError" ||
        RETRY_STATUS_CODES.includes(err.status));

    if (isRetryable) {
      // Retry once automatically after 3 seconds
      await new Promise((resolve) => setTimeout(resolve, 3000));
      try {
        const retryResult = await executeOnce();
        cleanup();
        return retryResult;
      } catch (retryErr) {
        cleanup();
        throw retryErr;
      }
    }

    cleanup();
    throw err;
  }
}

export async function fetchSchema() {
  return request("/api/schema");
}

export async function fetchExamples() {
  return request("/api/examples");
}

export async function fetchModelInfo() {
  return request("/api/model-info");
}

export async function predictScore(inputs, signal = null) {
  return request(
    "/api/predict",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs }),
    },
    signal,
    true
  );
}

export async function fetchWhatIf(inputs, feature, nPoints = 25, signal = null) {
  return request(
    "/api/whatif",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs, feature, n_points: nPoints }),
    },
    signal,
    true
  );
}
