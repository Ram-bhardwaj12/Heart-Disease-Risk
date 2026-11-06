/**
 * CardioLens API Client
 * Fetch wrappers with strict 20 s timeout and AbortController support.
 */

const TIMEOUT_MS = 20000;

async function request(url, options = {}, externalSignal = null) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  // Link external signal if provided
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
      const timeoutErr = new Error("Request timed out after 20 seconds");
      timeoutErr.isTimeout = true;
      throw timeoutErr;
    }
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
    signal
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
    signal
  );
}

