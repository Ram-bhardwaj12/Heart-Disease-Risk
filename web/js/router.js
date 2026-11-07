/**
 * CardioLens Hash Router
 * Hash routes: #/check, #/how, #/details, #/about
 * In-memory view switching; no query string serialization of sensitive patient answers.
 */

const routes = {};
let currentRoute = null;

const ROUTE_TITLES = {
  "#/check": "Check your risk",
  "#/how": "How it works",
  "#/details": "Model details",
  "#/about": "About",
};

export function registerRoute(path, handler) {
  routes[path] = handler;
}

export function initRouter() {
  window.addEventListener("hashchange", handleRouteChange);
  handleRouteChange();
}

export function navigateTo(path) {
  if (window.location.hash !== path) {
    window.location.hash = path;
  } else {
    handleRouteChange();
  }
}

function handleRouteChange() {
  const hash = window.location.hash || "#/check";
  // Normalize
  const matchedRoute = routes[hash] ? hash : "#/check";

  if (window.location.hash !== matchedRoute) {
    window.location.replace(matchedRoute);
    return;
  }

  currentRoute = matchedRoute;

  const pageName = ROUTE_TITLES[matchedRoute] || "Check your risk";
  document.title = `${pageName} - CardioLens`;

  // Update navigation link visual states
  document.querySelectorAll(".nav-item-link").forEach((link) => {
    const href = link.getAttribute("href");
    if (href === matchedRoute) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });

  const handler = routes[matchedRoute];
  if (handler) {
    handler();
  }

  window.scrollTo({ top: 0, behavior: "smooth" });
}

export function getCurrentRoute() {
  return currentRoute || "#/check";
}

