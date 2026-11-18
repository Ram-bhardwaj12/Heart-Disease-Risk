/**
 * CardioLens Theme Initializer
 * Runs synchronously in <head> to prevent flash of wrong theme.
 * Reads user choice from localStorage (wrapped in try/catch) or falls back to system preference.
 */
(function () {
  try {
    const saved = localStorage.getItem("cardiolens_theme");
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const theme = saved || (prefersDark ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "light");
  }
})();
