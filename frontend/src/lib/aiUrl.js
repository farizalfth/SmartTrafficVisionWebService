const ENV_URL = import.meta.env.VITE_AI_SERVER_URL || "";
const OVERRIDE_KEY = "aiServerUrl";

function isLocalhostUrl(url) {
  try {
    const host = new URL(url).hostname;
    return host === "localhost" || host === "127.0.0.1" || host === "::1";
  } catch {
    return true;
  }
}

function trimSlash(url) {
  return String(url || "").trim().replace(/\/+$/, "");
}

export function getAiUrl() {
  const override = trimSlash(localStorage.getItem(OVERRIDE_KEY) || "");
  if (override) return override;
  if (ENV_URL && !isLocalhostUrl(ENV_URL)) return trimSlash(ENV_URL);
  const hostname = typeof window !== "undefined" ? window.location.hostname : "localhost";
  return `http://${hostname}:5000`;
}

export function setAiServerUrl(url) {
  const clean = trimSlash(url);
  if (clean) {
    localStorage.setItem(OVERRIDE_KEY, clean);
  } else {
    localStorage.removeItem(OVERRIDE_KEY);
  }
}

export function clearAiServerUrl() {
  localStorage.removeItem(OVERRIDE_KEY);
}

export function getAiUrlSource() {
  const override = trimSlash(localStorage.getItem(OVERRIDE_KEY) || "");
  if (override) return { url: override, source: "override" };
  if (ENV_URL && !isLocalhostUrl(ENV_URL)) return { url: trimSlash(ENV_URL), source: "env" };
  return { url: getAiUrl(), source: "auto" };
}
