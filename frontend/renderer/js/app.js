"use strict";

async function checkLocalBackend() {
  const statusEl = document.getElementById("backend-status");
  const urlEl = document.getElementById("backend-url");

  if (!window.mindvault || typeof window.mindvault.getConfig !== "function") {
    statusEl.textContent = "Preload bridge unavailable.";
    return;
  }

  try {
    const config = await window.mindvault.getConfig();
    urlEl.textContent = `Endpoint: ${config.backendUrl}`;

    const response = await fetch(`${config.backendUrl}/health`, {
      method: "GET",
      headers: { Accept: "application/json" }
    });

    if (!response.ok) {
      statusEl.textContent = `Backend returned HTTP ${response.status}.`;
      return;
    }

    const payload = await response.json();
    statusEl.textContent = `Connected (${payload.service} v${payload.version}, offline-only: ${payload.offline_only})`;
  } catch (_err) {
    statusEl.textContent = "Local backend is not running. Start it with `python run.py`.";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  void checkLocalBackend();
});
