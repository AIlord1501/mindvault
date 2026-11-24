"use strict";

const THEME_STORAGE_KEY = "mindvault-theme";
const VALID_VIEWS = ["chat", "graph", "dashboard", "research", "settings"];
const VIEW_LABELS = {
  chat: "Chat",
  graph: "Graph",
  dashboard: "Dashboard",
  research: "Research",
  settings: "Settings"
};

let activeView = "chat";
let currentTheme = "dark";
let paletteOpen = false;
let paletteIndex = 0;
let filteredCommands = [];

function getCommands() {
  return [
    { id: "nav-chat", label: "Go to Chat", hint: "Navigation", run: () => setActiveView("chat") },
    { id: "nav-graph", label: "Go to Graph", hint: "Navigation", run: () => setActiveView("graph") },
    {
      id: "nav-dashboard",
      label: "Go to Dashboard",
      hint: "Navigation",
      run: () => setActiveView("dashboard")
    },
    {
      id: "nav-research",
      label: "Go to Research",
      hint: "Navigation",
      run: () => setActiveView("research")
    },
    {
      id: "nav-settings",
      label: "Go to Settings",
      hint: "Navigation",
      run: () => setActiveView("settings")
    },
    {
      id: "theme-toggle",
      label: "Toggle Light / Dark Theme",
      hint: "Appearance",
      run: () => applyTheme(currentTheme === "dark" ? "light" : "dark")
    },
    {
      id: "backend-retry",
      label: "Retry Local Backend Connection",
      hint: "System",
      run: () => {
        void retryLocalBackend();
      }
    }
  ];
}

function setActiveView(viewName) {
  if (!VALID_VIEWS.includes(viewName)) {
    return;
  }
  activeView = viewName;

  document.querySelectorAll(".nav-item").forEach((btn) => {
    const isMatch = btn.dataset.view === viewName;
    btn.classList.toggle("active", isMatch);
    if (isMatch) {
      btn.setAttribute("aria-current", "page");
    } else {
      btn.removeAttribute("aria-current");
    }
  });

  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === viewName);
  });

  const viewLabel = document.getElementById("status-view-label");
  if (viewLabel) {
    viewLabel.textContent = `View: ${VIEW_LABELS[viewName]}`;
  }
}

function applyTheme(theme) {
  const normalized = theme === "light" ? "light" : "dark";
  currentTheme = normalized;
  document.documentElement.setAttribute("data-theme", normalized);

  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, normalized);
  } catch (_err) {
    // Ignore storage unavailability in restricted contexts.
  }

  const label = normalized === "dark" ? "Dark" : "Light";
  const toggleBtn = document.getElementById("theme-toggle");
  if (toggleBtn) {
    toggleBtn.textContent = label;
  }
  const statusTheme = document.getElementById("status-theme-label");
  if (statusTheme) {
    statusTheme.textContent = `Theme: ${label}`;
  }
}

function loadSavedTheme() {
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === "light" || saved === "dark") {
      applyTheme(saved);
      return;
    }
  } catch (_err) {
    // Default to dark theme.
  }
  applyTheme("dark");
}

function renderBackendStatus(status) {
  const dot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-backend-text");
  const endpointEl = document.getElementById("status-endpoint");
  const banner = document.getElementById("backend-banner");
  const bannerText = document.getElementById("backend-banner-text");
  const settingsUrl = document.getElementById("settings-backend-url");
  const settingsState = document.getElementById("settings-backend-state");
  const settingsSchema = document.getElementById("settings-schema-version");

  if (!status) {
    return;
  }

  const endpoint = status.backendUrl || "http://127.0.0.1:8765";
  if (endpointEl) {
    endpointEl.textContent = endpoint.replace(/^http:\/\//, "");
  }
  if (settingsUrl) {
    settingsUrl.textContent = endpoint;
  }

  dot.classList.remove("dot-ok", "dot-pending", "dot-error");

  if (status.healthy && status.state === "running") {
    dot.classList.add("dot-ok");
    const schemaVer = status.health && status.health.schema_version != null
      ? `v${status.health.schema_version}`
      : "--";
    const processMode = status.managedProcess ? "managed" : "external";
    statusText.textContent = `Backend connected (${processMode})`;
    settingsState.textContent = `Running (${processMode})`;
    settingsSchema.textContent = schemaVer;
    banner.classList.add("hidden");
  } else if (status.state === "starting" || status.state === "idle") {
    dot.classList.add("dot-pending");
    statusText.textContent = "Starting local backend...";
    settingsState.textContent = "Starting...";
    settingsSchema.textContent = "--";
    banner.classList.add("hidden");
  } else {
    dot.classList.add("dot-error");
    const reason = status.error || "Local backend is unavailable.";
    statusText.textContent = "Local backend offline";
    settingsState.textContent = `Offline (${reason})`;
    settingsSchema.textContent = "--";
    bannerText.textContent = `Local backend unavailable: ${reason}`;
    banner.classList.remove("hidden");
  }
}

async function retryLocalBackend() {
  if (!window.mindvault || typeof window.mindvault.retryBackend !== "function") {
    return;
  }
  renderBackendStatus({
    state: "starting",
    healthy: false,
    backendUrl: document.getElementById("settings-backend-url")?.textContent || "http://127.0.0.1:8765"
  });
  try {
    const status = await window.mindvault.retryBackend();
    renderBackendStatus(status);
  } catch (err) {
    renderBackendStatus({
      state: "failed",
      healthy: false,
      error: err ? err.message : "Retry failed"
    });
  }
}

function renderPaletteItems() {
  const listEl = document.getElementById("palette-list");
  listEl.replaceChildren();

  if (filteredCommands.length === 0) {
    const emptyItem = document.createElement("li");
    emptyItem.className = "palette-item";
    emptyItem.textContent = "No matching commands";
    listEl.appendChild(emptyItem);
    return;
  }

  filteredCommands.forEach((cmd, idx) => {
    const li = document.createElement("li");
    li.className = idx === paletteIndex ? "palette-item selected" : "palette-item";
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", idx === paletteIndex ? "true" : "false");

    const titleSpan = document.createElement("span");
    titleSpan.textContent = cmd.label;

    const hintSpan = document.createElement("span");
    hintSpan.className = "palette-item-hint";
    hintSpan.textContent = cmd.hint;

    li.appendChild(titleSpan);
    li.appendChild(hintSpan);
    li.addEventListener("click", () => {
      closeCommandPalette();
      cmd.run();
    });
    listEl.appendChild(li);
  });
}

function filterPalette(query) {
  const needle = query.trim().toLowerCase();
  const all = getCommands();
  filteredCommands = needle
    ? all.filter(
        (cmd) =>
          cmd.label.toLowerCase().includes(needle) || cmd.hint.toLowerCase().includes(needle)
      )
    : all;
  paletteIndex = 0;
  renderPaletteItems();
}

function openCommandPalette() {
  const overlay = document.getElementById("command-palette");
  const input = document.getElementById("palette-input");
  paletteOpen = true;
  overlay.classList.remove("hidden");
  input.value = "";
  filterPalette("");
  input.focus();
}

function closeCommandPalette() {
  const overlay = document.getElementById("command-palette");
  paletteOpen = false;
  overlay.classList.add("hidden");
}

function wireEvents() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      setActiveView(btn.dataset.view);
    });
  });

  document.getElementById("theme-toggle").addEventListener("click", () => {
    applyTheme(currentTheme === "dark" ? "light" : "dark");
  });

  document.querySelectorAll("[data-set-theme]").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyTheme(btn.dataset.setTheme);
    });
  });

  document.querySelectorAll("[data-win-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (window.mindvault && typeof window.mindvault.windowControl === "function") {
        void window.mindvault.windowControl(btn.dataset.winAction);
      }
    });
  });

  document.getElementById("banner-retry-btn").addEventListener("click", () => {
    void retryLocalBackend();
  });

  document.getElementById("settings-retry-btn").addEventListener("click", () => {
    void retryLocalBackend();
  });

  document.getElementById("command-palette-trigger").addEventListener("click", () => {
    openCommandPalette();
  });

  const overlay = document.getElementById("command-palette");
  const input = document.getElementById("palette-input");

  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) {
      closeCommandPalette();
    }
  });

  input.addEventListener("input", () => {
    filterPalette(input.value);
  });

  window.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      if (paletteOpen) {
        closeCommandPalette();
      } else {
        openCommandPalette();
      }
      return;
    }

    if (!paletteOpen) {
      return;
    }

    if (event.key === "Escape") {
      event.preventDefault();
      closeCommandPalette();
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      if (filteredCommands.length > 0) {
        paletteIndex = (paletteIndex + 1) % filteredCommands.length;
        renderPaletteItems();
      }
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      if (filteredCommands.length > 0) {
        paletteIndex = (paletteIndex - 1 + filteredCommands.length) % filteredCommands.length;
        renderPaletteItems();
      }
    } else if (event.key === "Enter") {
      event.preventDefault();
      const selected = filteredCommands[paletteIndex];
      if (selected) {
        closeCommandPalette();
        selected.run();
      }
    }
  });
}

async function initDesktopShell() {
  loadSavedTheme();
  setActiveView("chat");
  wireEvents();

  if (!window.mindvault || typeof window.mindvault.getConfig !== "function") {
    renderBackendStatus({
      state: "failed",
      healthy: false,
      error: "Preload bridge unavailable."
    });
    return;
  }

  try {
    await window.mindvault.getConfig();
    if (typeof window.mindvault.onBackendStatus === "function") {
      window.mindvault.onBackendStatus((status) => {
        renderBackendStatus(status);
      });
    }
    const status = await window.mindvault.getBackendStatus();
    renderBackendStatus(status);
  } catch (err) {
    renderBackendStatus({
      state: "failed",
      healthy: false,
      error: err ? err.message : "Failed to initialize IPC bridge."
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  void initDesktopShell();
});
