"use strict";

const { contextBridge, ipcRenderer } = require("electron");

const ALLOWED_WINDOW_ACTIONS = new Set(["minimize", "maximize", "close"]);

function windowControl(action) {
  if (typeof action !== "string" || !ALLOWED_WINDOW_ACTIONS.has(action)) {
    return Promise.reject(new Error(`Unsupported window action: ${String(action)}`));
  }
  return ipcRenderer.invoke("mindvault:window-control", action);
}

function onBackendStatus(listener) {
  if (typeof listener !== "function") {
    throw new TypeError("Backend status listener must be a function.");
  }
  const wrapped = (_event, status) => {
    listener(status);
  };
  ipcRenderer.on("mindvault:backend-status", wrapped);
  return () => {
    ipcRenderer.removeListener("mindvault:backend-status", wrapped);
  };
}

// Expose only a minimal, frozen IPC surface to the sandboxed renderer process.
contextBridge.exposeInMainWorld(
  "mindvault",
  Object.freeze({
    getConfig: () => ipcRenderer.invoke("mindvault:get-config"),
    getBackendStatus: () => ipcRenderer.invoke("mindvault:get-backend-status"),
    retryBackend: () => ipcRenderer.invoke("mindvault:retry-backend"),
    windowControl,
    onBackendStatus
  })
);
