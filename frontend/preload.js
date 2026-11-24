"use strict";

const { contextBridge, ipcRenderer } = require("electron");

// Expose only a minimal, frozen IPC surface to the renderer process.
contextBridge.exposeInMainWorld(
  "mindvault",
  Object.freeze({
    getConfig: () => ipcRenderer.invoke("mindvault:get-config")
  })
);
