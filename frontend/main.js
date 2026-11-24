"use strict";

const path = require("node:path");
const { app, BrowserWindow, ipcMain } = require("electron");

const DEFAULT_BACKEND_PORT = 8765;

function resolveBackendUrl() {
  const rawPort = process.env.MINDVAULT_PORT || String(DEFAULT_BACKEND_PORT);
  const port = Number.parseInt(rawPort, 10);
  const safePort = Number.isInteger(port) && port >= 1 && port <= 65535 ? port : DEFAULT_BACKEND_PORT;
  return `http://127.0.0.1:${safePort}`;
}

function createMainWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: "MindVault",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
      allowRunningInsecureContent: false,
      enableRemoteModule: false
    }
  });

  // Block opening new windows or external links from the renderer.
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));

  // Prevent navigation away from local application files.
  win.webContents.on("will-navigate", (event, targetUrl) => {
    if (!targetUrl.startsWith("file://")) {
      event.preventDefault();
    }
  });

  win.loadFile(path.join(__dirname, "renderer", "index.html"));
  return win;
}

ipcMain.handle("mindvault:get-config", () => ({
  appName: "MindVault",
  appVersion: app.getVersion(),
  backendUrl: resolveBackendUrl(),
  offlineOnly: true
}));

app.whenReady().then(() => {
  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
