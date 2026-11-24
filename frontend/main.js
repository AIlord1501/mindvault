"use strict";

const path = require("node:path");
const { app, BrowserWindow, ipcMain } = require("electron");
const { BackendManager } = require("./backendManager");

const ALLOWED_WINDOW_ACTIONS = new Set(["minimize", "maximize", "close"]);

let mainWindow = null;
let splashWindow = null;
let isQuitting = false;
const backendManager = new BackendManager();

function attachNavigationGuards(win) {
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (event, targetUrl) => {
    if (!targetUrl.startsWith("file://")) {
      event.preventDefault();
    }
  });
}

function createSplashWindow() {
  const win = new BrowserWindow({
    width: 460,
    height: 300,
    frame: false,
    resizable: false,
    center: true,
    show: true,
    backgroundColor: "#0f1419",
    title: "MindVault",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
      allowRunningInsecureContent: false
    }
  });

  attachNavigationGuards(win);
  win.loadFile(path.join(__dirname, "renderer", "splash.html"));
  return win;
}

function createMainWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 960,
    minHeight: 640,
    frame: false,
    show: false,
    backgroundColor: "#0f1419",
    title: "MindVault",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
      allowRunningInsecureContent: false
    }
  });

  attachNavigationGuards(win);
  win.loadFile(path.join(__dirname, "renderer", "index.html"));

  win.on("closed", () => {
    if (mainWindow === win) {
      mainWindow = null;
    }
  });

  return win;
}

function broadcastBackendStatus(status) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("mindvault:backend-status", status);
  }
}

backendManager.on("status", broadcastBackendStatus);

ipcMain.handle("mindvault:get-config", () => ({
  appName: "MindVault",
  appVersion: app.getVersion(),
  backendUrl: backendManager.backendUrl,
  offlineOnly: true,
  platform: process.platform
}));

ipcMain.handle("mindvault:get-backend-status", () => backendManager.getStatus());

ipcMain.handle("mindvault:retry-backend", async () => backendManager.retry());

ipcMain.handle("mindvault:window-control", (event, action) => {
  if (typeof action !== "string" || !ALLOWED_WINDOW_ACTIONS.has(action)) {
    throw new Error(`Unsupported window action: ${String(action)}`);
  }
  const target = BrowserWindow.fromWebContents(event.sender);
  if (!target || target.isDestroyed()) {
    return false;
  }
  if (action === "minimize") {
    target.minimize();
  } else if (action === "maximize") {
    if (target.isMaximized()) {
      target.unmaximize();
    } else {
      target.maximize();
    }
  } else if (action === "close") {
    target.close();
  }
  return true;
});

async function bootstrapDesktop() {
  splashWindow = createSplashWindow();
  mainWindow = createMainWindow();

  const readyToShow = new Promise((resolve) => {
    mainWindow.once("ready-to-show", resolve);
  });

  const backendStartup = backendManager.start().catch((err) => ({
    state: "failed",
    healthy: false,
    managedProcess: false,
    backendUrl: backendManager.backendUrl,
    error: err ? err.message : "Backend startup failed"
  }));

  await Promise.all([readyToShow, backendStartup]);

  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.close();
    splashWindow = null;
  }

  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.show();
    broadcastBackendStatus(backendManager.getStatus());
  }
}

app.whenReady().then(() => {
  void bootstrapDesktop();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void bootstrapDesktop();
    }
  });
});

app.on("before-quit", (event) => {
  if (isQuitting) {
    return;
  }
  isQuitting = true;
  event.preventDefault();
  backendManager
    .stop()
    .catch(() => {})
    .finally(() => {
      app.exit(0);
    });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
