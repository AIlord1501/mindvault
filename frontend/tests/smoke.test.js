"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const FRONTEND_DIR = path.resolve(__dirname, "..");

function readFrontendFile(...segments) {
  return fs.readFileSync(path.join(FRONTEND_DIR, ...segments), "utf8");
}

test("package.json defines Electron entrypoint, D3 dependency, and scripts", () => {
  const pkg = JSON.parse(readFrontendFile("package.json"));
  assert.equal(pkg.main, "main.js");
  assert.ok(pkg.scripts.start, "start script must exist");
  assert.ok(pkg.scripts.test, "test script must exist");
  assert.ok(pkg.scripts.check, "check script must exist");
  assert.ok(pkg.dependencies && pkg.dependencies.d3, "d3 dependency must be declared");
  assert.ok(pkg.devDependencies && pkg.devDependencies.electron, "electron devDependency must be declared");
});

test("main.js enforces hardened Electron security settings and loopback backend", () => {
  const mainSrc = readFrontendFile("main.js");
  assert.match(mainSrc, /nodeIntegration:\s*false/);
  assert.match(mainSrc, /contextIsolation:\s*true/);
  assert.match(mainSrc, /sandbox:\s*true/);
  assert.match(mainSrc, /setWindowOpenHandler/);
  assert.match(mainSrc, /127\.0\.0\.1/);
});

test("preload.js exposes only minimal contextBridge API", () => {
  const preloadSrc = readFrontendFile("preload.js");
  assert.match(preloadSrc, /contextBridge\.exposeInMainWorld/);
  assert.doesNotMatch(preloadSrc, /nodeIntegration/);
  assert.doesNotMatch(preloadSrc, /exposeInMainWorld\([^,]+,\s*ipcRenderer\s*\)/);
});

test("renderer index.html sets strict offline CSP and loads only local assets", () => {
  const html = readFrontendFile("renderer", "index.html");
  assert.match(html, /Content-Security-Policy/);
  assert.match(html, /default-src 'self'/);
  assert.match(html, /connect-src http:\/\/127\.0\.0\.1:\*/);
  assert.doesNotMatch(html, /https?:\/\/(?!127\.0\.0\.1)/);
});

test("renderer app.js avoids unsafe innerHTML injection", () => {
  const appSrc = readFrontendFile("renderer", "js", "app.js");
  assert.doesNotMatch(appSrc, /\.innerHTML\s*=/);
  assert.match(appSrc, /window\.mindvault\.getConfig/);
});
