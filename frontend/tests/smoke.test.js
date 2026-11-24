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
  assert.ok(
    pkg.devDependencies && pkg.devDependencies.electron,
    "electron devDependency must be declared"
  );
});

test("main.js enforces frameless 1280x800 window, splash, security flags, and cleanup", () => {
  const mainSrc = readFrontendFile("main.js");
  assert.match(mainSrc, /width:\s*1280/);
  assert.match(mainSrc, /height:\s*800/);
  assert.match(mainSrc, /frame:\s*false/);
  assert.match(mainSrc, /nodeIntegration:\s*false/);
  assert.match(mainSrc, /contextIsolation:\s*true/);
  assert.match(mainSrc, /sandbox:\s*true/);
  assert.match(mainSrc, /setWindowOpenHandler/);
  assert.match(mainSrc, /createSplashWindow/);
  assert.match(mainSrc, /before-quit/);
  assert.match(mainSrc, /backendManager\s*\.stop/);
  assert.doesNotMatch(mainSrc, /shell:\s*true/);
});

test("backendManager.js binds only to 127.0.0.1 and never uses shell: true", () => {
  const mgrSrc = readFrontendFile("backendManager.js");
  assert.match(mgrSrc, /127\.0\.0\.1/);
  assert.match(mgrSrc, /shell:\s*false/);
  assert.doesNotMatch(mgrSrc, /shell:\s*true/);
});

test("renderer index.html has strict offline CSP, 5 sidebar items, status bar, and palette", () => {
  const html = readFrontendFile("renderer", "index.html");
  assert.match(html, /Content-Security-Policy/);
  assert.match(html, /default-src 'self'/);
  assert.match(html, /connect-src http:\/\/127\.0\.0\.1:\*/);
  assert.doesNotMatch(html, /https?:\/\/(?!127\.0\.0\.1)/);

  for (const view of ["chat", "graph", "dashboard", "research", "settings"]) {
    assert.match(html, new RegExp(`data-view="${view}"`));
    assert.match(html, new RegExp(`data-panel="${view}"`));
  }

  assert.match(html, /id="command-palette"/);
  assert.match(html, /Ctrl\+K/);
  assert.match(html, /class="statusbar"/);
});

test("splash.html enforces strict CSP with scripts disabled", () => {
  const splash = readFrontendFile("renderer", "splash.html");
  assert.match(splash, /Content-Security-Policy/);
  assert.match(splash, /script-src 'none'/);
  assert.doesNotMatch(splash, /https?:\/\//);
});

test("renderer app.js avoids innerHTML and wires Ctrl+K palette and themes", () => {
  const appSrc = readFrontendFile("renderer", "js", "app.js");
  assert.doesNotMatch(appSrc, /\.innerHTML\s*=/);
  assert.match(appSrc, /window\.mindvault\.getConfig/);
  assert.match(appSrc, /event\.key\.toLowerCase\(\)\s*===\s*"k"/);
  assert.match(appSrc, /data-theme/);
});
