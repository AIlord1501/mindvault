"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");
const vm = require("node:vm");

const { BackendManager, checkLocalHealth } = require("../backendManager");

const PRELOAD_PATH = path.resolve(__dirname, "..", "preload.js");

function loadPreloadWithMocks() {
  const exposed = {};
  const invokeCalls = [];
  const listeners = new Map();

  const electronMock = {
    contextBridge: {
      exposeInMainWorld(key, api) {
        exposed[key] = api;
      }
    },
    ipcRenderer: {
      invoke(channel, ...args) {
        invokeCalls.push({ channel, args });
        return Promise.resolve({ channel, args });
      },
      on(channel, fn) {
        if (!listeners.has(channel)) {
          listeners.set(channel, new Set());
        }
        listeners.get(channel).add(fn);
      },
      removeListener(channel, fn) {
        if (listeners.has(channel)) {
          listeners.get(channel).delete(fn);
        }
      }
    }
  };

  const code = fs.readFileSync(PRELOAD_PATH, "utf8");
  const wrapper = new vm.Script(
    `(function (require, module, exports) {\n${code}\n})`,
    { filename: PRELOAD_PATH }
  );
  const fn = wrapper.runInNewContext({});
  fn(
    (mod) => {
      if (mod === "electron") {
        return electronMock;
      }
      throw new Error(`Unexpected require in preload.js: ${mod}`);
    },
    { exports: {} },
    {}
  );

  return { exposed, invokeCalls, listeners };
}

function getFreeLoopbackPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
    server.on("error", reject);
  });
}

test("preload exposes only a minimal frozen IPC surface and rejects invalid window actions", async () => {
  const { exposed, invokeCalls, listeners } = loadPreloadWithMocks();

  assert.deepEqual(Object.keys(exposed), ["mindvault"]);
  const api = exposed.mindvault;
  assert.equal(Object.isFrozen(api), true);
  assert.deepEqual(Object.keys(api).sort(), [
    "getBackendStatus",
    "getConfig",
    "onBackendStatus",
    "retryBackend",
    "windowControl"
  ]);

  await api.getConfig();
  await api.getBackendStatus();
  await api.retryBackend();
  await api.windowControl("minimize");
  await api.windowControl("maximize");
  await api.windowControl("close");

  assert.deepEqual(
    invokeCalls.map((c) => c.channel),
    [
      "mindvault:get-config",
      "mindvault:get-backend-status",
      "mindvault:retry-backend",
      "mindvault:window-control",
      "mindvault:window-control",
      "mindvault:window-control"
    ]
  );

  const countBefore = invokeCalls.length;
  await assert.rejects(() => api.windowControl("exec"), /Unsupported window action/);
  assert.equal(invokeCalls.length, countBefore, "Rejected action must not invoke IPC");

  const received = [];
  const unsubscribe = api.onBackendStatus((status) => {
    received.push(status);
  });

  const channelListeners = listeners.get("mindvault:backend-status");
  assert.equal(channelListeners.size, 1);
  for (const listener of channelListeners) {
    listener({ sender: "internal-ipc-event" }, { state: "running", healthy: true });
  }
  assert.deepEqual(received, [{ state: "running", healthy: true }]);

  unsubscribe();
  assert.equal(channelListeners.size, 0);
});

test("BackendManager reuses an already-healthy local backend without spawning a child", async () => {
  const server = http.createServer((req, res) => {
    if (req.url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok", service: "mindvault-backend", schema_version: 1 }));
      return;
    }
    res.writeHead(404);
    res.end();
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();

  const manager = new BackendManager({
    port,
    pollIntervalMs: 60000
  });

  try {
    const status = await manager.start();
    assert.equal(status.state, "running");
    assert.equal(status.healthy, true);
    assert.equal(status.managedProcess, false);
    assert.equal(status.health.schema_version, 1);
  } finally {
    await manager.stop();
    await new Promise((resolve) => server.close(resolve));
  }
});

test("BackendManager handles missing executable startup failure gracefully", async () => {
  const port = await getFreeLoopbackPort();
  const manager = new BackendManager({
    port,
    pythonExecutable: "nonexistent-mindvault-python-binary-xyz",
    startupTimeoutMs: 1500,
    startupPollMs: 100,
    pollIntervalMs: 60000
  });

  try {
    const status = await manager.start();
    assert.equal(status.state, "failed");
    assert.equal(status.healthy, false);
    assert.equal(status.managedProcess, false);
    assert.ok(status.error && status.error.length > 0);
  } finally {
    await manager.stop();
  }
});

test("BackendManager spawns a managed loopback process, polls until healthy, and cleans it up", async () => {
  const port = await getFreeLoopbackPort();
  const inlineServer = [
    'const http = require("node:http");',
    `const s = http.createServer((req, res) => {`,
    '  res.writeHead(200, { "Content-Type": "application/json" });',
    '  res.end(JSON.stringify({ status: "ok", service: "stub-backend", schema_version: 1 }));',
    "});",
    `s.listen(${port}, "127.0.0.1");`,
    'process.on("SIGTERM", () => { s.close(() => process.exit(0)); });'
  ].join(" ");

  const manager = new BackendManager({
    port,
    pythonExecutable: process.execPath,
    spawnArgs: ["-e", inlineServer],
    startupTimeoutMs: 5000,
    startupPollMs: 100,
    pollIntervalMs: 60000
  });

  try {
    const status = await manager.start();
    assert.equal(status.state, "running");
    assert.equal(status.healthy, true);
    assert.equal(status.managedProcess, true);
    assert.equal(status.health.service, "stub-backend");
  } finally {
    await manager.stop();
  }

  const afterStop = await checkLocalHealth(port, 500);
  assert.equal(afterStop.ok, false, "Managed backend child process must be terminated on stop()");
});
