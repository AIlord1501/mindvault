"use strict";

const { EventEmitter } = require("node:events");
const http = require("node:http");
const path = require("node:path");
const { spawn } = require("node:child_process");

const DEFAULT_PORT = 8765;
const DEFAULT_POLL_INTERVAL_MS = 5000;
const DEFAULT_STARTUP_TIMEOUT_MS = 12000;
const DEFAULT_STARTUP_POLL_MS = 350;

function resolveBackendPort(env = process.env) {
  const raw = env.MINDVAULT_PORT || String(DEFAULT_PORT);
  const parsed = Number.parseInt(raw, 10);
  return Number.isInteger(parsed) && parsed >= 1 && parsed <= 65535 ? parsed : DEFAULT_PORT;
}

function buildBackendUrl(port) {
  return `http://127.0.0.1:${port}`;
}

function checkLocalHealth(port, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const req = http.get(
      {
        hostname: "127.0.0.1",
        port,
        path: "/health",
        timeout: timeoutMs,
        headers: { Accept: "application/json" }
      },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          if (body.length < 4096) {
            body += chunk;
          }
        });
        res.on("end", () => {
          if (res.statusCode !== 200) {
            resolve({ ok: false, error: `Health endpoint returned HTTP ${res.statusCode}` });
            return;
          }
          try {
            const payload = JSON.parse(body);
            resolve({ ok: payload.status === "ok", health: payload, error: null });
          } catch (_err) {
            resolve({ ok: false, error: "Invalid JSON from local backend health endpoint" });
          }
        });
      }
    );

    req.on("timeout", () => {
      req.destroy();
      resolve({ ok: false, error: "Health check timed out" });
    });

    req.on("error", (err) => {
      resolve({ ok: false, error: err.message });
    });
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class BackendManager extends EventEmitter {
  constructor(options = {}) {
    super();
    this.port = options.port || resolveBackendPort(options.env || process.env);
    this.backendUrl = buildBackendUrl(this.port);
    this.pythonExecutable =
      options.pythonExecutable ||
      process.env.MINDVAULT_PYTHON ||
      (process.platform === "win32" ? "python" : "python3");
    this.launcherPath =
      options.launcherPath || path.resolve(__dirname, "..", "run.py");
    this.spawnArgs = options.spawnArgs || [
      this.launcherPath,
      "--host",
      "127.0.0.1",
      "--port",
      String(this.port)
    ];
    this.cwd = options.cwd || path.resolve(__dirname, "..");
    this.skipSpawn = Boolean(
      options.skipSpawn ?? process.env.MINDVAULT_SKIP_BACKEND_SPAWN === "1"
    );
    this.pollIntervalMs = options.pollIntervalMs || DEFAULT_POLL_INTERVAL_MS;
    this.startupTimeoutMs = options.startupTimeoutMs || DEFAULT_STARTUP_TIMEOUT_MS;
    this.startupPollMs = options.startupPollMs || DEFAULT_STARTUP_POLL_MS;

    this._child = null;
    this._pollTimer = null;
    this._stopping = false;
    this._stderrBuffer = "";
    this._status = {
      state: "idle",
      healthy: false,
      managedProcess: false,
      backendUrl: this.backendUrl,
      error: null,
      health: null,
      lastCheckedAt: null
    };
  }

  getStatus() {
    return { ...this._status };
  }

  _updateStatus(patch) {
    this._status = {
      ...this._status,
      ...patch,
      backendUrl: this.backendUrl,
      lastCheckedAt: new Date().toISOString()
    };
    this.emit("status", this.getStatus());
  }

  async pollOnce() {
    const result = await checkLocalHealth(this.port);
    if (result.ok) {
      this._updateStatus({
        state: "running",
        healthy: true,
        managedProcess: Boolean(this._child),
        error: null,
        health: result.health
      });
      return true;
    }

    if (this._status.state !== "starting") {
      this._updateStatus({
        state: this._status.state === "failed" ? "failed" : "disconnected",
        healthy: false,
        managedProcess: Boolean(this._child),
        error: result.error || "Local backend unreachable",
        health: null
      });
    }
    return false;
  }

  startPolling() {
    this.stopPolling();
    this._pollTimer = setInterval(() => {
      void this.pollOnce();
    }, this.pollIntervalMs);
    if (typeof this._pollTimer.unref === "function") {
      this._pollTimer.unref();
    }
  }

  stopPolling() {
    if (this._pollTimer) {
      clearInterval(this._pollTimer);
      this._pollTimer = null;
    }
  }

  async start() {
    this._stopping = false;
    this._updateStatus({
      state: "starting",
      healthy: false,
      error: null
    });

    // Reuse an already-running local backend if healthy.
    const existing = await checkLocalHealth(this.port);
    if (existing.ok) {
      this._updateStatus({
        state: "running",
        healthy: true,
        managedProcess: false,
        error: null,
        health: existing.health
      });
      this.startPolling();
      return this.getStatus();
    }

    if (this.skipSpawn) {
      this._updateStatus({
        state: "failed",
        healthy: false,
        managedProcess: false,
        error: "Backend auto-start is disabled and no local backend is reachable."
      });
      this.startPolling();
      return this.getStatus();
    }

    await this._spawnChild();
    this.startPolling();
    return this.getStatus();
  }

  async _spawnChild() {
    this._stderrBuffer = "";
    let spawnError = null;
    let exitedEarly = null;

    try {
      this._child = spawn(this.pythonExecutable, this.spawnArgs, {
        cwd: this.cwd,
        shell: false,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...process.env,
          MINDVAULT_HOST: "127.0.0.1",
          MINDVAULT_PORT: String(this.port)
        }
      });
    } catch (err) {
      this._child = null;
      this._updateStatus({
        state: "failed",
        healthy: false,
        managedProcess: false,
        error: `Failed to spawn local backend: ${err.message}`
      });
      return;
    }

    const child = this._child;

    child.stderr.on("data", (chunk) => {
      const text = chunk.toString("utf8");
      this._stderrBuffer = (this._stderrBuffer + text).slice(-2000);
    });

    child.on("error", (err) => {
      spawnError = err;
      if (this._child === child) {
        this._child = null;
      }
      if (!this._stopping) {
        this._updateStatus({
          state: "failed",
          healthy: false,
          managedProcess: false,
          error: `Backend process error: ${err.message}`
        });
      }
    });

    child.on("exit", (code, signal) => {
      exitedEarly = { code, signal };
      if (this._child === child) {
        this._child = null;
      }
      if (!this._stopping && this._status.state !== "failed") {
        this._updateStatus({
          state: "failed",
          healthy: false,
          managedProcess: false,
          error: `Local backend exited unexpectedly (code=${code ?? "null"}, signal=${signal ?? "none"}).`
        });
      }
    });

    const deadline = Date.now() + this.startupTimeoutMs;
    while (Date.now() < deadline) {
      if (this._stopping || spawnError || exitedEarly) {
        break;
      }
      const probe = await checkLocalHealth(this.port, 800);
      if (probe.ok) {
        this._updateStatus({
          state: "running",
          healthy: true,
          managedProcess: true,
          error: null,
          health: probe.health
        });
        return;
      }
      await sleep(this.startupPollMs);
    }

    if (!this._stopping && this._status.state !== "running") {
      const detail = spawnError
        ? `Could not launch '${this.pythonExecutable}': ${spawnError.message}`
        : exitedEarly
          ? `Backend exited during startup (code=${exitedEarly.code ?? "null"}).`
          : `Timed out waiting for local backend on ${this.backendUrl}.`;

      this._updateStatus({
        state: "failed",
        healthy: false,
        managedProcess: Boolean(this._child),
        error: detail
      });
      await this.stopChildOnly();
    }
  }

  async retry() {
    await this.stopChildOnly();
    return this.start();
  }

  async stopChildOnly() {
    const child = this._child;
    if (!child) {
      return;
    }
    this._child = null;

    await new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (!settled) {
          settled = true;
          resolve();
        }
      };

      const forceTimer = setTimeout(() => {
        try {
          if (!child.killed) {
            child.kill("SIGKILL");
          }
        } catch (_err) {
          // Process already exited.
        }
        finish();
      }, 3000);
      if (typeof forceTimer.unref === "function") {
        forceTimer.unref();
      }

      child.once("exit", () => {
        clearTimeout(forceTimer);
        finish();
      });

      try {
        child.kill("SIGTERM");
      } catch (_err) {
        clearTimeout(forceTimer);
        finish();
      }
    });
  }

  async stop() {
    this._stopping = true;
    this.stopPolling();
    await this.stopChildOnly();
    this._updateStatus({
      state: "stopped",
      healthy: false,
      managedProcess: false,
      error: null,
      health: null
    });
  }
}

module.exports = {
  BackendManager,
  buildBackendUrl,
  checkLocalHealth,
  resolveBackendPort
};
