"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const d3Lite = require("../renderer/js/d3-lite");
const { WorkspaceController, parseSseBlock, renderGraphSvg } = require("../renderer/js/workspace");

const FRONTEND_DIR = path.resolve(__dirname, "..");

function readFrontendFile(...segments) {
  return fs.readFileSync(path.join(FRONTEND_DIR, ...segments), "utf8");
}

class MiniEl {
  constructor(doc, tag) {
    this.ownerDocument = doc;
    this.tagName = tag;
    this.children = [];
    this.parentNode = null;
    this._attrs = {};
    this.style = {};
    this.dataset = {};
    this.textContent = "";
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.files = [];
    this.clientWidth = 640;
    this._listeners = {};
    this._cls = new Set();
    const self = this;
    this.classList = {
      add: (...xs) => xs.forEach((x) => x && self._cls.add(x)),
      remove: (...xs) => xs.forEach((x) => self._cls.delete(x)),
      toggle: (x, f) => {
        if (f === undefined ? !self._cls.has(x) : f) {
          self._cls.add(x);
        } else {
          self._cls.delete(x);
        }
      },
      contains: (x) => self._cls.has(x)
    };
  }

  set id(v) {
    this._attrs.id = v;
    this.ownerDocument._byId.set(v, this);
  }

  get id() {
    return this._attrs.id || "";
  }

  set className(v) {
    this._cls = new Set(String(v || "").split(/\s+/).filter(Boolean));
  }

  get className() {
    return Array.from(this._cls).join(" ");
  }

  setAttribute(k, v) {
    this._attrs[k] = String(v);
    if (k === "id") {
      this.id = v;
    } else if (k === "class") {
      this.className = v;
    }
  }

  getAttribute(k) {
    return this._attrs[k];
  }

  removeAttribute(k) {
    delete this._attrs[k];
  }

  appendChild(ch) {
    ch.parentNode = this;
    this.children.push(ch);
    return ch;
  }

  removeChild(ch) {
    const idx = this.children.indexOf(ch);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      ch.parentNode = null;
    }
  }

  remove() {
    if (this.parentNode) {
      this.parentNode.removeChild(this);
    }
  }

  replaceChildren(...nodes) {
    this.children.forEach((c) => {
      c.parentNode = null;
    });
    this.children = [];
    nodes.forEach((n) => this.appendChild(n));
  }

  addEventListener(type, fn) {
    if (!this._listeners[type]) {
      this._listeners[type] = [];
    }
    this._listeners[type].push(fn);
  }

  dispatchEvent(evt) {
    (this._listeners[evt.type] || []).forEach((fn) => fn(evt));
  }

  querySelectorAll(selector) {
    const out = [];
    const walk = (n) => {
      for (const ch of n.children) {
        if (matchSel(ch, selector)) {
          out.push(ch);
        }
        walk(ch);
      }
    };
    walk(this);
    return out;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}

function matchSel(el, sel) {
  if (sel.startsWith("[")) {
    const m = sel.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);
    if (!m) {
      return false;
    }
    const v = el.getAttribute(m[1]);
    return m[2] !== undefined ? v === m[2] : v !== undefined;
  }
  const [tag, cls] = sel.split(".");
  if (tag && el.tagName !== tag) {
    return false;
  }
  if (cls && !el.classList.contains(cls)) {
    return false;
  }
  return true;
}

class MiniDoc {
  constructor() {
    this._byId = new Map();
    this.root = new MiniEl(this, "div");
  }

  createElement(tag) {
    return new MiniEl(this, tag);
  }

  createElementNS(_ns, tag) {
    return new MiniEl(this, tag);
  }

  getElementById(id) {
    return this._byId.get(id) || null;
  }

  querySelector(s) {
    return this.root.querySelector(s);
  }

  querySelectorAll(s) {
    return this.root.querySelectorAll(s);
  }
}

function createWorkspaceDoc() {
  const doc = new MiniDoc();
  [
    ["select", "chat-mode-select"],
    ["input", "chat-socratic-toggle"],
    ["input", "chat-epoch-min"],
    ["input", "chat-epoch-max"],
    ["form", "chat-query-form"],
    ["input", "chat-query-input"],
    ["button", "chat-cancel-btn"],
    ["div", "chat-messages"],
    ["div", "chat-citation-viewer"],
    ["select", "graph-type-select"],
    ["input", "graph-epoch-min"],
    ["input", "graph-epoch-max"],
    ["input", "graph-search-input"],
    ["button", "graph-refresh-btn"],
    ["button", "graph-rebuild-btn"],
    ["div", "graph-stats-bar"],
    ["div", "chart-knowledge-graph"],
    ["div", "graph-inspector-body"],
    ["div", "ingest-dropzone"],
    ["input", "ingest-file-input"],
    ["button", "ingest-browse-btn"],
    ["input", "ingest-epoch-input"],
    ["input", "ingest-note-title"],
    ["textarea", "ingest-note-content"],
    ["button", "ingest-submit-note-btn"],
    ["div", "ingest-status"],
    ["input", "research-search-input"],
    ["button", "research-search-btn"],
    ["div", "research-search-results"],
    ["button", "research-refresh-btn"],
    ["div", "research-documents-list"],
    ["button", "dashboard-brief-overlay-btn"],
    ["div", "morning-brief-overlay"],
    ["div", "brief-overlay-content"],
    ["button", "brief-overlay-close"],
    ["button", "brief-overlay-regenerate"],
    ["div", "settings-model-status"],
    ["div", "settings-index-status"],
    ["div", "settings-jobs-list"],
    ["button", "settings-rebuild-index-btn"],
    ["button", "settings-refresh-diagnostics"],
    ["span", "status-offline-badge"]
  ].forEach(([tag, id]) => {
    const el = doc.createElement(tag);
    el.id = id;
    if (id === "chat-mode-select") {
      el.value = "summary";
    }
    if (id === "graph-type-select") {
      el.value = "concepts";
    }
    if (id === "morning-brief-overlay") {
      el.classList.add("hidden");
    }
    doc.root.appendChild(el);
  });
  return doc;
}

test("index.html removes all placeholders and package.json excludes models", () => {
  const html = readFrontendFile("renderer", "index.html");
  const wsSrc = readFrontendFile("renderer", "js", "workspace.js");
  const appSrc = readFrontendFile("renderer", "js", "app.js");
  const pkg = JSON.parse(readFrontendFile("package.json"));

  assert.doesNotMatch(html, /placeholder-card/);
  assert.doesNotMatch(html, /Not Yet Enabled/);
  assert.doesNotMatch(wsSrc, /\.innerHTML\s*=/);
  assert.doesNotMatch(wsSrc, /https?:\/\/(?!127\.0\.0\.1|localhost)/);

  // Command palette actions cover navigation, search, ingestion, query, settings, and review.
  for (const needle of ["Query", "Ingestion", "Search", "Review", "Settings", "Morning Brief Overlay"]) {
    assert.match(appSrc, new RegExp(needle));
  }

  // Packaging config excludes huge weights and local vault dirs.
  assert.ok(pkg.build && Array.isArray(pkg.build.files));
  assert.ok(pkg.build.files.includes("!**/*.gguf"));
  assert.ok(pkg.build.files.includes("!models/**"));
  assert.ok(pkg.build.files.includes("!data/**"));
});

test("workspace handles SSE streaming, citations, Socratic mode, graph, and settings", async () => {
  const parsedSse = parseSseBlock('event: token\ndata: {"text": "Hello"}');
  assert.deepEqual(parsedSse, { event: "token", data: { text: "Hello" } });

  const doc = createWorkspaceDoc();
  doc.getElementById("chat-mode-select").value = "socratic";
  doc.getElementById("chat-query-input").value = "Why WAL?";
  doc.getElementById("chat-epoch-min").value = "1";

  const sseStream =
    'event: status\ndata: {"message": "Retrieving chunks..."}\n\n' +
    'event: subquestion\ndata: {"question": "What does WAL guarantee?"}\n\n' +
    'event: citation\ndata: {"label": "C1", "citation_id": "cit1", "chunk_id": "c1", "filename": "wal.md", "chunk_index": 0, "epoch": 1, "score": 0.9, "excerpt": "Durable writes."}\n\n' +
    'event: token\ndata: {"text": "What happens during crash recovery [1]?"}\n\n' +
    'event: done\ndata: {"ok": true}\n\n';

  const mockFetch = async (url, _init) => {
    if (url.endsWith("/api/v1/chat/stream")) {
      return { ok: true, text: async () => sseStream };
    }
    if (url.includes("/api/v1/graph/concepts")) {
      return {
        ok: true,
        json: async () => ({
          graph_type: "concepts",
          nodes: [
            { id: "n1", node_type: "chunk", label: "WAL", epoch: 1, degree: 1, orphan: false },
            { id: "n2", node_type: "belief", label: "Durable", epoch: 1, degree: 1, orphan: false },
            { id: "n3", node_type: "chunk", label: "Orphan", epoch: 2, degree: 0, orphan: true }
          ],
          edges: [{ id: "e1", source: "n1", target: "n2", edge_type: "derived_from", weight: 0.9 }],
          stats: { node_count: 3, edge_count: 1, orphan_count: 1 }
        })
      };
    }
    if (url.includes("/api/v1/graph/node/n1")) {
      return {
        ok: true,
        json: async () => ({
          nodes: [
            { id: "n1", node_type: "chunk", label: "WAL", epoch: 1, degree: 1 },
            { id: "n2", node_type: "belief", label: "Durable", epoch: 1, degree: 1 }
          ]
        })
      };
    }
    if (url.endsWith("/api/v1/briefs/generate")) {
      return {
        ok: true,
        json: async () => ({
          brief_date: "2026-11-24",
          insights: [{ category: "decaying_notes", title: "Review WAL", summary: "R=0.4", recommended_action: "Review" }]
        })
      };
    }
    if (url.includes("/api/v1/system/status")) {
      return {
        ok: true,
        json: async () => ({
          offline_only: true,
          models: {
            embedding: { available: false, reason: "Unconfigured" },
            llm: { available: true, provider: "llamacpp", model: "qwen.gguf" },
            nli: { available: true, model: "lexical" }
          },
          index: { sqlite_indexed_documents: 2, sqlite_indexed_chunks: 4, bm25_indexed_chunks: 4, vector_indexed_chunks: 0, synchronized: true },
          recovery: { database: { recovered: false } },
          scheduler: {
            jobs: [{ job_id: "retention_recalculation", name: "Retention", state: "idle", run_count: 1, interval_seconds: 180 }]
          }
        })
      };
    }
    return { ok: true, json: async () => ({ documents: [], results: [] }) };
  };

  const ws = new WorkspaceController({
    document: doc,
    fetch: mockFetch,
    d3: d3Lite,
    backendUrl: "http://127.0.0.1:8765"
  });
  await ws.submitChat();
  const chatMsgs = doc.getElementById("chat-messages");
  assert.ok(chatMsgs.querySelectorAll("details.progressive-disclosure").length === 1);
  const pill = chatMsgs.querySelectorAll("button.citation-pill")[0];
  assert.ok(pill);
  pill.dispatchEvent({ type: "click" });
  assert.match(doc.getElementById("chat-citation-viewer").children[0].textContent, /wal\.md/);

  await ws.loadGraph();
  const graphEl = doc.getElementById("chart-knowledge-graph");
  assert.equal(graphEl.querySelectorAll("circle.graph-node").length, 3);
  assert.equal(graphEl.querySelectorAll("circle.orphan-node").length, 1);

  await ws.openMorningBriefOverlay(false);
  assert.equal(doc.getElementById("morning-brief-overlay").classList.contains("hidden"), false);
  ws.closeMorningBriefOverlay();
  assert.equal(doc.getElementById("morning-brief-overlay").classList.contains("hidden"), true);

  await ws.loadSystemDiagnostics();
  assert.equal(doc.getElementById("status-offline-badge").textContent, "Offline-only");
  assert.equal(doc.getElementById("settings-jobs-list").children.length, 1);

  const stand = doc.createElement("div");
  renderGraphSvg(stand, { nodes: [], edges: [] }, { d3: d3Lite });
  assert.equal(stand.children[0].getAttribute("data-state"), "empty");
});
