"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const d3Lite = require("../renderer/js/d3-lite");
const {
  DashboardController,
  renderBeliefDriftChart,
  renderConceptVelocityChart,
  renderDecayHeatmap,
  renderHealthRadar
} = require("../renderer/js/dashboard");

const FRONTEND_DIR = path.resolve(__dirname, "..");

function readFrontendFile(...segments) {
  return fs.readFileSync(path.join(FRONTEND_DIR, ...segments), "utf8");
}

class FakeClassList {
  constructor(el) {
    this._el = el;
    this._set = new Set();
  }

  add(...names) {
    names.forEach((n) => {
      if (n) {
        this._set.add(n);
      }
    });
    this._sync();
  }

  remove(...names) {
    names.forEach((n) => this._set.delete(n));
    this._sync();
  }

  toggle(name, force) {
    if (force === undefined) {
      if (this._set.has(name)) {
        this._set.delete(name);
      } else {
        this._set.add(name);
      }
    } else if (force) {
      this._set.add(name);
    } else {
      this._set.delete(name);
    }
    this._sync();
  }

  contains(name) {
    return this._set.has(name);
  }

  _sync() {
    this._el._attrs.class = Array.from(this._set).join(" ");
  }
}

class FakeElement {
  constructor(doc, tagName) {
    this.ownerDocument = doc;
    this.tagName = tagName;
    this.children = [];
    this.parentNode = null;
    this._attrs = {};
    this.style = {};
    this.dataset = {};
    this.textContent = "";
    this.value = "";
    this.checked = false;
    this.clientWidth = 560;
    this._listeners = {};
    this.classList = new FakeClassList(this);
  }

  set id(val) {
    this._attrs.id = val;
    this.ownerDocument._byId.set(val, this);
  }

  get id() {
    return this._attrs.id || "";
  }

  set className(val) {
    this.classList._set = new Set(String(val || "").split(/\s+/).filter(Boolean));
    this.classList._sync();
  }

  get className() {
    return Array.from(this.classList._set).join(" ");
  }

  setAttribute(name, val) {
    const str = String(val);
    this._attrs[name] = str;
    if (name === "id") {
      this.ownerDocument._byId.set(str, this);
    } else if (name === "class") {
      this.className = str;
    } else if (name.startsWith("data-")) {
      const camel = name.slice(5).replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
      this.dataset[camel] = str;
    }
  }

  getAttribute(name) {
    return this._attrs[name];
  }

  removeAttribute(name) {
    delete this._attrs[name];
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      child.parentNode = null;
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
    const handlers = this._listeners[evt.type] || [];
    handlers.forEach((fn) => fn(evt));
  }

  click() {
    this.dispatchEvent({ type: "click", preventDefault() {} });
  }

  querySelectorAll(selector) {
    const results = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (matchesSelector(child, selector)) {
          results.push(child);
        }
        walk(child);
      }
    };
    walk(this);
    return results;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}

function matchesSelector(el, selector) {
  if (selector.startsWith("[data-")) {
    const m = selector.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);
    if (!m) {
      return false;
    }
    const attrVal = el.getAttribute(m[1]);
    return m[2] !== undefined ? attrVal === m[2] : attrVal !== undefined;
  }
  const [tagPart, classPart] = selector.split(".");
  if (tagPart && el.tagName !== tagPart) {
    return false;
  }
  if (classPart && !el.classList.contains(classPart)) {
    return false;
  }
  return true;
}

class FakeDocument {
  constructor() {
    this._byId = new Map();
    this.root = new FakeElement(this, "div");
    this.body = this.root;
  }

  createElement(tag) {
    return new FakeElement(this, tag);
  }

  createElementNS(_ns, tag) {
    return new FakeElement(this, tag);
  }

  getElementById(id) {
    return this._byId.get(id) || null;
  }

  querySelector(selector) {
    return this.root.querySelector(selector);
  }

  querySelectorAll(selector) {
    return this.root.querySelectorAll(selector);
  }
}

function createDashboardFixtureDoc() {
  const doc = new FakeDocument();
  const panel = doc.createElement("section");
  panel.setAttribute("data-panel", "dashboard");
  doc.root.appendChild(panel);

  const ids = [
    ["input", "dashboard-epoch-min"],
    ["input", "dashboard-epoch-max"],
    ["button", "dashboard-apply-epoch"],
    ["button", "dashboard-reset-epoch"],
    ["button", "dashboard-refresh-btn"],
    ["div", "dashboard-loading"],
    ["div", "dashboard-error"],
    ["span", "dashboard-error-text"],
    ["button", "dashboard-error-retry"],
    ["strong", "kpi-docs"],
    ["strong", "kpi-due"],
    ["strong", "kpi-contradictions"],
    ["strong", "kpi-retention"],
    ["strong", "kpi-health"],
    ["select", "brief-max-insights"],
    ["button", "brief-generate-btn"],
    ["button", "brief-regenerate-btn"],
    ["div", "morning-brief-container"],
    ["div", "chart-belief-drift"],
    ["div", "chart-concept-velocity"],
    ["div", "concept-velocity-tags"],
    ["div", "chart-decay-heatmap"],
    ["div", "chart-health-radar"],
    ["div", "flashcards-container"],
    ["div", "contradictions-container"],
    ["div", "dashboard-inspector-body"],
    ["button", "inspector-clear-btn"]
  ];
  ids.forEach(([tag, id]) => {
    const el = doc.createElement(tag);
    el.id = id;
    if (id === "brief-max-insights") {
      el.value = "3";
    }
    if (id === "dashboard-loading" || id === "dashboard-error") {
      el.classList.add("hidden");
    }
    panel.appendChild(el);
  });

  ["decaying_notes", "contradictions", "dormant_epochs", "major_clusters"].forEach((cat) => {
    const cb = doc.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.setAttribute("data-brief-category", cat);
    panel.appendChild(cb);
  });

  return doc;
}

function buildMockOverview() {
  return {
    kpis: {
      total_documents: 2,
      total_chunks: 2,
      due_flashcards: 1,
      open_contradictions: 1,
      active_beliefs: 2,
      average_retention: 0.64,
      health_score: 0.72
    },
    belief_drift: {
      epochs: [
        {
          epoch: 1,
          total_count: 2,
          active_count: 1,
          superseded_count: 1,
          uncertain_count: 0,
          contradiction_count: 1,
          average_confidence: 0.9,
          beliefs: [
            {
              id: "b1",
              status: "superseded",
              belief_text: "Old SQLite claim",
              filename: "notes.md",
              chunk_index: 0,
              confidence: 0.9
            }
          ]
        },
        {
          epoch: 2,
          total_count: 3,
          active_count: 3,
          superseded_count: 0,
          uncertain_count: 0,
          contradiction_count: 0,
          average_confidence: 0.95,
          beliefs: []
        }
      ]
    },
    concept_velocity: {
      epochs: [
        {
          epoch: 1,
          new_chunks: 1,
          emerging_concepts_count: 2,
          emerging_concepts: ["sqlite", "concurrency"],
          velocity_score: 2.5
        },
        {
          epoch: 2,
          new_chunks: 2,
          emerging_concepts_count: 3,
          emerging_concepts: ["chromadb", "vector", "hybrid"],
          velocity_score: 4.25
        }
      ],
      concepts: [
        {
          concept: "sqlite",
          total_mentions: 4,
          first_epoch: 1,
          last_epoch: 2,
          sample_chunks: [
            {
              chunk_id: "c1",
              filename: "notes.md",
              chunk_index: 0,
              epoch: 1,
              snippet: "SQLite WAL mode concurrency notes."
            }
          ]
        }
      ]
    },
    decay: {
      cells: [
        {
          chunk_id: "c1",
          document_id: "d1",
          filename: "notes.md",
          chunk_index: 0,
          epoch: 1,
          stability: 1.0,
          retention: 0.42,
          decay_band: "critical",
          section_heading: "WAL Mode",
          snippet: "WAL improves concurrent readers."
        }
      ]
    },
    health: {
      overall_score: 0.72,
      status_label: "good",
      total_chunks: 2,
      axes: [
        { key: "retention_health", label: "Retention", score: 0.64, value_pct: 64, detail: "ok" },
        { key: "belief_consistency", label: "Consistency", score: 0.7, value_pct: 70, detail: "ok" },
        { key: "graph_connectivity", label: "Graph", score: 0.8, value_pct: 80, detail: "ok" },
        { key: "review_freshness", label: "Freshness", score: 0.5, value_pct: 50, detail: "ok" },
        { key: "knowledge_coverage", label: "Coverage", score: 0.95, value_pct: 95, detail: "ok" }
      ]
    },
    review_queue: {
      items: [
        {
          chunk_id: "c1",
          filename: "notes.md",
          chunk_index: 0,
          epoch: 1,
          retention: 0.42,
          stability: 1.0,
          flashcard: {
            prompt: "Recall notes under WAL Mode.",
            hint: "Section: WAL Mode",
            answer_revealed: false,
            answer: null
          }
        }
      ]
    },
    contradictions: {
      total: 1,
      items: [
        {
          id: "con-1",
          score: 0.91,
          similarity_score: 0.72,
          belief_a_id: "b1",
          belief_b_id: "b2",
          belief_a: { belief_text: "Old WAL claim", epoch: 1, filename: "old.md" },
          belief_b: { belief_text: "Updated WAL claim", epoch: 2, filename: "new.md" }
        }
      ]
    },
    latest_brief: null
  };
}

test("dashboard frontend scripts avoid innerHTML and CSP leaks", () => {
  const html = readFrontendFile("renderer", "index.html");
  const d3Src = readFrontendFile("renderer", "js", "d3-lite.js");
  const dashSrc = readFrontendFile("renderer", "js", "dashboard.js");

  assert.doesNotMatch(d3Src, /\.innerHTML\s*=/);
  assert.doesNotMatch(dashSrc, /\.innerHTML\s*=/);
  assert.doesNotMatch(dashSrc, /https?:\/\/(?!127\.0\.0\.1|localhost)/);

  for (const viewId of [
    "belief-drift",
    "concept-velocity",
    "decay",
    "health",
    "flashcards",
    "contradictions"
  ]) {
    assert.match(html, new RegExp(`data-dashboard-view="${viewId}"`));
  }
  assert.match(html, /\.\/js\/d3-lite\.js/);
  assert.match(html, /\.\/js\/dashboard\.js/);
});

test("d3-lite scales, shapes, and colors compute deterministically", () => {
  const lin = d3Lite.scaleLinear().domain([0, 10]).range([0, 100]);
  assert.equal(lin(5), 50);

  const band = d3Lite.scaleBand().domain(["E1", "E2"]).range([0, 100]).padding(0.2);
  assert.ok(band.bandwidth() > 0);
  assert.ok(band("E2") > band("E1"));

  const pathD = d3Lite.line().x((d) => d[0]).y((d) => d[1])([[0, 0], [10, 20]]);
  assert.equal(pathD, "M0.00,0.00 L10.00,20.00");
  assert.equal(d3Lite.retentionColor(0.3), "#f85149");
  assert.equal(d3Lite.retentionColor(0.65), "#d29922");
  assert.equal(d3Lite.retentionColor(0.9), "#3fb950");
});

test("dashboard renders 6 views, click-to-explore, flashcards, contradictions, and brief", async () => {
  const doc = createDashboardFixtureDoc();
  const requests = [];
  let mockOverview = buildMockOverview();

  const mockFetch = async (url, init = {}) => {
    requests.push({ url, method: init.method || "GET", body: init.body });
    if (url.includes("/api/v1/dashboard/overview")) {
      return { ok: true, json: async () => mockOverview };
    }
    if (url.includes("/api/v1/briefs?limit=10")) {
      return { ok: true, json: async () => ({ items: [] }) };
    }
    if (url.includes("/api/v1/review/queue") && url.includes("include_answer=true")) {
      return {
        ok: true,
        json: async () => ({
          items: [
            {
              chunk_id: "c1",
              flashcard: {
                answer_revealed: true,
                answer: "WAL improves concurrent readers.",
                key_beliefs: []
              }
            }
          ]
        })
      };
    }
    if (url.includes("/api/v1/review/c1")) {
      return {
        ok: true,
        json: async () => ({
          outcome: "good",
          new_stability: 2.4,
          next_review_at: "2026-11-28T00:00:00+00:00"
        })
      };
    }
    if (url.includes("/api/v1/contradictions/con-1/resolve")) {
      return { ok: true, json: async () => ({ id: "con-1", status: "resolved" }) };
    }
    if (url.includes("/api/v1/briefs/generate")) {
      const parsed = JSON.parse(init.body || "{}");
      return {
        ok: true,
        json: async () => ({
          id: "brief-1",
          brief_date: "2026-11-24",
          max_insights: parsed.max_insights || 3,
          regenerated: Boolean(parsed.regenerate),
          insights: [
            {
              id: "i1",
              category: "decaying_notes",
              title: "1 memory chunk due for review",
              summary: "Lowest retention R=0.42.",
              recommended_action: "Review flashcards.",
              citations: [
                { chunk_id: "c1", filename: "notes.md", chunk_index: 0, epoch: 1, snippet: "WAL!" }
              ]
            }
          ]
        })
      };
    }
    return { ok: false, status: 404, json: async () => ({ detail: "Not found" }) };
  };

  const ctrl = new DashboardController({
    document: doc,
    fetch: mockFetch,
    d3: d3Lite,
    backendUrl: "http://127.0.0.1:8765"
  });
  await ctrl.refresh();

  assert.equal(doc.getElementById("kpi-docs").textContent, "2");
  assert.equal(doc.getElementById("kpi-retention").textContent, "64%");
  assert.ok(doc.getElementById("chart-belief-drift").querySelectorAll("circle.drift-node").length === 2);
  assert.ok(doc.getElementById("chart-concept-velocity").querySelectorAll("rect.velocity-bar").length === 2);
  assert.ok(doc.getElementById("chart-decay-heatmap").querySelectorAll("rect.heatmap-cell").length === 1);
  assert.ok(doc.getElementById("chart-health-radar").querySelectorAll("polygon.radar-shape").length === 1);

  // Click-to-explore on heatmap cell populates inspector.
  doc.getElementById("chart-decay-heatmap").querySelectorAll("rect.heatmap-cell")[0].click();
  const inspectorBody = doc.getElementById("dashboard-inspector-body");
  assert.match(inspectorBody.children[0].textContent, /notes\.md/);
  assert.match(inspectorBody.children[1].textContent, /Retention R=0\.42/);

  // Flashcard reveal & outcome submission.
  const revealBtn = doc.getElementById("flashcards-container").querySelector('[data-action="reveal-flashcard"]');
  assert.ok(revealBtn);
  revealBtn.click();
  await new Promise((r) => setTimeout(r, 10));
  const goodBtn = doc.getElementById("flashcards-container").querySelector('[data-outcome="good"]');
  assert.ok(goodBtn);
  goodBtn.click();
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(requests.some((r) => r.url.includes("/api/v1/review/c1") && r.method === "POST"));

  // Contradiction resolution.
  const acceptNewerBtn = doc.getElementById("contradictions-container").querySelector('[data-resolution="accept_newer"]');
  assert.ok(acceptNewerBtn);
  acceptNewerBtn.click();
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(requests.some((r) => r.url.includes("/api/v1/contradictions/con-1/resolve")));

  // Morning Brief generation & regeneration.
  doc.getElementById("brief-generate-btn").click();
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(doc.getElementById("morning-brief-container").querySelectorAll("li.brief-insight-card").length === 1);

  // Epoch filter passes query params.
  doc.getElementById("dashboard-epoch-min").value = "1";
  doc.getElementById("dashboard-epoch-max").value = "2";
  doc.getElementById("dashboard-apply-epoch").click();
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(requests.some((r) => r.url.includes("epoch_min=1") && r.url.includes("epoch_max=2")));

  // Empty state rendering for each chart.
  mockOverview = {
    kpis: { total_documents: 0, total_chunks: 0, due_flashcards: 0, open_contradictions: 0, active_beliefs: 0, average_retention: 0, health_score: 0 },
    belief_drift: { epochs: [] },
    concept_velocity: { epochs: [], concepts: [] },
    decay: { cells: [] },
    health: { overall_score: 0, total_chunks: 0, axes: [] },
    review_queue: { items: [] },
    contradictions: { items: [] },
    latest_brief: null
  };
  await ctrl.refresh();
  assert.equal(doc.getElementById("chart-belief-drift").children[0].getAttribute("data-state"), "empty");
  assert.equal(doc.getElementById("chart-concept-velocity").children[0].getAttribute("data-state"), "empty");
  assert.equal(doc.getElementById("chart-decay-heatmap").children[0].getAttribute("data-state"), "empty");
  assert.equal(doc.getElementById("chart-health-radar").children[0].getAttribute("data-state"), "empty");

  // Error state handling.
  const failingCtrl = new DashboardController({
    document: doc,
    fetch: async () => {
      throw new Error("Backend offline");
    },
    d3: d3Lite
  });
  await failingCtrl.refresh();
  assert.equal(doc.getElementById("dashboard-error-text").textContent, "Backend offline");
  assert.equal(doc.getElementById("dashboard-error").classList.contains("hidden"), false);

  // Direct render functions also work cleanly on standalone containers.
  const isolated = doc.createElement("div");
  renderBeliefDriftChart(isolated, { epochs: [] }, { d3: d3Lite });
  renderConceptVelocityChart(isolated, null, { epochs: [], concepts: [] }, { d3: d3Lite });
  renderDecayHeatmap(isolated, { cells: [] }, { d3: d3Lite });
  renderHealthRadar(isolated, { total_chunks: 0, axes: [] }, { d3: d3Lite });
});
