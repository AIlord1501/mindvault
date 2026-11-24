(function (globalScope) {
  "use strict";

  const DEFAULT_CATEGORIES = [
    "decaying_notes",
    "contradictions",
    "dormant_epochs",
    "major_clusters"
  ];

  const CATEGORY_LABELS = {
    decaying_notes: "Decaying Notes",
    contradictions: "Contradictions",
    dormant_epochs: "Dormant Epochs",
    major_clusters: "Major Clusters"
  };

  function getD3() {
    if (globalScope && globalScope.d3) {
      return globalScope.d3;
    }
    if (typeof require === "function") {
      return require("./d3-lite");
    }
    return null;
  }

  function isLocalLoopbackUrl(url) {
    return typeof url === "string" && /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/i.test(url.trim());
  }

  function createEl(doc, tag, className, text) {
    const el = doc.createElement(tag);
    if (className) {
      el.className = className;
    }
    if (text !== undefined && text !== null) {
      el.textContent = String(text);
    }
    return el;
  }

  function makeAccessibleActivation(node, onActivate, label) {
    if (!node) {
      return;
    }
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "button");
    if (label) {
      node.setAttribute("aria-label", label);
    }
    node.addEventListener("keydown", (evt) => {
      if (evt.key === "Enter" || evt.key === " ") {
        evt.preventDefault();
        onActivate(evt);
      }
    });
  }

  function formatPct(val) {
    const num = Number(val);
    if (!Number.isFinite(num)) {
      return "0%";
    }
    return `${Math.round(num * 100)}%`;
  }

  function formatFloat(val, digits) {
    const num = Number(val);
    if (!Number.isFinite(num)) {
      return "--";
    }
    return num.toFixed(digits === undefined ? 2 : digits);
  }

  function buildEmptyState(doc, title, description) {
    const box = createEl(doc, "div", "chart-empty-state");
    box.setAttribute("data-state", "empty");
    box.setAttribute("role", "status");
    box.appendChild(createEl(doc, "strong", "empty-title", title));
    box.appendChild(createEl(doc, "p", "empty-desc", description));
    return box;
  }

  function renderBeliefDriftChart(container, data, options) {
    const doc = container.ownerDocument;
    const d3 = (options && options.d3) || getD3();
    const onSelectEpoch = options && options.onSelectEpoch;
    container.replaceChildren();

    const epochs = (data && Array.isArray(data.epochs) ? data.epochs : []).slice();
    if (epochs.length === 0 || !d3) {
      container.appendChild(
        buildEmptyState(
          doc,
          "No belief timeline yet",
          "Extract beliefs from local documents or adjust the epoch filter to view drift."
        )
      );
      return;
    }

    const width = Math.max(340, container.clientWidth || 520);
    const height = 220;
    const pad = { top: 20, right: 20, bottom: 34, left: 38 };

    const svg = d3
      .select(container)
      .append("svg")
      .attr("class", "mv-chart-svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("role", "img")
      .attr("aria-label", "Belief Drift Timeline across epochs");

    const xVals = epochs.map((e) => e.epoch);
    const xMin = Math.min(...xVals);
    const xMax = xVals.length === 1 ? xMin + 1 : Math.max(...xVals);
    const maxCount = Math.max(1, d3.max(epochs, (e) => e.total_count) || 1);

    const xScale = d3.scaleLinear().domain([xMin, xMax]).range([pad.left, width - pad.right]);
    const yScale = d3.scaleLinear().domain([0, maxCount]).range([height - pad.bottom, pad.top]);

    const areaTotal = d3
      .area()
      .x((e) => xScale(e.epoch))
      .y0(() => height - pad.bottom)
      .y1((e) => yScale(e.total_count));

    const lineActive = d3
      .line()
      .x((e) => xScale(e.epoch))
      .y((e) => yScale(e.active_count));

    const lineDrift = d3
      .line()
      .x((e) => xScale(e.epoch))
      .y((e) => yScale(e.superseded_count + e.uncertain_count));

    svg
      .append("line")
      .attr("x1", pad.left)
      .attr("y1", height - pad.bottom)
      .attr("x2", width - pad.right)
      .attr("y2", height - pad.bottom)
      .attr("class", "chart-axis-line");

    svg
      .append("path")
      .attr("d", areaTotal(epochs))
      .attr("class", "drift-area-total");

    svg
      .append("path")
      .attr("d", lineActive(epochs))
      .attr("class", "drift-line-active");

    svg
      .append("path")
      .attr("d", lineDrift(epochs))
      .attr("class", "drift-line-superseded");

    const points = svg
      .selectAll("circle.drift-node")
      .data(epochs)
      .join("circle")
      .attr("class", "drift-node")
      .attr("cx", (e) => xScale(e.epoch))
      .attr("cy", (e) => yScale(e.total_count))
      .attr("r", (e) => (e.contradiction_count > 0 || e.superseded_count > 0 ? 7 : 5))
      .on("click", (_evt, datum) => {
        if (onSelectEpoch) {
          onSelectEpoch(datum);
        }
      });

    points.nodes().forEach((circleEl) => {
      const datum = circleEl.__data__;
      const label =
        `Epoch ${datum.epoch}: ${datum.total_count} beliefs, ` +
        `${datum.superseded_count} superseded, ${datum.contradiction_count} contradictions. ` +
        "Activate to inspect.";
      makeAccessibleActivation(
        circleEl,
        () => {
          if (onSelectEpoch) {
            onSelectEpoch(datum);
          }
        },
        label
      );
    });

    svg
      .selectAll("text.chart-x-label")
      .data(epochs)
      .join("text")
      .attr("class", "chart-x-label")
      .attr("x", (e) => xScale(e.epoch))
      .attr("y", height - 10)
      .attr("text-anchor", "middle")
      .text((e) => `E${e.epoch}`);
  }

  function renderConceptVelocityChart(container, tagsContainer, data, options) {
    const doc = container.ownerDocument;
    const d3 = (options && options.d3) || getD3();
    const onSelectEpoch = options && options.onSelectEpoch;
    const onSelectConcept = options && options.onSelectConcept;

    container.replaceChildren();
    if (tagsContainer) {
      tagsContainer.replaceChildren();
    }

    const epochs = (data && Array.isArray(data.epochs) ? data.epochs : []).slice();
    const concepts = (data && Array.isArray(data.concepts) ? data.concepts : []).slice();

    if (epochs.length === 0 || !d3) {
      container.appendChild(
        buildEmptyState(
          doc,
          "No concept velocity yet",
          "Ingest documents across epochs to measure concept emergence and velocity."
        )
      );
      return;
    }

    const width = Math.max(340, container.clientWidth || 520);
    const height = 220;
    const pad = { top: 20, right: 20, bottom: 34, left: 38 };

    const svg = d3
      .select(container)
      .append("svg")
      .attr("class", "mv-chart-svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("role", "img")
      .attr("aria-label", "Concept Velocity chart across epochs");

    const xBand = d3
      .scaleBand()
      .domain(epochs.map((e) => `E${e.epoch}`))
      .range([pad.left, width - pad.right])
      .padding(0.25);

    const maxVel = Math.max(1, d3.max(epochs, (e) => e.velocity_score) || 1);
    const yScale = d3.scaleLinear().domain([0, maxVel]).range([height - pad.bottom, pad.top]);

    const bars = svg
      .selectAll("rect.velocity-bar")
      .data(epochs)
      .join("rect")
      .attr("class", "velocity-bar")
      .attr("x", (e) => xBand(`E${e.epoch}`))
      .attr("y", (e) => yScale(e.velocity_score))
      .attr("width", () => xBand.bandwidth())
      .attr("height", (e) => Math.max(4, height - pad.bottom - yScale(e.velocity_score)))
      .attr("rx", 4)
      .on("click", (_evt, datum) => {
        if (onSelectEpoch) {
          onSelectEpoch(datum);
        }
      });

    bars.nodes().forEach((rectEl) => {
      const datum = rectEl.__data__;
      const label =
        `Epoch ${datum.epoch}: velocity ${datum.velocity_score}, ` +
        `${datum.emerging_concepts_count} emerging concepts. Activate to inspect.`;
      makeAccessibleActivation(
        rectEl,
        () => {
          if (onSelectEpoch) {
            onSelectEpoch(datum);
          }
        },
        label
      );
    });

    svg
      .selectAll("text.chart-x-label")
      .data(epochs)
      .join("text")
      .attr("class", "chart-x-label")
      .attr("x", (e) => xBand(`E${e.epoch}`) + xBand.bandwidth() / 2)
      .attr("y", height - 10)
      .attr("text-anchor", "middle")
      .text((e) => `E${e.epoch}`);

    if (tagsContainer && concepts.length > 0) {
      concepts.slice(0, 8).forEach((conceptItem) => {
        const chip = createEl(
          doc,
          "button",
          "concept-chip",
          `${conceptItem.concept} (${conceptItem.total_mentions})`
        );
        chip.type = "button";
        chip.setAttribute(
          "aria-label",
          `Explore concept ${conceptItem.concept}, ${conceptItem.total_mentions} mentions`
        );
        chip.addEventListener("click", () => {
          if (onSelectConcept) {
            onSelectConcept(conceptItem);
          }
        });
        tagsContainer.appendChild(chip);
      });
    }
  }

  function renderDecayHeatmap(container, data, options) {
    const doc = container.ownerDocument;
    const d3 = (options && options.d3) || getD3();
    const onSelectCell = options && options.onSelectCell;
    container.replaceChildren();

    const cells = (data && Array.isArray(data.cells) ? data.cells : []).slice(0, 60);
    if (cells.length === 0 || !d3) {
      container.appendChild(
        buildEmptyState(
          doc,
          "No memory cells tracked yet",
          "Ingest notes or clear epoch filters to inspect memory decay."
        )
      );
      return;
    }

    const width = Math.max(340, container.clientWidth || 520);
    const cols = Math.max(4, Math.min(10, Math.floor((width - 24) / 48)));
    const rows = Math.ceil(cells.length / cols);
    const cellSize = Math.floor((width - 24 - (cols - 1) * 6) / cols);
    const boxSize = Math.max(28, Math.min(44, cellSize));
    const height = Math.max(140, rows * (boxSize + 6) + 20);

    const svg = d3
      .select(container)
      .append("svg")
      .attr("class", "mv-chart-svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("role", "img")
      .attr("aria-label", "Knowledge Decay Heatmap showing memory retention per chunk");

    const laidOut = cells.map((cell, idx) => ({
      ...cell,
      _col: idx % cols,
      _row: Math.floor(idx / cols)
    }));

    const rects = svg
      .selectAll("rect.heatmap-cell")
      .data(laidOut)
      .join("rect")
      .attr("class", (c) => `heatmap-cell band-${c.decay_band}`)
      .attr("x", (c) => 12 + c._col * (boxSize + 6))
      .attr("y", (c) => 10 + c._row * (boxSize + 6))
      .attr("width", boxSize)
      .attr("height", boxSize)
      .attr("rx", 5)
      .style("fill", (c) => d3.retentionColor(c.retention))
      .on("click", (_evt, datum) => {
        if (onSelectCell) {
          onSelectCell(datum);
        }
      });

    rects.nodes().forEach((rectEl) => {
      const datum = rectEl.__data__;
      const label =
        `${datum.filename} chunk ${datum.chunk_index}, epoch ${datum.epoch}, ` +
        `retention ${formatPct(datum.retention)}. Activate to inspect.`;
      makeAccessibleActivation(
        rectEl,
        () => {
          if (onSelectCell) {
            onSelectCell(datum);
          }
        },
        label
      );
    });
  }

  function renderHealthRadar(container, data, options) {
    const doc = container.ownerDocument;
    const d3 = (options && options.d3) || getD3();
    const onSelectAxis = options && options.onSelectAxis;
    container.replaceChildren();

    const axes = data && Array.isArray(data.axes) ? data.axes : [];
    if (axes.length === 0 || !data.total_chunks || !d3) {
      container.appendChild(
        buildEmptyState(
          doc,
          "Knowledge Health Radar waiting for data",
          "Ingest notes to evaluate retention, consistency, graph links, and coverage."
        )
      );
      return;
    }

    const width = Math.max(320, container.clientWidth || 440);
    const height = 240;
    const cx = width / 2;
    const cy = height / 2 + 4;
    const radius = Math.min(width, height) * 0.34;
    const n = axes.length;

    const svg = d3
      .select(container)
      .append("svg")
      .attr("class", "mv-chart-svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("role", "img")
      .attr("aria-label", `Knowledge Health Radar: overall score ${formatPct(data.overall_score)}`);

    function angleAt(idx) {
      return -Math.PI / 2 + (idx * 2 * Math.PI) / n;
    }

    [0.25, 0.5, 0.75, 1.0].forEach((lvl) => {
      const ringPts = axes
        .map((_, idx) => {
          const a = angleAt(idx);
          const x = cx + Math.cos(a) * radius * lvl;
          const y = cy + Math.sin(a) * radius * lvl;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
      svg.append("polygon").attr("points", ringPts).attr("class", "radar-grid-ring");
    });

    axes.forEach((axis, idx) => {
      const a = angleAt(idx);
      const x2 = cx + Math.cos(a) * radius;
      const y2 = cy + Math.sin(a) * radius;
      svg
        .append("line")
        .attr("x1", cx)
        .attr("y1", cy)
        .attr("x2", x2)
        .attr("y2", y2)
        .attr("class", "radar-spoke");

      const lx = cx + Math.cos(a) * (radius + 22);
      const ly = cy + Math.sin(a) * (radius + 16);
      const anchor = Math.cos(a) > 0.25 ? "start" : Math.cos(a) < -0.25 ? "end" : "middle";
      svg
        .append("text")
        .attr("x", lx)
        .attr("y", ly)
        .attr("text-anchor", anchor)
        .attr("class", "radar-axis-label")
        .text(`${axis.label} (${axis.value_pct}%)`);
    });

    const dataPts = axes
      .map((axis, idx) => {
        const a = angleAt(idx);
        const r = Math.max(0.08, Number(axis.score) || 0) * radius;
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    svg.append("polygon").attr("points", dataPts).attr("class", "radar-shape");

    const vertices = axes.map((axis, idx) => {
      const a = angleAt(idx);
      const r = Math.max(0.08, Number(axis.score) || 0) * radius;
      return {
        ...axis,
        _vx: cx + Math.cos(a) * r,
        _vy: cy + Math.sin(a) * r
      };
    });

    const nodes = svg
      .selectAll("circle.radar-vertex")
      .data(vertices)
      .join("circle")
      .attr("class", "radar-vertex")
      .attr("cx", (v) => v._vx)
      .attr("cy", (v) => v._vy)
      .attr("r", 6)
      .on("click", (_evt, datum) => {
        if (onSelectAxis) {
          onSelectAxis(datum);
        }
      });

    nodes.nodes().forEach((el) => {
      const datum = el.__data__;
      makeAccessibleActivation(
        el,
        () => {
          if (onSelectAxis) {
            onSelectAxis(datum);
          }
        },
        `${datum.label}: ${datum.value_pct}%. ${datum.detail}`
      );
    });
  }

  class DashboardController {
    constructor(options) {
      this.doc = (options && options.document) || (typeof document !== "undefined" ? document : null);
      this.fetchFn =
        (options && options.fetch) ||
        (typeof fetch === "function" ? fetch.bind(globalScope) : null);
      this.d3 = (options && options.d3) || getD3();
      this.backendUrl = (options && options.backendUrl) || "http://127.0.0.1:8765";
      this.state = {
        loading: false,
        error: null,
        epochMin: "",
        epochMax: "",
        asOf: null,
        overview: null,
        briefCategories: DEFAULT_CATEGORIES.slice(),
        maxInsights: 3,
        briefsHistory: [],
        activeBrief: null,
        flashcardIndex: 0,
        flashcardRevealed: false,
        flashcardAnswer: null,
        flashcardKeyBeliefs: null,
        flashcardFeedback: null,
        selectedInspector: null
      };
      this._eventsWired = false;
      this._resizeObserver = null;
    }

    setBackendUrl(url) {
      if (isLocalLoopbackUrl(url)) {
        this.backendUrl = url.trim().replace(/\/$/, "");
      }
    }

    buildQueryParams(extra) {
      const params = new URLSearchParams();
      if (this.state.epochMin !== "" && this.state.epochMin !== null) {
        params.set("epoch_min", String(this.state.epochMin));
      }
      if (this.state.epochMax !== "" && this.state.epochMax !== null) {
        params.set("epoch_max", String(this.state.epochMax));
      }
      if (this.state.asOf) {
        params.set("as_of", String(this.state.asOf));
      }
      if (extra) {
        Object.keys(extra).forEach((k) => {
          const val = extra[k];
          if (val !== undefined && val !== null && val !== "") {
            params.set(k, String(val));
          }
        });
      }
      const qs = params.toString();
      return qs ? `?${qs}` : "";
    }

    async apiRequest(path, init) {
      if (!this.fetchFn) {
        throw new Error("Fetch API is unavailable in this environment.");
      }
      if (!isLocalLoopbackUrl(this.backendUrl)) {
        throw new Error(`Blocked non-loopback backend URL: ${this.backendUrl}`);
      }
      const url = `${this.backendUrl}${path}`;
      const response = await this.fetchFn(url, {
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json"
        },
        ...init
      });
      if (!response.ok) {
        let message = `Request failed (${response.status})`;
        try {
          const errBody = await response.json();
          if (errBody && (errBody.detail || errBody.message)) {
            message = String(errBody.detail || errBody.message);
          }
        } catch (_ignored) {
          // Use status fallback
        }
        throw new Error(message);
      }
      return response.json();
    }

    wireUI() {
      if (!this.doc || this._eventsWired) {
        return;
      }
      this._eventsWired = true;

      const applyBtn = this.doc.getElementById("dashboard-apply-epoch");
      const resetBtn = this.doc.getElementById("dashboard-reset-epoch");
      const refreshBtn = this.doc.getElementById("dashboard-refresh-btn");
      const retryBtn = this.doc.getElementById("dashboard-error-retry");
      const genBtn = this.doc.getElementById("brief-generate-btn");
      const regenBtn = this.doc.getElementById("brief-regenerate-btn");
      const clearInspectorBtn = this.doc.getElementById("inspector-clear-btn");

      if (applyBtn) {
        applyBtn.addEventListener("click", () => {
          const minEl = this.doc.getElementById("dashboard-epoch-min");
          const maxEl = this.doc.getElementById("dashboard-epoch-max");
          this.state.epochMin = minEl ? minEl.value.trim() : "";
          this.state.epochMax = maxEl ? maxEl.value.trim() : "";
          void this.refresh();
        });
      }

      if (resetBtn) {
        resetBtn.addEventListener("click", () => {
          const minEl = this.doc.getElementById("dashboard-epoch-min");
          const maxEl = this.doc.getElementById("dashboard-epoch-max");
          if (minEl) {
            minEl.value = "";
          }
          if (maxEl) {
            maxEl.value = "";
          }
          this.state.epochMin = "";
          this.state.epochMax = "";
          void this.refresh();
        });
      }

      if (refreshBtn) {
        refreshBtn.addEventListener("click", () => {
          void this.refresh();
        });
      }

      if (retryBtn) {
        retryBtn.addEventListener("click", () => {
          void this.refresh();
        });
      }

      if (genBtn) {
        genBtn.addEventListener("click", () => {
          void this.generateMorningBrief(false);
        });
      }

      if (regenBtn) {
        regenBtn.addEventListener("click", () => {
          void this.generateMorningBrief(true);
        });
      }

      if (clearInspectorBtn) {
        clearInspectorBtn.addEventListener("click", () => {
          this.state.selectedInspector = null;
          this.renderInspector();
        });
      }

      const maxSelect = this.doc.getElementById("brief-max-insights");
      if (maxSelect) {
        maxSelect.addEventListener("change", () => {
          const parsed = parseInt(maxSelect.value, 10);
          this.state.maxInsights = Number.isFinite(parsed) && parsed > 0 ? parsed : 3;
        });
      }

      this.doc.querySelectorAll("[data-brief-category]").forEach((cb) => {
        cb.addEventListener("change", () => {
          const selected = [];
          this.doc.querySelectorAll("[data-brief-category]").forEach((box) => {
            if (box.checked) {
              selected.push(box.dataset.briefCategory);
            }
          });
          this.state.briefCategories = selected.length > 0 ? selected : [cb.dataset.briefCategory];
          if (selected.length === 0) {
            cb.checked = true;
          }
        });
      });

      const panel = this.doc.querySelector('[data-panel="dashboard"]');
      if (panel && typeof globalScope.ResizeObserver === "function") {
        this._resizeObserver = new globalScope.ResizeObserver(() => {
          if (this.state.overview) {
            this.renderChartsOnly();
          }
        });
        this._resizeObserver.observe(panel);
      } else if (globalScope && typeof globalScope.addEventListener === "function") {
        globalScope.addEventListener("resize", () => {
          if (this.state.overview) {
            this.renderChartsOnly();
          }
        });
      }
    }

    setLoading(flag) {
      this.state.loading = Boolean(flag);
      if (!this.doc) {
        return;
      }
      const loadingEl = this.doc.getElementById("dashboard-loading");
      const panelEl = this.doc.querySelector('[data-panel="dashboard"]');
      if (loadingEl) {
        loadingEl.classList.toggle("hidden", !this.state.loading);
      }
      if (panelEl) {
        panelEl.setAttribute("aria-busy", this.state.loading ? "true" : "false");
      }
    }

    setError(message) {
      this.state.error = message || null;
      if (!this.doc) {
        return;
      }
      const errBanner = this.doc.getElementById("dashboard-error");
      const errText = this.doc.getElementById("dashboard-error-text");
      if (errBanner && errText) {
        if (this.state.error) {
          errText.textContent = this.state.error;
          errBanner.classList.remove("hidden");
        } else {
          errText.textContent = "";
          errBanner.classList.add("hidden");
        }
      }
    }

    async refresh() {
      this.wireUI();
      this.setError(null);
      this.setLoading(true);
      try {
        const qs = this.buildQueryParams();
        const [overview, briefsResp] = await Promise.all([
          this.apiRequest(`/api/v1/dashboard/overview${qs}`),
          this.apiRequest("/api/v1/briefs?limit=10")
        ]);
        this.state.overview = overview;
        this.state.briefsHistory = Array.isArray(briefsResp.items) ? briefsResp.items : [];
        if (!this.state.activeBrief && overview.latest_brief) {
          this.state.activeBrief = overview.latest_brief;
        }
        this.state.flashcardIndex = 0;
        this.state.flashcardRevealed = false;
        this.state.flashcardAnswer = null;
        this.state.flashcardKeyBeliefs = null;
        this.render();
      } catch (err) {
        this.setError(err ? err.message : "Failed to load dashboard insights.");
      } finally {
        this.setLoading(false);
      }
    }

    async generateMorningBrief(regenerate) {
      this.setError(null);
      this.setLoading(true);
      try {
        const body = {
          categories: this.state.briefCategories,
          max_insights: this.state.maxInsights,
          regenerate: Boolean(regenerate)
        };
        if (this.state.epochMin !== "") {
          body.epoch_min = parseInt(this.state.epochMin, 10);
        }
        if (this.state.epochMax !== "") {
          body.epoch_max = parseInt(this.state.epochMax, 10);
        }
        const brief = await this.apiRequest("/api/v1/briefs/generate", {
          method: "POST",
          body: JSON.stringify(body)
        });
        this.state.activeBrief = brief;
        const briefsResp = await this.apiRequest("/api/v1/briefs?limit=10");
        this.state.briefsHistory = Array.isArray(briefsResp.items) ? briefsResp.items : [brief];
        this.renderMorningBrief();
      } catch (err) {
        this.setError(err ? err.message : "Failed to generate Morning Brief.");
      } finally {
        this.setLoading(false);
      }
    }

    async revealCurrentFlashcard() {
      const queue = this.state.overview && this.state.overview.review_queue;
      const items = (queue && queue.items) || [];
      const current = items[this.state.flashcardIndex];
      if (!current) {
        return;
      }
      try {
        const qs = this.buildQueryParams({ include_answer: "true", limit: "20" });
        const revealedQueue = await this.apiRequest(`/api/v1/review/queue${qs}`);
        const match = (revealedQueue.items || []).find((x) => x.chunk_id === current.chunk_id);
        if (match && match.flashcard) {
          this.state.flashcardRevealed = true;
          this.state.flashcardAnswer = match.flashcard.answer || "";
          this.state.flashcardKeyBeliefs = match.flashcard.key_beliefs || [];
          this.renderFlashcards();
        }
      } catch (err) {
        this.setError(err ? err.message : "Failed to reveal flashcard answer.");
      }
    }

    async submitFlashcardOutcome(chunkId, outcome) {
      this.setError(null);
      try {
        const result = await this.apiRequest(`/api/v1/review/${encodeURIComponent(chunkId)}`, {
          method: "POST",
          body: JSON.stringify({ outcome, reveal_answer: true })
        });
        this.state.flashcardFeedback =
          `Recorded '${result.outcome}' — stability S=${formatFloat(result.new_stability)}d, ` +
          `next review scheduled for ${result.next_review_at.slice(0, 10)}.`;
        await this.refresh();
      } catch (err) {
        this.setError(err ? err.message : "Failed to submit review outcome.");
      }
    }

    async resolveContradiction(contradictionId, action, isDismiss) {
      this.setError(null);
      const noteEl = this.doc
        ? this.doc.getElementById(`contradiction-note-${contradictionId}`)
        : null;
      const note = noteEl ? noteEl.value.trim() : "";
      const path = isDismiss
        ? `/api/v1/contradictions/${encodeURIComponent(contradictionId)}/dismiss`
        : `/api/v1/contradictions/${encodeURIComponent(contradictionId)}/resolve`;
      const payload = isDismiss
        ? { note: note || null }
        : { resolution: action, note: note || null };
      try {
        await this.apiRequest(path, {
          method: "POST",
          body: JSON.stringify(payload)
        });
        await this.refresh();
      } catch (err) {
        this.setError(err ? err.message : "Failed to update contradiction.");
      }
    }

    inspect(selection) {
      this.state.selectedInspector = selection;
      this.renderInspector();
    }

    render() {
      if (!this.doc || !this.state.overview) {
        return;
      }
      this.renderKpis();
      this.renderChartsOnly();
      this.renderFlashcards();
      this.renderContradictions();
      this.renderMorningBrief();
      this.renderInspector();
    }

    renderChartsOnly() {
      if (!this.doc || !this.state.overview) {
        return;
      }
      const ov = this.state.overview;

      const driftContainer = this.doc.getElementById("chart-belief-drift");
      if (driftContainer) {
        renderBeliefDriftChart(driftContainer, ov.belief_drift, {
          d3: this.d3,
          onSelectEpoch: (epochData) => {
            this.inspect({
              type: "epoch_beliefs",
              title: `Belief Drift — Epoch ${epochData.epoch}`,
              subtitle:
                `${epochData.active_count} active, ${epochData.superseded_count} superseded, ` +
                `${epochData.contradiction_count} contradictions (avg conf ${formatFloat(epochData.average_confidence)})`,
              items: (epochData.beliefs || []).map((b) => ({
                label: `[${b.status}] ${b.belief_text}`,
                meta: `${b.filename || "doc"} (chunk #${b.chunk_index}, conf ${formatFloat(b.confidence)})`
              }))
            });
          }
        });
      }

      const velContainer = this.doc.getElementById("chart-concept-velocity");
      const velTags = this.doc.getElementById("concept-velocity-tags");
      if (velContainer) {
        renderConceptVelocityChart(velContainer, velTags, ov.concept_velocity, {
          d3: this.d3,
          onSelectEpoch: (epochData) => {
            this.inspect({
              type: "epoch_velocity",
              title: `Concept Velocity — Epoch ${epochData.epoch}`,
              subtitle:
                `Velocity ${epochData.velocity_score}, ${epochData.new_chunks} chunk(s), ` +
                `${epochData.emerging_concepts_count} emerging concept(s).`,
              items: (epochData.emerging_concepts || []).map((c) => ({
                label: c,
                meta: `First surfaced in Epoch ${epochData.epoch}`
              }))
            });
          },
          onSelectConcept: (conceptItem) => {
            this.inspect({
              type: "concept",
              title: `Concept: ${conceptItem.concept}`,
              subtitle:
                `${conceptItem.total_mentions} mentions across epochs ` +
                `${conceptItem.first_epoch}..${conceptItem.last_epoch}`,
              items: (conceptItem.sample_chunks || []).map((ch) => ({
                label: `${ch.filename} (chunk #${ch.chunk_index}, epoch ${ch.epoch})`,
                meta: ch.snippet
              }))
            });
          }
        });
      }

      const decayContainer = this.doc.getElementById("chart-decay-heatmap");
      if (decayContainer) {
        renderDecayHeatmap(decayContainer, ov.decay, {
          d3: this.d3,
          onSelectCell: (cell) => {
            this.inspect({
              type: "decay_cell",
              title: `${cell.filename} — Chunk #${cell.chunk_index}`,
              subtitle:
                `Retention R=${formatFloat(cell.retention)} (${cell.decay_band}), ` +
                `Stability S=${formatFloat(cell.stability)}d, Epoch ${cell.epoch}`,
              chunkId: cell.chunk_id,
              items: [
                {
                  label: cell.section_heading || "Chunk Excerpt",
                  meta: cell.snippet || "(empty excerpt)"
                }
              ]
            });
          }
        });
      }

      const healthContainer = this.doc.getElementById("chart-health-radar");
      if (healthContainer) {
        renderHealthRadar(healthContainer, ov.health, {
          d3: this.d3,
          onSelectAxis: (axis) => {
            this.inspect({
              type: "health_axis",
              title: `Health Axis: ${axis.label} (${axis.value_pct}%)`,
              subtitle: axis.detail,
              items: [
                {
                  label: `Overall Vault Health: ${formatPct(ov.health.overall_score)}`,
                  meta: `Status: ${ov.health.status_label}`
                }
              ]
            });
          }
        });
      }
    }

    renderKpis() {
      const kpis = this.state.overview && this.state.overview.kpis;
      if (!kpis || !this.doc) {
        return;
      }
      const mapping = {
        "kpi-docs": kpis.total_documents,
        "kpi-due": kpis.due_flashcards,
        "kpi-contradictions": kpis.open_contradictions,
        "kpi-retention": formatPct(kpis.average_retention),
        "kpi-health": formatPct(kpis.health_score)
      };
      Object.keys(mapping).forEach((id) => {
        const el = this.doc.getElementById(id);
        if (el) {
          el.textContent = String(mapping[id]);
        }
      });
    }

    renderFlashcards() {
      const container = this.doc.getElementById("flashcards-container");
      if (!container) {
        return;
      }
      container.replaceChildren();

      if (this.state.flashcardFeedback) {
        container.appendChild(
          createEl(this.doc, "div", "flashcard-feedback", this.state.flashcardFeedback)
        );
      }

      const queue = this.state.overview && this.state.overview.review_queue;
      const items = (queue && queue.items) || [];
      if (items.length === 0) {
        container.appendChild(
          buildEmptyState(
            this.doc,
            "All caught up",
            "No memory chunks are currently below the review threshold."
          )
        );
        return;
      }

      if (this.state.flashcardIndex >= items.length) {
        this.state.flashcardIndex = 0;
      }
      const current = items[this.state.flashcardIndex];
      const card = createEl(this.doc, "div", "flashcard-box");

      const meta = createEl(
        this.doc,
        "div",
        "flashcard-meta",
        `${current.filename} | Chunk #${current.chunk_index} | Epoch ${current.epoch} | ` +
          `R=${formatFloat(current.retention)} | S=${formatFloat(current.stability)}d`
      );
      card.appendChild(meta);
      card.appendChild(createEl(this.doc, "p", "flashcard-prompt", current.flashcard.prompt));
      if (current.flashcard.hint) {
        card.appendChild(
          createEl(this.doc, "p", "flashcard-hint", `Hint: ${current.flashcard.hint}`)
        );
      }

      if (!this.state.flashcardRevealed) {
        const revealBtn = createEl(this.doc, "button", "btn-primary", "Reveal Answer");
        revealBtn.type = "button";
        revealBtn.setAttribute("data-action", "reveal-flashcard");
        revealBtn.addEventListener("click", () => {
          void this.revealCurrentFlashcard();
        });
        card.appendChild(revealBtn);
      } else {
        const answerBox = createEl(this.doc, "div", "flashcard-answer");
        answerBox.appendChild(createEl(this.doc, "strong", "", "Answer:"));
        answerBox.appendChild(createEl(this.doc, "p", "", this.state.flashcardAnswer || ""));
        card.appendChild(answerBox);

        const actions = createEl(this.doc, "div", "flashcard-outcomes");
        [
          { outcome: "forgot", label: "Forgot (1)" },
          { outcome: "hard", label: "Hard (2)" },
          { outcome: "good", label: "Good (3)" },
          { outcome: "easy", label: "Easy (4)" }
        ].forEach((opt) => {
          const btn = createEl(this.doc, "button", "btn-secondary", opt.label);
          btn.type = "button";
          btn.setAttribute("data-outcome", opt.outcome);
          btn.addEventListener("click", () => {
            void this.submitFlashcardOutcome(current.chunk_id, opt.outcome);
          });
          actions.appendChild(btn);
        });
        card.appendChild(actions);
      }

      container.appendChild(card);
    }

    renderContradictions() {
      const container = this.doc.getElementById("contradictions-container");
      if (!container) {
        return;
      }
      container.replaceChildren();

      const cons = (this.state.overview && this.state.overview.contradictions) || {};
      const items = Array.isArray(cons.items) ? cons.items : [];
      if (items.length === 0) {
        container.appendChild(
          buildEmptyState(
            this.doc,
            "No open contradictions",
            "Belief consistency is clear in the selected epoch window."
          )
        );
        return;
      }

      items.slice(0, 6).forEach((item) => {
        const card = createEl(this.doc, "article", "contradiction-card");
        card.setAttribute("data-contradiction-id", item.id);

        const header = createEl(
          this.doc,
          "div",
          "contradiction-header",
          `Conflict Score: ${formatFloat(item.score)} (Similarity ${formatFloat(item.similarity_score)})`
        );
        card.appendChild(header);

        const pair = createEl(this.doc, "div", "contradiction-pair");
        const ba = item.belief_a || {};
        const bb = item.belief_b || {};

        const boxA = createEl(this.doc, "div", "belief-side");
        boxA.appendChild(
          createEl(
            this.doc,
            "span",
            "belief-epoch-tag",
            `Older (Epoch ${ba.epoch ?? "--"}): ${ba.filename || ""}`
          )
        );
        boxA.appendChild(createEl(this.doc, "p", "", ba.belief_text || item.belief_a_id));

        const boxB = createEl(this.doc, "div", "belief-side");
        boxB.appendChild(
          createEl(
            this.doc,
            "span",
            "belief-epoch-tag",
            `Newer (Epoch ${bb.epoch ?? "--"}): ${bb.filename || ""}`
          )
        );
        boxB.appendChild(createEl(this.doc, "p", "", bb.belief_text || item.belief_b_id));

        pair.appendChild(boxA);
        pair.appendChild(boxB);
        card.appendChild(pair);

        const noteInput = this.doc.createElement("input");
        noteInput.type = "text";
        noteInput.id = `contradiction-note-${item.id}`;
        noteInput.className = "contradiction-note-input";
        noteInput.placeholder = "Optional resolution note or context...";
        noteInput.setAttribute("aria-label", "Contradiction resolution note");
        card.appendChild(noteInput);

        const btnRow = createEl(this.doc, "div", "contradiction-actions");
        [
          { key: "accept_newer", label: "Accept Newer", dismiss: false },
          { key: "accept_older", label: "Accept Older", dismiss: false },
          { key: "keep_both", label: "Keep Both", dismiss: false },
          { key: "context_dependent", label: "Context Dependent", dismiss: false },
          { key: "false_positive", label: "Dismiss", dismiss: true }
        ].forEach((act) => {
          const btn = createEl(this.doc, "button", "btn-secondary", act.label);
          btn.type = "button";
          btn.setAttribute("data-resolution", act.key);
          btn.addEventListener("click", () => {
            void this.resolveContradiction(item.id, act.key, act.dismiss);
          });
          btnRow.appendChild(btn);
        });
        card.appendChild(btnRow);
        container.appendChild(card);
      });
    }

    renderMorningBrief() {
      const container = this.doc.getElementById("morning-brief-container");
      if (!container) {
        return;
      }
      container.replaceChildren();

      const brief = this.state.activeBrief;
      if (!brief) {
        container.appendChild(
          buildEmptyState(
            this.doc,
            "No Morning Brief generated yet",
            "Select categories and click 'Generate Brief' to synthesize up to 3 local insights."
          )
        );
        return;
      }

      const metaBar = createEl(
        this.doc,
        "div",
        "brief-meta",
        `Date: ${brief.brief_date} | Insights: ${brief.insights.length}/${brief.max_insights}` +
          (brief.regenerated ? " (regenerated)" : "")
      );
      container.appendChild(metaBar);

      if (!brief.insights || brief.insights.length === 0) {
        container.appendChild(
          buildEmptyState(
            this.doc,
            "Vault has no active insights for today",
            "Ingest documents or enable additional categories and regenerate."
          )
        );
        return;
      }

      const list = createEl(this.doc, "ol", "brief-insights-list");
      brief.insights.forEach((ins) => {
        const li = createEl(this.doc, "li", "brief-insight-card");
        const top = createEl(this.doc, "div", "brief-insight-header");
        top.appendChild(
          createEl(
            this.doc,
            "span",
            `brief-badge cat-${ins.category}`,
            CATEGORY_LABELS[ins.category] || ins.category
          )
        );
        top.appendChild(createEl(this.doc, "strong", "", ins.title));
        li.appendChild(top);
        li.appendChild(createEl(this.doc, "p", "brief-insight-summary", ins.summary));
        li.appendChild(createEl(this.doc, "p", "brief-insight-action", ins.recommended_action));

        if (Array.isArray(ins.citations) && ins.citations.length > 0) {
          const citWrap = createEl(this.doc, "div", "brief-citations");
          ins.citations.forEach((cit) => {
            const citBtn = createEl(
              this.doc,
              "button",
              "citation-pill",
              `${cit.filename} #${cit.chunk_index} (E${cit.epoch})`
            );
            citBtn.type = "button";
            citBtn.addEventListener("click", () => {
              this.inspect({
                type: "citation",
                title: `Citation: ${cit.filename} (Chunk #${cit.chunk_index})`,
                subtitle: `Epoch ${cit.epoch}`,
                chunkId: cit.chunk_id,
                items: [{ label: "Source Excerpt", meta: cit.snippet }]
              });
            });
            citWrap.appendChild(citBtn);
          });
          li.appendChild(citWrap);
        }
        list.appendChild(li);
      });
      container.appendChild(list);
    }

    renderInspector() {
      const container = this.doc.getElementById("dashboard-inspector-body");
      if (!container) {
        return;
      }
      container.replaceChildren();
      const sel = this.state.selectedInspector;
      if (!sel) {
        container.appendChild(
          createEl(
            this.doc,
            "p",
            "muted",
            "Click any timeline node, velocity bar, decay cell, radar vertex, or brief citation to explore details."
          )
        );
        return;
      }

      container.appendChild(createEl(this.doc, "h3", "inspector-title", sel.title));
      if (sel.subtitle) {
        container.appendChild(createEl(this.doc, "p", "inspector-subtitle", sel.subtitle));
      }
      if (Array.isArray(sel.items) && sel.items.length > 0) {
        const ul = createEl(this.doc, "ul", "inspector-list");
        sel.items.forEach((item) => {
          const li = createEl(this.doc, "li", "");
          li.appendChild(createEl(this.doc, "strong", "", item.label));
          if (item.meta) {
            li.appendChild(createEl(this.doc, "p", "muted", item.meta));
          }
          ul.appendChild(li);
        });
        container.appendChild(ul);
      }
    }
  }

  let singletonController = null;

  function initDashboard(options) {
    if (!singletonController) {
      singletonController = new DashboardController(options || {});
      singletonController.wireUI();
    } else if (options && options.backendUrl) {
      singletonController.setBackendUrl(options.backendUrl);
    }
    return singletonController;
  }

  const exported = {
    DEFAULT_CATEGORIES,
    DashboardController,
    initDashboard,
    renderBeliefDriftChart,
    renderConceptVelocityChart,
    renderDecayHeatmap,
    renderHealthRadar
  };

  if (globalScope) {
    globalScope.MindVaultDashboard = exported;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = exported;
  }
})(typeof window !== "undefined" ? window : globalThis);
