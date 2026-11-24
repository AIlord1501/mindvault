(function (globalScope) {
  "use strict";

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

  function parseSseBlock(rawBlock) {
    const lines = rawBlock.split(/\r?\n/);
    let eventName = "message";
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (dataLines.length === 0) {
      return null;
    }
    const rawData = dataLines.join("\n");
    try {
      return { event: eventName, data: JSON.parse(rawData) };
    } catch (_err) {
      return { event: eventName, data: { raw: rawData } };
    }
  }

  function renderGraphSvg(container, graphData, options) {
    const doc = container.ownerDocument;
    const d3 = (options && options.d3) || getD3();
    const onSelectNode = options && options.onSelectNode;
    container.replaceChildren();

    const nodes = (graphData && Array.isArray(graphData.nodes) ? graphData.nodes : []).slice();
    const edges = (graphData && Array.isArray(graphData.edges) ? graphData.edges : []).slice();

    if (nodes.length === 0 || !d3) {
      const empty = createEl(doc, "div", "chart-empty-state");
      empty.setAttribute("data-state", "empty");
      empty.appendChild(createEl(doc, "strong", "empty-title", "No graph nodes available"));
      empty.appendChild(
        createEl(
          doc,
          "p",
          "empty-desc",
          "Ingest local documents and click Rebuild Graph to visualize entities and orphans."
        )
      );
      container.appendChild(empty);
      return;
    }

    const width = Math.max(460, container.clientWidth || 680);
    const height = 340;
    const cx = width / 2;
    const cy = height / 2;

    const connected = nodes.filter((n) => !n.orphan);
    const orphans = nodes.filter((n) => n.orphan);
    const placed = new Map();

    connected.forEach((node, idx) => {
      const angle = (idx * 2 * Math.PI) / Math.max(1, connected.length);
      const r = Math.min(width, height) * 0.30;
      const pos = {
        ...node,
        _x: cx + Math.cos(angle) * r,
        _y: cy + Math.sin(angle) * r
      };
      placed.set(node.id, pos);
    });

    orphans.forEach((node, idx) => {
      const col = idx % 6;
      const row = Math.floor(idx / 6);
      const pos = {
        ...node,
        _x: 42 + col * 72,
        _y: height - 34 - row * 28
      };
      placed.set(node.id, pos);
    });

    const positionedNodes = nodes.map((n) => placed.get(n.id));
    const positionedEdges = edges
      .map((e) => ({
        ...e,
        _src: placed.get(e.source),
        _dst: placed.get(e.target)
      }))
      .filter((e) => e._src && e._dst);

    const svg = d3
      .select(container)
      .append("svg")
      .attr("class", "mv-chart-svg graph-canvas-svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("role", "img")
      .attr("aria-label", `${graphData.graph_type || "concept"} knowledge graph`);

    svg
      .selectAll("line.graph-edge")
      .data(positionedEdges)
      .join("line")
      .attr("class", "graph-edge")
      .attr("x1", (e) => e._src._x)
      .attr("y1", (e) => e._src._y)
      .attr("x2", (e) => e._dst._x)
      .attr("y2", (e) => e._dst._y);

    const circles = svg
      .selectAll("circle.graph-node")
      .data(positionedNodes)
      .join("circle")
      .attr("class", (n) => (n.orphan ? "graph-node orphan-node" : `graph-node type-${n.node_type}`))
      .attr("cx", (n) => n._x)
      .attr("cy", (n) => n._y)
      .attr("r", (n) => (n.orphan ? 6 : Math.min(14, 7 + (n.degree || 0))))
      .on("click", (_evt, datum) => {
        if (onSelectNode) {
          onSelectNode(datum);
        }
      });

    circles.nodes().forEach((el) => {
      const datum = el.__data__;
      el.setAttribute("tabindex", "0");
      el.setAttribute("role", "button");
      el.setAttribute(
        "aria-label",
        `${datum.node_type} ${datum.label} (epoch ${datum.epoch}, degree ${datum.degree})`
      );
      el.addEventListener("keydown", (evt) => {
        if (evt.key === "Enter" || evt.key === " ") {
          evt.preventDefault();
          if (onSelectNode) {
            onSelectNode(datum);
          }
        }
      });
    });

    svg
      .selectAll("text.graph-node-label")
      .data(positionedNodes.slice(0, 24))
      .join("text")
      .attr("class", "graph-node-label")
      .attr("x", (n) => n._x)
      .attr("y", (n) => n._y - 12)
      .attr("text-anchor", "middle")
      .text((n) => (n.label || n.id).slice(0, 18));
  }

  class WorkspaceController {
    constructor(options) {
      this.doc =
        (options && options.document) || (typeof document !== "undefined" ? document : null);
      this.fetchFn =
        (options && options.fetch) ||
        (typeof fetch === "function" ? fetch.bind(globalScope) : null);
      this.d3 = (options && options.d3) || getD3();
      this.backendUrl = (options && options.backendUrl) || "http://127.0.0.1:8765";
      this.onAfterMutate = (options && options.onAfterMutate) || null;
      this._chatAbortController = null;
      this._eventsWired = false;
      this.state = {
        chatStreaming: false,
        chatCitations: [],
        activeCitationIndex: null,
        graphType: "concepts",
        documents: [],
        selectedDocument: null
      };
    }

    setBackendUrl(url) {
      if (isLocalLoopbackUrl(url)) {
        this.backendUrl = url.trim().replace(/\/$/, "");
      }
    }

    async apiRequest(path, init) {
      if (!this.fetchFn) {
        throw new Error("Fetch API unavailable.");
      }
      if (!isLocalLoopbackUrl(this.backendUrl)) {
        throw new Error(`Blocked non-loopback backend URL: ${this.backendUrl}`);
      }
      const url = `${this.backendUrl}${path}`;
      const isFormData = typeof FormData !== "undefined" && init && init.body instanceof FormData;
      const defaultHeaders = isFormData
        ? { Accept: "application/json" }
        : { Accept: "application/json", "Content-Type": "application/json" };
      const response = await this.fetchFn(url, {
        ...init,
        headers: {
          ...defaultHeaders,
          ...(init && init.headers ? init.headers : {})
        }
      });
      if (!response.ok) {
        let message = `HTTP ${response.status}`;
        try {
          const err = await response.json();
          if (err && (err.detail || err.message)) {
            message = String(err.detail || err.message);
          }
        } catch (_ignored) {
          // Fallback to status message
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

      // Chat controls
      const chatForm = this.doc.getElementById("chat-query-form");
      const chatCancel = this.doc.getElementById("chat-cancel-btn");
      const chatMode = this.doc.getElementById("chat-mode-select");
      const socraticToggle = this.doc.getElementById("chat-socratic-toggle");

      if (chatMode && socraticToggle) {
        chatMode.addEventListener("change", () => {
          socraticToggle.checked = chatMode.value === "socratic";
        });
        socraticToggle.addEventListener("change", () => {
          if (socraticToggle.checked) {
            chatMode.value = "socratic";
          } else if (chatMode.value === "socratic") {
            chatMode.value = "summary";
          }
        });
      }

      if (chatForm) {
        chatForm.addEventListener("submit", (evt) => {
          evt.preventDefault();
          void this.submitChat();
        });
      }
      if (chatCancel) {
        chatCancel.addEventListener("click", () => {
          this.cancelChat();
        });
      }

      // Graph controls
      const graphRefresh = this.doc.getElementById("graph-refresh-btn");
      const graphRebuild = this.doc.getElementById("graph-rebuild-btn");
      if (graphRefresh) {
        graphRefresh.addEventListener("click", () => {
          void this.loadGraph();
        });
      }
      if (graphRebuild) {
        graphRebuild.addEventListener("click", () => {
          void this.rebuildGraph();
        });
      }

      // Research / Ingestion controls
      const dropzone = this.doc.getElementById("ingest-dropzone");
      const fileInput = this.doc.getElementById("ingest-file-input");
      const browseBtn = this.doc.getElementById("ingest-browse-btn");
      const noteBtn = this.doc.getElementById("ingest-submit-note-btn");
      const searchBtn = this.doc.getElementById("research-search-btn");
      const docsRefresh = this.doc.getElementById("research-refresh-btn");

      if (browseBtn && fileInput) {
        browseBtn.addEventListener("click", () => {
          if (typeof fileInput.click === "function") {
            fileInput.click();
          }
        });
      }
      if (fileInput) {
        fileInput.addEventListener("change", () => {
          if (fileInput.files && fileInput.files.length > 0) {
            void this.ingestFiles(fileInput.files);
          }
        });
      }
      if (dropzone) {
        dropzone.addEventListener("dragover", (evt) => {
          evt.preventDefault();
          dropzone.classList.add("drag-over");
        });
        dropzone.addEventListener("dragleave", () => {
          dropzone.classList.remove("drag-over");
        });
        dropzone.addEventListener("drop", (evt) => {
          evt.preventDefault();
          dropzone.classList.remove("drag-over");
          const files = evt.dataTransfer && evt.dataTransfer.files;
          if (files && files.length > 0) {
            void this.ingestFiles(files);
          }
        });
      }
      if (noteBtn) {
        noteBtn.addEventListener("click", () => {
          void this.ingestTextNote();
        });
      }
      if (searchBtn) {
        searchBtn.addEventListener("click", () => {
          void this.runHybridSearch();
        });
      }
      if (docsRefresh) {
        docsRefresh.addEventListener("click", () => {
          void this.loadDocuments();
        });
      }

      // Morning brief overlay
      const openOverlayBtn = this.doc.getElementById("dashboard-brief-overlay-btn");
      const overlay = this.doc.getElementById("morning-brief-overlay");
      const closeOverlayBtn = this.doc.getElementById("brief-overlay-close");
      const regenOverlayBtn = this.doc.getElementById("brief-overlay-regenerate");

      if (openOverlayBtn) {
        openOverlayBtn.addEventListener("click", () => {
          void this.openMorningBriefOverlay(false);
        });
      }
      if (closeOverlayBtn) {
        closeOverlayBtn.addEventListener("click", () => {
          this.closeMorningBriefOverlay();
        });
      }
      if (regenOverlayBtn) {
        regenOverlayBtn.addEventListener("click", () => {
          void this.openMorningBriefOverlay(true);
        });
      }
      if (overlay) {
        overlay.addEventListener("click", (evt) => {
          if (evt.target === overlay) {
            this.closeMorningBriefOverlay();
          }
        });
      }

      // Settings diagnostics
      const rebuildIndexBtn = this.doc.getElementById("settings-rebuild-index-btn");
      const refreshSettingsBtn = this.doc.getElementById("settings-refresh-diagnostics");
      if (rebuildIndexBtn) {
        rebuildIndexBtn.addEventListener("click", () => {
          void this.rebuildIndex();
        });
      }
      if (refreshSettingsBtn) {
        refreshSettingsBtn.addEventListener("click", () => {
          void this.loadSystemDiagnostics();
        });
      }
    }

    cancelChat() {
      if (this._chatAbortController) {
        this._chatAbortController.abort();
        this._chatAbortController = null;
      }
      this.state.chatStreaming = false;
      const cancelBtn = this.doc && this.doc.getElementById("chat-cancel-btn");
      if (cancelBtn) {
        cancelBtn.disabled = true;
      }
    }

    async submitChat() {
      this.wireUI();
      if (!this.doc) {
        return;
      }
      const inputEl = this.doc.getElementById("chat-query-input");
      const modeEl = this.doc.getElementById("chat-mode-select");
      const socraticEl = this.doc.getElementById("chat-socratic-toggle");
      const minEl = this.doc.getElementById("chat-epoch-min");
      const maxEl = this.doc.getElementById("chat-epoch-max");
      const messagesEl = this.doc.getElementById("chat-messages");
      const cancelBtn = this.doc.getElementById("chat-cancel-btn");

      const query = inputEl ? inputEl.value.trim() : "";
      if (!query) {
        return;
      }

      const selectedMode = modeEl ? modeEl.value : "summary";
      const isSocratic =
        selectedMode === "socratic" || Boolean(socraticEl && socraticEl.checked);
      const answerMode = selectedMode === "socratic" ? "summary" : selectedMode;
      const payload = {
        query,
        answer_mode: answerMode,
        socratic: isSocratic,
        top_k: 6
      };
      if (minEl && minEl.value.trim() !== "") {
        payload.epoch_min = parseInt(minEl.value.trim(), 10);
      }
      if (maxEl && maxEl.value.trim() !== "") {
        payload.epoch_max = parseInt(maxEl.value.trim(), 10);
      }

      if (messagesEl) {
        const emptyHint = messagesEl.querySelector('[data-state="empty"]');
        if (emptyHint) {
          emptyHint.remove();
        }
        const userCard = createEl(this.doc, "article", "chat-msg chat-msg-user");
        const modeBadge = isSocratic ? "SOCRATIC" : answerMode.toUpperCase();
        userCard.appendChild(createEl(this.doc, "div", "chat-msg-role", `You (${modeBadge})`));
        userCard.appendChild(createEl(this.doc, "p", "chat-msg-text", query));
        messagesEl.appendChild(userCard);
      }

      const assistantCard = messagesEl
        ? createEl(this.doc, "article", "chat-msg chat-msg-assistant")
        : null;
      const disclosureEl = this.doc.createElement("details");
      disclosureEl.className = "progressive-disclosure";
      disclosureEl.open = true;
      const summaryEl = createEl(this.doc, "summary", "", "Reasoning & Retrieval Steps");
      const stepsList = createEl(this.doc, "ul", "disclosure-steps");
      disclosureEl.appendChild(summaryEl);
      disclosureEl.appendChild(stepsList);

      const answerEl = createEl(this.doc, "div", "chat-answer-text", "");
      const citationsRow = createEl(this.doc, "div", "chat-citations-row");

      if (assistantCard && messagesEl) {
        const modeBadge = isSocratic ? "SOCRATIC" : answerMode.toUpperCase();
        assistantCard.appendChild(
          createEl(this.doc, "div", "chat-msg-role", `MindVault (${modeBadge})`)
        );
        assistantCard.appendChild(disclosureEl);
        assistantCard.appendChild(answerEl);
        assistantCard.appendChild(citationsRow);
        messagesEl.appendChild(assistantCard);
      }

      if (inputEl) {
        inputEl.value = "";
      }

      this.state.chatStreaming = true;
      this.state.chatCitations = [];
      if (cancelBtn) {
        cancelBtn.disabled = false;
      }
      if (typeof AbortController === "function") {
        this._chatAbortController = new AbortController();
      }

      const handleEvent = (eventType, data) => {
        if (eventType === "status" && data && (data.message || data.stage)) {
          stepsList.appendChild(
            createEl(this.doc, "li", "step-status", data.message || `Stage: ${data.stage}`)
          );
        } else if (eventType === "subquestion" && data && data.question) {
          stepsList.appendChild(
            createEl(this.doc, "li", "step-subquestion", `Subquestion: ${data.question}`)
          );
        } else if (eventType === "citation" && data) {
          this.state.chatCitations.push(data);
          this.appendCitationChip(citationsRow, data);
        } else if (eventType === "answer_section" && data) {
          stepsList.appendChild(
            createEl(
              this.doc,
              "li",
              "step-section",
              `${data.title || "Section"}: ${data.content || ""}`
            )
          );
        } else if (eventType === "token" && data && data.text) {
          answerEl.textContent += data.text;
        } else if (eventType === "error" && data) {
          const errMsg = data.message || data.detail || "Local model unavailable.";
          answerEl.appendChild(createEl(this.doc, "p", "chat-error-note", errMsg));
        }
      };

      try {
        if (!this.fetchFn || !isLocalLoopbackUrl(this.backendUrl)) {
          throw new Error("Local loopback fetch is unavailable.");
        }
        const resp = await this.fetchFn(`${this.backendUrl}/api/v1/chat/stream`, {
          method: "POST",
          headers: {
            Accept: "text/event-stream",
            "Content-Type": "application/json"
          },
          body: JSON.stringify(payload),
          signal: this._chatAbortController ? this._chatAbortController.signal : undefined
        });

        if (!resp.ok) {
          let errDetail = `Chat failed (HTTP ${resp.status})`;
          try {
            const j = await resp.json();
            if (j && (j.detail || j.message)) {
              errDetail = String(j.detail || j.message);
            }
          } catch (_ignored) {
            // Keep HTTP status
          }
          throw new Error(errDetail);
        }

        if (resp.body && typeof resp.body.getReader === "function") {
          const reader = resp.body.getReader();
          const decoder = new TextDecoder("utf-8");
          let buffer = "";
          while (true) {
            const { value, done } = await reader.read();
            if (done) {
              break;
            }
            buffer += decoder.decode(value, { stream: true });
            const blocks = buffer.split(/\r?\n\r?\n/);
            buffer = blocks.pop() || "";
            for (const block of blocks) {
              const parsed = parseSseBlock(block);
              if (parsed) {
                handleEvent(parsed.event, parsed.data);
              }
            }
          }
          if (buffer.trim()) {
            const parsed = parseSseBlock(buffer.trim());
            if (parsed) {
              handleEvent(parsed.event, parsed.data);
            }
          }
        } else if (typeof resp.text === "function") {
          const raw = await resp.text();
          for (const block of raw.split(/\r?\n\r?\n/)) {
            const parsed = parseSseBlock(block);
            if (parsed) {
              handleEvent(parsed.event, parsed.data);
            }
          }
        }
      } catch (err) {
        if (err && err.name === "AbortError") {
          stepsList.appendChild(createEl(this.doc, "li", "step-cancelled", "Generation cancelled."));
        } else {
          answerEl.appendChild(
            createEl(
              this.doc,
              "p",
              "chat-error-note",
              err ? err.message : "Failed to query local chat engine."
            )
          );
        }
      } finally {
        this.state.chatStreaming = false;
        this._chatAbortController = null;
        if (cancelBtn) {
          cancelBtn.disabled = true;
        }
      }
    }

    appendCitationChip(container, citation) {
      if (!this.doc || !container) {
        return;
      }
      const badge = citation.label || citation.index || "*";
      const label = `[${badge}] ${citation.filename || "doc"} #${citation.chunk_index ?? 0}`;
      const btn = createEl(this.doc, "button", "citation-pill", label);
      btn.type = "button";
      btn.setAttribute("aria-label", `Inspect citation ${label}`);
      btn.addEventListener("click", () => {
        this.showCitationDetails(citation);
      });
      container.appendChild(btn);
    }

    showCitationDetails(citation) {
      const viewer = this.doc && this.doc.getElementById("chat-citation-viewer");
      if (!viewer) {
        return;
      }
      viewer.replaceChildren();
      viewer.appendChild(
        createEl(
          this.doc,
          "h3",
          "inspector-title",
          `${citation.filename || "Document"} (Chunk #${citation.chunk_index ?? 0})`
        )
      );
      viewer.appendChild(
        createEl(
          this.doc,
          "p",
          "inspector-subtitle",
          `Epoch ${citation.epoch ?? "--"} | Score ${Number(citation.score || 0).toFixed(3)}`
        )
      );
      viewer.appendChild(
        createEl(this.doc, "p", "citation-excerpt", citation.excerpt || citation.content || "")
      );
    }

    async loadGraph() {
      this.wireUI();
      if (!this.doc) {
        return;
      }
      const typeSel = this.doc.getElementById("graph-type-select");
      const minEl = this.doc.getElementById("graph-epoch-min");
      const maxEl = this.doc.getElementById("graph-epoch-max");
      const searchEl = this.doc.getElementById("graph-search-input");
      const host = this.doc.getElementById("chart-knowledge-graph");
      const statsEl = this.doc.getElementById("graph-stats-bar");

      const graphType = typeSel ? typeSel.value : "concepts";
      const endpoint = graphType === "citations" ? "/api/v1/graph/citations" : "/api/v1/graph/concepts";
      const params = new URLSearchParams();
      if (minEl && minEl.value.trim()) {
        params.set("epoch_min", minEl.value.trim());
      }
      if (maxEl && maxEl.value.trim()) {
        params.set("epoch_max", maxEl.value.trim());
      }
      if (searchEl && searchEl.value.trim()) {
        params.set("search", searchEl.value.trim());
      }
      const qs = params.toString() ? `?${params.toString()}` : "";

      try {
        const data = await this.apiRequest(`${endpoint}${qs}`);
        if (statsEl && data.stats) {
          statsEl.textContent =
            `Nodes: ${data.stats.node_count} | Edges: ${data.stats.edge_count} | ` +
            `Orphans: ${data.stats.orphan_count}`;
        }
        if (host) {
          renderGraphSvg(host, data, {
            d3: this.d3,
            onSelectNode: (node) => {
              void this.inspectGraphNode(node.id, graphType);
            }
          });
        }
      } catch (err) {
        if (host) {
          host.replaceChildren(
            createEl(
              this.doc,
              "div",
              "dashboard-error",
              err ? err.message : "Failed to load knowledge graph."
            )
          );
        }
      }
    }

    async inspectGraphNode(nodeId, graphType) {
      const box = this.doc && this.doc.getElementById("graph-inspector-body");
      if (!box) {
        return;
      }
      try {
        const sub = await this.apiRequest(
          `/api/v1/graph/node/${encodeURIComponent(nodeId)}?graph_type=${encodeURIComponent(graphType)}`
        );
        box.replaceChildren();
        const center = (sub.nodes || []).find((n) => n.id === nodeId) || (sub.nodes || [])[0];
        if (!center) {
          box.appendChild(createEl(this.doc, "p", "muted", "Node not found."));
          return;
        }
        box.appendChild(
          createEl(this.doc, "h3", "inspector-title", `[${center.node_type}] ${center.label}`)
        );
        box.appendChild(
          createEl(
            this.doc,
            "p",
            "inspector-subtitle",
            `Epoch ${center.epoch} | Degree ${center.degree} | Connected neighbors: ${Math.max(0, (sub.nodes || []).length - 1)}`
          )
        );
      } catch (err) {
        box.replaceChildren(
          createEl(this.doc, "p", "muted", err ? err.message : "Node inspection failed.")
        );
      }
    }

    async rebuildGraph() {
      try {
        await this.apiRequest("/api/v1/graph/rebuild", { method: "POST", body: "{}" });
        await this.loadGraph();
      } catch (err) {
        const statsEl = this.doc && this.doc.getElementById("graph-stats-bar");
        if (statsEl) {
          statsEl.textContent = err ? err.message : "Graph rebuild failed.";
        }
      }
    }

    async ingestFiles(fileList) {
      const statusEl = this.doc && this.doc.getElementById("ingest-status");
      const epochEl = this.doc && this.doc.getElementById("ingest-epoch-input");
      const epochVal = epochEl && epochEl.value.trim() ? epochEl.value.trim() : null;
      const files = Array.from(fileList || []);
      if (files.length === 0) {
        return;
      }
      if (statusEl) {
        statusEl.textContent = `Ingesting ${files.length} local file(s)...`;
      }
      let indexedCount = 0;
      let lastError = null;
      for (const file of files) {
        try {
          const form = new FormData();
          form.append("file", file);
          if (epochVal) {
            form.append("epoch", epochVal);
          }
          await this.apiRequest("/api/v1/ingest/file", { method: "POST", body: form });
          indexedCount += 1;
        } catch (err) {
          lastError = err ? err.message : "Ingestion error";
        }
      }
      if (statusEl) {
        statusEl.textContent = lastError
          ? `Indexed ${indexedCount}/${files.length} file(s). Last error: ${lastError}`
          : `Successfully processed ${indexedCount} local file(s).`;
      }
      await this.loadDocuments();
      if (this.onAfterMutate) {
        this.onAfterMutate();
      }
    }

    async ingestTextNote() {
      if (!this.doc) {
        return;
      }
      const titleEl = this.doc.getElementById("ingest-note-title");
      const contentEl = this.doc.getElementById("ingest-note-content");
      const epochEl = this.doc.getElementById("ingest-epoch-input");
      const statusEl = this.doc.getElementById("ingest-status");

      const rawTitle = titleEl && titleEl.value.trim() ? titleEl.value.trim() : "quick_note.md";
      const filename = rawTitle.endsWith(".md") || rawTitle.endsWith(".txt") ? rawTitle : `${rawTitle}.md`;
      const content = contentEl ? contentEl.value : "";
      if (!content.trim()) {
        if (statusEl) {
          statusEl.textContent = "Note content must not be empty.";
        }
        return;
      }

      try {
        const blob = new Blob([content], { type: "text/markdown" });
        const form = new FormData();
        form.append("file", blob, filename);
        if (epochEl && epochEl.value.trim()) {
          form.append("epoch", epochEl.value.trim());
        }
        const res = await this.apiRequest("/api/v1/ingest/file", { method: "POST", body: form });
        if (titleEl) {
          titleEl.value = "";
        }
        if (contentEl) {
          contentEl.value = "";
        }
        if (statusEl) {
          statusEl.textContent = `Ingested '${res.filename}' (${res.chunk_count} chunk(s), status: ${res.status}).`;
        }
        await this.loadDocuments();
      } catch (err) {
        if (statusEl) {
          statusEl.textContent = err ? err.message : "Failed to ingest note.";
        }
      }
    }

    async loadDocuments() {
      this.wireUI();
      const listEl = this.doc && this.doc.getElementById("research-documents-list");
      if (!listEl) {
        return;
      }
      try {
        const resp = await this.apiRequest("/api/v1/documents?limit=50");
        const docs = Array.isArray(resp.documents) ? resp.documents : [];
        this.state.documents = docs;
        listEl.replaceChildren();
        if (docs.length === 0) {
          listEl.appendChild(
            createEl(this.doc, "p", "muted", "No local documents indexed yet. Drop files above to start.")
          );
          return;
        }
        docs.forEach((d) => {
          const item = createEl(this.doc, "div", "research-doc-row");
          item.appendChild(
            createEl(
              this.doc,
              "span",
              "research-doc-name",
              `${d.filename} (${d.file_type}, ${d.chunk_count} chunks)`
            )
          );
          const delBtn = createEl(this.doc, "button", "btn-secondary", "Delete");
          delBtn.type = "button";
          delBtn.addEventListener("click", () => {
            void this.deleteDocument(d.id);
          });
          item.appendChild(delBtn);
          listEl.appendChild(item);
        });
      } catch (err) {
        listEl.replaceChildren(
          createEl(this.doc, "p", "muted", err ? err.message : "Failed to load documents.")
        );
      }
    }

    async deleteDocument(docId) {
      try {
        await this.apiRequest(`/api/v1/documents/${encodeURIComponent(docId)}`, {
          method: "DELETE"
        });
        await this.loadDocuments();
      } catch (_ignored) {
        // Reload view silently
      }
    }

    async runHybridSearch() {
      if (!this.doc) {
        return;
      }
      const queryEl = this.doc.getElementById("research-search-input");
      const resultsEl = this.doc.getElementById("research-search-results");
      const query = queryEl ? queryEl.value.trim() : "";
      if (!query || !resultsEl) {
        return;
      }
      resultsEl.replaceChildren(createEl(this.doc, "p", "muted", "Searching local hybrid index..."));
      try {
        const res = await this.apiRequest("/api/v1/search", {
          method: "POST",
          body: JSON.stringify({ query, mode: "hybrid", top_k: 8 })
        });
        resultsEl.replaceChildren();
        const hits = Array.isArray(res.results) ? res.results : [];
        if (hits.length === 0) {
          resultsEl.appendChild(createEl(this.doc, "p", "muted", "No matching local chunks found."));
          return;
        }
        hits.forEach((hit) => {
          const row = createEl(this.doc, "div", "search-hit-card");
          row.appendChild(
            createEl(
              this.doc,
              "strong",
              "",
              `${hit.filename} #${hit.chunk_index} (Epoch ${hit.epoch}, score ${Number(hit.score).toFixed(3)})`
            )
          );
          row.appendChild(createEl(this.doc, "p", "muted", hit.content.slice(0, 220)));
          resultsEl.appendChild(row);
        });
      } catch (err) {
        resultsEl.replaceChildren(
          createEl(this.doc, "p", "muted", err ? err.message : "Search failed.")
        );
      }
    }

    async openMorningBriefOverlay(regenerate) {
      this.wireUI();
      if (!this.doc) {
        return;
      }
      const overlay = this.doc.getElementById("morning-brief-overlay");
      const bodyEl = this.doc.getElementById("brief-overlay-content");
      if (overlay) {
        overlay.classList.remove("hidden");
      }
      if (bodyEl) {
        bodyEl.replaceChildren(createEl(this.doc, "p", "muted", "Loading daily local brief..."));
      }
      try {
        const brief = await this.apiRequest("/api/v1/briefs/generate", {
          method: "POST",
          body: JSON.stringify({ max_insights: 3, regenerate: Boolean(regenerate) })
        });
        if (bodyEl) {
          bodyEl.replaceChildren();
          bodyEl.appendChild(
            createEl(
              this.doc,
              "div",
              "brief-meta",
              `Morning Brief ${brief.brief_date} (${(brief.insights || []).length} insight(s))`
            )
          );
          (brief.insights || []).forEach((ins) => {
            const card = createEl(this.doc, "div", "brief-insight-card");
            card.appendChild(createEl(this.doc, "strong", "", `[${ins.category}] ${ins.title}`));
            card.appendChild(createEl(this.doc, "p", "brief-insight-summary", ins.summary));
            card.appendChild(
              createEl(this.doc, "p", "brief-insight-action", ins.recommended_action)
            );
            bodyEl.appendChild(card);
          });
        }
      } catch (err) {
        if (bodyEl) {
          bodyEl.replaceChildren(
            createEl(this.doc, "p", "muted", err ? err.message : "Could not generate brief.")
          );
        }
      }
    }

    closeMorningBriefOverlay() {
      const overlay = this.doc && this.doc.getElementById("morning-brief-overlay");
      if (overlay) {
        overlay.classList.add("hidden");
      }
    }

    async loadSystemDiagnostics() {
      this.wireUI();
      if (!this.doc) {
        return;
      }
      try {
        const status = await this.apiRequest("/api/v1/system/status?include_paths=true");
        const modelEl = this.doc.getElementById("settings-model-status");
        const indexEl = this.doc.getElementById("settings-index-status");
        const jobsEl = this.doc.getElementById("settings-jobs-list");
        const offlineBadge = this.doc.getElementById("status-offline-badge");

        if (offlineBadge) {
          offlineBadge.textContent = status.offline_only ? "Offline-only" : "Local";
        }

        if (modelEl && status.models) {
          modelEl.replaceChildren();
          const emb = status.models.embedding || {};
          const llm = status.models.llm || {};
          const nli = status.models.nli || {};
          [
            `Embeddings: ${emb.available ? "Ready" : `Unavailable (${emb.reason || "unconfigured"})`}`,
            `Local LLM (${llm.provider || "llamacpp"}): ${llm.available ? `Ready (${llm.model || ""})` : `Unavailable (${llm.reason || "unconfigured"})`}`,
            `NLI Classifier: ${nli.available ? "Ready" : `Fallback (${nli.reason || "lexical"})`}`
          ].forEach((line) => {
            modelEl.appendChild(createEl(this.doc, "div", "diag-row", line));
          });
        }

        if (indexEl) {
          indexEl.replaceChildren();
          const idx = status.index || {};
          const rec = status.recovery || {};
          [
            `Documents: ${idx.sqlite_indexed_documents ?? 0} | Chunks: ${idx.sqlite_indexed_chunks ?? 0}`,
            `BM25: ${idx.bm25_indexed_chunks ?? 0} | Vector: ${idx.vector_indexed_chunks ?? 0}`,
            `Synchronized: ${idx.synchronized ? "Yes" : "No"}`,
            `SQLite Recovery: ${rec.database && rec.database.recovered ? "Recovered corrupt file" : "Clean"}`
          ].forEach((line) => {
            indexEl.appendChild(createEl(this.doc, "div", "diag-row", line));
          });
        }

        if (jobsEl && status.scheduler) {
          jobsEl.replaceChildren();
          const jobs = Array.isArray(status.scheduler.jobs) ? status.scheduler.jobs : [];
          jobs.forEach((job) => {
            const row = createEl(this.doc, "div", "job-status-row");
            row.appendChild(
              createEl(
                this.doc,
                "span",
                "",
                `${job.name} [${job.state}] (runs: ${job.run_count}, every ${job.interval_seconds}s)`
              )
            );
            const runBtn = createEl(this.doc, "button", "btn-secondary", "Run Now");
            runBtn.type = "button";
            runBtn.setAttribute("data-run-job", job.job_id);
            runBtn.addEventListener("click", () => {
              void this.runJob(job.job_id);
            });
            row.appendChild(runBtn);
            jobsEl.appendChild(row);
          });
        }
      } catch (_ignored) {
        // Keep Settings diagnostics responsive even if backend is offline
      }
    }

    async rebuildIndex() {
      try {
        await this.apiRequest("/api/v1/index/rebuild", { method: "POST", body: "{}" });
        await this.loadSystemDiagnostics();
      } catch (_ignored) {
        // Ignore gracefully
      }
    }

    async runJob(jobId) {
      try {
        await this.apiRequest(`/api/v1/jobs/${encodeURIComponent(jobId)}/run`, {
          method: "POST",
          body: "{}"
        });
        await this.loadSystemDiagnostics();
      } catch (_ignored) {
        // Ignore gracefully
      }
    }
  }

  let singleton = null;

  function initWorkspace(options) {
    if (!singleton) {
      singleton = new WorkspaceController(options || {});
      singleton.wireUI();
    } else if (options && options.backendUrl) {
      singleton.setBackendUrl(options.backendUrl);
    }
    return singleton;
  }

  const exported = {
    WorkspaceController,
    initWorkspace,
    renderGraphSvg,
    parseSseBlock
  };

  if (globalScope) {
    globalScope.MindVaultWorkspace = exported;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = exported;
  }
})(typeof window !== "undefined" ? window : globalThis);
