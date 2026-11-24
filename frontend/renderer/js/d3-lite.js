(function (globalScope) {
  "use strict";

  const SVG_TAGS = new Set([
    "svg",
    "g",
    "path",
    "rect",
    "circle",
    "line",
    "polyline",
    "polygon",
    "text",
    "tspan",
    "title"
  ]);
  const SVG_NS = "http://www.w3.org/2000/svg";

  function createElementForTag(doc, tag) {
    if (SVG_TAGS.has(tag)) {
      return doc.createElementNS(SVG_NS, tag);
    }
    return doc.createElement(tag);
  }

  function resolveValue(valOrFn, datum, index) {
    return typeof valOrFn === "function" ? valOrFn(datum, index) : valOrFn;
  }

  class Selection {
    constructor(nodes, parentNode = null) {
      this._nodes = Array.isArray(nodes) ? nodes.filter(Boolean) : [nodes].filter(Boolean);
      this._parentNode = parentNode || null;
    }

    node() {
      return this._nodes[0] || null;
    }

    nodes() {
      return this._nodes.slice();
    }

    selectAll(selector) {
      const parent = this.node() || this._parentNode;
      if (!parent || typeof parent.querySelectorAll !== "function") {
        return new Selection([], parent || null);
      }
      const matched = Array.from(parent.querySelectorAll(selector));
      return new Selection(matched, parent);
    }

    append(tag) {
      const created = [];
      for (const parent of this._nodes) {
        const doc = parent.ownerDocument || (typeof document !== "undefined" ? document : null);
        if (!doc) {
          continue;
        }
        const child = createElementForTag(doc, tag);
        if ("__data__" in parent) {
          child.__data__ = parent.__data__;
        }
        parent.appendChild(child);
        created.push(child);
      }
      return new Selection(created);
    }

    attr(name, valOrFn) {
      this._nodes.forEach((el, idx) => {
        const value = resolveValue(valOrFn, el.__data__, idx);
        if (value === null || value === undefined) {
          el.removeAttribute(name);
        } else {
          el.setAttribute(name, String(value));
        }
      });
      return this;
    }

    style(name, valOrFn) {
      this._nodes.forEach((el, idx) => {
        const value = resolveValue(valOrFn, el.__data__, idx);
        if (el.style) {
          el.style[name] = value === null || value === undefined ? "" : String(value);
        }
      });
      return this;
    }

    text(valOrFn) {
      this._nodes.forEach((el, idx) => {
        const value = resolveValue(valOrFn, el.__data__, idx);
        el.textContent = value === null || value === undefined ? "" : String(value);
      });
      return this;
    }

    classed(classNames, flagOrFn) {
      const names = String(classNames || "").trim().split(/\s+/).filter(Boolean);
      this._nodes.forEach((el, idx) => {
        const enabled = Boolean(resolveValue(flagOrFn, el.__data__, idx));
        if (el.classList) {
          names.forEach((c) => el.classList.toggle(c, enabled));
        }
      });
      return this;
    }

    on(eventType, handler) {
      this._nodes.forEach((el) => {
        el.addEventListener(eventType, (evt) => {
          handler(evt, el.__data__);
        });
      });
      return this;
    }

    remove() {
      this._nodes.forEach((el) => {
        if (el.parentNode) {
          el.parentNode.removeChild(el);
        }
      });
      return this;
    }

    data(items) {
      const parent = this._parentNode || this.node();
      const arr = Array.isArray(items) ? items : [];
      return new DataJoin(parent, this._nodes, arr);
    }
  }

  class DataJoin {
    constructor(parent, existingNodes, dataItems) {
      this._parent = parent;
      this._existing = existingNodes;
      this._data = dataItems;
    }

    join(tag) {
      this._existing.forEach((el) => {
        if (el.parentNode) {
          el.parentNode.removeChild(el);
        }
      });
      if (!this._parent) {
        return new Selection([]);
      }
      const doc =
        this._parent.ownerDocument || (typeof document !== "undefined" ? document : null);
      if (!doc) {
        return new Selection([]);
      }
      const created = this._data.map((datum) => {
        const el = createElementForTag(doc, tag);
        el.__data__ = datum;
        this._parent.appendChild(el);
        return el;
      });
      return new Selection(created);
    }
  }

  function select(target) {
    if (typeof target === "string") {
      if (typeof document === "undefined") {
        return new Selection([]);
      }
      return new Selection([document.querySelector(target)]);
    }
    return new Selection([target]);
  }

  function min(arr, accessor) {
    if (!Array.isArray(arr) || arr.length === 0) {
      return undefined;
    }
    let best = Infinity;
    for (let i = 0; i < arr.length; i += 1) {
      const v = Number(accessor ? accessor(arr[i], i) : arr[i]);
      if (Number.isFinite(v) && v < best) {
        best = v;
      }
    }
    return best === Infinity ? undefined : best;
  }

  function max(arr, accessor) {
    if (!Array.isArray(arr) || arr.length === 0) {
      return undefined;
    }
    let best = -Infinity;
    for (let i = 0; i < arr.length; i += 1) {
      const v = Number(accessor ? accessor(arr[i], i) : arr[i]);
      if (Number.isFinite(v) && v > best) {
        best = v;
      }
    }
    return best === -Infinity ? undefined : best;
  }

  function extent(arr, accessor) {
    return [min(arr, accessor), max(arr, accessor)];
  }

  function mean(arr, accessor) {
    if (!Array.isArray(arr) || arr.length === 0) {
      return undefined;
    }
    let sum = 0;
    let count = 0;
    for (let i = 0; i < arr.length; i += 1) {
      const v = Number(accessor ? accessor(arr[i], i) : arr[i]);
      if (Number.isFinite(v)) {
        sum += v;
        count += 1;
      }
    }
    return count > 0 ? sum / count : undefined;
  }

  function scaleLinear() {
    let dom = [0, 1];
    let rng = [0, 1];
    let clamped = false;

    function scale(val) {
      const d0 = Number(dom[0]);
      const d1 = Number(dom[1]);
      const r0 = Number(rng[0]);
      const r1 = Number(rng[1]);
      const span = d1 - d0;
      let t = span === 0 ? 0.5 : (Number(val) - d0) / span;
      if (clamped) {
        t = Math.max(0, Math.min(1, t));
      }
      return r0 + t * (r1 - r0);
    }

    scale.domain = function (d) {
      if (!arguments.length) {
        return dom.slice();
      }
      dom = [Number(d[0]), Number(d[1])];
      if (dom[0] === dom[1]) {
        dom[1] = dom[0] + 1;
      }
      return scale;
    };

    scale.range = function (r) {
      if (!arguments.length) {
        return rng.slice();
      }
      rng = [Number(r[0]), Number(r[1])];
      return scale;
    };

    scale.clamp = function (flag) {
      clamped = Boolean(flag);
      return scale;
    };

    scale.nice = function () {
      return scale;
    };

    scale.ticks = function (count) {
      const n = Math.max(2, Number(count) || 5);
      const d0 = dom[0];
      const d1 = dom[1];
      const out = [];
      for (let i = 0; i < n; i += 1) {
        out.push(d0 + (i / (n - 1)) * (d1 - d0));
      }
      return out;
    };

    return scale;
  }

  function scaleBand() {
    let dom = [];
    let rng = [0, 1];
    let pad = 0.1;

    function metrics() {
      const n = dom.length;
      const span = rng[1] - rng[0];
      if (n === 0) {
        return { step: 0, bw: 0, offset: rng[0] };
      }
      const step = span / (n + pad);
      const bw = Math.max(1, step * (1 - pad));
      const offset = rng[0] + step * pad * 0.5;
      return { step, bw, offset };
    }

    function scale(val) {
      const idx = dom.indexOf(val);
      if (idx < 0) {
        return undefined;
      }
      const { step, offset } = metrics();
      return offset + idx * step;
    }

    scale.domain = function (d) {
      if (!arguments.length) {
        return dom.slice();
      }
      dom = Array.isArray(d) ? d.slice() : [];
      return scale;
    };

    scale.range = function (r) {
      if (!arguments.length) {
        return rng.slice();
      }
      rng = [Number(r[0]), Number(r[1])];
      return scale;
    };

    scale.padding = function (p) {
      pad = Math.max(0, Math.min(0.9, Number(p) || 0));
      return scale;
    };

    scale.bandwidth = function () {
      return metrics().bw;
    };

    return scale;
  }

  function line() {
    let xAcc = (d) => d[0];
    let yAcc = (d) => d[1];
    let definedFn = () => true;

    function generator(items) {
      if (!Array.isArray(items) || items.length === 0) {
        return "";
      }
      const parts = [];
      let penDown = false;
      for (let i = 0; i < items.length; i += 1) {
        const item = items[i];
        if (!definedFn(item, i)) {
          penDown = false;
          continue;
        }
        const xv = Number(xAcc(item, i)).toFixed(2);
        const yv = Number(yAcc(item, i)).toFixed(2);
        parts.push(`${penDown ? "L" : "M"}${xv},${yv}`);
        penDown = true;
      }
      return parts.join(" ");
    }

    generator.x = function (fn) {
      xAcc = typeof fn === "function" ? fn : () => fn;
      return generator;
    };

    generator.y = function (fn) {
      yAcc = typeof fn === "function" ? fn : () => fn;
      return generator;
    };

    generator.defined = function (fn) {
      definedFn = typeof fn === "function" ? fn : () => Boolean(fn);
      return generator;
    };

    return generator;
  }

  function area() {
    let xAcc = (d) => d[0];
    let y0Acc = () => 0;
    let y1Acc = (d) => d[1];

    function generator(items) {
      if (!Array.isArray(items) || items.length === 0) {
        return "";
      }
      const upper = [];
      const lower = [];
      for (let i = 0; i < items.length; i += 1) {
        const xv = Number(xAcc(items[i], i)).toFixed(2);
        const y1v = Number(y1Acc(items[i], i)).toFixed(2);
        const y0v = Number(y0Acc(items[i], i)).toFixed(2);
        upper.push(`${i === 0 ? "M" : "L"}${xv},${y1v}`);
        lower.unshift(`L${xv},${y0v}`);
      }
      return `${upper.join(" ")} ${lower.join(" ")} Z`;
    }

    generator.x = function (fn) {
      xAcc = typeof fn === "function" ? fn : () => fn;
      return generator;
    };

    generator.y0 = function (fn) {
      y0Acc = typeof fn === "function" ? fn : () => fn;
      return generator;
    };

    generator.y1 = function (fn) {
      y1Acc = typeof fn === "function" ? fn : () => fn;
      return generator;
    };

    return generator;
  }

  function retentionColor(retention) {
    const r = Math.max(0, Math.min(1, Number(retention) || 0));
    if (r < 0.5) {
      return "#f85149";
    }
    if (r <= 0.75) {
      return "#d29922";
    }
    return "#3fb950";
  }

  const d3Compat = {
    select,
    min,
    max,
    extent,
    mean,
    scaleLinear,
    scaleBand,
    line,
    area,
    retentionColor
  };

  if (globalScope && !globalScope.d3) {
    globalScope.d3 = d3Compat;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = d3Compat;
  }
})(typeof window !== "undefined" ? window : globalThis);
