(() => {
  "use strict";
  const D = JSON.parse(document.getElementById("site-data").textContent);
  const REDUCE = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ vocabulary
  const DS = ["commonsenseqa", "truthfulqa", "bigbenchhard", "gsm8k", "humaneval"];
  const DSL = { all: "All tasks", commonsenseqa: "CommonsenseQA", truthfulqa: "TruthfulQA", bigbenchhard: "BIG-Bench Hard", gsm8k: "GSM8K", humaneval: "HumanEval" };
  const DSS = { all: "All tasks", commonsenseqa: "CommonsenseQA", truthfulqa: "TruthfulQA", bigbenchhard: "BBH", gsm8k: "GSM8K", humaneval: "HumanEval" };
  const TASKS = {
    commonsenseqa: { what: "Everyday knowledge. Multiple choice.", metric: "accuracy", n: "200 of 12,102" },
    truthfulqa: { what: "Spot the true answer among popular misconceptions.", metric: "match + similarity", n: "200 of 817" },
    bigbenchhard: { what: "Reasoning puzzles: logic, dates, tracking objects.", metric: "accuracy", n: "200 of 6,511" },
    gsm8k: { what: "Grade-school math word problems.", metric: "final answer", n: "200 of 8,500" },
    humaneval: { what: "Write a Python function; unit tests decide.", metric: "pass@1", n: "all 164" },
  };
  const FAMS = ["qwen2.5_0.5b", "llama3.2_1b", "qwen2.5_1.5b", "gemma2_2b"];
  const FL = D.meta.familyLabel;
  const FSHORT = { "qwen2.5_0.5b": "Qwen 0.5B", "llama3.2_1b": "Llama 1B", "qwen2.5_1.5b": "Qwen 1.5B", "gemma2_2b": "Gemma 2B" };
  const FC = { "qwen2.5_0.5b": "var(--f1)", "llama3.2_1b": "var(--f2)", "qwen2.5_1.5b": "var(--f3)", "gemma2_2b": "var(--f4)" };
  const FSYM = { "qwen2.5_0.5b": d3.symbolCircle, "llama3.2_1b": d3.symbolSquare, "qwen2.5_1.5b": d3.symbolTriangle, "gemma2_2b": d3.symbolDiamond };
  const QUANTS = ["fp16", "q8_0", "q4_1", "q4_K_M", "q4_0", "q4_K_S", "q3_K_L", "q3_K_M", "q3_K_S"];
  const M = D.models.map((m, i) => Object.assign({}, m, { i, fl: FL[m.fam], fs: FSHORT[m.fam], color: FC[m.fam] }));
  const MID = Object.fromEntries(M.map((m) => [m.id, m]));
  const ag = (m, ds) => D.agg[m.id][ds];
  const IDLE = D.meta.idleW;
  const PHONE_WH = 15;

  // ------------------------------------------------------------------ helpers
  const fmtN = d3.format(",.0f");
  const f1 = d3.format(".1f");
  const f2 = d3.format(".2f");
  const pct = (v) => `${Math.round(v * 100)}%`;
  const joule = (v) => (v < 10 ? f2(v) : v < 100 ? f1(v) : fmtN(v)) + " J";
  const secs = (v) => (v < 100 ? f1(v) : fmtN(v)) + " s";
  const METRICS = {
    E: { label: "Energy per answer", short: "J / answer", fmt: joule, ramp: "cost" },
    jpt: { label: "Energy per token", short: "J / token", fmt: (v) => f2(v) + " J", ramp: "cost" },
    acc: { label: "Accuracy", short: "Accuracy", fmt: pct, ramp: "acc" },
    lat: { label: "Seconds per answer", short: "Seconds", fmt: secs, ramp: "cost" },
    tps: { label: "Tokens per second", short: "Tokens/s", fmt: (v) => f2(v), ramp: "acc" },
    tok: { label: "Tokens per answer", short: "Tokens", fmt: (v) => f1(v), ramp: "cost" },
  };
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function modelName(m) {
    const f = el("span");
    f.append(document.createTextNode(m.fl + " "));
    const c = el("code", null, m.q);
    f.append(c);
    return f;
  }
  function swatch(color) {
    const s = el("span", "swatch");
    s.style.background = color;
    return s;
  }
  function seg(container, options, value, onChange, label) {
    const wrap = el("div", "seg");
    wrap.setAttribute("role", "group");
    if (label) wrap.setAttribute("aria-label", label);
    const btns = options.map((o) => {
      const b = el("button", null, o.label);
      b.type = "button";
      b.dataset.v = o.v;
      b.setAttribute("aria-pressed", String(o.v === value));
      b.addEventListener("click", () => {
        set(o.v);
        onChange(o.v);
      });
      wrap.append(b);
      return b;
    });
    function set(v) {
      btns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === String(v))));
    }
    function disable(pred) {
      btns.forEach((b) => (b.disabled = pred(b.dataset.v)));
    }
    container.append(wrap);
    return { el: wrap, set, disable, btns };
  }
  function onResize(node, fn) {
    let w = 0;
    new ResizeObserver(() => {
      const nw = node.clientWidth;
      if (nw && Math.abs(nw - w) > 1) {
        w = nw;
        fn();
      }
    }).observe(node);
  }
  const tr = (sel, animate) => (animate && !REDUCE ? sel.transition().duration(550).ease(d3.easeCubicOut) : sel);
  function copyText(text, btn) {
    const done = () => {
      const old = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = old), 1400);
    };
    try {
      navigator.clipboard.writeText(text).then(done, () => fallbackSelect(btn));
    } catch (e) {
      fallbackSelect(btn);
    }
  }
  function fallbackSelect(btn) {
    const target = document.getElementById(btn.dataset.target || "");
    if (target) {
      const r = document.createRange();
      r.selectNodeContents(target);
      const s = getSelection();
      s.removeAllRanges();
      s.addRange(r);
    }
    btn.textContent = "Press Ctrl/⌘+C";
  }

  // ------------------------------------------------------------------ tooltip
  const tip = el("div", "tip");
  tip.setAttribute("role", "tooltip");
  document.body.append(tip);
  function tipShow(evt, { title, color, rows = [], note }) {
    tip.replaceChildren();
    const t = el("div", "t-title");
    if (color) t.append(swatch(color));
    t.append(typeof title === "string" ? document.createTextNode(title) : title);
    tip.append(t);
    for (const [k, v] of rows) {
      const r = el("div", "t-row");
      r.append(el("span", null, k), el("b", null, v));
      tip.append(r);
    }
    if (note) tip.append(el("div", "t-note", note));
    tip.classList.add("on");
    tipMove(evt);
  }
  function tipMove(evt) {
    let x, y;
    if (evt && evt.clientX) {
      x = evt.clientX;
      y = evt.clientY;
    } else {
      const r = (evt.currentTarget || evt.target).getBoundingClientRect();
      x = r.left + r.width / 2;
      y = r.top;
    }
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let L = x + 14, T = y - h - 12;
    if (L + w > innerWidth - 8) L = Math.max(8, x - w - 14);
    if (T < 60) T = y + 18;
    tip.style.left = L + "px";
    tip.style.top = T + "px";
  }
  const tipHide = () => tip.classList.remove("on");
  function bindTip(sel, content) {
    sel
      .on("pointerenter", (e, d) => tipShow(e, content(d)))
      .on("pointermove", (e) => tipMove(e))
      .on("pointerleave", tipHide)
      .on("focus", (e, d) => tipShow(e, content(d)))
      .on("blur", tipHide);
  }
  function modelTip(m, ds, extra = []) {
    const a = ag(m, ds);
    return {
      title: modelName(m),
      color: m.color,
      rows: [
        ["Accuracy", pct(a.acc)],
        ["Energy per answer", joule(a.E)],
        ["Energy per token", f2(a.jpt) + " J"],
        ["Seconds per answer", secs(a.lat)],
        ["Tokens per second", f2(a.tps)],
        ...extra,
      ],
      note: `${DSL[ds]} · ${fmtN(a.n)} answers · ${fmtN(m.size)} MB file`,
    };
  }

  // ------------------------------------------------------------------ questions
  function qTitle(q) {
    const p = q.prompt.trim();
    if (q.ds === "humaneval") {
      const m = p.match(/def\s+(\w+)\s*\(([^)]*)\)/);
      return m ? `Write ${m[1]}(${m[2].replace(/:\s*[^,]+/g, "").replace(/\s+/g, " ")})` : p.slice(0, 80);
    }
    let s = p.replace(/^Question:\s*/, "").split("\n")[0];
    if (q.ds === "gsm8k") {
      const parts = s.split(/(?<=[.?!])\s+/);
      s = parts[parts.length - 1] || s;
    }
    return s.length > 96 ? s.slice(0, 94).trimEnd() + "…" : s;
  }
  function promptBody(q) {
    return q.prompt.replace(/\n*Print only the answer[\s\S]*$/, "").replace(/\n*# Complete the function[\s\S]*$/, "").trim();
  }
  const Q = D.questions.map((q, i) => Object.assign(q, { i, title: qTitle(q), nOk: q.answers.filter((a) => a.ok).length }));
  const anime = Q.find((q) => /everyone loves anime/.test(q.prompt)) || Q[0];

  // ------------------------------------------------------------------ nav highlight
  (function nav() {
    const links = [...document.querySelectorAll(".nav a")];
    const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            links.forEach((a) => a.classList.remove("active"));
            const a = map.get(e.target.id);
            if (a) a.classList.add("active");
          }
        }
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    map.forEach((_, id) => {
      const s = document.getElementById(id);
      if (s) io.observe(s);
    });
  })();
  document.getElementById("fact-n").textContent = fmtN(D.meta.nInferences);

  // ================================================================== HERO TRACE
  (function hero() {
    const root = document.getElementById("hero-chart");
    const ans = anime.answers.find((a) => M[a.m].id === "llama3.2_1b_instruct_fp16");
    const tr0 = ans.tr;
    const pts = tr0.p.map((p, k) => ({ t: tr0.t0 + k * tr0.dt, w: p / 100 }));
    const readout = document.getElementById("hero-readout");
    const avgW = d3.mean(pts.filter((d) => d.t >= 0 && d.t <= ans.lat), (d) => d.w);
    function setReadout(w, t) {
      readout.replaceChildren();
      const items = t == null
        ? [["avg", f2(avgW) + " W"], ["energy", f1(ans.E) + " J"], ["time", f1(ans.lat) + " s"]]
        : [["now", f2(w) + " W"], ["t", f1(Math.max(0, t)) + " s"], ["answer", f1(ans.E) + " J"]];
      for (const [k, v] of items) {
        const s = el("span");
        s.append(document.createTextNode(k + " "), el("b", null, v));
        readout.append(s);
      }
    }
    setReadout();
    document.getElementById("hero-cap").textContent =
      `Power drawn by the Raspberry Pi while Llama 3.2 1B (FP16) answers “${anime.title}”, measured twice per second. The shaded area is the energy counted for the answer.`;
    let svg, x, y, dot, vline, H = 150;
    function draw() {
      const W = root.clientWidth;
      if (!W) return;
      x = d3.scaleLinear().domain(d3.extent(pts, (d) => d.t)).range([0, W]);
      y = d3.scaleLinear().domain([0, 8]).range([H - 4, 6]);
      d3.select(root).selectAll("svg").remove();
      svg = d3.select(root).append("svg").attr("width", W).attr("height", H);
      const inside = pts.filter((d) => d.t >= 0 && d.t <= ans.lat);
      svg.append("path").datum(inside).attr("d", d3.area().x((d) => x(d.t)).y0(y(IDLE)).y1((d) => y(Math.max(IDLE, d.w))).curve(d3.curveMonotoneX)).style("fill", "var(--energy-wash)");
      svg.append("line").attr("x1", 0).attr("x2", W).attr("y1", y(IDLE)).attr("y2", y(IDLE)).style("stroke", "var(--muted)").style("stroke-dasharray", "3 4").style("stroke-width", 1);
      svg.append("text").attr("x", 2).attr("y", y(IDLE) + 15).text(`idle ${IDLE} W`).style("font-size", "12px").style("fill", "var(--muted)");
      svg.append("path").datum(pts).attr("d", d3.line().x((d) => x(d.t)).y((d) => y(d.w)).curve(d3.curveMonotoneX)).style("fill", "none").style("stroke", "var(--energy)").style("stroke-width", 2).style("stroke-linejoin", "round");
      const peak = d3.max(pts, (d) => d.w);
      svg.append("text").attr("x", W - 2).attr("y", y(peak) - 8).attr("text-anchor", "end").text(`${f1(peak)} W peak`).style("font-size", "12px").style("fill", "var(--ink-2)");
      vline = svg.append("line").attr("y1", 0).attr("y2", H).style("stroke", "var(--line-2)").style("opacity", 0);
      dot = svg.append("circle").attr("r", 4.5).style("fill", "var(--energy)").style("stroke", "var(--bg)").style("stroke-width", 2).style("opacity", 0);
    }
    draw();
    onResize(root, draw);
    if (REDUCE) return;
    // ambient playhead that sweeps the trace, paused when off-screen
    let visible = false, t0 = null, raf = null;
    const dur = 9000, pause = 2500;
    function step(now) {
      if (!visible) { raf = null; return; }
      if (t0 == null) t0 = now;
      const k = ((now - t0) % (dur + pause)) / dur;
      if (k <= 1 && dot) {
        const t = x.domain()[0] + k * (x.domain()[1] - x.domain()[0]);
        const i = Math.min(pts.length - 1, Math.max(0, Math.round((t - tr0.t0) / tr0.dt)));
        dot.attr("cx", x(t)).attr("cy", y(pts[i].w)).style("opacity", 1);
        vline.attr("x1", x(t)).attr("x2", x(t)).style("opacity", 0.8);
        setReadout(pts[i].w, t);
      } else if (dot) {
        dot.style("opacity", 0);
        vline.style("opacity", 0);
        setReadout();
      }
      raf = requestAnimationFrame(step);
    }
    new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
      if (visible && !raf) raf = requestAnimationFrame(step);
    }).observe(root);
  })();

  // ================================================================== SETUP
  (function modelGrid() {
    const root = document.getElementById("model-grid");
    const t = el("table");
    const thead = el("thead"), hr = el("tr");
    hr.append(el("th"));
    QUANTS.forEach((q) => hr.append(el("th", null, q)));
    thead.append(hr);
    const tb = el("tbody");
    const maxS = d3.max(M, (m) => m.size);
    for (const f of FAMS) {
      const r = el("tr");
      const th = el("th");
      const fh = el("div", "famhead");
      fh.append(swatch(FC[f]), document.createTextNode(FL[f]), el("small", null, D.meta.params[f]));
      th.append(fh);
      r.append(th);
      for (const q of QUANTS) {
        const m = MID[`${f}_instruct_${q}`];
        const td = el("td", m ? "has" : "none");
        if (m) {
          const side = Math.max(8, Math.sqrt(m.size / maxS) * 46);
          const sq = el("div", "sq");
          sq.style.width = sq.style.height = side.toFixed(1) + "px";
          sq.style.background = m.color;
          td.append(sq, el("div", "mb", fmtN(m.size) + " MB"));
          td.tabIndex = 0;
          const content = () => ({ title: m.ollama, color: m.color, rows: [["File size", fmtN(m.size) + " MB"], ["Parameters", D.meta.params[f]], ["Energy per answer", joule(ag(m, "all").E)], ["Accuracy", pct(ag(m, "all").acc)]], note: "Averages over the five tasks" });
          td.addEventListener("pointerenter", (e) => tipShow(e, content()));
          td.addEventListener("pointermove", tipMove);
          td.addEventListener("pointerleave", tipHide);
          td.addEventListener("focus", (e) => tipShow(e, content()));
          td.addEventListener("blur", tipHide);
        } else {
          td.textContent = "–";
          td.setAttribute("aria-label", "not tested");
        }
        r.append(td);
      }
      tb.append(r);
    }
    t.append(thead, tb);
    root.append(t);

    const tasks = document.getElementById("tasks");
    for (const ds of DS) {
      const d = el("div", "task");
      d.append(el("h4", null, DSL[ds]), el("div", "what", TASKS[ds].what), el("div", "meta", `${TASKS[ds].n} · ${TASKS[ds].metric}`));
      tasks.append(d);
    }
  })();

  // ================================================================== QUANTIZATION DEMO
  (function quantDemo() {
    // deterministic example weights (seeded normal draws plus one outlier)
    let s = 20250;
    const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const W = d3.range(32).map(() => {
      const u = rand() || 1e-9, v = rand();
      return 0.32 * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    });
    W[11] = 0.98;
    W[23] = -0.71;
    const S = { bits: 4, scheme: "_0" };
    const SCHEMES = { 16: [""], 8: ["_0"], 4: ["_0", "_1", "K"], 3: ["K"] };
    const FORMATS = { 16: { "": ["fp16"] }, 8: { _0: ["q8_0"] }, 4: { _0: ["q4_0"], _1: ["q4_1"], K: ["q4_K_S", "q4_K_M"] }, 3: { K: ["q3_K_S", "q3_K_M", "q3_K_L"] } };

    function quantize(ws, bits, scheme) {
      if (bits === 16) return { q: ws.slice(), levels: [] };
      const n = 2 ** bits;
      const out = [], levels = [];
      const groups = scheme === "K" ? [[0, 8], [8, 16], [16, 24], [24, 32]] : [[0, 32]];
      for (const [a, b] of groups) {
        const g = ws.slice(a, b);
        let lv;
        if (scheme === "_0") {
          // llama.cpp q*_0: scale from the signed value with the largest magnitude
          const mx = g.reduce((p, c) => (Math.abs(c) > Math.abs(p) ? c : p), 0);
          const d = mx / -(n / 2);
          lv = d3.range(-n / 2, n / 2).map((k) => k * d);
        } else {
          // _1 and K: scale plus offset (min), per block or per group of 8
          const lo = d3.min(g), hi = d3.max(g), d = (hi - lo) / (n - 1);
          lv = d3.range(n).map((k) => lo + k * d);
        }
        levels.push({ a, b, lv });
        for (const w of g) out.push(lv.reduce((p, c) => (Math.abs(c - w) < Math.abs(p - w) ? c : p), lv[0]));
      }
      return { q: out, levels };
    }
    const controls = document.getElementById("q-controls");
    const bitsSeg = seg(controls, [{ v: "16", label: "16-bit FP16" }, { v: "8", label: "8-bit" }, { v: "4", label: "4-bit" }, { v: "3", label: "3-bit" }], String(S.bits), (v) => { S.bits = +v; if (!SCHEMES[S.bits].includes(S.scheme)) S.scheme = SCHEMES[S.bits][SCHEMES[S.bits].length - 1]; schemeSeg.set(S.scheme); update(); }, "Bits per weight");
    const schemeSeg = seg(controls, [{ v: "_0", label: "_0 one scale" }, { v: "_1", label: "_1 scale + offset" }, { v: "K", label: "K small groups" }], S.scheme, (v) => { S.scheme = v; update(); }, "Format");
    const chart = document.getElementById("q-chart");
    const stats = document.getElementById("q-stats");
    const sizes = document.getElementById("q-sizes");
    const llama = M.filter((m) => m.fam === "llama3.2_1b");
    const fp = MID["llama3.2_1b_instruct_fp16"].size;
    const sizeRows = llama.map((m) => {
      const r = el("div", "row");
      const bar = el("div", "bar"), fill = el("i");
      fill.style.width = ((m.size / fp) * 100).toFixed(1) + "%";
      bar.append(fill);
      r.append(el("code", null, m.q), bar, el("span", "v", fmtN(m.size) + " MB"));
      sizes.append(r);
      return { m, r };
    });
    sizes.prepend(el("div", "source-note", "Llama 3.2 1B file size by format"));
    function update() {
      schemeSeg.disable((v) => !SCHEMES[S.bits].includes(v));
      if (S.bits === 16) schemeSeg.set("");
      const { q, levels } = quantize(W, S.bits, S.scheme);
      const err = d3.mean(W, (w, i) => Math.abs(w - q[i])) / d3.mean(W, (w) => Math.abs(w));
      const fmts = FORMATS[S.bits][S.scheme] || FORMATS[S.bits][""];
      const fm = MID[`llama3.2_1b_instruct_${fmts[0]}`];
      stats.replaceChildren();
      const cells = [
        ["Values a weight can take", S.bits === 16 ? "65,536" : fmtN(2 ** S.bits)],
        ["Average rounding error", S.bits === 16 ? "≈ 0%" : (err * 100).toFixed(1) + "%"],
        ["Llama 3.2 1B file", fmtN(fm.size) + " MB"],
        ["Share of FP16 size", Math.round((fm.size / fp) * 100) + "%"],
      ];
      for (const [k, v] of cells) {
        const d = el("div");
        d.append(el("span", null, k), el("b", null, v));
        stats.append(d);
      }
      sizeRows.forEach(({ m, r }) => r.classList.toggle("on", fmts.includes(m.q)));
      draw(q, levels);
    }
    function draw(q, levels) {
      const Wd = chart.clientWidth;
      if (!Wd) return;
      const H = 230, mL = 34, mR = 6, mT = 8, mB = 22;
      const x = d3.scaleBand().domain(d3.range(32)).range([mL, Wd - mR]).padding(0.3);
      const y = d3.scaleLinear().domain([-1.05, 1.05]).range([H - mB, mT]);
      d3.select(chart).selectAll("svg").remove();
      const svg = d3.select(chart).append("svg").attr("width", Wd).attr("height", H).attr("role", "img").attr("aria-label", `32 example weights quantized to ${S.bits} bits`);
      svg.append("g").attr("class", "axis").attr("transform", `translate(${mL - 6},0)`).call(d3.axisLeft(y).tickValues([-1, -0.5, 0, 0.5, 1]).tickSize(0).tickPadding(4)).call((g) => g.select(".domain").remove());
      // levels
      if (S.bits <= 4) {
        for (const g of levels) {
          const x0 = x(g.a) - x.step() * 0.15, x1 = x(g.b - 1) + x.bandwidth() + x.step() * 0.15;
          svg.append("g").selectAll("line").data(g.lv.filter((v) => v >= -1.05 && v <= 1.05)).join("line").attr("x1", x0).attr("x2", x1).attr("y1", y).attr("y2", y).style("stroke", "var(--line-2)").style("stroke-width", 1);
        }
        if (levels.length > 1) {
          svg.append("g").selectAll("line").data(levels.slice(1)).join("line").attr("x1", (g) => x(g.a) - x.step() * 0.15).attr("x2", (g) => x(g.a) - x.step() * 0.15).attr("y1", mT).attr("y2", H - mB).style("stroke", "var(--ink-2)").style("stroke-width", 1).style("opacity", 0.35);
          svg.append("g").selectAll("text").data(levels).join("text").attr("x", (g) => (x(g.a) + x(g.b - 1) + x.bandwidth()) / 2).attr("y", H - 4).attr("text-anchor", "middle").text((g, i) => `group ${i + 1}`).style("font-size", "11px").style("fill", "var(--muted)");
        }
      }
      svg.append("line").attr("x1", mL).attr("x2", Wd - mR).attr("y1", y(0)).attr("y2", y(0)).style("stroke", "var(--line-2)");
      const cx = (i) => x(i) + x.bandwidth() / 2;
      const g = svg.append("g");
      g.selectAll("line.err").data(W).join("line").attr("class", "err").attr("x1", (d, i) => cx(i)).attr("x2", (d, i) => cx(i)).attr("y1", (d) => y(d)).attr("y2", (d, i) => y(q[i])).style("stroke", "var(--energy)").style("stroke-width", 2).style("opacity", 0.55);
      g.selectAll("circle.o").data(W).join("circle").attr("class", "o").attr("cx", (d, i) => cx(i)).attr("cy", (d) => y(d)).attr("r", Math.min(5, x.bandwidth() / 2)).style("fill", "var(--surface)").style("stroke", "var(--ink-2)").style("stroke-width", 1.5);
      g.selectAll("circle.q").data(q).join("circle").attr("class", "q").attr("cx", (d, i) => cx(i)).attr("cy", (d) => y(d)).attr("r", Math.min(3.6, x.bandwidth() / 2.6)).style("fill", "var(--energy)");
      if (S.bits === 8) svg.append("text").attr("x", Wd - mR).attr("y", mT + 10).attr("text-anchor", "end").text("256 values: too close together to draw").style("font-size", "11.5px").style("fill", "var(--muted)");
      if (S.bits === 16) svg.append("text").attr("x", Wd - mR).attr("y", mT + 10).attr("text-anchor", "end").text("Full precision: nothing is rounded away").style("font-size", "11.5px").style("fill", "var(--muted)");
    }
    update();
    onResize(chart, update);

    // tag decoder
    const parts = [
      { k: "q4", text: "<b>Bits per weight.</b> q8 stores 8 bits, q4 stores 4 and q3 stores 3. FP16, the unquantized baseline, stores 16-bit floating-point numbers. Fewer bits give a smaller file and usually lower accuracy." },
      { k: "_K", text: "<b>Quantization scheme.</b> The older llama.cpp formats _0 and _1 give each block of 32 weights one scale (_0) or a scale plus an offset (_1). K marks the newer <i>k-quant</i> formats, which split each block into smaller groups with their own scales, so less precision is lost for the same number of bits." },
      { k: "_M", text: "<b>Size of the mix.</b> S, M and L are small, medium and large variants of a k-quant format. Larger variants keep some of the most sensitive layers at higher precision: a slightly bigger file in exchange for accuracy." },
    ];
    const dec = document.getElementById("decoder");
    const txt = document.getElementById("decoder-text");
    const btns = parts.map((p, i) => {
      const b = el("button", null, p.k);
      b.type = "button";
      b.setAttribute("aria-pressed", String(i === 0));
      b.addEventListener("click", () => { btns.forEach((x) => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); txt.innerHTML = p.text; });
      dec.append(b);
      return b;
    });
    txt.innerHTML = parts[0].text;
  })();

  // ================================================================== DOT PLOT (RQ1, RQ3)
  function DotPlot(chart, opt) {
    const S = Object.assign({ metric: "jpt", ds: "all", hl: [] }, opt);
    const svg = d3.select(chart).append("svg");
    const gGrid = svg.append("g").attr("class", "grid");
    const gAxis = svg.append("g").attr("class", "axis");
    const axLabel = svg.append("text").attr("class", "axis-label").attr("text-anchor", "end");
    const gFam = svg.append("g");
    const gRows = svg.append("g");
    function draw(animate) {
      const W = chart.clientWidth;
      if (!W) return;
      const narrow = W < 560, labelW = narrow ? 62 : 110, rowH = 21, mR = narrow ? 50 : 70;
      let y = 4;
      const fams = [], rows = [];
      FAMS.forEach((f) => {
        fams.push({ f, y: y + 13 });
        y += 24;
        M.filter((m) => m.fam === f).forEach((m) => { rows.push({ m, y: y + rowH / 2 }); y += rowH; });
        y += 10;
      });
      const plotB = y, H = y + 40;
      svg.attr("width", W).attr("height", H).attr("role", "img").attr("aria-label", `${METRICS[S.metric].label} for all 28 model versions, ${DSL[S.ds]}`);
      const data = rows.map((r) => { const a = ag(r.m, S.ds); return { id: r.m.id, m: r.m, y: r.y, v: a[S.metric], box: S.ds === "all" ? null : a.box[S.metric] }; });
      const xmax = d3.max(data, (d) => Math.max(d.v, d.box ? d.box[3] : 0));
      const x = d3.scaleLinear().domain([0, xmax * 1.02]).nice().range([labelW + 10, W - mR]);
      const ticks = x.ticks(narrow ? 4 : 6);
      gGrid.selectAll("line").data(ticks).join("line").attr("y1", 0).attr("y2", plotB).attr("x1", x).attr("x2", x);
      gAxis.attr("transform", `translate(0,${plotB})`).call(d3.axisBottom(x).tickValues(ticks).tickSize(0).tickPadding(8).tickFormat(d3.format("~g"))).call((g) => g.select(".domain").remove());
      axLabel.attr("x", W - mR).attr("y", H - 2).text(S.axisLabel || METRICS[S.metric].label);
      gFam.selectAll("g.fh").data(fams, (d) => d.f).join((en) => { const g = en.append("g").attr("class", "fh"); g.append("rect").attr("width", 10).attr("height", 10).attr("rx", 3).attr("y", -9); g.append("text").attr("x", 16).style("font-weight", 650).style("font-size", "13px").style("fill", "var(--ink)"); return g; })
        .attr("transform", (d) => `translate(0,${d.y})`)
        .call((g) => g.select("rect").style("fill", (d) => FC[d.f]))
        .call((g) => g.select("text").text((d) => FL[d.f]));
      const row = gRows.selectAll("g.r").data(data, (d) => d.id).join((en) => {
        const g = en.append("g").attr("class", "r").attr("tabindex", 0);
        g.append("rect").attr("class", "hit").style("fill", "transparent");
        g.append("text").attr("class", "ql").attr("text-anchor", "end").attr("dy", "0.32em").style("font-family", "var(--mono)").style("font-size", "12px").style("fill", "var(--ink-2)");
        g.append("line").attr("class", "stem").style("stroke", "var(--line)").style("stroke-width", 1);
        g.append("line").attr("class", "iqr").style("stroke-width", 5).style("stroke-linecap", "round").style("opacity", 0.3);
        g.append("circle").attr("class", "dot").attr("r", 5.5).style("stroke", "var(--surface)").style("stroke-width", 2);
        g.append("text").attr("class", "val").attr("dy", "0.32em").style("font-size", "12px").style("font-weight", 650).style("fill", "var(--ink)");
        return g;
      });
      row.attr("transform", (d) => `translate(0,${d.y})`).attr("aria-label", (d) => `${d.m.fl} ${d.m.q}: ${METRICS[S.metric].fmt(d.v)}`);
      row.select(".hit").attr("x", 0).attr("y", -rowH / 2).attr("width", W).attr("height", rowH);
      row.select(".ql").attr("x", labelW).text((d) => d.m.q);
      tr(row.select(".stem"), animate).attr("x1", x(0)).attr("x2", (d) => x(d.v));
      tr(row.select(".iqr"), animate).attr("x1", (d) => (d.box ? x(d.box[1]) : x(d.v))).attr("x2", (d) => (d.box ? x(d.box[3]) : x(d.v))).style("stroke", (d) => d.m.color).style("opacity", (d) => (d.box ? 0.3 : 0));
      tr(row.select(".dot"), animate).attr("cx", (d) => x(d.v)).style("fill", (d) => d.m.color);
      const hl = new Set(S.hl);
      tr(row.select(".val"), animate).attr("x", (d) => Math.max(x(d.v), d.box ? x(d.box[3]) : 0) + 10).text((d) => (hl.has(d.id) ? METRICS[S.metric].fmt(d.v) : ""));
      row.style("opacity", (d) => (hl.size && !hl.has(d.id) ? 0.3 : 1));
      bindTip(row, (d) => {
        const extra = d.box ? [["Middle 50%", `${METRICS[S.metric].fmt(d.box[1])} – ${METRICS[S.metric].fmt(d.box[3])}`]] : [];
        return modelTip(d.m, S.ds, extra);
      });
    }
    draw(false);
    onResize(chart, () => draw(false));
    return { update(p) { Object.assign(S, p); draw(true); }, S };
  }

  // ---------------------------------------------------------------- RQ1 energy chart
  (function energy() {
    const chart = DotPlot(document.getElementById("energy-chart"), { metric: "jpt", ds: "all", axisLabel: "Joules per generated token" });
    const callout = document.getElementById("energy-callout");
    const v = (id, ds) => ag(MID[id], ds).jpt;
    const stories = [
      {
        label: "Llama 3.2 1B: FP16 to q3_K_S",
        hl: ["llama3.2_1b_instruct_fp16", "llama3.2_1b_instruct_q8_0", "llama3.2_1b_instruct_q3_K_S"],
        text: (ds) => {
          const a = v("llama3.2_1b_instruct_fp16", ds), b = v("llama3.2_1b_instruct_q3_K_S", ds), c = v("llama3.2_1b_instruct_q8_0", ds);
          return `<b>Llama 3.2 1B</b> drops from <b>${f2(a)} J</b> per token at FP16 to <b>${f2(b)} J</b> as q3_K_S (${Math.round((1 - b / a) * 100)}% less${ds === "all" ? "" : ` on ${DSL[ds]}`}). The 8-bit version already saves ${Math.round((1 - c / a) * 100)}%.`;
        },
      },
      {
        label: "Same bits, different cost",
        hl: ["qwen2.5_0.5b_instruct_q3_K_M", "qwen2.5_0.5b_instruct_q3_K_S"],
        text: (ds) => {
          const a = v("qwen2.5_0.5b_instruct_q3_K_M", ds), b = v("qwen2.5_0.5b_instruct_q3_K_S", ds);
          const r = b / a;
          return `<b>Qwen 2.5 0.5B</b> q3_K_M and q3_K_S both store 3 bits per weight, yet ${r >= 1 ? `q3_K_S used <b>${f1(r)}×</b> the energy per token (${f2(b)} vs ${f2(a)} J)` : `here q3_K_M used <b>${f1(1 / r)}×</b> the energy per token (${f2(a)} vs ${f2(b)} J)`}. The format matters, not only the bit count.`;
        },
      },
      {
        label: "When 3-bit costs more than 4-bit",
        hl: M.filter((m) => m.fam === "qwen2.5_1.5b").map((m) => m.id),
        text: (ds) => {
          const q = M.filter((m) => m.fam === "qwen2.5_1.5b");
          const q3 = q.filter((m) => m.q.startsWith("q3")).map((m) => ag(m, ds).jpt), q4 = q.filter((m) => m.q.startsWith("q4")).map((m) => ag(m, ds).jpt);
          const all = d3.min(q3) > d3.max(q4);
          return `For <b>Qwen 2.5 1.5B</b>, the 3-bit versions averaged <b>${f2(d3.mean(q3))} J</b> per token against <b>${f2(d3.mean(q4))} J</b> for 4-bit${all ? "; every 3-bit version cost more than every 4-bit one" : ""}. Unpacking very low-bit weights takes extra work that can eat the savings.`;
        },
      },
      { label: "Show all", hl: [], text: () => "All 28 versions. Hover or focus a row for accuracy, energy, speed and file size." },
    ];
    let cur = 0;
    const box = document.getElementById("energy-stories");
    const btns = stories.map((s, i) => {
      const b = el("button", "chip", s.label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(i === cur));
      b.addEventListener("click", () => { cur = i; btns.forEach((x, j) => x.setAttribute("aria-pressed", String(j === i))); apply(); });
      box.append(b);
      return b;
    });
    function apply() {
      callout.innerHTML = stories[cur].text(chart.S.ds);
      chart.update({ hl: stories[cur].hl });
    }
    seg(document.getElementById("energy-ds"), ["all", ...DS].map((d) => ({ v: d, label: DSS[d] })), "all", (ds) => { chart.S.ds = ds; apply(); }, "Task");
    apply();
  })();

  // ================================================================== REPLAY
  function Replay(root, opt) {
    const S = { q: opt.q, lanes: opt.lanes.slice(), speed: opt.speed || 4, T: null, playing: false, raf: null };
    const head = el("div", "figure-head");
    const ht = el("div");
    ht.append(el("div", "figure-title", opt.title || "Energy meter replay"), el("div", "figure-sub", "Recorded power at 2 samples per second. Text appears at the model's measured writing speed."));
    head.append(ht);
    root.append(head);
    const ctr = el("div", "replay-controls");
    const mkSel = (label, id) => { const l = el("label"); l.htmlFor = id; l.append(document.createTextNode(label)); const s = el("select"); s.id = id; l.append(s); ctr.append(l); return s; };
    const uid = opt.id;
    const qSel = mkSel("Question", uid + "-q");
    const laneSel = [mkSel("Model A", uid + "-a"), mkSel("Model B", uid + "-b")];
    DS.forEach((ds) => {
      const g = el("optgroup");
      g.label = DSL[ds];
      Q.filter((q) => q.ds === ds).forEach((q) => { const o = el("option", null, q.title); o.value = q.i; g.append(o); });
      qSel.append(g);
    });
    laneSel.forEach((s) => {
      FAMS.forEach((f) => {
        const g = el("optgroup");
        g.label = FL[f];
        M.filter((m) => m.fam === f).forEach((m) => { const o = el("option", null, `${m.fl} · ${m.q}`); o.value = m.i; g.append(o); });
        s.append(g);
      });
    });
    root.append(ctr);
    const bar = el("div", "replay-bar");
    const playBtn = el("button", "btn energy", "▶ Play");
    playBtn.type = "button";
    bar.append(playBtn);
    const spd = el("div", "control");
    spd.append(el("span", null, "Speed"));
    seg(spd, [{ v: "1", label: "1×" }, { v: "4", label: "4×" }, { v: "16", label: "16×" }], String(S.speed), (v) => (S.speed = +v), "Playback speed");
    bar.append(spd);
    const pl = el("div", "phase-legend");
    pl.innerHTML = '<span><i style="background:var(--line-2)"></i>Loading</span><span><i style="background:var(--ink-2);opacity:.45"></i>Reading the question</span><span><i style="background:var(--energy)"></i>Writing the answer</span>';
    bar.append(pl);
    root.append(bar);
    const qBox = el("div", "question-box");
    root.append(qBox);
    const lanesEl = el("div", "lanes");
    root.append(lanesEl);
    let lanes = [], x, tStart, tEnd;

    qSel.addEventListener("change", () => { S.q = Q[+qSel.value]; stop(); build(); });
    laneSel.forEach((s, i) => s.addEventListener("change", () => { S.lanes[i] = +s.value; stop(); build(); }));
    playBtn.addEventListener("click", () => (S.playing ? pause() : play()));

    function prep(ans) {
      const t = ans.tr;
      const pts = t.p.map((p, k) => ({ t: t.t0 + k * t.dt, w: p / 100 }));
      const cum = [0];
      for (let k = 1; k < pts.length; k++) {
        const a = pts[k - 1], b = pts[k];
        const lo = Math.max(0, a.t), hi = Math.min(ans.lat, b.t);
        const e = hi > lo ? ((Math.max(0, a.w - IDLE) + Math.max(0, b.w - IDLE)) / 2) * (hi - lo) : 0;
        cum.push(cum[k - 1] + e);
      }
      const total = cum[cum.length - 1] || 1;
      const genStart = ans.load + ans.pre;
      const avgW = d3.mean(pts.filter((d) => d.t >= 0 && d.t <= ans.lat), (d) => d.w) || IDLE;
      return { pts, cum, total, genStart, avgW };
    }
    function valueAt(L, T) {
      const { pts } = L.p;
      const k = (T - pts[0].t) / (pts[1].t - pts[0].t);
      const i = Math.max(0, Math.min(pts.length - 2, Math.floor(k)));
      const f = Math.max(0, Math.min(1, k - i));
      return { w: pts[i].w + (pts[i + 1].w - pts[i].w) * f, cum: L.p.cum[i] + (L.p.cum[i + 1] - L.p.cum[i]) * f };
    }
    function build() {
      qSel.value = S.q.i;
      laneSel.forEach((s, i) => (s.value = S.lanes[i]));
      qBox.replaceChildren();
      qBox.append(el("span", "ds-tag", `${DSL[S.q.ds]} · ${S.q.nOk} of 28 models answered correctly`), document.createTextNode(promptBody(S.q)));
      lanes = S.lanes.map((mi) => {
        const ans = S.q.answers.find((a) => a.m === mi);
        return { m: M[mi], ans, p: prep(ans) };
      });
      tStart = d3.min(lanes, (L) => L.p.pts[0].t);
      tEnd = d3.max(lanes, (L) => Math.min(L.p.pts[L.p.pts.length - 1].t, L.ans.lat + 3));
      lanesEl.replaceChildren();
      lanes.forEach((L) => {
        const lane = el("div", "lane");
        const lh = el("div", "lane-head");
        const nm = el("div", "lane-name");
        nm.append(swatch(L.m.color), document.createTextNode(L.m.fl), el("code", null, L.m.q));
        const vd = el("span", "verdict " + (L.ans.ok ? "ok" : "no"), L.ans.ok ? "✓ Correct" : "✗ Incorrect");
        nm.append(vd);
        const ro = el("div", "readouts");
        L.ro = {};
        [["w", "Power"], ["e", "Energy"], ["t", "Time"], ["k", "Tokens"]].forEach(([k, lab]) => {
          const r = el("span", "readout " + k);
          const lb = el("span", null, lab + " ");
          const b = el("b");
          r.append(lb, b);
          ro.append(r);
          L.ro[k] = { b, lb };
        });
        lh.append(nm, ro);
        const body = el("div", "lane-body");
        L.chartEl = el("div", "chart");
        L.answerEl = el("div", "answer");
        L.answerEl.setAttribute("aria-live", "off");
        body.append(L.chartEl, L.answerEl);
        lane.append(lh, body);
        lanesEl.append(lane);
      });
      drawCharts();
      renderT();
    }
    function drawCharts() {
      const W = lanes[0] && lanes[0].chartEl.clientWidth;
      if (!W) return;
      const H = 142, mL = 30, mR = 8, mT = 10, plotB = H - 44;
      x = d3.scaleLinear().domain([tStart, tEnd]).range([mL, W - mR]);
      const ymax = Math.max(8, d3.max(lanes, (L) => d3.max(L.p.pts, (d) => d.w)));
      const y = d3.scaleLinear().domain([0, ymax]).range([plotB, mT]);
      lanes.forEach((L) => {
        d3.select(L.chartEl).selectAll("svg").remove();
        const svg = d3.select(L.chartEl).append("svg").attr("width", W).attr("height", H).attr("role", "img").attr("aria-label", `Power trace for ${L.m.fl} ${L.m.q}: ${f1(L.ans.E)} joules over ${f1(L.ans.lat)} seconds`);
        svg.append("g").attr("class", "grid").selectAll("line").data([2, 4, 6, 8].filter((v) => v <= ymax)).join("line").attr("x1", mL).attr("x2", W - mR).attr("y1", y).attr("y2", y);
        svg.append("g").attr("class", "axis").attr("transform", `translate(${mL - 4},0)`).call(d3.axisLeft(y).tickValues([0, 4, 8]).tickSize(0).tickPadding(4).tickFormat((d) => d + " W")).call((g) => g.select(".domain").remove());
        const xt = x.ticks(W < 420 ? 4 : 7).filter((t) => t >= 0);
        svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 14})`).call(d3.axisBottom(x).tickValues(xt).tickSize(0).tickPadding(2).tickFormat((d) => d + " s")).call((g) => g.select(".domain").remove());
        const cid = `clip-${uid}-${L.m.i}-${Math.random().toString(36).slice(2, 7)}`;
        const cp = svg.append("clipPath").attr("id", cid).append("rect").attr("x", 0).attr("y", 0).attr("height", H).attr("width", W);
        const inside = L.p.pts.filter((d) => d.t >= 0 && d.t <= L.ans.lat);
        svg.append("path").datum(inside).attr("clip-path", `url(#${cid})`).attr("d", d3.area().x((d) => x(d.t)).y0(y(IDLE)).y1((d) => y(Math.max(IDLE, d.w)))).style("fill", "var(--energy-wash)");
        svg.append("line").attr("x1", mL).attr("x2", W - mR).attr("y1", y(IDLE)).attr("y2", y(IDLE)).style("stroke", "var(--muted)").style("stroke-dasharray", "3 4");
        svg.append("path").datum(L.p.pts.filter((d) => d.t <= tEnd)).attr("clip-path", `url(#${cid})`).attr("d", d3.line().x((d) => x(d.t)).y((d) => y(d.w))).style("fill", "none").style("stroke", "var(--energy)").style("stroke-width", 2).style("stroke-linejoin", "round");
        // phase strip
        const py = plotB + 8, ph = 9;
        const phases = [
          { a: 0, b: L.ans.load, c: "var(--line-2)", o: 1, n: "Loading" },
          { a: L.ans.load, b: L.p.genStart, c: "var(--ink-2)", o: 0.45, n: "Reading the question", k: `${L.ans.ptok} tokens` },
          { a: L.p.genStart, b: L.ans.lat, c: "var(--energy)", o: 1, n: "Writing the answer", k: `${L.ans.tok} tokens` },
        ];
        const pg = svg.append("g");
        pg.selectAll("rect").data(phases).join("rect").attr("x", (d) => x(d.a)).attr("width", (d) => Math.max(1, x(d.b) - x(d.a))).attr("y", py).attr("height", ph).attr("rx", 2).style("fill", (d) => d.c).style("opacity", (d) => d.o)
          .each(function () { d3.select(this).attr("tabindex", 0); });
        bindTip(pg.selectAll("rect"), (d) => ({ title: d.n, rows: [["Duration", f1(d.b - d.a) + " s"], ...(d.k ? [["Size", d.k]] : [])] }));
        L.cp = cp;
        L.head = svg.append("line").attr("y1", mT).attr("y2", py + ph).style("stroke", "var(--ink)").style("stroke-width", 1).style("opacity", 0);
      });
    }
    function renderT() {
      const atRest = S.T === null;
      lanes.forEach((L) => {
        const a = L.ans;
        const T = atRest ? Infinity : S.T;
        if (L.cp && x) L.cp.attr("width", atRest ? 99999 : Math.max(0, x(Math.min(T, tEnd))));
        if (L.head && x) L.head.style("opacity", atRest ? 0 : 0.6).attr("x1", x(Math.min(T, tEnd))).attr("x2", x(Math.min(T, tEnd)));
        let w, eJ, tS, tok, textN, phase;
        if (atRest) {
          w = L.p.avgW; eJ = a.E; tS = a.lat; tok = a.tok; textN = a.resp.length; phase = "done";
          L.ro.w.lb.textContent = "Avg power ";
        } else {
          const v = valueAt(L, Math.min(T, L.p.pts[L.p.pts.length - 1].t));
          w = v.w;
          eJ = T <= 0 ? 0 : T >= a.lat ? a.E : (a.E * v.cum) / L.p.total;
          tS = Math.max(0, Math.min(T, a.lat));
          const g = a.lat - L.p.genStart > 0 ? (T - L.p.genStart) / (a.lat - L.p.genStart) : 1;
          const gf = Math.max(0, Math.min(1, g));
          tok = Math.round(a.tok * gf);
          textN = Math.round(a.resp.length * gf);
          phase = T < 0 ? "wait" : T < a.load ? "load" : T < L.p.genStart ? "read" : T < a.lat ? "write" : "done";
          L.ro.w.lb.textContent = "Power ";
        }
        L.ro.w.b.textContent = f2(w) + " W";
        L.ro.e.b.textContent = f1(eJ) + " J";
        L.ro.t.b.textContent = f1(tS) + " s";
        L.ro.k.b.textContent = String(tok);
        const ae = L.answerEl;
        ae.replaceChildren();
        if (phase === "wait") ae.append(el("span", "wait", "Waiting for the question…"));
        else if (phase === "load") ae.append(el("span", "wait", "Loading the model…"));
        else if (phase === "read") ae.append(el("span", "wait", `Reading the question (${a.ptok} tokens)…`));
        else {
          ae.append(document.createTextNode(a.resp.slice(0, textN)));
          if (phase === "write") ae.append(el("span", "caret"));
          if (phase === "write") ae.scrollTop = ae.scrollHeight;
        }
      });
    }
    function play() {
      if (S.T === null || S.T >= tEnd) S.T = tStart;
      S.playing = true;
      playBtn.textContent = "❚❚ Pause";
      let last = performance.now();
      const step = (now) => {
        if (!S.playing) return;
        S.T += ((now - last) / 1000) * S.speed;
        last = now;
        if (S.T >= tEnd) {
          S.T = tEnd;
          S.playing = false;
          playBtn.textContent = "↺ Replay";
          renderT();
          return;
        }
        renderT();
        S.raf = requestAnimationFrame(step);
      };
      S.raf = requestAnimationFrame(step);
    }
    function pause() {
      S.playing = false;
      if (S.raf) cancelAnimationFrame(S.raf);
      playBtn.textContent = "▶ Resume";
    }
    function stop() {
      S.playing = false;
      if (S.raf) cancelAnimationFrame(S.raf);
      S.T = null;
      playBtn.textContent = "▶ Play";
    }
    build();
    onResize(lanesEl, () => { drawCharts(); renderT(); });
    return {
      set(q, lanesIdx) {
        stop();
        if (q) S.q = q;
        if (lanesIdx) S.lanes = lanesIdx;
        build();
      },
      S,
    };
  }

  const storyReplay = Replay(document.getElementById("story-replay"), {
    id: "story",
    title: "Energy meter replay",
    q: anime,
    lanes: [MID["llama3.2_1b_instruct_fp16"].i, MID["llama3.2_1b_instruct_q4_K_M"].i],
    speed: 4,
  });

  (function readVsWrite() {
    const pre = d3.mean(M, (m) => d3.mean(DS, (ds) => ag(m, ds).prefill));
    const gen = d3.mean(M, (m) => d3.mean(DS, (ds) => ag(m, ds).gen));
    const short = DS.filter((d) => d !== "humaneval");
    const shareShort = d3.mean(M, (m) => d3.mean(short, (ds) => ag(m, ds).prefill / ag(m, ds).lat));
    document.getElementById("read-vs-write").innerHTML =
      `The replay shows something the averages hide: on a Raspberry Pi, much of the time goes into <strong>reading the question</strong> rather than writing the answer. Across all models and tasks, reading took ${f1(pre)} seconds per answer on average and writing ${f1(gen)} seconds. On the four short-answer tasks, reading alone took ${Math.round(shareShort * 100)}% of the time. (This split comes from Ollama's timing logs for the same runs.)`;
  })();

  // ---------------------------------------------------------------- task panels
  (function taskPanels() {
    const root = document.getElementById("task-panels");
    const rows = DS.map((ds) => ({
      ds,
      jpt: d3.mean(M, (m) => ag(m, ds).jpt),
      tok: d3.mean(M, (m) => ag(m, ds).tok),
      pre: d3.mean(M, (m) => ag(m, ds).prefill),
      gen: d3.mean(M, (m) => ag(m, ds).gen),
      load: d3.mean(M, (m) => ag(m, ds).load),
    }));
    const panels = [
      { title: "Energy per token", key: "jpt", fmt: (v) => f2(v) + " J" },
      { title: "Tokens per answer", key: "tok", fmt: (v) => f1(v) },
      { title: "Seconds per answer", key: "time", fmt: (v) => f1(v) + " s" },
    ];
    const els = panels.map((p) => {
      const d = el("div");
      d.append(el("div", "sm-title", p.title));
      const c = el("div", "chart");
      d.append(c);
      root.append(d);
      return { p, c };
    });
    function draw() {
      els.forEach(({ p, c }) => {
        const W = c.clientWidth;
        if (!W) return;
        const rowH = 30, mL = 104, mR = 58, H = rowH * rows.length + 8;
        d3.select(c).selectAll("svg").remove();
        const svg = d3.select(c).append("svg").attr("width", W).attr("height", H).attr("role", "img").attr("aria-label", `${p.title} by task`);
        const xmax = p.key === "time" ? d3.max(rows, (r) => r.load + r.pre + r.gen) : d3.max(rows, (r) => r[p.key]);
        const x = d3.scaleLinear().domain([0, xmax]).range([mL, W - mR]);
        const y = d3.scaleBand().domain(rows.map((r) => r.ds)).range([4, H - 4]).padding(0.45);
        svg.append("g").selectAll("text").data(rows).join("text").attr("x", mL - 8).attr("y", (r) => y(r.ds) + y.bandwidth() / 2).attr("dy", "0.32em").attr("text-anchor", "end").text((r) => DSS[r.ds]).style("font-size", "12.5px").style("fill", "var(--ink-2)");
        svg.append("line").attr("x1", mL).attr("x2", mL).attr("y1", 0).attr("y2", H).style("stroke", "var(--line-2)");
        const bh = Math.min(14, y.bandwidth());
        const rr = (x0, x1, yy, rounded) => {
          const w = Math.max(0, x1 - x0), r = rounded ? Math.min(4, w, bh / 2) : 0;
          return `M${x0},${yy}h${w - r}${r ? `a${r},${r} 0 0 1 ${r},${r}` : ""}v${bh - 2 * r}${r ? `a${r},${r} 0 0 1 ${-r},${r}` : ""}h${-(w - r)}z`;
        };
        const g = svg.append("g");
        rows.forEach((r) => {
          const yy = y(r.ds) + (y.bandwidth() - bh) / 2;
          const gg = g.append("g").attr("tabindex", 0);
          if (p.key === "time") {
            const x0 = x(0), x1 = x(r.load + r.pre), x2 = x(r.load + r.pre + r.gen);
            gg.append("path").attr("d", rr(x0, x1 - 1, yy, false)).style("fill", "var(--ink-2)").style("opacity", 0.45);
            gg.append("path").attr("d", rr(x1 + 1, x2, yy, true)).style("fill", "var(--energy)");
            gg.append("text").attr("x", x2 + 6).attr("y", yy + bh / 2).attr("dy", "0.32em").text(f1(r.load + r.pre + r.gen) + " s").style("font-size", "12px").style("fill", "var(--ink)").style("font-variant-numeric", "tabular-nums");
            bindTip(gg.datum(r), (d) => ({ title: DSL[d.ds], rows: [["Reading the question", f1(d.pre + d.load) + " s"], ["Writing the answer", f1(d.gen) + " s"]], note: "Mean over the 28 model versions" }));
          } else {
            gg.append("path").attr("d", rr(x(0), x(r[p.key]), yy, true)).style("fill", r.ds === "humaneval" || r.ds === "bigbenchhard" ? "var(--ink)" : "var(--ink-2)").style("opacity", r.ds === "humaneval" || r.ds === "bigbenchhard" ? 0.85 : 0.45);
            gg.append("text").attr("x", x(r[p.key]) + 6).attr("y", yy + bh / 2).attr("dy", "0.32em").text(p.fmt(r[p.key])).style("font-size", "12px").style("fill", "var(--ink)").style("font-variant-numeric", "tabular-nums");
            bindTip(gg.datum(r), (d) => ({ title: DSL[d.ds], rows: [[p.title, p.fmt(d[p.key])]], note: "Mean over the 28 model versions" }));
          }
        });
      });
    }
    draw();
    onResize(root, draw);
    // phone-charge comparison
    const best = M.reduce((p, m) => (ag(m, "all").E < ag(p, "all").E ? m : p), M[0]);
    const gemma = D.paper.t5.avg["gemma2_2b"];
    const wh = (ag(best, "all").E * 1000) / 3600;
    document.getElementById("scale-note").innerHTML =
      `For scale: an average Gemma 2 2B answer used <b>${fmtN(gemma)} joules</b>, the same as a 5&nbsp;W LED bulb burning for ${Math.round(gemma / 5)} seconds. ` +
      `At the frugal end, a thousand answers from Qwen 2.5 0.5B ${best.q} (${f1(ag(best, "all").E)} J each) add up to ${f1(wh)} Wh, about <b>${Math.round((wh / PHONE_WH) * 100)}% of a phone charge</b> (assuming a ${PHONE_WH} Wh battery).`;
  })();

  // ================================================================== HEAT TABLE (RQ2, explorer)
  function HeatTable(root, opt) {
    const S = Object.assign({ metric: "acc", sort: null, dir: -1 }, opt);
    function render() {
      const cols = [...DS, "all"];
      const met = METRICS[S.metric];
      const vals = M.flatMap((m) => cols.map((c) => ag(m, c)[S.metric]));
      const lo = d3.min(vals), hi = d3.max(vals);
      const scaleT = (v) => (hi > lo ? (v - lo) / (hi - lo) : 0);
      const t = el("table", "heat");
      const thead = el("thead"), hr = el("tr");
      const mh = el("th", "sortable", "Model");
      mh.tabIndex = 0;
      const resetSort = () => { S.sort = null; render(); };
      mh.addEventListener("click", resetSort);
      mh.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); resetSort(); } });
      hr.append(mh);
      cols.forEach((c) => {
        const th = el("th", "sortable", c === "all" ? "Average" : DSS[c]);
        th.tabIndex = 0;
        th.setAttribute("aria-sort", S.sort === c ? (S.dir < 0 ? "descending" : "ascending") : "none");
        if (S.sort === c) th.append(el("span", "arrow", S.dir < 0 ? "↓" : "↑"));
        const go = () => { if (S.sort === c) S.dir *= -1; else { S.sort = c; S.dir = -1; } render(); };
        th.addEventListener("click", go);
        th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
        hr.append(th);
      });
      thead.append(hr);
      const tb = el("tbody");
      let order = M.slice();
      if (S.sort) order.sort((a, b) => S.dir * (ag(a, S.sort)[S.metric] - ag(b, S.sort)[S.metric]));
      order.forEach((m, i) => {
        if (!S.sort && i > 0 && order[i - 1].fam !== m.fam) {
          const g = el("tr", "gap");
          g.append(el("td"));
          tb.append(g);
        }
        const r = el("tr");
        const nm = el("td", "name");
        nm.append(swatch(m.color), el("small", null, m.fs), document.createTextNode(m.q));
        r.append(nm);
        cols.forEach((c) => {
          const v = ag(m, c)[S.metric];
          const k = scaleT(v);
          const td = el("td", c === "all" ? "avg" : null, met.fmt(v));
          td.style.background = `color-mix(in oklab, var(--ramp-${met.ramp}-hi) ${Math.round(k * 100)}%, var(--ramp-${met.ramp}-lo))`;
          td.style.color = k > 0.55 ? "var(--on-ramp-hi)" : "var(--ink)";
          td.title = `${m.fl} ${m.q} · ${DSL[c]}: ${met.fmt(v)}`;
          r.append(td);
        });
        tb.append(r);
      });
      t.append(thead, tb);
      root.replaceChildren(t);
      if (opt.legend) {
        const lg = opt.legend;
        lg.replaceChildren();
        const sw = (k) => { const s = el("span", "swatch"); s.style.background = `color-mix(in oklab, var(--ramp-${met.ramp}-hi) ${k}%, var(--ramp-${met.ramp}-lo))`; s.style.width = "18px"; return s; };
        const a = el("span"); a.append(sw(0), document.createTextNode(met.fmt(lo)));
        const b = el("span"); b.append(sw(50), document.createTextNode(met.fmt((lo + hi) / 2)));
        const c = el("span"); c.append(sw(100), document.createTextNode(met.fmt(hi)));
        lg.append(a, b, c);
      }
    }
    render();
    return { update(p) { Object.assign(S, p); render(); } };
  }
  HeatTable(document.getElementById("acc-table"), { metric: "acc", legend: document.getElementById("acc-legend") });

  // ================================================================== PARETO
  function Pareto(chart, opt) {
    const S = Object.assign({ ds: "all", mini: false, constraint: null, pick: null, dimmed: null }, opt);
    function draw() {
      const W = chart.clientWidth;
      if (!W) return;
      const narrow = W < 560;
      const H = S.mini ? 240 : narrow ? 340 : 420;
      const mL = 44, mR = S.mini ? 12 : narrow ? 16 : 150, mT = 14, mB = 38;
      const data = M.map((m) => { const a = ag(m, S.ds); return { m, E: a.E, acc: a.acc }; });
      const front = new Set(D.fronts[S.ds]);
      const x = d3.scaleLog().domain([d3.min(data, (d) => d.E) * 0.8, d3.max(data, (d) => d.E) * 1.2]).range([mL, W - mR]);
      const y = d3.scaleLinear().domain([0, Math.min(1, d3.max(data, (d) => d.acc) * 1.12)]).nice().range([H - mB, mT]);
      d3.select(chart).selectAll("svg").remove();
      const svg = d3.select(chart).append("svg").attr("width", W).attr("height", H).attr("role", "img").attr("aria-label", `Accuracy against energy per answer, ${DSL[S.ds]}. ${front.size} models are Pareto-optimal.`);
      const xt = [5, 10, 20, 50, 100, 200, 500, 1000].filter((v) => v >= x.domain()[0] && v <= x.domain()[1]);
      svg.append("g").attr("class", "grid").selectAll("line").data(xt).join("line").attr("x1", x).attr("x2", x).attr("y1", mT).attr("y2", H - mB);
      svg.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line").attr("x1", mL).attr("x2", W - mR).attr("y1", y).attr("y2", y);
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - mB})`).call(d3.axisBottom(x).tickValues(xt).tickSize(0).tickPadding(8).tickFormat((d) => d + " J")).call((g) => g.select(".domain").remove());
      svg.append("g").attr("class", "axis").attr("transform", `translate(${mL - 4},0)`).call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(6).tickFormat(d3.format(".0%"))).call((g) => g.select(".domain").remove());
      svg.append("text").attr("class", "axis-label").attr("x", W - mR).attr("y", H - 4).attr("text-anchor", "end").text("Energy per answer (log scale)");
      if (!S.mini) svg.append("text").attr("class", "axis-label").attr("x", mL).attr("y", mT - 2).text("Accuracy");
      // constraint region
      if (S.constraint) {
        const c = S.constraint;
        if (c.type === "budget" && c.v >= x.domain()[0]) {
          const xv = Math.min(x(c.v), W - mR);
          svg.append("rect").attr("x", mL).attr("y", mT).attr("width", Math.max(0, xv - mL)).attr("height", H - mB - mT).style("fill", "var(--energy-wash)").style("opacity", 0.6);
          svg.append("line").attr("x1", xv).attr("x2", xv).attr("y1", mT).attr("y2", H - mB).style("stroke", "var(--energy)").style("stroke-dasharray", "4 3");
        } else if (c.type === "target") {
          svg.append("rect").attr("x", mL).attr("y", mT).attr("width", W - mR - mL).attr("height", Math.max(0, y(c.v) - mT)).style("fill", "var(--energy-wash)").style("opacity", 0.6);
          svg.append("line").attr("x1", mL).attr("x2", W - mR).attr("y1", y(c.v)).attr("y2", y(c.v)).style("stroke", "var(--energy)").style("stroke-dasharray", "4 3");
        }
      }
      // frontier
      const fp = data.filter((d) => front.has(d.m.id)).sort((a, b) => a.E - b.E);
      svg.append("path").datum(fp).attr("d", d3.line().x((d) => x(d.E)).y((d) => y(d.acc)).curve(d3.curveStepAfter)).style("fill", "none").style("stroke", "var(--ink)").style("stroke-width", 1.5).style("opacity", 0.5);
      // points
      const sym = d3.symbol().size(S.mini ? 58 : 84);
      const pts = svg.append("g").selectAll("path.pt").data(data).join("path").attr("class", "pt")
        .attr("transform", (d) => `translate(${x(d.E)},${y(d.acc)})`)
        .attr("d", (d) => sym.type(FSYM[d.m.fam])())
        .style("fill", (d) => d.m.color).style("stroke", "var(--surface)").style("stroke-width", 1.5)
        .style("opacity", (d) => (S.dimmed && S.dimmed(d.m) ? 0.22 : front.has(d.m.id) ? 1 : 0.55));
      if (S.pick) {
        const d = data.find((d) => d.m.id === S.pick);
        if (d) svg.append("circle").attr("cx", x(d.E)).attr("cy", y(d.acc)).attr("r", 11).style("fill", "none").style("stroke", "var(--ink)").style("stroke-width", 2);
      }
      // labels for frontier points (greedy collision avoidance)
      if (!S.mini) {
        const placed = [];
        const segs = [];
        for (let i = 0; i < fp.length - 1; i++) {
          const a = fp[i], b = fp[i + 1];
          segs.push([x(a.E), y(a.acc), x(b.E), y(a.acc)], [x(b.E), y(a.acc), x(b.E), y(b.acc)]);
        }
        const lab = svg.append("g");
        fp.slice().sort((a, b) => b.acc - a.acc).forEach((d) => {
          const text = narrow ? d.m.q : `${d.m.fs} ${d.m.q}`;
          const w = text.length * 6.4 + 4, h = 14;
          const cands = [[7, 5], [-w - 7, -h - 3], [7, -h - 3], [-w - 7, 5], [-w / 2, -h - 9], [-w / 2, 10], [7, 16], [-w - 7, 16]];
          const hitsLine = (bx, by) => segs.some(([x1, y1, x2, y2]) => bx < Math.max(x1, x2) + 2 && bx + w > Math.min(x1, x2) - 2 && by < Math.max(y1, y2) + 2 && by + h > Math.min(y1, y2) - 2);
          for (const [dx, dy] of cands) {
            const bx = x(d.E) + dx, by = y(d.acc) + dy;
            if (bx < mL || bx + w > W - 2 || by < 0 || by + h > H - mB) continue;
            if (placed.some((p) => bx < p.x + p.w && bx + w > p.x && by < p.y + p.h && by + h > p.y)) continue;
            if (data.some((o) => x(o.E) > bx - 5 && x(o.E) < bx + w + 5 && y(o.acc) > by - 4 && y(o.acc) < by + h + 4)) continue;
            if (hitsLine(bx, by)) continue;
            placed.push({ x: bx, y: by, w, h });
            lab.append("text").attr("x", bx).attr("y", by + 11).text(text).style("font-size", "11.5px").style("fill", "var(--ink)").style("font-weight", 550)
              .style("paint-order", "stroke").style("stroke", "var(--surface)").style("stroke-width", 3).style("stroke-linejoin", "round");
            break;
          }
        });
      }
      // nearest-point hover layer
      const del = d3.Delaunay.from(data, (d) => x(d.E), (d) => y(d.acc));
      const ring = svg.append("circle").attr("r", 9).style("fill", "none").style("stroke", "var(--ink)").style("stroke-width", 1.5).style("opacity", 0).style("pointer-events", "none");
      svg.append("rect").attr("x", mL).attr("y", mT).attr("width", W - mR - mL).attr("height", H - mB - mT).style("fill", "transparent")
        .on("pointermove", (e) => {
          const [px, py] = d3.pointer(e);
          const i = del.find(px, py);
          const d = data[i];
          if (Math.hypot(x(d.E) - px, y(d.acc) - py) > 36) { ring.style("opacity", 0); tipHide(); return; }
          ring.attr("cx", x(d.E)).attr("cy", y(d.acc)).style("opacity", 1);
          tipShow(e, modelTip(d.m, S.ds, front.has(d.m.id) ? [["Pareto-optimal", "yes"]] : []));
        })
        .on("pointerleave", () => { ring.style("opacity", 0); tipHide(); });
      pts.attr("tabindex", S.mini ? null : 0).on("focus", (e, d) => tipShow(e, modelTip(d.m, S.ds))).on("blur", tipHide);
    }
    draw();
    onResize(chart, draw);
    return { update(p) { Object.assign(S, p); draw(); }, S };
  }
  function paretoLegend(root) {
    root.replaceChildren();
    FAMS.forEach((f) => {
      const s = el("span");
      const sv = d3.create("svg").attr("width", 14).attr("height", 14).attr("aria-hidden", "true");
      sv.append("path").attr("transform", "translate(7,7)").attr("d", d3.symbol().type(FSYM[f]).size(70)()).style("fill", FC[f]);
      s.append(sv.node(), document.createTextNode(FL[f]));
      root.append(s);
    });
    const s = el("span");
    s.innerHTML = '<svg width="22" height="12" aria-hidden="true"><path d="M1 10h7V5h7V2h6" fill="none" stroke="var(--ink)" stroke-width="1.5" opacity=".6"/></svg>Pareto frontier';
    root.append(s);
  }
  (function rq2Pareto() {
    const p = Pareto(document.getElementById("pareto-chart"), { ds: "all" });
    paretoLegend(document.getElementById("pareto-legend"));
    const callout = document.getElementById("pareto-callout");
    const nf = (ds) => D.fronts[ds].length;
    const T = {
      all: () => { const g = ag(MID["gemma2_2b_instruct_q3_K_M"], "all"); return `Averaged over the five tasks, <b>${nf("all")} of 28</b> versions are on the frontier. Qwen 2.5 0.5B covers the cheap end; Gemma 2 2B q3_K_M is the most accurate (${pct(g.acc)}) at about ${fmtN(g.E)} J per answer.`; },
      commonsenseqa: () => `<b>${nf("commonsenseqa")} versions</b> on the frontier, the most of any task. Accuracy climbs steadily with model size, up to 70% for Gemma 2 2B q3_K_S at 117 J per answer.`,
      truthfulqa: () => `<b>${nf("truthfulqa")} versions</b> on the frontier. Llama 3.2 1B q4_K_S nearly matches Gemma 2 2B (39% against 40%) at about a third of the energy.`,
      bigbenchhard: () => `<b>${nf("bigbenchhard")} versions</b> on the frontier. Qwen 2.5 0.5B q8_0 gets 29% at just 28 J per answer; the larger Qwen and Gemma versions add a few points for 3 to 6 times the energy.`,
      gsm8k: () => `<b>${nf("gsm8k")} versions</b> on the frontier, all Qwen. The best, Qwen 2.5 1.5B q4_K_S, solves 13% of the problems at 63 J per answer.`,
      humaneval: () => `<b>${nf("humaneval")} versions</b> on the frontier. Llama 3.2 1B q4_K_M writes working code for 87% of problems at 134 J; Qwen 2.5 0.5B versions reach 72–75% for a little less.`,
    };
    function apply(ds) { p.update({ ds }); callout.innerHTML = T[ds](); }
    seg(document.getElementById("pareto-ds"), ["all", ...DS].map((d) => ({ v: d, label: DSS[d] })), "all", apply, "Task");
    apply("all");
  })();

  // ---------------------------------------------------------------- RQ3 speed
  (function speed() {
    const S = { metric: "lat", ds: "all", story: 0 };
    const chart = DotPlot(document.getElementById("speed-chart"), { metric: "lat", ds: "all", axisLabel: "Seconds per answer" });
    const callout = document.getElementById("speed-callout");
    const title = document.getElementById("speed-title"), sub = document.getElementById("speed-sub");
    const val = (id) => ag(MID[id], S.ds)[S.metric];
    const fmt = (v) => METRICS[S.metric].fmt(v) + (S.metric === "tps" ? " tokens/s" : "");
    const change = (a, b) => (S.metric === "lat" ? `${Math.round((1 - b / a) * 100)}% less waiting` : `${f1(b / a)}× the writing speed`);
    const stories = [
      { label: "Qwen 2.5 0.5B: FP16 to q8_0", hl: ["qwen2.5_0.5b_instruct_fp16", "qwen2.5_0.5b_instruct_q8_0"], text: () => { const a = val("qwen2.5_0.5b_instruct_fp16"), b = val("qwen2.5_0.5b_instruct_q8_0"); return `<b>Qwen 2.5 0.5B</b>: ${fmt(a)} at FP16, ${fmt(b)} at q8_0 (${change(a, b)}).`; } },
      { label: "Llama 3.2 1B: FP16 to q8_0", hl: ["llama3.2_1b_instruct_fp16", "llama3.2_1b_instruct_q8_0"], text: () => { const a = val("llama3.2_1b_instruct_fp16"), b = val("llama3.2_1b_instruct_q8_0"); return `<b>Llama 3.2 1B</b>: ${fmt(a)} at FP16, ${fmt(b)} at q8_0 (${change(a, b)}). Its 4-bit and 3-bit versions are no faster than 8-bit.`; } },
      { label: "Gemma 2 2B, the slowest", hl: M.filter((m) => m.fam === "gemma2_2b").map((m) => m.id), text: () => { const v = M.filter((m) => m.fam === "gemma2_2b").map((m) => ag(m, S.ds)[S.metric]); return `<b>Gemma 2 2B</b>, the largest model, ranged from ${fmt(d3.min(v))} to ${fmt(d3.max(v))} even in its 3-bit versions.`; } },
      { label: "Show all", hl: [], text: () => "All 28 versions. Hover or focus a row for the full numbers." },
    ];
    const box = document.getElementById("speed-stories");
    const btns = stories.map((s, i) => {
      const b = el("button", "chip", s.label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(i === 0));
      b.addEventListener("click", () => { S.story = i; btns.forEach((x, j) => x.setAttribute("aria-pressed", String(j === i))); apply(); });
      box.append(b);
      return b;
    });
    function apply() {
      title.textContent = METRICS[S.metric].label;
      sub.textContent = `Dot: mean. Bar: middle 50% of answers (single tasks only). ${S.metric === "lat" ? "Lower" : "Higher"} is better.`;
      callout.innerHTML = stories[S.story].text();
      chart.update({ metric: S.metric, ds: S.ds, hl: stories[S.story].hl, axisLabel: S.metric === "lat" ? "Seconds per answer" : "Tokens generated per second" });
    }
    seg(document.getElementById("speed-metric"), [{ v: "lat", label: "Seconds per answer" }, { v: "tps", label: "Tokens per second" }], "lat", (v) => { S.metric = v; apply(); }, "Measure");
    seg(document.getElementById("speed-ds"), ["all", ...DS].map((d) => ({ v: d, label: DSS[d] })), "all", (v) => { S.ds = v; apply(); }, "Task");
    apply();
  })();

  // ================================================================== ADVISOR
  (function advisor() {
    const S = { ds: "all", goal: "budget", budget: 100, target: 0.4, maxLat: Infinity };
    const onTask = () => (S.ds === "all" ? "across the five tasks" : `on ${DSL[S.ds]}`);
    const bIn = document.getElementById("adv-budget"), bOut = document.getElementById("adv-budget-out");
    const tIn = document.getElementById("adv-target"), tOut = document.getElementById("adv-target-out");
    const lIn = document.getElementById("adv-lat"), lOut = document.getElementById("adv-lat-out");
    const toJ = (v) => 10 * Math.pow(90, v / 1000);
    const fromJ = (j) => Math.round((Math.log(j / 10) / Math.log(90)) * 1000);
    const toS = (v) => (v >= 100 ? Infinity : 5 * Math.pow(60, v / 100));
    bIn.value = fromJ(S.budget);
    tIn.value = Math.round(S.target * 100);
    lIn.value = 100;
    const niceJ = (j) => (j < 20 ? Math.round(j) : j < 100 ? Math.round(j / 5) * 5 : Math.round(j / 10) * 10);
    const result = document.getElementById("adv-result");
    const chart = Pareto(document.getElementById("adv-chart"), { ds: "all", mini: true });
    seg(document.getElementById("adv-task"), ["all", ...DS].map((d) => ({ v: d, label: d === "all" ? "A mix of everything" : DSL[d] })), "all", (v) => { S.ds = v; update(); }, "Kind of work");
    seg(document.getElementById("adv-goal"), [{ v: "budget", label: "Most accurate within a budget" }, { v: "target", label: "Cheapest that meets a target" }], "budget", (v) => { S.goal = v; update(); }, "Goal");
    bIn.addEventListener("input", () => { S.budget = niceJ(toJ(+bIn.value)); update(); });
    tIn.addEventListener("input", () => { S.target = +tIn.value / 100; update(); });
    lIn.addEventListener("input", () => { S.maxLat = toS(+lIn.value); update(); });
    function update() {
      document.getElementById("adv-budget-field").hidden = S.goal !== "budget";
      document.getElementById("adv-target-field").hidden = S.goal !== "target";
      bOut.textContent = S.budget + " J";
      tOut.textContent = Math.round(S.target * 100) + "%";
      lOut.textContent = S.maxLat === Infinity ? "no limit" : secs(S.maxLat);
      const C = M.map((m) => { const a = ag(m, S.ds); return { m, E: a.E, acc: a.acc, lat: a.lat }; });
      const okLat = (c) => c.lat <= S.maxLat;
      const okGoal = (c) => (S.goal === "budget" ? c.E <= S.budget : c.acc >= S.target - 1e-9);
      const feas = C.filter((c) => okLat(c) && okGoal(c));
      let pick = null;
      if (feas.length) pick = S.goal === "budget" ? feas.reduce((p, c) => (c.acc > p.acc || (c.acc === p.acc && c.E < p.E) ? c : p)) : feas.reduce((p, c) => (c.E < p.E || (c.E === p.E && c.acc > p.acc) ? c : p));
      chart.update({ ds: S.ds, constraint: S.goal === "budget" ? { type: "budget", v: S.budget } : { type: "target", v: S.target }, pick: pick && pick.m.id, dimmed: (m) => !(okLat(C[m.i]) && okGoal(C[m.i])) });
      result.replaceChildren();
      if (!pick) {
        const n = el("div", "empty-note");
        const cheapest = C.filter(okLat).sort((a, b) => a.E - b.E)[0];
        const best = C.filter(okLat).sort((a, b) => b.acc - a.acc)[0];
        n.textContent = S.goal === "budget"
          ? `No model stays under ${S.budget} J per answer${S.maxLat < Infinity ? ` and answers within ${secs(S.maxLat)}` : ""} ${onTask()}. The cheapest option uses ${cheapest ? joule(cheapest.E) : "more"}; raise the budget or relax the time limit.`
          : `No model reaches ${Math.round(S.target * 100)}% ${onTask()}${S.maxLat < Infinity ? ` within ${secs(S.maxLat)}` : ""}. The most accurate option gets ${best ? pct(best.acc) : "less"}; lower the target or relax the time limit.`;
        result.append(n);
        return;
      }
      const card = el("div", "pick");
      card.append(el("div", "label", "Suggested model"));
      const h = el("h3");
      h.append(swatch(pick.m.color), document.createTextNode(`${pick.m.fl} · ${pick.m.q}`));
      card.append(h);
      const tl = el("div", "tagline");
      const code = el("code", null, `ollama run ${pick.m.ollama}`);
      code.id = "adv-cmd";
      const cp = el("button", "copy", "Copy");
      cp.type = "button";
      cp.dataset.target = "adv-cmd";
      cp.addEventListener("click", () => copyText(`ollama run ${pick.m.ollama}`, cp));
      tl.append(code, cp);
      card.append(tl);
      const st = el("div", "pick-stats");
      [["Accuracy", pct(pick.acc)], ["Energy per answer", joule(pick.E)], ["Time per answer", secs(pick.lat)], ["File size", fmtN(pick.m.size) + " MB"]].forEach(([k, v]) => { const d = el("div"); d.append(el("span", null, k), el("b", null, v)); st.append(d); });
      card.append(st);
      const wh = (pick.E * 1000) / 3600;
      const why = el("p", "why");
      let next = "";
      if (S.goal === "budget") {
        const up = C.filter((c) => okLat(c) && c.acc > pick.acc).sort((a, b) => a.E - b.E)[0];
        if (up) next = ` The next step up, <b>${up.m.fl} ${up.m.q}</b>, reaches ${pct(up.acc)} but needs ${joule(up.E)} per answer (${f1(up.E / pick.E)}× as much).`;
        why.innerHTML = `The most accurate of the ${feas.length} versions that stay under <b>${S.budget} J</b> per answer ${onTask()}${S.maxLat < Infinity ? ` and answer within ${secs(S.maxLat)}` : ""}.${next}`;
      } else {
        const down = C.filter((c) => okLat(c) && c.E < pick.E).sort((a, b) => b.acc - a.acc)[0];
        if (down) next = ` Dropping to <b>${down.m.fl} ${down.m.q}</b> would use ${f1(pick.E / down.E)}× less energy but gets ${pct(down.acc)}.`;
        why.innerHTML = `The cheapest of the ${feas.length} versions that reach <b>${Math.round(S.target * 100)}%</b> ${onTask()}${S.maxLat < Infinity ? ` within ${secs(S.maxLat)}` : ""}.${next}`;
      }
      card.append(why);
      const k = el("p", "why");
      k.innerHTML = `A thousand answers would use <b>${f1(wh)} Wh</b>: about ${f1(wh / PHONE_WH)} full phone charges, or a 5&nbsp;W LED bulb left on for ${f1(wh / 5)} hours.`;
      card.append(k);
      result.append(card);
      // frontier table
      const alts = el("div", "alts");
      alts.append(el("div", "source-note", `Pareto-optimal versions ${onTask()}${D.fronts[S.ds].includes(pick.m.id) ? "" : ", plus the suggestion"}`));
      const t = el("table");
      t.innerHTML = '<thead><tr><th>Model</th><th class="r">Accuracy</th><th class="r">J / answer</th><th class="r">Seconds</th></tr></thead>';
      const tb = el("tbody");
      const ids = [...new Set([...D.fronts[S.ds], pick.m.id])];
      ids.map((id) => C[MID[id].i]).sort((a, b) => a.E - b.E).forEach((c) => {
        const r = el("tr", c === pick ? "sel" : okLat(c) && okGoal(c) ? "" : "dim");
        const n = el("td");
        n.append(swatch(c.m.color), document.createTextNode(` ${c.m.fl} `), el("code", null, c.m.q));
        r.append(n, el("td", "r", pct(c.acc)), el("td", "r", joule(c.E)), el("td", "r", secs(c.lat)));
        tb.append(r);
      });
      t.append(tb);
      alts.append(t);
      result.append(alts);
    }
    update();
  })();

  // ================================================================== PI 5
  (function pi5() {
    const root = document.getElementById("pi5-chart");
    const P = D.paper;
    const precs = [["fp16", "FP16"], ["q8_0", "8-bit"], ["q4", "4-bit"], ["q3", "3-bit"]];
    const rows = [];
    FAMS.forEach((f) => precs.forEach(([k, lab]) => { if (P.t5[k][f] != null && P.t11[k][f] != null) rows.push({ f, k, lab, a: P.t5[k][f], b: P.t11[k][f] }); }));
    function draw() {
      const W = root.clientWidth;
      if (!W) return;
      const rowH = 26, mL = 150, mR = 50, H = rows.length * rowH + 34;
      d3.select(root).selectAll("svg").remove();
      const svg = d3.select(root).append("svg").attr("width", W).attr("height", H).attr("role", "img").attr("aria-label", "Energy per answer on Raspberry Pi 4 and Pi 5 by model and precision");
      const x = d3.scaleLinear().domain([0, 280]).range([mL, W - mR]);
      const y = (i) => 8 + i * rowH + rowH / 2;
      svg.append("g").attr("class", "grid").selectAll("line").data(x.ticks(5)).join("line").attr("x1", x).attr("x2", x).attr("y1", 0).attr("y2", H - 26);
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 26})`).call(d3.axisBottom(x).ticks(5).tickSize(0).tickPadding(8).tickFormat((d) => d + " J")).call((g) => g.select(".domain").remove());
      const g = svg.append("g").selectAll("g").data(rows).join("g").attr("transform", (d, i) => `translate(0,${y(i)})`).attr("tabindex", 0);
      g.append("text").attr("x", mL - 10).attr("dy", "0.32em").attr("text-anchor", "end").text((d) => `${FSHORT[d.f]} · ${d.lab}`).style("font-size", "12.5px").style("fill", "var(--ink-2)");
      g.append("line").attr("x1", (d) => x(Math.min(d.a, d.b))).attr("x2", (d) => x(Math.max(d.a, d.b))).style("stroke", (d) => FC[d.f]).style("stroke-width", 2).style("opacity", 0.5);
      g.append("circle").attr("cx", (d) => x(d.a)).attr("r", 5).style("fill", "var(--surface)").style("stroke", (d) => FC[d.f]).style("stroke-width", 2);
      g.append("circle").attr("cx", (d) => x(d.b)).attr("r", 5.5).style("fill", (d) => FC[d.f]).style("stroke", "var(--surface)").style("stroke-width", 1.5);
      bindTip(g, (d) => ({ title: `${FL[d.f]} · ${d.lab}`, color: FC[d.f], rows: [["Raspberry Pi 4", joule(d.a)], ["Raspberry Pi 5", joule(d.b)]], note: "Mean energy per answer (Tables 5 and 11)" }));
    }
    draw();
    onResize(root, draw);
  })();

  // ================================================================== EXPLORER
  const exploreReplay = (function () {
    const q0 = Q.find((q) => q.ds === "commonsenseqa");
    return Replay(document.getElementById("explore-replay"), { id: "explore", title: "Replay the selected answers", q: q0, lanes: defaultLanes(q0), speed: 4 });
  })();
  function defaultLanes(q) {
    const ok = q.answers.filter((a) => a.ok).sort((a, b) => a.E - b.E);
    const all = q.answers.slice().sort((a, b) => a.E - b.E);
    const a = (ok[0] || all[0]).m;
    const b = all[all.length - 1].m === a ? all[all.length - 2].m : all[all.length - 1].m;
    return [a, b];
  }
  (function inspector() {
    const S = { ds: "commonsenseqa", q: null, sort: { k: "m", dir: 1 }, picked: null };
    const list = document.getElementById("insp-list");
    const qBox = document.getElementById("insp-question");
    const table = document.getElementById("insp-table");
    seg(document.getElementById("insp-ds"), DS.map((d) => ({ v: d, label: DSL[d] })), S.ds, (v) => { S.ds = v; renderList(); }, "Task");
    function renderList() {
      list.replaceChildren();
      const qs = Q.filter((q) => q.ds === S.ds);
      qs.forEach((q) => {
        const b = el("button");
        b.type = "button";
        b.append(document.createTextNode(q.title), el("span", "score", `${q.nOk} of 28 correct`));
        b.addEventListener("click", () => selectQ(q));
        b.dataset.i = q.i;
        list.append(b);
      });
      selectQ(qs[0]);
    }
    function selectQ(q) {
      S.q = q;
      [...list.children].forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.i === q.i)));
      qBox.replaceChildren(el("span", "ds-tag", `${DSL[q.ds]} · ${q.nOk} of 28 models answered correctly`), document.createTextNode(promptBody(q)));
      const lanes = defaultLanes(q);
      S.picked = lanes[0];
      exploreReplay.set(q, lanes);
      renderTable();
    }
    function renderTable() {
      const q = S.q;
      const maxE = d3.max(q.answers, (a) => a.E);
      const cols = [
        { k: "m", label: "Model", get: (a) => a.m },
        { k: "resp", label: "Answer", get: (a) => a.resp },
        { k: "ok", label: "Result", get: (a) => (a.ok ? 1 : 0) },
        { k: "E", label: "Energy", get: (a) => a.E, r: true },
        { k: "lat", label: "Time", get: (a) => a.lat, r: true },
        { k: "tok", label: "Tokens", get: (a) => a.tok, r: true },
      ];
      table.replaceChildren();
      const thead = el("thead"), hr = el("tr");
      cols.forEach((c) => {
        const th = el("th", c.r ? "r" : null, c.label);
        th.tabIndex = 0;
        if (S.sort.k === c.k) th.append(document.createTextNode(S.sort.dir > 0 ? " ↑" : " ↓"));
        th.setAttribute("aria-sort", S.sort.k === c.k ? (S.sort.dir > 0 ? "ascending" : "descending") : "none");
        const go = () => { S.sort = { k: c.k, dir: S.sort.k === c.k ? -S.sort.dir : c.k === "ok" ? -1 : 1 }; renderTable(); };
        th.addEventListener("click", go);
        th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
        hr.append(th);
      });
      thead.append(hr);
      const tb = el("tbody");
      const col = cols.find((c) => c.k === S.sort.k);
      const rows = q.answers.slice().sort((a, b) => { const va = col.get(a), vb = col.get(b); return S.sort.dir * (typeof va === "string" ? va.localeCompare(vb) : va - vb); });
      rows.forEach((a) => {
        const m = M[a.m];
        const r = el("tr", a.m === S.picked ? "picked" : null);
        r.tabIndex = 0;
        const mc = el("td", "model");
        mc.append(swatch(m.color), document.createTextNode(` ${m.fs} `), el("code", null, m.q));
        const rc = el("td", "resp");
        rc.append(el("div", null, a.resp || "(empty)"));
        const vc = el("td");
        vc.append(el("span", "verdict " + (a.ok ? "ok" : "no"), a.ok ? "✓ Correct" : "✗ Wrong"));
        const ec = el("td", "num");
        ec.append(document.createTextNode(joule(a.E)));
        const bar = el("span", "ebar");
        bar.style.width = Math.max(2, (a.E / maxE) * 46).toFixed(0) + "px";
        ec.append(bar);
        r.append(mc, rc, vc, ec, el("td", "num", secs(a.lat)), el("td", "num", String(a.tok)));
        const pick = () => {
          S.picked = a.m;
          const other = exploreReplay.S.lanes[1] === a.m ? exploreReplay.S.lanes[0] : exploreReplay.S.lanes[1];
          exploreReplay.set(null, [a.m, other]);
          r.classList.toggle("open");
          [...tb.children].forEach((x) => x.classList.toggle("picked", x === r));
        };
        r.addEventListener("click", pick);
        r.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
        tb.append(r);
      });
      table.append(thead, tb);
    }
    renderList();
  })();

  (function allResults() {
    const t = HeatTable(document.getElementById("all-table"), { metric: "E" });
    const title = document.getElementById("all-title");
    seg(document.getElementById("all-metric"), Object.entries(METRICS).map(([k, v]) => ({ v: k, label: v.short })), "E", (v) => { title.textContent = METRICS[v].label; t.update({ metric: v }); }, "Measure");
  })();

  (function everyDot() {
    const sel = document.getElementById("dots-model");
    FAMS.forEach((f) => {
      const g = el("optgroup");
      g.label = FL[f];
      M.filter((m) => m.fam === f).forEach((m) => { const o = el("option", null, `${m.fl} · ${m.q}`); o.value = m.id; g.append(o); });
      sel.append(g);
    });
    sel.value = "llama3.2_1b_instruct_q4_K_M";
    const root = document.getElementById("dots");
    const panels = DS.map((ds) => {
      const d = el("div");
      const t = el("div", "sm-title");
      const c = el("div", "chart");
      d.append(t, c);
      root.append(d);
      return { ds, t, c };
    });
    function draw() {
      const m = MID[sel.value];
      const xAll = [], yAll = [];
      DS.forEach((ds) => { const p = D.points[m.id][ds]; p.tok.forEach((v, i) => { if (p.E[i] > 0) { xAll.push(v); yAll.push(p.E[i] / 10); } }); });
      const x0 = Math.max(1, d3.min(xAll) * 0.8), x1 = d3.max(xAll) * 1.25, y0 = Math.max(0.3, d3.min(yAll) * 0.8), y1 = d3.max(yAll) * 1.25;
      panels.forEach(({ ds, t, c }) => {
        const p = D.points[m.id][ds];
        const a = ag(m, ds);
        t.replaceChildren(document.createTextNode(DSS[ds]), el("span", null, `r = ${f2(a.r)}`));
        const W = c.clientWidth;
        if (!W) return;
        const H = 180, mL = 34, mR = 6, mT = 6, mB = 22;
        const x = d3.scaleLog().domain([x0, x1]).range([mL, W - mR]);
        const y = d3.scaleLog().domain([y0, y1]).range([H - mB, mT]);
        d3.select(c).selectAll("svg").remove();
        const svg = d3.select(c).append("svg").attr("width", W).attr("height", H).attr("role", "img").attr("aria-label", `${DSL[ds]}: ${p.E.length} answers, correlation ${f2(a.r)}`);
        const xt = [1, 10, 100, 1000].filter((v) => v >= x0 && v <= x1), yt = [1, 10, 100, 1000].filter((v) => v >= y0 && v <= y1);
        svg.append("g").attr("class", "grid").selectAll("line").data(xt).join("line").attr("x1", x).attr("x2", x).attr("y1", mT).attr("y2", H - mB);
        svg.append("g").attr("class", "grid").selectAll("line").data(yt).join("line").attr("x1", mL).attr("x2", W - mR).attr("y1", y).attr("y2", y);
        svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - mB})`).call(d3.axisBottom(x).tickValues(xt).tickSize(0).tickPadding(6).tickFormat(d3.format("~s"))).call((g) => g.select(".domain").remove());
        svg.append("g").attr("class", "axis").attr("transform", `translate(${mL - 4},0)`).call(d3.axisLeft(y).tickValues(yt).tickSize(0).tickPadding(4).tickFormat(d3.format("~s"))).call((g) => g.select(".domain").remove());
        const pts = p.tok.map((v, i) => ({ t: v, e: p.E[i] / 10, ok: p.ok[i] === "1" })).filter((d) => d.e > 0);
        svg.append("g").selectAll("circle").data(pts).join("circle").attr("cx", (d) => x(d.t)).attr("cy", (d) => y(d.e)).attr("r", 2.6).style("fill", m.color).style("opacity", 0.45);
      });
      root.parentElement.querySelector(".figure-sub").textContent = `Each dot is one answer by ${m.fl} ${m.q}. x: tokens generated; y: joules. Both axes on log scales. r = Pearson correlation.`;
    }
    sel.addEventListener("change", draw);
    draw();
    onResize(root, draw);
  })();

  // ================================================================== cite
  document.getElementById("copy-bib").addEventListener("click", (e) => {
    e.currentTarget.dataset.target = "bibtex";
    copyText(document.getElementById("bibtex").textContent, e.currentTarget);
  });
})();
