/* Interview QA editorial site — vanilla runtime. */
/* Depends on window.__SEARCH_INDEX__ being defined by a prior <script> tag. */

(function () {
  "use strict";

  /* ---------- Theme ---------- */
  function readTheme() {
    try {
      var t = localStorage.getItem("iqa.theme");
      if (t === "light" || t === "dark") return t;
    } catch (e) {}
    return "dark";
  }
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("iqa.theme", t); } catch (e) {}
    var btn = document.querySelector("[data-action='toggle-theme']");
    if (btn) btn.textContent = t === "dark" ? "☾  Light" : "☀  Dark";
  }

  function initThemeToggle() {
    applyTheme(readTheme());
    var btn = document.querySelector("[data-action='toggle-theme']");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var current = document.documentElement.getAttribute("data-theme") || "dark";
      applyTheme(current === "dark" ? "light" : "dark");
    });
  }

  /* ---------- Reading progress ---------- */
  function initProgress() {
    var bar = document.querySelector(".progress > i");
    if (!bar) return;
    var article = document.querySelector("article.content");
    if (!article) return;
    function update() {
      var rect = article.getBoundingClientRect();
      var total = rect.height - window.innerHeight;
      if (total <= 0) { bar.style.width = "100%"; return; }
      var passed = Math.max(0, Math.min(total, -rect.top));
      bar.style.width = (passed / total * 100).toFixed(2) + "%";
    }
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  /* ---------- Sticky TOC highlighting ---------- */
  function initTocHighlight() {
    /* New ID schema:
         h2 id = "v<NN>-q-<n>"          e.g. v04-q-2
         h3 id = "v<NN>-q-<n>-s-<m>"    e.g. v04-q-2-s-3
       data-q on left TOC links matches full h2 id.
       data-sec on right TOC links is the numeric section index ("1".."8"). */
    var leftLinks = document.querySelectorAll("aside.toc:not(.right) a[data-q]");
    var rightLinks = document.querySelectorAll("aside.toc.right a[data-sec]");
    if (!leftLinks.length && !rightLinks.length) return;

    function setActiveQ(qid) {
      leftLinks.forEach(function (l) {
        l.classList.toggle("active", l.getAttribute("data-q") === qid);
      });
      var base = qid.replace(/-s-\d+$/, "");
      rightLinks.forEach(function (a) {
        a.href = "#" + base + "-s-" + a.getAttribute("data-sec");
      });
    }
    function setActiveSec(secN) {
      rightLinks.forEach(function (l) {
        l.classList.toggle("active", l.getAttribute("data-sec") === secN);
      });
    }

    var headings = document.querySelectorAll("article.content h2[id]");
    var qHeadings = [];
    headings.forEach(function (h) {
      if (/^v\d{2}-q-\d+$/.test(h.id)) qHeadings.push(h);
    });
    var qio = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) setActiveQ(e.target.id);
      });
    }, { rootMargin: "0px 0px -70% 0px" });
    qHeadings.forEach(function (h) { qio.observe(h); });

    var subheadings = [];
    document.querySelectorAll("article.content h3[id]").forEach(function (h) {
      if (/^v\d{2}-q-\d+-s-\d+$/.test(h.id)) subheadings.push(h);
    });
    var rio = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var parts = e.target.id.split("-");
        setActiveSec(parts[parts.length - 1]);
      });
    }, { rootMargin: "0px 0px -70% 0px" });
    subheadings.forEach(function (h) { rio.observe(h); });
  }

  /* ---------- Code block copy buttons ---------- */
  function initCodeCopy() {
    document.querySelectorAll("article.content pre").forEach(function (pre) {
      if (pre.querySelector(".copy-btn")) return;
      var btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.type = "button";
      btn.textContent = "Copy";
      btn.addEventListener("click", function () {
        var code = pre.querySelector("code");
        var text = code ? code.innerText : pre.innerText;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(function () {
            btn.textContent = "Copied";
            setTimeout(function () { btn.textContent = "Copy"; }, 1600);
          });
        } else {
          var ta = document.createElement("textarea");
          ta.value = text; document.body.appendChild(ta);
          ta.select(); document.execCommand("copy");
          document.body.removeChild(ta);
          btn.textContent = "Copied";
          setTimeout(function () { btn.textContent = "Copy"; }, 1600);
        }
      });
      pre.appendChild(btn);
    });
  }

  /* ---------- Anchor link injection on Q headings ---------- */
  function initAnchorLinks() {
    document.querySelectorAll("article.content h2[id]").forEach(function (h) {
      if (!/^v\d{2}-q-\d+$/.test(h.id)) return;
      if (h.querySelector(".anchor")) return;
      var a = document.createElement("a");
      a.className = "anchor";
      a.href = "#" + h.id;
      a.textContent = "#";
      a.title = "复制链接";
      a.addEventListener("click", function (ev) {
        ev.preventDefault();
        var url = location.origin + location.pathname + "#" + h.id;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(url);
        }
        history.replaceState(null, "", "#" + h.id);
      });
      h.appendChild(a);
    });
  }

  /* ---------- Command palette ---------- */
  function fuzzyMatch(needle, hay) {
    if (!needle) return 0;
    var n = needle.toLowerCase();
    var h = hay.toLowerCase();
    if (h === n) return 100;
    if (h.indexOf(n) >= 0) return 80 - Math.min(40, h.indexOf(n));
    /* simple subsequence score */
    var ni = 0;
    for (var i = 0; i < h.length && ni < n.length; i++) {
      if (h[i] === n[ni]) ni++;
    }
    if (ni < n.length) return 0;
    return Math.max(1, 30 - (h.length - n.length) / 4);
  }

  function search(idx, query) {
    if (!query) {
      return idx.slice(0, 8).map(function (e) { return { e: e, score: 1 }; });
    }
    /* Direct numeric Q-id shortcut: '47', 'q47', 'Q47'. */
    var num = query.replace(/^[Qq]\s*/, "").trim();
    if (/^\d+$/.test(num)) {
      var n = parseInt(num, 10);
      var hit = idx.find(function (e) { return e.q === n; });
      if (hit) return [{ e: hit, score: 200 }];
    }
    var scored = [];
    idx.forEach(function (e) {
      var s = 0;
      s += 3 * fuzzyMatch(query, e.title);
      s += 2 * fuzzyMatch(query, e.lede);
      s += fuzzyMatch(query, e.vol_title);
      if (s > 0) scored.push({ e: e, score: s });
    });
    scored.sort(function (a, b) { return b.score - a.score; });
    return scored.slice(0, 8);
  }

  function ensurePalette() {
    var existing = document.querySelector(".cmdk-overlay");
    if (existing) return existing;
    var html =
      '<div class="cmdk-overlay" role="dialog" aria-modal="true">' +
      '  <div class="cmdk">' +
      '    <input type="text" placeholder="搜索题号、关键词、卷名…  (按 Esc 关闭)" />' +
      '    <ul class="cmdk-results"></ul>' +
      '  </div>' +
      '</div>';
    var wrap = document.createElement("div");
    wrap.innerHTML = html;
    document.body.appendChild(wrap.firstChild);
    return document.querySelector(".cmdk-overlay");
  }

  function renderPaletteResults(ul, results) {
    if (!results.length) {
      ul.innerHTML = '<li class="cmdk-empty">没有匹配项</li>';
      return;
    }
    ul.innerHTML = results.map(function (r, i) {
      var e = r.e;
      var anchor = e.anchor || ("v" + (e.vol_num < 10 ? "0" : "") + e.vol_num + "-q-" + e.q);
      return '<li data-href="' + e.vol + '.html#' + anchor + '"' +
             (i === 0 ? ' class="active"' : '') + '>' +
             '<span class="qchip">Vol ' + e.vol_num + ' · Q' + e.q + '</span>' +
             '<span class="ttl">' + escapeHtml(e.title) + '</span>' +
             '<span class="vol">' + escapeHtml(e.vol_title) + '</span>' +
             '</li>';
    }).join("");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function initPalette() {
    var idx = window.__SEARCH_INDEX__ || [];

    function open() {
      var overlay = ensurePalette();
      overlay.classList.add("open");
      var input = overlay.querySelector("input");
      var ul = overlay.querySelector(".cmdk-results");
      input.value = "";
      renderPaletteResults(ul, search(idx, ""));
      input.focus();
    }
    function close() {
      var overlay = document.querySelector(".cmdk-overlay");
      if (overlay) overlay.classList.remove("open");
    }
    function activate(li) {
      var href = li.getAttribute("data-href");
      if (href) location.href = href;
    }

    document.addEventListener("keydown", function (ev) {
      if ((ev.key === "k" || ev.key === "K") && (ev.metaKey || ev.ctrlKey)) {
        ev.preventDefault();
        open();
        return;
      }
      var overlay = document.querySelector(".cmdk-overlay.open");
      if (!overlay) return;
      if (ev.key === "Escape") { ev.preventDefault(); close(); return; }
      if (ev.key === "Enter") {
        ev.preventDefault();
        var active = overlay.querySelector(".cmdk-results li.active");
        if (active) activate(active);
        return;
      }
      if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
        ev.preventDefault();
        var items = overlay.querySelectorAll(".cmdk-results li[data-href]");
        if (!items.length) return;
        var idx0 = -1;
        items.forEach(function (li, i) { if (li.classList.contains("active")) idx0 = i; });
        var next = ev.key === "ArrowDown" ? idx0 + 1 : idx0 - 1;
        if (next < 0) next = items.length - 1;
        if (next >= items.length) next = 0;
        items.forEach(function (li, i) { li.classList.toggle("active", i === next); });
        items[next].scrollIntoView({ block: "nearest" });
      }
    });

    document.addEventListener("click", function (ev) {
      var overlay = ev.target.closest(".cmdk-overlay");
      if (overlay && ev.target === overlay) { close(); return; }
      var li = ev.target.closest(".cmdk-results li[data-href]");
      if (li) activate(li);
    });

    document.addEventListener("input", function (ev) {
      if (!ev.target.closest(".cmdk")) return;
      var overlay = document.querySelector(".cmdk-overlay.open");
      if (!overlay) return;
      var ul = overlay.querySelector(".cmdk-results");
      renderPaletteResults(ul, search(idx, ev.target.value.trim()));
    });

    var trigger = document.querySelector("[data-action='open-palette']");
    if (trigger) trigger.addEventListener("click", open);
  }

  /* ---------- Boot ---------- */
  function boot() {
    initThemeToggle();
    initAnchorLinks();
    initCodeCopy();
    initTocHighlight();
    initProgress();
    initPalette();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
