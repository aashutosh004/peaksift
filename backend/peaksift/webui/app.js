/* PeakSift web app — plain JavaScript, no build step.
 *
 * Routes (hash):
 *   #/                 new clips: paste a link, drop a file, or start OBS input
 *   #/videos           every video and live session
 *   #/job/<id>         one video: progress, the moments strip, its clips
 *   #/live/<id>        a running live session
 *   #/edit/<clipId>    caption editor and subtitle downloads
 *   #/settings         AI judge, transcription, defaults
 */
(() => {
  "use strict";

  const app = document.getElementById("app");
  let pollTimer = null;
  let captionStyles = [];

  // ------------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------------

  const h = (tag, attrs = {}, ...children) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false || child === "") continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  };

  const ICONS = {
    download: "M12 4v11m0 0-4-4m4 4 4-4M5 19.5h14",
    upload: "M12 19V8m0 0-4 4m4-4 4 4M5 4.5h14",
    edit: "M4 20h4L19 9l-4-4L4 16v4ZM13 7l4 4",
    redo: "M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4.5H15",
    more: "M5.5 12h.01M12 12h.01M18.5 12h.01",
    back: "M15 18l-6-6 6-6",
    chevron: "M9 6l6 6-6 6",
    layout: "M7 3v14h14M3 7h14v14",
    play: "M8 5.5v13l10.5-6.5L8 5.5Z",
    stop: "M7 7h10v10H7z",
    check: "M5 12.5l4.5 4.5L19 7.5",
    file: "M14 3.5H7a1.5 1.5 0 0 0-1.5 1.5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8L14 3.5ZM14 3.5V8h4.5",
    subtitles: "M4 6.5h16v11H4zM7.5 13.5h4M13.5 13.5h3M7.5 10.5h2M11.5 10.5h5",
    broadcast: "M12 12h.01M8.5 15.5a5 5 0 0 1 0-7M15.5 8.5a5 5 0 0 1 0 7M5.6 18.4a9 9 0 0 1 0-12.8M18.4 5.6a9 9 0 0 1 0 12.8",
  };
  const icon = (name, extra = "") => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", name === "more" ? "3" : "1.75");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    if (extra) svg.setAttribute("class", extra);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", ICONS[name]);
    svg.append(path);
    return svg;
  };

  async function api(path, options = {}) {
    const init = { ...options };
    if (init.json !== undefined) {
      init.body = JSON.stringify(init.json);
      init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
      delete init.json;
    }
    const res = await fetch(path, init);
    if (res.status === 204) return null;
    let data = null;
    try { data = await res.json(); } catch { data = null; }
    if (!res.ok) {
      const detail = data && data.detail;
      const message = typeof detail === "string" ? detail : (detail && detail.message) || res.statusText;
      const err = new Error(message);
      err.hint = detail && detail.hint;
      throw err;
    }
    return data;
  }

  function toast(message, bad = false) {
    const t = document.getElementById("toast");
    t.textContent = message;
    t.className = "toast" + (bad ? " bad" : "");
    t.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { t.hidden = true; }, 4500);
  }

  const hms = (s) => {
    s = Math.max(0, Math.floor(s || 0));
    const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
    return (hh ? hh + ":" + String(mm).padStart(2, "0") : mm) + ":" + String(ss).padStart(2, "0");
  };
  const ago = (iso) => {
    if (!iso) return "";
    const diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
    return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  };
  const PLATFORM_NAMES = { youtube: "YouTube", twitch: "Twitch", kick: "Kick", tiktok: "TikTok", instagram: "Instagram", facebook: "Facebook" };
  const platformName = (p) => PLATFORM_NAMES[p] || (p ? p[0].toUpperCase() + p.slice(1) : "");
  const STATUS_TEXT = { done: "Ready", running: "Working", queued: "Waiting", failed: "Failed", cancelled: "Stopped" };
  const statusChip = (status) => h("span", { class: `status ${status}` }, STATUS_TEXT[status] || status);
  const bar = (fraction) => h("div", { class: "bar", role: "progressbar", "aria-valuenow": Math.round((fraction || 0) * 100), "aria-valuemin": 0, "aria-valuemax": 100 },
    h("i", { style: `width:${Math.round((fraction || 0) * 100)}%` }));
  const btn = (label, attrs = {}, iconName = null) =>
    h(attrs.href ? "a" : "button", { type: attrs.href ? undefined : "button", ...attrs, class: `btn ${attrs.class || ""}`.trim() },
      iconName ? icon(iconName) : "", label);

  function seg(options, value, onChange) {
    const wrap = h("div", { class: "seg", role: "group" });
    const render = (current) => {
      wrap.replaceChildren(...options.map(([v, label]) =>
        h("button", { type: "button", class: v === current ? "on" : "", "aria-pressed": String(v === current), onclick: () => { render(v); onChange(v); } }, label)));
    };
    render(value);
    return wrap;
  }
  const field = (label, control, hint) => h("label", { class: "field" }, h("span", { class: "label" }, label), control, hint ? h("span", { class: "hint" }, hint) : "");
  const group = (label, control, hint) => h("div", { class: "field" }, h("span", { class: "label" }, label), control, hint ? h("span", { class: "hint" }, hint) : "");
  const check = (checked, text, onChange) => h("label", { class: "check" },
    h("input", { type: "checkbox", checked, onchange: (e) => onChange(e.target.checked) }), h("span", {}, text));

  function menu(trigger, items) {
    const wrap = h("div", { class: "menu-wrap" });
    let panel = null;
    const close = () => { if (panel) { panel.remove(); panel = null; document.removeEventListener("click", outside, true); } };
    const outside = (e) => { if (!wrap.contains(e.target)) close(); };
    trigger.setAttribute("aria-haspopup", "menu");
    trigger.addEventListener("click", () => {
      if (panel) return close();
      panel = h("div", { class: "menu", role: "menu" }, items.map((item) => item === "-" ? h("hr") :
        h(item.href ? "a" : "button", { role: "menuitem", href: item.href, download: item.download, onclick: () => { close(); if (item.onclick) item.onclick(); } },
          item.icon ? icon(item.icon) : "", item.label)));
      wrap.append(panel);
      setTimeout(() => document.addEventListener("click", outside, true));
    });
    wrap.append(trigger);
    return wrap;
  }

  function stopPolling() { clearTimeout(pollTimer); pollTimer = null; }
  function poll(fn, ms) {
    stopPolling();
    const tick = async () => {
      try { await fn(); } catch (err) { console.warn(err); }
      if (pollTimer !== null) pollTimer = setTimeout(tick, ms);
    };
    pollTimer = setTimeout(tick, 0);
  }

  // Windows can't replace a clip file the page is still streaming.
  function releaseVideos() {
    document.querySelectorAll("video").forEach((v) => { v.pause(); v.removeAttribute("src"); v.load(); });
  }

  // ------------------------------------------------------------------------
  // New clips
  // ------------------------------------------------------------------------

  function hindiModelSelect(obj, key, models) {
    const known = models.some((m) => m.key === obj[key] || m.repo === obj[key]);
    const choices = models.map((m) => [m.key, `${m.label} (${m.download_gb} GB)`]);
    if (obj[key] && !known) choices.push([obj[key], `${obj[key]} (custom)`]);
    const note = h("span", { class: "hint" });
    const describe = () => {
      const m = models.find((x) => x.key === obj[key] || x.repo === obj[key]);
      note.textContent = m ? `${m.note} Based on ${m.based_on}; ${m.license}.` : "";
    };
    const sel = h("select", { onchange: (e) => { obj[key] = e.target.value; describe(); } },
      choices.map(([v, l]) => h("option", { value: v, selected: obj[key] === v || (models.find((m) => m.key === v) || {}).repo === obj[key] }, l)));
    describe();
    return [sel, note];
  }

  async function viewNew() {
    let settings = {};
    try { settings = await api("/api/settings"); } catch { /* defaults below */ }
    if (!captionStyles.length) {
      try { captionStyles = await api("/api/caption-styles"); } catch { captionStyles = []; }
    }
    const det = settings.detection || {}, sel = settings.selection || {}, exp = settings.export || {}, wh = settings.whisper || {};
    const liveDefaults = settings.live || {};
    const opts = {
      profile: det.profile || "auto",
      view: sel.view || "overall",
      per_bucket: sel.per_bucket || 3,
      max_clips: (settings.clips || {}).max_clips || 10,
      layout: exp.layout || "auto",
      split_ratio: exp.split_ratio || 0.35,
      facecam_position: exp.facecam_position || "top",
      two_person: exp.two_person || "split",
      auto_switch_layout: exp.auto_switch_layout !== false,
      llm_rerank: det.llm_rerank || "auto",
      ratio: exp.ratio || "9:16",
      caption_style: exp.caption_style || "bold_pop",
      hindi_model: wh.hindi_model || "hinglish-prime",
      caption_source: wh.caption_source || "whisper",
    };
    const live = { backfill: liveDefaults.backfill !== false, clips_per_hour: liveDefaults.clips_per_hour || 4, max_hours: "" };

    const jobSettings = () => {
      const s = {
        profile: opts.profile, view: opts.view, layout: opts.layout, split_ratio: opts.split_ratio,
        facecam_position: opts.facecam_position, two_person: opts.two_person, llm_rerank: opts.llm_rerank,
        auto_switch_layout: opts.auto_switch_layout, ratio: opts.ratio, caption_style: opts.caption_style,
        hindi_model: opts.hindi_model, caption_source: opts.caption_source,
      };
      if (opts.view === "overall") s.max_clips = Number(opts.max_clips);
      else s.per_bucket = Number(opts.per_bucket);
      return s;
    };
    const liveOptions = () => ({
      backfill: live.backfill,
      clips_per_hour: Number(live.clips_per_hour) || null,
      max_hours: live.max_hours ? Number(live.max_hours) : null,
    });

    const link = h("input", { type: "url", name: "link", placeholder: "Paste a link to a video or stream", "aria-label": "Video or stream link", autofocus: true, autocomplete: "off" });
    const go = btn("Find clips", { type: "submit", class: "btn-primary btn-lg" });
    const status = h("div", { class: "progress-line", "aria-live": "polite" });
    const showError = (message, hint) => status.replaceChildren(h("div", { class: "notice error" },
      h("div", {}, h("strong", {}, message), hint ? h("div", { class: "hint", style: "margin-top:4px" }, hint) : "")));

    async function startLink(url) {
      go.disabled = true;
      status.replaceChildren(h("span", { class: "muted" }, "Checking the link…"));
      try {
        const { task_id } = await api("/api/ui/start", { method: "POST", json: { url, settings: jobSettings(), ...liveOptions() } });
        poll(async () => {
          const task = await api(`/api/ui/tasks/${task_id}`);
          if (task.status === "live") { stopPolling(); location.hash = `#/live/${task.live_id}`; }
          else if (task.status === "queued") { stopPolling(); location.hash = `#/job/${task.job_id}`; }
          else if (task.status === "failed") { stopPolling(); go.disabled = false; showError(task.error, task.hint); }
          else if (task.status === "downloading") {
            status.replaceChildren(h("div", { class: "num" }, `Downloading the video and its chat… ${Math.round(task.progress * 100)}%`), bar(task.progress));
          }
        }, 1000);
      } catch (err) { go.disabled = false; showError(err.message, err.hint); }
    }

    function uploadFile(file) {
      status.replaceChildren(`Uploading ${file.name}…`);
      const form = new FormData();
      form.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/sources/upload");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) status.replaceChildren(h("div", { class: "num" }, `Uploading ${file.name}… ${Math.round(e.loaded / e.total * 100)}%`), bar(e.loaded / e.total));
      };
      xhr.onload = async () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch { /* handled below */ }
        if (xhr.status >= 300) {
          const d = data.detail || {};
          showError(typeof d === "string" ? d : (d.message || "The upload didn't go through."), d.hint);
          return;
        }
        try {
          const job = await api("/api/jobs", { method: "POST", json: { source_id: data.id, settings: jobSettings() } });
          location.hash = `#/job/${job.id}`;
        } catch (err) { showError(err.message, err.hint); }
      };
      xhr.onerror = () => showError("The upload didn't go through.", "Check that PeakSift is still running, then try again.");
      xhr.send(form);
    }

    const fileInput = h("input", { type: "file", accept: "video/*,audio/*", hidden: true, onchange: (e) => e.target.files[0] && uploadFile(e.target.files[0]) });
    const form = h("form", { class: "dropzone", onsubmit: (e) => { e.preventDefault(); const url = link.value.trim(); if (url) startLink(url); } },
      h("div", { class: "linkbox" }, link, go),
      h("div", { class: "alt" }, "or", btn("Upload a video file", { class: "btn-sm", onclick: () => fileInput.click() }, "upload"), h("span", {}, "or drop one here")),
      fileInput, status);
    form.addEventListener("dragover", (e) => { e.preventDefault(); form.classList.add("over"); });
    form.addEventListener("dragleave", (e) => { if (!form.contains(e.relatedTarget)) form.classList.remove("over"); });
    form.addEventListener("drop", (e) => { e.preventDefault(); form.classList.remove("over"); if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); });

    // Options
    const splitOut = h("span", { class: "num" }, `${Math.round(opts.split_ratio * 100)}%`);
    const countLabel = h("span", { class: "label" });
    const countInput = h("input", { type: "number", min: 1, max: 50, oninput: (e) => { if (opts.view === "overall") opts.max_clips = e.target.value; else opts.per_bucket = e.target.value; } });
    const syncCount = () => {
      countLabel.textContent = opts.view === "overall" ? "Number of clips" : opts.view === "per_hour" ? "Clips per hour" : "Clips per chapter";
      countInput.value = opts.view === "overall" ? opts.max_clips : opts.per_bucket;
    };
    syncCount();
    const styleSelect = h("select", { onchange: (e) => { opts.caption_style = e.target.value; } },
      (captionStyles.length ? captionStyles : [{ key: "bold_pop", label: "Bold pop" }]).map((s) =>
        h("option", { value: s.key, selected: s.key === opts.caption_style }, s.label || s.key)));
    const optgroup = (title, note, ...fields) => h("div", { class: "optgroup" },
      h("div", { class: "optgroup-title" }, h("h3", {}, title), note ? h("p", { class: "hint" }, note) : ""),
      h("div", { class: "fields" }, fields));

    const options = h("details", { class: "disclosure" },
      h("summary", {}, icon("chevron", "chev"), "Options", h("span", { class: "summary-note" }, "Defaults come from Settings")),
      optgroup("Finding moments", "How PeakSift scores the video and which moments become clips.",
        group("Type of video", seg([["auto", "Detect"], ["gaming", "Gaming"], ["just_chatting", "Just chatting"], ["podcast", "Podcast"]], opts.profile, (v) => { opts.profile = v; })),
        group("Which clips", seg([["overall", "Best overall"], ["per_hour", "Best of each hour"], ["per_chapter", "Best of each chapter"]], opts.view, (v) => { opts.view = v; syncCount(); })),
        h("div", { class: "two" }, h("label", { class: "field" }, countLabel, countInput),
          group("AI judge", seg([["auto", "When available"], ["on", "Always"], ["off", "Off"]], opts.llm_rerank, (v) => { opts.llm_rerank = v; }),
            "Re-ranks moments and writes titles."))),
      optgroup("Framing", "How the wide video becomes a vertical clip.",
        group("Layout", seg([["auto", "Automatic"], ["gaming", "Facecam over gameplay"], ["full", "Full scene"], ["track", "Follow the speaker"], ["center", "Centre"]], opts.layout, (v) => { opts.layout = v; })),
        check(opts.auto_switch_layout, "Show the whole game while the webcam is off, and the facecam again when it's back", (v) => { opts.auto_switch_layout = v; }),
        h("div", { class: "two" },
          h("label", { class: "field" }, h("span", { class: "label" }, "Facecam height ", splitOut),
            h("input", { type: "range", min: 25, max: 70, value: Math.round(opts.split_ratio * 100), oninput: (e) => { opts.split_ratio = e.target.value / 100; splitOut.textContent = `${e.target.value}%`; } })),
          group("Facecam position", seg([["top", "Top"], ["bottom", "Bottom"]], opts.facecam_position, (v) => { opts.facecam_position = v; }))),
        h("div", { class: "two" },
          group("Two people on screen", seg([["split", "Split screen"], ["track", "Follow the speaker"]], opts.two_person, (v) => { opts.two_person = v; })),
          group("Shape", seg([["9:16", "9:16"], ["1:1", "1:1"], ["16:9", "16:9"]], opts.ratio, (v) => { opts.ratio = v; })))),
      optgroup("Captions", "",
        h("div", { class: "two" }, field("Style", styleSelect),
          group("Words come from", seg([["whisper", "This computer"], ["gemini", "Gemini"]], opts.caption_source, (v) => { opts.caption_source = v; }),
            "Gemini receives only the finished clips' audio.")),
        field("Speech model for Hindi", ...hindiModelSelect(opts, "hindi_model", settings.hindi_models || []))),
      optgroup("If the link is live", "YouTube, Twitch and Kick streams that are live right now.",
        h("div", { class: "three" },
          field("Clips per hour", h("input", { type: "number", min: 1, max: 20, value: live.clips_per_hour, oninput: (e) => { live.clips_per_hour = e.target.value; } })),
          field("Stop after (hours)", h("input", { type: "number", min: 0.1, step: 0.5, placeholder: "When the stream ends", oninput: (e) => { live.max_hours = e.target.value; } }))),
        check(live.backfill, "Also clip what happened before I joined", (v) => { live.backfill = v; })));

    // OBS
    const chatInput = h("input", { type: "url", placeholder: "https://www.twitch.tv/yourchannel" });
    const obsStart = btn("Start listening for OBS", { class: "btn-sm", onclick: async () => {
      obsStart.disabled = true;
      try {
        const { live_id } = await api("/api/ui/live", { method: "POST", json: { url: "obs", chat_url: chatInput.value.trim() || null, settings: jobSettings(), backfill: false, clips_per_hour: Number(live.clips_per_hour) || null } });
        location.hash = `#/live/${live_id}`;
      } catch (err) { toast(err.message, true); obsStart.disabled = false; }
    } }, "broadcast");
    const obs = h("details", { class: "disclosure" },
      h("summary", {}, icon("chevron", "chev"), "Clip your own stream from OBS"),
      h("div", { class: "optgroup" },
        h("div", { class: "optgroup-title" }, h("p", { class: "hint" }, "PeakSift records what OBS sends it and clips as you stream. It listens on this computer only.")),
        h("div", { class: "fields" },
          h("p", { class: "small" }, "In OBS, open Settings › Stream, choose Custom, and enter server ", h("code", {}, "rtmp://127.0.0.1:1935/live"),
            " with stream key ", h("code", {}, "stream"), ". Start listening here first, then start streaming in OBS."),
          field("Your channel, to read its chat (optional)", chatInput),
          h("div", {}, obsStart))));

    const recent = h("div", { class: "vlist" }, h("p", { class: "loading" }, "Loading…"));
    app.replaceChildren(
      h("div", { class: "start" },
        h("h1", {}, "Find the moments worth clipping"),
        h("p", { class: "lede" }, "Paste a video, a finished stream or one that's live right now. PeakSift watches, listens and reads the chat, then cuts captioned vertical clips of the best parts."),
        form, options, obs,
        h("div", { class: "section" }, h("div", { class: "section-head" }, h("h2", {}, "Recent"), h("a", { href: "#/videos", class: "small" }, "All videos")), recent)));
    renderVideos(recent, 5);
  }

  // ------------------------------------------------------------------------
  // Videos
  // ------------------------------------------------------------------------

  async function renderVideos(target, limit) {
    const [jobs, lives] = await Promise.all([api(`/api/jobs?limit=${limit}`), api("/api/ui/live").catch(() => [])]);
    const rows = [];
    for (const l of lives.filter((x) => x.running)) {
      rows.push(h("a", { class: "vrow", href: `#/live/${l.id}` },
        h("span", { class: "vtitle" }, l.title || l.url),
        h("span", { class: "vmeta num" }, h("span", {}, platformName(l.platform) || (l.obs ? "OBS" : "")), h("span", {}, `${hms(l.recorded_s)} recorded`), h("span", {}, `${l.clips} clips so far`)),
        h("span", { class: "vstate" }, h("span", { class: "live-badge" }, "Live"))));
    }
    for (const j of jobs) {
      const src = j.source || {};
      const isLive = j.current_stage === "live";
      const active = j.status === "running" || j.status === "queued";
      rows.push(h("a", { class: "vrow", href: `#/job/${j.id}` },
        h("span", { class: "vtitle" }, src.title || "Untitled video"),
        h("span", { class: "vmeta num" },
          src.platform ? h("span", {}, platformName(src.platform) + (isLive ? " live" : "")) : h("span", {}, src.type === "upload" ? "File" : ""),
          src.duration_s ? h("span", {}, hms(src.duration_s)) : "",
          h("span", {}, ago(j.created_at))),
        h("span", { class: "vstate" }, statusChip(j.status), active && !isLive ? bar(j.progress) : "")));
    }
    target.replaceChildren(...(rows.length ? rows : [h("p", { class: "empty" }, "No videos yet. Paste a link on the New clips page to make your first ones.")]));
  }

  async function viewVideos() {
    const list = h("div", { class: "vlist" }, h("p", { class: "loading" }, "Loading…"));
    app.replaceChildren(
      h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Videos"), h("p", { class: "lede" }, "Everything you've clipped, newest first.")),
        btn("New clips", { href: "#/", class: "btn-primary" })),
      list);
    poll(() => renderVideos(list, 50), 4000);
  }

  // ------------------------------------------------------------------------
  // One video
  // ------------------------------------------------------------------------

  const DRIVER_LABELS = {
    loudness: "a volume spike", flux: "a sudden sound", shout: "shouting", motion: "on-screen action", scene: "a scene change",
    mouth_open: "a big vocal reaction", face_surprise: "a surprised face", face_joy: "laughing", head_motion: "a physical reaction",
    speech_rate: "rapid talking", exclaim: "exclamations", chat_rate: "chat exploding", chat_unique: "many chatters reacting",
    chat_laugh: "chat laughing", chat_hype: "chat hype", chat_shock: "chat in shock", chat_events: "subs and donations",
    heatmap: "the most-replayed part", comments: "viewers' timestamps",
  };
  const STAGES = [["prepare", "Prepare"], ["transcribe", "Transcribe"], ["signals", "Read signals"], ["highlights", "Find moments"], ["reframe", "Frame"], ["captions", "Captions"], ["export", "Export"]];

  function stepper(current) {
    const at = STAGES.findIndex(([key]) => key === current);
    return h("div", { class: "stepper", "aria-label": "Progress" }, STAGES.map(([key, label], i) =>
      h("span", { class: i < at ? "done" : i === at ? "now" : "" }, i < at ? icon("check") : "", label)));
  }

  function whyText(clip) {
    const drivers = ((clip.signals || {}).drivers || []).filter(([, share]) => share >= 0.1).slice(0, 3).map(([name]) => DRIVER_LABELS[name] || name);
    if (!drivers.length) return clip.reason || "";
    const list = drivers.length > 1 ? `${drivers.slice(0, -1).join(", ")} and ${drivers[drivers.length - 1]}` : drivers[0];
    return `Picked for ${list}.`;
  }

  async function rerender(jobId, ranks) {
    releaseVideos();
    try {
      await api(`/api/ui/jobs/${jobId}/rerender`, { method: "POST", json: { ranks: ranks || null } });
      toast(ranks ? `Re-rendering clip ${ranks[0]}…` : "Re-rendering every clip…");
      setTimeout(() => route(), 800);
    } catch (err) { toast(err.message, true); }
  }

  function clipCard(clip, job) {
    const exp = (clip.exports || [])[0];
    const ratioClass = { "1:1": "r1x1", "16:9": "r16x9" }[exp ? exp.ratio : "9:16"] || "";
    const more = h("button", { type: "button", class: "btn btn-sm btn-icon btn-quiet", "aria-label": `More for clip ${clip.rank}` }, icon("more"));
    return h("article", { class: `clip ${ratioClass}`, id: `clip-${clip.id}` },
      h("div", { class: "tile" }, exp
        // "#t=0.5" makes the browser draw a real frame instead of a black tile.
        ? h("video", { src: `${exp.download_url}#t=0.5`, controls: true, preload: "metadata", playsinline: true, "aria-label": clip.title || `Clip ${clip.rank}` })
        : h("div", { class: "pending" }, job.status === "running" ? "Rendering…" : "Not rendered")),
      h("div", { class: "clip-top num" }, h("span", { class: "rank" }, `Clip ${clip.rank}`),
        h("span", {}, `${hms(clip.start_s)}–${hms(clip.end_s)}`)),
      h("h3", {}, clip.title || "Untitled moment"),
      h("p", { class: "why" }, whyText(clip)),
      h("div", { class: "actions" },
        exp ? btn("Download", { href: exp.download_url, download: true, class: "btn-sm btn-primary" }, "download") : "",
        btn("Edit captions", { href: `#/edit/${clip.id}`, class: "btn-sm" }),
        menu(more, [
          { label: "Re-render this clip", icon: "redo", onclick: () => rerender(job.id, [clip.rank]) },
          "-",
          { label: "Subtitles (.srt)", icon: "subtitles", href: `/api/ui/clips/${clip.id}/subtitles.srt`, download: true },
          { label: "Subtitles (.vtt)", icon: "subtitles", href: `/api/ui/clips/${clip.id}/subtitles.vtt`, download: true },
        ])));
  }

  function momentsStrip(data) {
    const duration = data.duration_s || 1;
    const pts = data.points || [];
    const strip = h("div", { class: "strip" });
    if (pts.length > 1) {
      const n = pts.length - 1;
      const line = pts.map((p, i) => `${(i / n * 1000).toFixed(1)},${(100 - p * 96).toFixed(1)}`).join(" L");
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "curve");
      svg.setAttribute("viewBox", "0 0 1000 100");
      svg.setAttribute("preserveAspectRatio", "none");
      svg.setAttribute("aria-hidden", "true");
      svg.innerHTML = `<line class="baseline" x1="0" y1="100" x2="1000" y2="100"/><path class="curve-fill" d="M0,100 L${line} L1000,100 Z"/><path class="curve-line" d="M${line}"/>`;
      strip.append(svg);
    }
    for (const c of data.clips) {
      const at = (c.peak_s ?? (c.start_s + c.end_s) / 2) / duration * 100;
      strip.append(h("button", {
        type: "button", class: "marker", style: `left:${Math.min(99, Math.max(1, at))}%`,
        title: `Clip ${c.rank}${c.title ? `: ${c.title}` : ""} (${hms(c.start_s)})`,
        "aria-label": `Go to clip ${c.rank}`,
        onclick: () => {
          const card = document.getElementById(`clip-${c.id}`);
          if (!card) return;
          card.scrollIntoView({ behavior: "smooth", block: "center" });
          card.classList.add("flash");
          setTimeout(() => card.classList.remove("flash"), 1400);
        },
      }, h("span", {}, String(c.rank))));
    }
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => h("span", {}, hms(duration * f)));
    return h("div", { class: "moments" },
      h("div", { class: "moments-head" }, h("h2", {}, "Moments"), h("p", {}, "How much was happening across the whole video. Numbers mark the clips.")),
      strip, h("div", { class: "axis" }, ticks));
  }

  async function viewJob(jobId) {
    const header = h("div");
    const progress = h("div");
    const body = h("div");
    app.replaceChildren(header, progress, body);
    let lastStatus = null;

    poll(async () => {
      const job = await api(`/api/jobs/${jobId}`);
      const src = job.source || {};
      const active = job.status === "running" || job.status === "queued";
      header.replaceChildren(h("div", { class: "page-head" },
        h("div", { style: "min-width:0" },
          h("a", { class: "back", href: "#/videos" }, icon("back"), "Videos"),
          h("h1", {}, src.title || "Untitled video"),
          h("div", { class: "meta num" }, statusChip(job.status), src.platform ? h("span", {}, platformName(src.platform)) : "",
            src.duration_s ? h("span", {}, hms(src.duration_s)) : "", h("span", {}, ago(job.created_at)))),
        h("div", { class: "actions" },
          btn("Adjust layout", { href: `/layout-picker?job=${job.id}`, target: "_blank", rel: "noopener" }, "layout"),
          btn("Re-render all", { onclick: () => rerender(job.id, null), disabled: active }, "redo"))));

      if (active) {
        const stage = (STAGES.find(([key]) => key === job.current_stage) || [null, job.current_stage === "rerender" ? "Re-rendering" : "Waiting to start"])[1];
        progress.replaceChildren(h("div", { class: "run-panel" },
          h("div", { class: "row num" }, h("span", {}, stage), h("span", { class: "muted" }, `${Math.round(job.progress * 100)}%`)),
          bar(job.progress), job.current_stage !== "rerender" ? stepper(job.current_stage) : ""));
      } else if (job.error) {
        progress.replaceChildren(h("div", { class: "notice error" }, h("div", {}, h("strong", {}, job.status === "failed" ? "This video didn't finish. " : "The last re-render didn't finish. "), job.error)));
      } else {
        progress.replaceChildren();
      }

      if (job.status !== lastStatus) {
        if (!active || lastStatus === null) await renderClips(job, body);
        lastStatus = job.status;
      }
      if (!active) stopPolling();
    }, 2000);
  }

  async function renderClips(job, target) {
    const [clips, moments] = await Promise.all([
      api(`/api/jobs/${job.id}/clips`),
      api(`/api/ui/jobs/${job.id}/moments`).catch(() => null),
    ]);
    if (!clips.length) {
      target.replaceChildren(job.status === "done" ? h("p", { class: "empty" }, "No clips were made from this video.") : "");
      return;
    }
    const groups = new Map();
    for (const c of clips) {
      const key = c.group_label || "";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(c);
    }
    const keys = [...groups.keys()].sort((a, b) => (a === "Snapshots") - (b === "Snapshots") || a.localeCompare(b, undefined, { numeric: true }));
    const sections = keys.map((key) => {
      const items = groups.get(key).sort((a, b) => a.rank - b.rank);
      return h("div", { class: "section" },
        h("div", { class: "section-head" }, h("h2", {}, key || `${items.length} clip${items.length === 1 ? "" : "s"}, best first`),
          key ? h("span", { class: "muted small" }, `${items.length} clip${items.length === 1 ? "" : "s"}`) : ""),
        h("div", { class: "clips" }, items.map((c) => clipCard(c, job))));
    });
    target.replaceChildren(moments && moments.points && moments.points.length ? h("div", { class: "section" }, momentsStrip(moments)) : "", ...sections);
  }

  // ------------------------------------------------------------------------
  // Caption editor
  // ------------------------------------------------------------------------

  const ORIGIN_TEXT = { edited: "Your edits", gemini: "Words from Gemini", whisper: "Automatic words" };

  async function viewEdit(clipId) {
    let data = await api(`/api/ui/clips/${clipId}/captions`);
    let words = data.words.map((w) => ({ ...w }));
    let selected = -1;
    let dirty = false;
    const clip = data.clip;
    const hasVideo = Boolean(data.video_url);

    const video = hasVideo
      ? h("video", { src: data.video_url, controls: true, preload: "auto", playsinline: true })
      : h("div", { class: "pending" }, "Not rendered yet");
    const origin = h("span");
    const unsaved = h("span", { class: "muted" });
    const wordsBox = h("div", { class: "words", role: "group", "aria-label": "Caption words" });
    const tools = h("div", { class: "wordtools" });
    const inputs = [];

    const measure = document.createElement("canvas").getContext("2d");
    measure.font = '400 17px Plex, system-ui, sans-serif';
    const fit = (input) => { input.style.width = `${Math.ceil(Math.max(12, measure.measureText(input.value).width) + 10)}px`; };
    const seek = (i) => { if (hasVideo && words[i]) video.currentTime = Math.max(0, words[i].start - clip.start_s) + 0.01; };
    const showOrigin = () => origin.replaceChildren(ORIGIN_TEXT[data.origin] || data.origin);

    function select(i) {
      selected = i;
      inputs.forEach((el, k) => el.classList.toggle("sel", k === i));
      renderTools();
    }
    function renderWords(focus = -1) {
      inputs.length = 0;
      wordsBox.replaceChildren(...words.map((w, i) => {
        const input = h("input", { type: "text", value: w.text, spellcheck: "false", "aria-label": `Word at ${(w.start - clip.start_s).toFixed(1)} seconds` });
        fit(input);
        input.addEventListener("input", () => { words[i].text = input.value; fit(input); setDirty(true); renderTools(); });
        input.addEventListener("focus", () => { select(i); if (!hasVideo || video.paused) seek(i); });
        input.addEventListener("keydown", (e) => {
          if (e.key === "Enter") { e.preventDefault(); (inputs[i + 1] || input).focus(); }
          else if (e.key === "Backspace" && input.value === "" && words.length > 1) { e.preventDefault(); removeWord(i); }
        });
        inputs.push(input);
        return input;
      }));
      if (!words.length) wordsBox.replaceChildren(h("span", { class: "muted" }, "This clip has no words."));
      if (focus >= 0 && inputs[focus]) inputs[focus].focus(); else select(-1);
    }
    function removeWord(i) { words.splice(i, 1); setDirty(true); renderWords(Math.min(Math.max(0, i - 1), words.length - 1)); }
    function insertAfter(i) {
      const w = words[i], next = words[i + 1];
      const start = w ? w.end : clip.start_s;
      let end = next ? next.start : start + 0.3;
      if (end <= start) end = start + 0.3;
      words.splice(i + 1, 0, { text: "", start, end, speaker: w ? w.speaker : null });
      setDirty(true);
      renderWords(i + 1);
    }
    function nudge(key, delta) {
      const w = words[selected];
      if (!w) return;
      w[key] = Math.max(0, Math.round((w[key] + delta) * 100) / 100);
      if (w.end < w.start) { if (key === "start") w.end = w.start; else w.start = w.end; }
      setDirty(true); renderTools(); seek(selected);
    }
    function renderTools() {
      const w = words[selected];
      if (!w) { tools.replaceChildren(h("span", { class: "muted" }, "Select a word to change its timing, remove it, or add one after it.")); return; }
      const nudger = (label, key) => h("span", { class: "nudge" }, h("span", { class: "muted" }, label),
        btn("−", { class: "btn-sm btn-icon", "aria-label": `${label} earlier`, onclick: () => nudge(key, -0.1) }),
        h("code", { class: "num" }, `${(w[key] - clip.start_s).toFixed(2)} s`),
        btn("+", { class: "btn-sm btn-icon", "aria-label": `${label} later`, onclick: () => nudge(key, 0.1) }));
      tools.replaceChildren(
        h("span", { class: "nudge" }, nudger("Starts", "start"), nudger("Ends", "end")),
        h("span", { class: "actions" },
          hasVideo ? btn("Play from here", { class: "btn-sm btn-quiet", onclick: () => { seek(selected); video.play(); } }, "play") : "",
          btn("Add word after", { class: "btn-sm btn-quiet", onclick: () => insertAfter(selected) }),
          btn("Remove", { class: "btn-sm btn-quiet btn-danger", onclick: () => removeWord(selected) })));
    }
    if (hasVideo) {
      video.addEventListener("timeupdate", () => {
        const t = video.currentTime + clip.start_s;
        inputs.forEach((el, i) => el.classList.toggle("now", Boolean(words[i]) && t >= words[i].start && t < words[i].end + 0.05));
      });
    }

    async function save() {
      data = await api(`/api/ui/clips/${clipId}/captions`, { method: "PUT", json: { words } });
      words = data.words.map((w) => ({ ...w }));
      showOrigin(); renderWords(); setDirty(false);
    }
    const saveBtn = btn("Save", { class: "btn-primary", onclick: () => save().then(() => toast("Captions saved. Re-render the clip to put them in the video.")).catch((err) => toast(err.message, true)) });
    const renderBtn = btn("Save and re-render", { onclick: async () => {
      try {
        if (dirty) await save();
        releaseVideos();
        await api(`/api/ui/jobs/${clip.job_id}/rerender`, { method: "POST", json: { ranks: [clip.rank] } });
        toast(`Re-rendering clip ${clip.rank} with your captions…`);
        location.hash = `#/job/${clip.job_id}`;
      } catch (err) { toast(err.message, true); }
    } }, "redo");
    const resetBtn = btn("Reset to automatic", { class: "btn-quiet", onclick: async () => {
      if (!confirm("Discard your caption edits for this clip and go back to the automatic words?")) return;
      try {
        data = await api(`/api/ui/clips/${clipId}/captions`, { method: "DELETE" });
        words = data.words.map((w) => ({ ...w }));
        showOrigin(); renderWords(); setDirty(false);
        toast("Back to the automatic words.");
      } catch (err) { toast(err.message, true); }
    } });
    function setDirty(value) { dirty = value; unsaved.textContent = dirty ? "Unsaved changes" : ""; saveBtn.disabled = !dirty; }

    const download = (fmt) => btn(`.${fmt}`, { href: data.subtitles[fmt], class: "btn-sm", onclick: async (e) => {
      if (!dirty) return;
      e.preventDefault();
      try { await save(); location.href = data.subtitles[fmt]; } catch (err) { toast(err.message, true); }
    } }, "download");

    const findInput = h("input", { type: "text", placeholder: "Misheard word, e.g. lakshy", "aria-label": "Word to replace" });
    const replaceInput = h("input", { type: "text", placeholder: "Correct word, e.g. Lakshya", "aria-label": "Replacement" });
    const replaceBtn = btn("Replace in all clips", { onclick: async () => {
      const find = findInput.value.trim();
      if (!find) return;
      try {
        if (dirty) await save();
        const out = await api(`/api/ui/jobs/${clip.job_id}/replace`, { method: "POST", json: { find, replace: replaceInput.value.trim() } });
        data = await api(`/api/ui/clips/${clipId}/captions`);
        words = data.words.map((w) => ({ ...w }));
        showOrigin(); renderWords(); setDirty(false);
        toast(out.replaced
          ? `Replaced ${out.replaced} time${out.replaced === 1 ? "" : "s"} in clip${out.clips.length === 1 ? "" : "s"} ${out.clips.join(", ")}. Re-render to put it in the videos.`
          : `"${find}" isn't in any of this video's clips.`);
      } catch (err) { toast(err.message, true); }
    } });

    window.onbeforeunload = () => (dirty ? "You have unsaved caption edits." : undefined);
    window.addEventListener("hashchange", () => { window.onbeforeunload = null; }, { once: true });

    showOrigin(); renderWords(); setDirty(false);
    app.replaceChildren(
      h("div", { class: "page-head" },
        h("div", { style: "min-width:0" },
          h("a", { class: "back", href: `#/job/${clip.job_id}` }, icon("back"), "Back to the clips"),
          h("h1", {}, `Captions for clip ${clip.rank}`),
          h("div", { class: "meta num" }, h("span", {}, clip.title || "Untitled moment"), h("span", {}, `${hms(clip.start_s)}–${hms(clip.end_s)}`), origin, unsaved)),
        h("div", { class: "actions" }, resetBtn, renderBtn, saveBtn)),
      h("div", { class: "editor" },
        h("div", { class: "editor-side" },
          h("div", { class: "tile" }, video),
          h("p", { class: "hint" }, "The video shows the last render. Save and re-render to see your changes in it; subtitle files use them as soon as you save."),
          h("div", { class: "actions" }, h("span", { class: "muted small" }, "Subtitles"), download("srt"), download("vtt"))),
        h("div", { class: "editor-main" },
          h("div", {}, h("h2", { style: "margin-bottom:6px" }, "Words"),
            h("p", { class: "hint", style: "margin-bottom:14px" }, "Click a word to jump there and retype it. Enter moves to the next word; Backspace on an empty word removes it. The spoken word is highlighted while the clip plays."),
            wordsBox, h("div", { style: "margin-top:10px" }, tools)),
          h("div", {}, h("h2", { style: "margin-bottom:6px" }, "Fix a word everywhere"),
            h("p", { class: "hint", style: "margin-bottom:14px" }, "For a name or term that keeps coming out wrong. This changes every clip of the video."),
            h("div", { class: "replace" }, findInput, replaceInput, replaceBtn)))));
  }

  // ------------------------------------------------------------------------
  // Live session
  // ------------------------------------------------------------------------

  async function viewLive(liveId) {
    const header = h("div"), stats = h("div", { class: "stats" }), problem = h("div"), log = h("div", { class: "log" }), clipsBox = h("div");
    const snapButtons = [15, 30, 60].map((s) => btn(`Last ${s} s`, { onclick: () => api(`/api/ui/live/${liveId}/snapshot`, { method: "POST", json: { seconds: s } })
      .then(() => toast(`Clipping the last ${s} seconds. It appears below once that part is recorded.`)).catch((err) => toast(err.message, true)) }));
    app.replaceChildren(header, stats, problem,
      h("div", { class: "snap" }, h("span", { class: "label", style: "font-weight:500" }, "Clip it now"), snapButtons),
      h("div", { class: "section" }, h("div", { class: "section-head" }, h("h2", {}, "Activity")), log),
      clipsBox);
    let lastClipCount = -1;

    poll(async () => {
      const s = await api(`/api/ui/live/${liveId}`);
      header.replaceChildren(h("div", { class: "page-head" },
        h("div", { style: "min-width:0" },
          h("a", { class: "back", href: "#/videos" }, icon("back"), "Videos"),
          h("h1", {}, s.title || s.url),
          h("div", { class: "meta" }, s.running ? h("span", { class: "live-badge" }, "Live") : statusChip("done"),
            h("span", {}, platformName(s.platform) || (s.obs ? "OBS" : "")))),
        h("div", { class: "actions" },
          s.job_id ? btn("Adjust layout", { href: `/layout-picker?job=${s.job_id}`, target: "_blank", rel: "noopener" }, "layout") : "",
          s.running ? btn("Stop and finish", { class: "btn-danger", onclick: () => api(`/api/ui/live/${liveId}/stop`, { method: "POST" })
            .then(() => toast("Stopping. The final ranking runs now.")) }, "stop") : "")));
      const stat = (value, label) => h("div", { class: "stat" }, h("b", {}, value), h("span", {}, label));
      stats.replaceChildren(
        stat(hms(s.recorded_s), "recorded since joining"),
        stat(s.backfill_total_s ? `${hms(s.backfill_done_s)} / ${hms(s.backfill_total_s)}` : hms(s.processed_s), s.backfill_total_s ? "earlier part filled in" : "analysed"),
        stat(String(s.clips), "clips so far"),
        stat(String(s.chat_messages), "chat messages read"));
      problem.replaceChildren(s.error ? h("div", { class: "notice error", style: "margin-top:16px" }, h("strong", {}, s.error)) : "");
      snapButtons.forEach((b) => { b.disabled = !s.running; });
      log.replaceChildren(...(s.events.length ? s.events.slice().reverse().map((e) => h("div", {}, e)) : [h("div", { class: "muted" }, "Starting up…")]));
      if (s.job_id && (s.clips !== lastClipCount || !s.running)) {
        lastClipCount = s.clips;
        const job = await api(`/api/jobs/${s.job_id}`);
        await renderClips(job, clipsBox);
      }
      if (!s.running) stopPolling();
    }, 3000);
  }

  // ------------------------------------------------------------------------
  // Settings
  // ------------------------------------------------------------------------

  const PROVIDER_NAMES = { anthropic: "Anthropic", openai: "OpenAI", gemini: "Google Gemini", ollama: "Ollama" };

  async function viewSettings() {
    const [settings, providers] = await Promise.all([api("/api/settings"), api("/api/providers/status").catch(() => [])]);
    const draft = JSON.parse(JSON.stringify(settings));
    let dirty = false;
    const savedNote = h("span", { class: "muted small" });
    const touch = () => { dirty = true; savedNote.textContent = "Unsaved changes"; saveBtn.disabled = false; };
    const text = (obj, key, attrs = {}) => h("input", { type: attrs.type || "text", value: obj[key] ?? "", ...attrs,
      oninput: (e) => { obj[key] = attrs.type === "number" ? Number(e.target.value) : e.target.value; touch(); } });
    const select = (obj, key, options) => h("select", { onchange: (e) => { obj[key] = e.target.value; touch(); } },
      options.map(([v, l]) => h("option", { value: v, selected: obj[key] === v }, l)));
    const box = (obj, key, label) => check(Boolean(obj[key]), label, (v) => { obj[key] = v; touch(); });
    const section = (title, note, ...fields) => h("section", {},
      h("div", {}, h("h2", { style: "margin-bottom:6px" }, title), h("p", { class: "hint" }, note)),
      h("div", { class: "fields" }, fields));

    const providerRows = providers.map((p) => {
      const keyInput = p.requires_key ? h("input", { type: "password", placeholder: p.has_key ? "Key saved. Paste to replace" : "Paste an API key", autocomplete: "off" }) : null;
      return h("div", { class: "provider" },
        h("label", { class: "name" },
          h("input", { type: "radio", name: "provider", checked: draft.active_provider === p.name, onchange: () => { draft.active_provider = p.name; touch(); } }),
          h("span", {}, PROVIDER_NAMES[p.name] || p.name, h("br"), h("span", { class: "state" }, p.available ? "Ready" : (p.detail ? p.detail.slice(0, 48) : "Not reachable")))),
        h("div", {}, h("input", { type: "text", value: (draft.providers[p.name] || {}).model || "", placeholder: p.models[0] || "Model", list: `models-${p.name}`, "aria-label": `${p.name} model`,
          oninput: (e) => { draft.providers[p.name] = { ...(draft.providers[p.name] || {}), model: e.target.value }; touch(); } }),
          h("datalist", { id: `models-${p.name}` }, p.models.map((m) => h("option", { value: m })))),
        keyInput ? h("div", { class: "keyrow" }, keyInput, btn("Save key", { class: "btn-sm", onclick: async () => {
          if (!keyInput.value.trim()) return;
          try { await api("/api/settings/secrets", { method: "PUT", json: { key: p.name, value: keyInput.value.trim() } }); keyInput.value = ""; toast(`${PROVIDER_NAMES[p.name] || p.name} key saved in this PeakSift folder.`); }
          catch (err) { toast(err.message, true); }
        } })) : h("span", { class: "hint" }, "Runs on this computer. No key needed."));
    });

    const saveBtn = btn("Save settings", { class: "btn-primary", disabled: true, onclick: async () => {
      try {
        await api("/api/settings", { method: "PUT", json: {
          active_provider: draft.active_provider, providers: draft.providers, whisper: draft.whisper, clips: draft.clips,
          ingest: draft.ingest, export: draft.export, detection: draft.detection, selection: draft.selection, live: draft.live,
        } });
        dirty = false; saveBtn.disabled = true; savedNote.textContent = "Saved";
        toast("Settings saved.");
      } catch (err) { toast(err.message, true); }
    } });
    window.onbeforeunload = () => (dirty ? "You have unsaved settings." : undefined);
    window.addEventListener("hashchange", () => { window.onbeforeunload = null; }, { once: true });

    const hindi = hindiModelSelect(draft.whisper, "hindi_model", settings.hindi_models || []);
    hindi[0].addEventListener("change", touch);

    app.replaceChildren(
      h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Settings"), h("p", { class: "lede" }, "Defaults for every new video. You can change most of them per video under Options."))),
      h("div", { class: "settings" },
        section("AI judge", "Optional. An AI model re-ranks the moments PeakSift finds and writes titles. Ollama runs on this computer; the others need an API key and receive only transcript text.",
          h("div", { class: "providers" }, providerRows)),
        section("Transcription", "Speech to text runs on this computer. Larger models are more accurate; with an NVIDIA GPU, large-v3-turbo is a good choice.",
          h("div", { class: "two" },
            field("Speech model", select(draft.whisper, "model", [["tiny", "tiny (fastest)"], ["base", "base"], ["small", "small"], ["medium", "medium"], ["large-v3-turbo", "large-v3-turbo"], ["large-v3", "large-v3 (most accurate)"]])),
            field("Language", text(draft.whisper, "language", { placeholder: "Detect automatically" }), "A code such as en or hi.")),
          h("div", { class: "two" },
            field("Hindi captions", select(draft.whisper, "hindi_script", [["hinglish", "Hinglish (kya scene hai)"], ["devanagari", "Devanagari (क्या सीन है)"]])),
            field("Caption words come from", select(draft.whisper, "caption_source", [["whisper", "This computer"], ["gemini", "Gemini, for finished clips"]]), "Gemini needs a Gemini key above.")),
          field("Speech model for Hindi", ...hindi),
          box(draft.export, "write_srt", "Also save an .srt subtitle file next to each clip"),
          box(draft.export, "write_vtt", "Also save a .vtt subtitle file next to each clip")),
        section("Clips", "Length in seconds, and how many clips a video gets when you choose best overall.",
          h("div", { class: "three" },
            field("Shortest", text(draft.clips, "min_duration_s", { type: "number", min: 5 })),
            field("Longest", text(draft.clips, "max_duration_s", { type: "number", min: 10 })),
            field("Number of clips", text(draft.clips, "max_clips", { type: "number", min: 1, max: 50 })))),
        section("Downloads", "How videos are fetched from links.",
          field("Use cookies from this browser", select(draft.ingest, "cookies_from_browser", [["", "None"], ["chrome", "Chrome"], ["edge", "Edge"], ["firefox", "Firefox"], ["brave", "Brave"]]),
            "Only for videos that need you to be signed in. Close that browser while downloading."),
          box(draft.ingest, "fetch_chat", "Download the chat replay with finished streams"),
          box(draft.ingest, "fetch_comments", "Read YouTube comments for timestamps viewers left")),
        section("Live streams", "Recording and clipping while a stream is live.",
          h("div", { class: "three" },
            field("Clips per hour", text(draft.live, "clips_per_hour", { type: "number", min: 1 })),
            field("Recording height", text(draft.live, "max_height", { type: "number" }), "720 is plenty for vertical clips."),
            field("Wait after a moment (s)", text(draft.live, "finalize_after_s", { type: "number" }))),
          box(draft.live, "backfill", "Also clip what happened before I joined"),
          box(draft.live, "render_while_live", "Make clips during the stream, not only at the end"))),
      h("div", { class: "savebar" }, savedNote, saveBtn));
  }

  // ------------------------------------------------------------------------
  // Router
  // ------------------------------------------------------------------------

  function route() {
    stopPolling();
    window.onbeforeunload = null;
    const [, page, id] = (location.hash || "#/").split("/");
    const section = { "": "new", videos: "videos", jobs: "videos", job: "videos", live: "videos", edit: "videos", settings: "settings" }[page || ""] || "new";
    document.querySelectorAll(".nav a").forEach((a) => {
      const on = a.dataset.nav === section;
      a.classList.toggle("on", on);
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    const views = { "": viewNew, videos: viewVideos, jobs: viewVideos, job: () => viewJob(id), live: () => viewLive(id), edit: () => viewEdit(id), settings: viewSettings };
    const view = views[page || ""] || viewNew;
    app.replaceChildren(h("p", { class: "loading" }, "Loading…"));
    window.scrollTo(0, 0);
    Promise.resolve(view()).catch((err) => {
      app.replaceChildren(h("div", { class: "notice error" }, h("div", {}, h("strong", {}, "This page couldn't load. "), err.message)));
    });
  }
  window.addEventListener("hashchange", route);
  route();
})();
