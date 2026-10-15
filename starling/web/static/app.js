(() => {
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

  /** @type {Map<string, {id: string, label: string, started: number, timer?: number}>} */
  const jobs = new Map();
  /** modelId -> Set of ops */
  const rowOps = new Map();

  function apiKey() {
    return sessionStorage.getItem("starling_api_key") || "";
  }

  function setApiKey(key) {
    if (key) sessionStorage.setItem("starling_api_key", key);
    else sessionStorage.removeItem("starling_api_key");
  }

  function authHeaders(extra = {}) {
    const h = { ...extra };
    const key = apiKey();
    if (key) h.Authorization = `Bearer ${key}`;
    return h;
  }

  async function api(path, opts = {}) {
    const headers = authHeaders(opts.headers || {});
    if (opts.json !== undefined) {
      headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.json);
      delete opts.json;
    }
    const res = await fetch(path, { ...opts, headers });
    const ct = res.headers.get("content-type") || "";
    const data = ct.includes("application/json") ? await res.json() : await res.text();
    if (!res.ok) {
      const detail = typeof data === "object" ? data.detail || JSON.stringify(data) : data;
      throw new Error(detail || res.statusText);
    }
    return data;
  }

  function showMsg(el, text, kind = "ok") {
    if (!el) return;
    el.className = "msg " + kind;
    el.textContent = typeof text === "string" ? text : JSON.stringify(text, null, 2);
    el.hidden = false;
  }

  function fmtElapsed(ms) {
    const s = Math.floor(ms / 1000);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}m ${r.toString().padStart(2, "0")}s`;
  }

  function renderActivity() {
    const strip = $("#activity");
    const list = $("#activity-list");
    const count = $("#activity-count");
    if (!strip || !list) return;

    if (jobs.size === 0) {
      strip.hidden = true;
      document.body.classList.remove("busy");
      list.innerHTML = "";
      return;
    }

    strip.hidden = false;
    document.body.classList.add("busy");
    count.textContent = `${jobs.size} active`;
    list.innerHTML = "";
    for (const job of jobs.values()) {
      const li = document.createElement("li");
      li.className = "activity-item";
      li.dataset.jobId = job.id;
      li.innerHTML = `
        <div class="activity-row">
          <span class="activity-label">${esc(job.label)}</span>
          <span class="activity-meta" data-elapsed>${fmtElapsed(Date.now() - job.started)}</span>
        </div>
        <div class="progress" aria-hidden="true"><i></i></div>`;
      list.appendChild(li);
    }
  }

  function tickActivity() {
    $$("[data-elapsed]").forEach((el) => {
      const item = el.closest(".activity-item");
      const job = item && jobs.get(item.dataset.jobId);
      if (job) el.textContent = fmtElapsed(Date.now() - job.started);
    });
  }

  setInterval(tickActivity, 500);

  function beginJob(key, label) {
    const id = key;
    jobs.set(id, { id, label, started: Date.now() });
    renderActivity();
    return id;
  }

  function endJob(id) {
    jobs.delete(id);
    renderActivity();
  }

  function markRowBusy(modelId, op, busy) {
    const set = rowOps.get(modelId) || new Set();
    if (busy) set.add(op);
    else set.delete(op);
    if (set.size) rowOps.set(modelId, set);
    else rowOps.delete(modelId);
  }

  function rowIsBusy(modelId) {
    return (rowOps.get(modelId) || new Set()).size > 0;
  }

  async function runJob({ key, label, modelId, op, fn, msgEl, after, successText }) {
    if (modelId && op) markRowBusy(modelId, op, true);
    beginJob(key, label);
    // Re-render models so the row shows working state
    if (modelId && $("#panel-models")?.classList.contains("active")) {
      /* refreshModels will pick up rowOps */
    }
    const busyButtons = modelId
      ? $$(`[data-model-id="${CSS.escape(modelId)}"] button`)
      : [];
    busyButtons.forEach((b) => {
      b.disabled = true;
      if (b.dataset.op === op) {
        b.classList.add("busy");
        b.dataset.label = b.textContent;
        b.textContent = op === "download" ? "Downloading…" : op === "load" ? "Loading…" : "Working…";
      }
    });
    // Also paint the row if present
    const tr = modelId ? $(`tr[data-model-id="${CSS.escape(modelId)}"]`) : null;
    if (tr) {
      tr.classList.add("row-busy");
      const state = tr.querySelector("[data-state]");
      if (state) {
        state.dataset.prev = state.innerHTML;
        state.innerHTML = `<span class="badge working">${esc(op)}</span>`;
      }
    }
    if (msgEl) showMsg(msgEl, `${label}…`, "info");

    let out;
    let ok = false;
    try {
      out = await fn();
      ok = true;
    } catch (e) {
      if (msgEl) showMsg(msgEl, e.message, "error");
      throw e;
    } finally {
      endJob(key);
      if (modelId && op) markRowBusy(modelId, op, false);
      if (tr) {
        tr.classList.remove("row-busy");
        const state = tr.querySelector("[data-state]");
        if (state && state.dataset.prev != null) state.innerHTML = state.dataset.prev;
      }
      busyButtons.forEach((b) => {
        b.disabled = false;
        b.classList.remove("busy");
        if (b.dataset.label) b.textContent = b.dataset.label;
      });
    }

    if (ok) {
      if (msgEl && successText !== false) {
        showMsg(
          msgEl,
          successText || (typeof out === "object" ? JSON.stringify(out) : String(out)),
          "ok",
        );
      }
      if (after) await after(out);
      return out;
    }
  }

  function switchTab(name) {
    $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
    if (name === "models") refreshModels();
    if (name === "registry") refreshRegistry();
    if (name === "settings") loadSettings();
    if (name === "status") loadStatus();
    if (name === "try") fillTryModels();
  }

  async function refreshModels() {
    const tbody = $("#models-body");
    const msg = $("#models-msg");
    try {
      const data = await api("/v1/registry");
      tbody.innerHTML = "";
      for (const m of data.data || []) {
        const busy = rowIsBusy(m.id);
        const tr = document.createElement("tr");
        tr.dataset.modelId = m.id;
        if (busy) tr.classList.add("row-busy");
        const ops = rowOps.get(m.id);
        const workingOp = ops && [...ops][0];
        tr.innerHTML = `
          <td><code>${esc(m.id)}</code></td>
          <td>${esc(m.backend)}</td>
          <td>${esc(m.repo)}</td>
          <td>${m.vram_mb ?? "—"}</td>
          <td data-state>
            ${
              workingOp
                ? `<span class="badge working">${esc(workingOp)}</span>`
                : `<span class="badge ${m.downloaded ? "on" : "off"}">${m.downloaded ? "disk" : "remote"}</span>
                   <span class="badge ${m.loaded ? "on" : "off"}">${m.loaded ? "vram" : "cold"}</span>`
            }
          </td>
          <td class="actions" data-model-id="${esc(m.id)}"></td>`;
        const actions = tr.querySelector(".actions");
        const mk = (label, op, methodPath, method, cls = "") => {
          const b = btn(label, () => {
            if (rowIsBusy(m.id)) return;
            runJob({
              key: `${op}:${m.id}`,
              label: `${label} ${m.id}`,
              modelId: m.id,
              op,
              fn: () => api(methodPath, { method }),
              msgEl: msg,
              after: () => refreshModels(),
              successText:
                op === "download"
                  ? `Downloaded ${m.id}`
                  : op === "load"
                    ? `Loaded ${m.id} into VRAM`
                    : op === "unload"
                      ? `Unloaded ${m.id}`
                      : `Purged ${m.id}`,
            }).catch(() => {});
          }, cls);
          b.dataset.op = op;
          if (busy) {
            b.disabled = true;
            if (workingOp === op) {
              b.classList.add("busy");
              b.textContent =
                op === "download" ? "Downloading…" : op === "load" ? "Loading…" : "Working…";
            }
          }
          return b;
        };
        actions.append(
          mk("Download", "download", `/v1/models/${encodeURIComponent(m.id)}`, "POST"),
          mk("Load", "load", `/v1/models/${encodeURIComponent(m.id)}/load`, "POST"),
          mk("Unload", "unload", `/v1/models/${encodeURIComponent(m.id)}/unload`, "POST", "secondary"),
          mk("Purge", "purge", `/v1/models/${encodeURIComponent(m.id)}?purge=true`, "DELETE", "danger"),
        );
        tbody.appendChild(tr);
      }
      if (msg && !msg.classList.contains("info") && !msg.classList.contains("error")) {
        msg.hidden = true;
      }
    } catch (e) {
      showMsg(msg, e.message, "error");
    }
  }

  async function refreshRegistry() {
    const tbody = $("#registry-body");
    const msg = $("#registry-msg");
    try {
      const data = await api("/v1/registry");
      tbody.innerHTML = "";
      for (const m of data.data || []) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${esc(m.id)}</code></td>
          <td>${esc(m.backend)}</td>
          <td>${esc(m.repo)}${m.file ? `<br><span class="hint">${esc(m.file)}</span>` : ""}</td>
          <td class="actions"></td>`;
        const actions = tr.querySelector(".actions");
        actions.append(
          btn("Edit", () => fillRegistryForm(m), "secondary"),
          btn(
            "Delete",
            () => {
              if (!confirm(`Remove ${m.id} from registry?`)) return;
              runJob({
                key: `registry-del:${m.id}`,
                label: `Delete registry ${m.id}`,
                fn: () => api(`/v1/registry/${encodeURIComponent(m.id)}`, { method: "DELETE" }),
                msgEl: msg,
                after: () => {
                  refreshRegistry();
                  refreshModels();
                },
                successText: `Removed ${m.id}`,
              }).catch(() => {});
            },
            "danger",
          ),
        );
        tbody.appendChild(tr);
      }
    } catch (e) {
      showMsg(msg, e.message, "error");
    }
  }

  function fillRegistryForm(m) {
    const f = $("#registry-form");
    f.id.value = m.id;
    f.id.readOnly = true;
    f.backend.value = m.backend;
    f.repo.value = m.repo || "";
    f.file.value = m.file || "";
    f.revision.value = m.revision || "";
    f.vram_mb.value = m.vram_mb ?? 4096;
    f.requires_language.checked = !!m.requires_language;
    f.gated.checked = !!m.gated;
    f.supports_translate.checked = !!m.supports_translate;
    f.supports_timestamps.checked = !!m.supports_timestamps;
    f.license.value = m.license || "";
    f.dataset.mode = "edit";
    $("#registry-submit").textContent = "Update entry";
    f.scrollIntoView({ behavior: "smooth" });
  }

  function resetRegistryForm() {
    const f = $("#registry-form");
    f.reset();
    f.id.readOnly = false;
    f.dataset.mode = "create";
    $("#registry-submit").textContent = "Add entry";
  }

  async function submitRegistry(ev) {
    ev.preventDefault();
    const f = ev.target;
    const msg = $("#registry-msg");
    const body = {
      backend: f.backend.value,
      repo: f.repo.value.trim(),
      file: f.file.value.trim() || null,
      revision: f.revision.value.trim() || null,
      vram_mb: Number(f.vram_mb.value) || 4096,
      requires_language: f.requires_language.checked,
      gated: f.gated.checked,
      supports_translate: f.supports_translate.checked,
      supports_timestamps: f.supports_timestamps.checked,
      license: f.license.value.trim() || null,
      languages: "auto",
    };
    const id = f.id.value.trim();
    const editing = f.dataset.mode === "edit";
    runJob({
      key: `registry-save:${id}`,
      label: editing ? `Update ${id}` : `Create ${id}`,
      fn: () =>
        editing
          ? api(`/v1/registry/${encodeURIComponent(id)}`, { method: "PUT", json: body })
          : api("/v1/registry", { method: "POST", json: { id, ...body } }),
      msgEl: msg,
      after: () => {
        resetRegistryForm();
        refreshRegistry();
        refreshModels();
      },
      successText: editing ? `Updated ${id}` : `Created ${id}`,
    }).catch(() => {});
  }

  async function loadSettings() {
    const msg = $("#settings-msg");
    try {
      const s = await api("/v1/admin/settings");
      const f = $("#settings-form");
      f.stt_model_ttl.value = s.stt_model_ttl;
      f.device_override.value = s.device_override || "";
      f.whisper_compute_type.value = s.whisper_compute_type || "default";
      f.preload_models.value = (s.preload_models || []).join(", ");
      f.metrics_enabled.checked = !!s.metrics_enabled;
      f.hf_hub_cache.value = s.hf_hub_cache || "";
      f.hf_token.value = "";
      f.hf_token.placeholder = s.hf_token_set ? "(set — leave blank to keep)" : "(not set)";
      $("#settings-meta").textContent = `API key ${s.api_key_set ? "is" : "is not"} configured · UI enabled=${s.enable_ui}`;
    } catch (e) {
      showMsg(msg, e.message, "error");
    }
  }

  async function saveSettings(ev) {
    ev.preventDefault();
    const f = ev.target;
    const msg = $("#settings-msg");
    const body = {
      stt_model_ttl: Number(f.stt_model_ttl.value),
      device_override: f.device_override.value.trim(),
      whisper_compute_type: f.whisper_compute_type.value.trim() || "default",
      preload_models: f.preload_models.value
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      metrics_enabled: f.metrics_enabled.checked,
      hf_hub_cache: f.hf_hub_cache.value.trim(),
    };
    if (f.hf_token.value !== "") body.hf_token = f.hf_token.value;
    const submit = f.querySelector('button[type="submit"]');
    runJob({
      key: "settings-save",
      label: "Save settings",
      fn: async () => {
        if (submit) {
          submit.disabled = true;
          submit.classList.add("busy");
        }
        try {
          return await api("/v1/admin/settings", { method: "PATCH", json: body });
        } finally {
          if (submit) {
            submit.disabled = false;
            submit.classList.remove("busy");
          }
        }
      },
      msgEl: msg,
      after: () => loadSettings(),
      successText: "Settings saved",
    }).catch(() => {});
  }

  async function loadStatus() {
    const el = $("#status-json");
    const msg = $("#status-msg");
    try {
      const s = await api("/v1/admin/status");
      el.textContent = JSON.stringify(s, null, 2);
      msg.hidden = true;
    } catch (e) {
      showMsg(msg, e.message, "error");
    }
  }

  async function fillTryModels() {
    const sel = $("#try-model");
    try {
      const data = await api("/v1/registry");
      const cur = sel.value;
      sel.innerHTML = "";
      for (const m of data.data || []) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = `${m.id} (${m.backend})`;
        sel.appendChild(opt);
      }
      if (cur) sel.value = cur;
    } catch (_) {
      /* ignore until auth */
    }
  }

  async function tryTranscribe(ev) {
    ev.preventDefault();
    const f = ev.target;
    const msg = $("#try-msg");
    const submit = $("#try-submit");
    const file = f.file.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("model", f.model.value);
    fd.append("response_format", f.response_format.value);
    if (f.language.value.trim()) fd.append("language", f.language.value.trim());

    runJob({
      key: `transcribe:${f.model.value}`,
      label: `Transcribe with ${f.model.value}`,
      fn: async () => {
        if (submit) {
          submit.disabled = true;
          submit.classList.add("busy");
          submit.textContent = "Transcribing…";
        }
        try {
          const res = await fetch("/v1/audio/transcriptions", {
            method: "POST",
            headers: authHeaders(),
            body: fd,
          });
          const ct = res.headers.get("content-type") || "";
          const data = ct.includes("application/json") ? await res.json() : await res.text();
          if (!res.ok) {
            const detail = typeof data === "object" ? data.detail || JSON.stringify(data) : data;
            throw new Error(detail || res.statusText);
          }
          return data;
        } finally {
          if (submit) {
            submit.disabled = false;
            submit.classList.remove("busy");
            submit.textContent = "Transcribe";
          }
        }
      },
      msgEl: msg,
      successText: false,
      after: (data) => {
        showMsg(msg, typeof data === "string" ? data : JSON.stringify(data, null, 2), "ok");
      },
    }).catch(() => {});
  }

  function btn(label, onClick, cls = "") {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    if (cls) b.className = cls;
    b.addEventListener("click", onClick);
    return b;
  }

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function init() {
    const keyInput = $("#api-key");
    keyInput.value = apiKey();
    $("#save-key").addEventListener("click", () => {
      setApiKey(keyInput.value.trim());
      showMsg(
        $("#auth-msg"),
        keyInput.value.trim() ? "API key stored for this browser session" : "API key cleared",
        "ok",
      );
      refreshModels();
    });

    $$(".tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

    $("#refresh-models")?.addEventListener("click", refreshModels);
    $("#registry-form")?.addEventListener("submit", submitRegistry);
    $("#registry-reset")?.addEventListener("click", resetRegistryForm);
    $("#settings-form")?.addEventListener("submit", saveSettings);
    $("#refresh-status")?.addEventListener("click", loadStatus);
    $("#try-form")?.addEventListener("submit", tryTranscribe);

    switchTab("models");
  }

  document.addEventListener("DOMContentLoaded", init);
})();
