(() => {
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

  /** @type {Map<string, object>} job_id -> last job payload */
  const tracked = new Map();
  let pollTimer = null;

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

  function fmtBytes(n) {
    if (!n || n <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i += 1;
    }
    return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  }

  function fmtElapsed(s) {
    const sec = Math.floor(s);
    if (sec < 60) return `${sec}s`;
    return `${Math.floor(sec / 60)}m ${String(sec % 60).padStart(2, "0")}s`;
  }

  function renderActivity() {
    const strip = $("#activity");
    const list = $("#activity-list");
    const count = $("#activity-count");
    if (!strip || !list) return;

    const jobs = [...tracked.values()].filter((j) => j.status === "running");
    if (jobs.length === 0) {
      strip.hidden = true;
      list.innerHTML = "";
      return;
    }

    strip.hidden = false;
    count.textContent = `${jobs.length} active`;
    list.innerHTML = "";
    for (const job of jobs) {
      const li = document.createElement("li");
      li.className = "activity-item";
      const pct = job.percent;
      const hasPct = pct != null && Number.isFinite(pct);
      const meta = hasPct
        ? `${pct.toFixed(0)}% · ${fmtBytes(job.bytes_done)} / ${fmtBytes(job.bytes_total)} · ${fmtElapsed(job.elapsed_s || 0)}`
        : `${job.message || job.op} · ${fmtElapsed(job.elapsed_s || 0)}`;
      li.innerHTML = `
        <div class="activity-row">
          <span class="activity-label">${esc(job.model_id)} — ${esc(job.op)}</span>
          <span class="activity-meta">${esc(meta)}</span>
        </div>
        ${
          hasPct
            ? `<div class="progress" role="progressbar" aria-valuenow="${pct.toFixed(0)}" aria-valuemin="0" aria-valuemax="100"><i style="width:${pct}%"></i></div>`
            : ""
        }`;
      list.appendChild(li);
    }
  }

  async function pollJobs() {
    try {
      const data = await api("/v1/jobs?active=true");
      const activeIds = new Set();
      for (const job of data.data || []) {
        activeIds.add(job.id);
        tracked.set(job.id, job);
      }
      // Drop finished from tracked unless we just need one last paint
      for (const id of [...tracked.keys()]) {
        if (!activeIds.has(id) && tracked.get(id)?.status === "running") {
          // fetch final state
          try {
            const finalJob = await api(`/v1/jobs/${encodeURIComponent(id)}`);
            tracked.set(id, finalJob);
            if (finalJob.status !== "running") {
              setTimeout(() => {
                tracked.delete(id);
                renderActivity();
                if ($("#panel-models")?.classList.contains("active")) refreshModels();
              }, 800);
            }
          } catch (_) {
            tracked.delete(id);
          }
        }
      }
      renderActivity();
      if (activeIds.size === 0 && tracked.size === 0) {
        stopPolling();
      }
    } catch (_) {
      /* ignore poll errors (e.g. auth) */
    }
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollJobs, 500);
    pollJobs();
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function trackJob(jobId) {
    if (!jobId) return;
    tracked.set(jobId, {
      id: jobId,
      model_id: "",
      op: "download",
      status: "running",
      percent: null,
      bytes_done: 0,
      bytes_total: 0,
      message: "Starting…",
      elapsed_s: 0,
    });
    renderActivity();
    startPolling();
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

  function modelBusy(modelId) {
    return [...tracked.values()].some((j) => j.model_id === modelId && j.status === "running");
  }

  async function refreshModels() {
    const tbody = $("#models-body");
    const msg = $("#models-msg");
    try {
      const data = await api("/v1/registry");
      tbody.innerHTML = "";
      for (const m of data.data || []) {
        const busyJob = [...tracked.values()].find(
          (j) => j.model_id === m.id && j.status === "running",
        );
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${esc(m.id)}</code></td>
          <td>${esc(m.backend)}</td>
          <td>${esc(m.repo)}</td>
          <td>${m.vram_mb ?? "—"}</td>
          <td data-state>
            ${
              busyJob
                ? `<span class="badge working">${esc(busyJob.op)}${
                    busyJob.percent != null ? ` ${busyJob.percent.toFixed(0)}%` : ""
                  }</span>`
                : `<span class="badge ${m.downloaded ? "on" : "off"}">${m.downloaded ? "disk" : "remote"}</span>
                   <span class="badge ${m.loaded ? "on" : "off"}">${m.loaded ? "vram" : "cold"}</span>`
            }
          </td>
          <td class="actions"></td>`;
        const actions = tr.querySelector(".actions");
        const disabled = !!busyJob;

        actions.append(
          btn("Download", () => downloadModel(m.id, msg), "", disabled),
          btn("Load", () => loadModel(m.id, msg), "", disabled),
          btn("Unload", () => simpleOp(m.id, "unload", msg), "secondary", disabled),
          btn("Purge", () => simpleOp(m.id, "purge", msg), "danger", disabled),
        );
        tbody.appendChild(tr);
      }
    } catch (e) {
      showMsg(msg, e.message, "error");
    }
  }

  async function downloadModel(modelId, msgEl) {
    try {
      showMsg(msgEl, `Starting download of ${modelId}…`, "info");
      const result = await api(`/v1/models/${encodeURIComponent(modelId)}?wait=false`, {
        method: "POST",
      });
      if (result.status === "already_downloaded") {
        showMsg(msgEl, `${modelId} is already on disk`, "ok");
        refreshModels();
        return;
      }
      if (result.job_id) trackJob(result.job_id);
      showMsg(msgEl, `Downloading ${modelId}…`, "info");
      refreshModels();
      // Wait until job completes via polling, then toast
      await waitForJob(result.job_id, msgEl, modelId);
    } catch (e) {
      showMsg(msgEl, e.message, "error");
    }
  }

  async function waitForJob(jobId, msgEl, modelId) {
    if (!jobId) return;
    for (;;) {
      await sleep(500);
      let job;
      try {
        job = await api(`/v1/jobs/${encodeURIComponent(jobId)}`);
      } catch (_) {
        break;
      }
      tracked.set(jobId, job);
      renderActivity();
      if (job.status === "done") {
        tracked.delete(jobId);
        renderActivity();
        showMsg(msgEl, `Downloaded ${modelId}`, "ok");
        refreshModels();
        return;
      }
      if (job.status === "error") {
        tracked.delete(jobId);
        renderActivity();
        showMsg(msgEl, job.error || `Download failed for ${modelId}`, "error");
        refreshModels();
        return;
      }
    }
  }

  async function loadModel(modelId, msgEl) {
    showMsg(msgEl, `Loading ${modelId}…`, "info");
    startPolling();
    refreshModels();
    try {
      await api(`/v1/models/${encodeURIComponent(modelId)}/load`, { method: "POST" });
      showMsg(msgEl, `Loaded ${modelId}`, "ok");
    } catch (e) {
      showMsg(msgEl, e.message, "error");
    } finally {
      await pollJobs();
      refreshModels();
    }
  }

  async function simpleOp(modelId, op, msgEl) {
    try {
      if (op === "unload") {
        showMsg(msgEl, `Unloading ${modelId}…`, "info");
        await api(`/v1/models/${encodeURIComponent(modelId)}/unload`, { method: "POST" });
        showMsg(msgEl, `Unloaded ${modelId}`, "ok");
      } else {
        if (
          !confirm(
            `Delete downloaded files for "${modelId}" from disk?\n\nThe registry entry stays; you can download again later.`,
          )
        ) {
          return;
        }
        showMsg(msgEl, `Purging ${modelId} from disk…`, "info");
        const result = await api(`/v1/models/${encodeURIComponent(modelId)}?purge=true`, {
          method: "DELETE",
        });
        if (result.still_downloaded) {
          showMsg(
            msgEl,
            `Purge ran for ${modelId} but files still look present (method=${result.purge?.method || "?"}).`,
            "error",
          );
        } else if (!result.purged) {
          showMsg(msgEl, `${modelId} had nothing on disk to purge`, "info");
        } else {
          const freed = result.purge?.freed_bytes;
          const extra =
            freed != null && freed > 0 ? ` (~${fmtBytes(freed)} freed)` : "";
          showMsg(msgEl, `Purged ${modelId} from disk${extra}`, "ok");
        }
      }
      await refreshModels();
      if ($("#panel-status")?.classList.contains("active")) await loadStatus();
    } catch (e) {
      showMsg(msgEl, e.message, "error");
      await refreshModels();
    }
  }

  function setRegistryMode(mode, id) {
    const f = $("#registry-form");
    const banner = $("#registry-mode");
    const submit = $("#registry-submit");
    f.dataset.mode = mode;
    if (mode === "edit") {
      f.id.readOnly = true;
      banner.textContent = `Editing ${id}`;
      banner.classList.add("editing");
      submit.textContent = "Save changes";
    } else {
      f.id.readOnly = false;
      banner.textContent = "Adding new entry";
      banner.classList.remove("editing");
      submit.textContent = "Add entry";
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
        const source = m.file ? `${m.repo} · ${m.file}` : m.repo;
        tr.innerHTML = `
          <td><code>${esc(m.id)}</code>${m.gated ? ' <span class="badge">gated</span>' : ""}</td>
          <td>${esc(m.backend)}</td>
          <td class="source-cell">${esc(source)}</td>
          <td class="actions"></td>`;
        const actions = tr.querySelector(".actions");
        actions.append(
          btn("Edit", () => fillRegistryForm(m), "secondary"),
          btn(
            "Delete",
            async () => {
              if (!confirm(`Remove ${m.id} from registry?`)) return;
              try {
                await api(`/v1/registry/${encodeURIComponent(m.id)}`, { method: "DELETE" });
                showMsg(msg, `Removed ${m.id}`, "ok");
                refreshRegistry();
                refreshModels();
              } catch (e) {
                showMsg(msg, e.message, "error");
              }
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
    setRegistryMode("edit", m.id);
    f.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function resetRegistryForm() {
    const f = $("#registry-form");
    f.reset();
    f.vram_mb.value = 4096;
    setRegistryMode("create");
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
    try {
      if (f.dataset.mode === "edit") {
        await api(`/v1/registry/${encodeURIComponent(id)}`, { method: "PUT", json: body });
        showMsg(msg, `Updated ${id}`, "ok");
      } else {
        await api("/v1/registry", { method: "POST", json: { id, ...body } });
        showMsg(msg, `Created ${id}`, "ok");
      }
      resetRegistryForm();
      refreshRegistry();
      refreshModels();
    } catch (e) {
      showMsg(msg, e.message, "error");
    }
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
      f.hf_token.placeholder = s.hf_token_set
        ? "Token is set — leave blank to keep"
        : "Optional — only for gated/private models";
      $("#settings-meta").textContent = s.api_key_set
        ? "API key is configured (env). Bind address stays env-only."
        : "No API key configured. Bind address stays env-only.";
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
    try {
      await api("/v1/admin/settings", { method: "PATCH", json: body });
      showMsg(msg, "Settings saved", "ok");
      loadSettings();
    } catch (e) {
      showMsg(msg, e.message, "error");
    }
  }

  async function loadStatus() {
    const view = $("#status-view");
    const msg = $("#status-msg");
    try {
      const s = await api("/v1/admin/status");
      msg.hidden = true;
      const devices = s.devices || [];
      const loaded = s.loaded || [];
      const backends = s.backends || [];
      const deviceCards = devices
        .map((d) => {
          const total = d.total_mb || 0;
          const free = d.free_mb || 0;
          const used = Math.max(0, total - free);
          const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0;
          const label = d.index == null ? "CPU" : d.name || d.device;
          const uuid = d.uuid ? `<div class="status-meta">${esc(d.uuid)}</div>` : "";
          return `
            <article class="status-card">
              <h3>${esc(label)}</h3>
              <div class="status-meta">${esc(d.device || "cpu")}${d.index != null ? ` · index ${d.index}` : ""}</div>
              ${uuid}
              ${
                total > 0
                  ? `<div class="vram-bar" role="meter" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100">
                       <i style="width:${pct}%"></i>
                     </div>
                     <div class="status-meta">${used} / ${total} MB used · ${free} MB free</div>`
                  : `<div class="status-meta">No VRAM reported</div>`
              }
            </article>`;
        })
        .join("");
      const loadedList =
        loaded.length === 0
          ? `<p class="hint">Nothing in VRAM — models load on first use.</p>`
          : `<ul class="status-list">${loaded
              .map(
                (m) =>
                  `<li><code>${esc(m.id)}</code> · ${esc(m.backend || "?")} · ${esc(m.device || "?")}${
                    m.vram_mb != null ? ` · ~${m.vram_mb} MB` : ""
                  }</li>`,
              )
              .join("")}</ul>`;
      view.innerHTML = `
        <div class="status-summary">
          <article class="status-card">
            <h3>Starling ${esc(s.version || "")}</h3>
            <div class="status-meta">${backends.length} backend${backends.length === 1 ? "" : "s"} · ${esc(
              backends.join(", ") || "none",
            )}</div>
            <div class="status-meta">${s.downloaded_count ?? 0} on disk · ${s.registry_count ?? 0} in registry</div>
            <div class="status-meta">TTL ${s.settings?.stt_model_ttl ?? "—"}s · device ${esc(
              s.settings?.device_override || "auto",
            )}</div>
          </article>
        </div>
        <h3 class="status-section">GPUs</h3>
        <div class="status-grid">${deviceCards || `<p class="hint">No devices reported.</p>`}</div>
        <h3 class="status-section">Loaded in VRAM</h3>
        ${loadedList}`;
    } catch (e) {
      showMsg(msg, e.message, "error");
    }
  }

  async function fillTryModels() {
    const sel = $("#try-model");
    const preferred = "orukeet";
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
      const ids = [...sel.options].map((o) => o.value);
      if (cur && ids.includes(cur)) sel.value = cur;
      else if (ids.includes(preferred)) sel.value = preferred;
    } catch (_) {
      /* ignore */
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
    submit.disabled = true;
    submit.textContent = "Transcribing…";
    showMsg(msg, "Transcribing…", "info");
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
      showMsg(msg, typeof data === "string" ? data : JSON.stringify(data, null, 2), "ok");
    } catch (e) {
      showMsg(msg, e.message, "error");
    } finally {
      submit.disabled = false;
      submit.textContent = "Transcribe";
    }
  }

  function btn(label, onClick, cls = "", disabled = false) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    if (cls) b.className = cls;
    b.disabled = disabled;
    b.addEventListener("click", onClick);
    return b;
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
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
    $("#registry-new")?.addEventListener("click", () => {
      resetRegistryForm();
      $("#registry-form")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    $("#settings-form")?.addEventListener("submit", saveSettings);
    $("#refresh-status")?.addEventListener("click", loadStatus);
    $("#try-form")?.addEventListener("submit", tryTranscribe);

    startPolling();
    switchTab("models");
  }

  document.addEventListener("DOMContentLoaded", init);
})();
