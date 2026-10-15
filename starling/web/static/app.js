(() => {
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

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
    if (key) h["Authorization"] = `Bearer ${key}`;
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

  function showMsg(el, text, ok = true) {
    if (!el) return;
    el.className = "msg " + (ok ? "ok" : "error");
    el.textContent = typeof text === "string" ? text : JSON.stringify(text, null, 2);
    el.hidden = false;
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

  // ---- Models ----
  async function refreshModels() {
    const tbody = $("#models-body");
    const msg = $("#models-msg");
    try {
      const data = await api("/v1/registry");
      tbody.innerHTML = "";
      for (const m of data.data || []) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${esc(m.id)}</code></td>
          <td>${esc(m.backend)}</td>
          <td>${esc(m.repo)}</td>
          <td>${m.vram_mb ?? "—"}</td>
          <td><span class="badge ${m.downloaded ? "on" : "off"}">${m.downloaded ? "disk" : "remote"}</span>
              <span class="badge ${m.loaded ? "on" : "off"}">${m.loaded ? "vram" : "cold"}</span></td>
          <td class="actions"></td>`;
        const actions = tr.querySelector(".actions");
        actions.append(
          btn("Download", () => act(() => api(`/v1/models/${encodeURIComponent(m.id)}`, { method: "POST" }), msg, refreshModels)),
          btn("Load", () => act(() => api(`/v1/models/${encodeURIComponent(m.id)}/load`, { method: "POST" }), msg, refreshModels)),
          btn("Unload", () => act(() => api(`/v1/models/${encodeURIComponent(m.id)}/unload`, { method: "POST" }), msg, refreshModels), "secondary"),
          btn("Purge", () => act(() => api(`/v1/models/${encodeURIComponent(m.id)}?purge=true`, { method: "DELETE" }), msg, refreshModels), "danger"),
        );
        tbody.appendChild(tr);
      }
      showMsg(msg, `Loaded ${data.data.length} registry entries`, true);
    } catch (e) {
      showMsg(msg, e.message, false);
    }
  }

  // ---- Registry ----
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
          btn("Delete", () => act(async () => {
            if (!confirm(`Remove ${m.id} from registry?`)) return;
            await api(`/v1/registry/${encodeURIComponent(m.id)}`, { method: "DELETE" });
          }, msg, () => { refreshRegistry(); refreshModels(); }), "danger"),
        );
        tbody.appendChild(tr);
      }
    } catch (e) {
      showMsg(msg, e.message, false);
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
    try {
      if (f.dataset.mode === "edit") {
        await api(`/v1/registry/${encodeURIComponent(f.id.value)}`, { method: "PUT", json: body });
        showMsg(msg, `Updated ${f.id.value}`, true);
      } else {
        await api("/v1/registry", { method: "POST", json: { id: f.id.value.trim(), ...body } });
        showMsg(msg, `Created ${f.id.value}`, true);
      }
      resetRegistryForm();
      refreshRegistry();
      refreshModels();
    } catch (e) {
      showMsg(msg, e.message, false);
    }
  }

  // ---- Settings ----
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
      showMsg(msg, e.message, false);
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
    if (f.hf_token.value !== "") {
      body.hf_token = f.hf_token.value;
    }
    try {
      await api("/v1/admin/settings", { method: "PATCH", json: body });
      showMsg(msg, "Settings saved", true);
      loadSettings();
    } catch (e) {
      showMsg(msg, e.message, false);
    }
  }

  // ---- Status ----
  async function loadStatus() {
    const el = $("#status-json");
    const msg = $("#status-msg");
    try {
      const s = await api("/v1/admin/status");
      el.textContent = JSON.stringify(s, null, 2);
      showMsg(msg, "Status refreshed", true);
    } catch (e) {
      showMsg(msg, e.message, false);
    }
  }

  // ---- Try ----
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
    const fd = new FormData();
    fd.append("file", f.file.files[0]);
    fd.append("model", f.model.value);
    fd.append("response_format", f.response_format.value);
    if (f.language.value.trim()) fd.append("language", f.language.value.trim());
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
      showMsg(msg, typeof data === "string" ? data : JSON.stringify(data, null, 2), true);
    } catch (e) {
      showMsg(msg, e.message, false);
    }
  }

  // helpers
  function btn(label, onClick, cls = "") {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    if (cls) b.className = cls;
    b.addEventListener("click", onClick);
    return b;
  }

  async function act(fn, msgEl, after) {
    try {
      const out = await fn();
      if (out !== undefined) showMsg(msgEl, out, true);
      if (after) after();
    } catch (e) {
      showMsg(msgEl, e.message, false);
    }
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
      showMsg($("#auth-msg"), keyInput.value.trim() ? "API key stored in sessionStorage" : "API key cleared", true);
      refreshModels();
    });

    $$(".tabs button").forEach((b) =>
      b.addEventListener("click", () => switchTab(b.dataset.tab)),
    );

    $("#refresh-models")?.addEventListener("click", refreshModels);
    $("#registry-form")?.addEventListener("submit", submitRegistry);
    $("#registry-reset")?.addEventListener("click", resetRegistryForm);
    $("#settings-form")?.addEventListener("submit", saveSettings);
    $("#refresh-status")?.addEventListener("click", loadStatus);
    $("#try-form")?.addEventListener("submit", tryTranscribe);

    // HTMX global headers if present
    document.body.addEventListener("htmx:configRequest", (e) => {
      const key = apiKey();
      if (key) e.detail.headers["Authorization"] = `Bearer ${key}`;
    });

    switchTab("models");
  }

  document.addEventListener("DOMContentLoaded", init);
})();
