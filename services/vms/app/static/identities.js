/* Identities view — auto-discovered objects (people, cars, dogs, …).
 *
 * Loaded after app.js, so it shares the global helpers (el, $, $$, api, modal,
 * toast, fmtTime). Exposes window.Identities = { init, render, teardown } which
 * app.js's view router calls.
 *
 * Each identity is one remembered object: its appearance is matched
 * orientation-invariantly across appearances/cameras, and its total time in
 * view (dwell) is summed from the per-camera presence segments.
 */
"use strict";

(function () {
  const state = { all: [], filterClass: "", selected: new Set() };

  function fmtDuration(sec) {
    sec = Math.max(0, Math.round(Number(sec) || 0));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h) return `${h}h ${m}m ${s}s`;
    if (m) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function classBadge(cls) {
    return el("span", { class: "id-class-badge" }, cls || "object");
  }

  function colorBadge(it) {
    if (!it.color || it.color === "unknown") return null;
    const sw = el("span", { class: "id-color-swatch" });
    if (it.color_hex) sw.style.background = it.color_hex;
    return el("span", { class: "id-color-badge" }, sw, it.color);
  }

  async function render() {
    const grid = $("#identities-grid");
    grid.innerHTML = "";
    grid.append(el("div", { class: "empty" }, "Loading…"));
    let resp;
    try {
      resp = await api.get("/api/identities?limit=300");
    } catch (e) {
      grid.innerHTML = "";
      grid.append(el("div", { class: "empty" }, "Error: " + e.message));
      return;
    }
    state.all = resp.items || [];
    paint();
  }

  function paint() {
    const grid = $("#identities-grid");
    grid.innerHTML = "";

    // Class filter chips derived from what's actually present.
    const classes = Array.from(new Set(state.all.map((i) => i.object_class || "object"))).sort();
    const bar = el("div", { class: "id-filterbar" });
    const chip = (label, val) =>
      el("button", {
        class: "btn btn-sm" + (state.filterClass === val ? " active" : ""),
        onclick: () => { state.filterClass = val; paint(); },
      }, label);
    bar.append(chip(`All (${state.all.length})`, ""));
    for (const c of classes) {
      const n = state.all.filter((i) => (i.object_class || "object") === c).length;
      bar.append(chip(`${c} (${n})`, c));
    }
    grid.append(bar);

    const items = state.filterClass
      ? state.all.filter((i) => (i.object_class || "object") === state.filterClass)
      : state.all;

    if (!items.length) {
      grid.append(el("div", { class: "empty" },
        "No objects remembered yet. They appear automatically as the cameras see people/objects you selected as triggers."));
      updateMergeBtn();
      return;
    }
    const cards = el("div", { class: "id-cards" });
    for (const it of items) cards.append(card(it));
    grid.append(cards);
    updateMergeBtn();
  }

  function card(it) {
    const thumb = it.rep_thumb_url
      ? el("img", { src: it.rep_thumb_url, alt: it.name, loading: "lazy" })
      : el("div", { class: "no-thumb" }, "no image");

    const sel = el("input", {
      type: "checkbox", class: "id-select", title: "Select for merge",
      ...(state.selected.has(it.id) ? { checked: "" } : {}),
      onclick: (e) => {
        e.stopPropagation();
        if (e.target.checked) state.selected.add(it.id);
        else state.selected.delete(it.id);
        updateMergeBtn();
      },
    });

    return el("div", { class: "id-card", onclick: () => openDetail(it.id) },
      el("div", { class: "id-thumb" }, thumb, sel),
      el("div", { class: "id-info" },
        el("div", { class: "id-line1" },
          el("span", { class: "id-name" }, it.name || `#${it.id}`),
          classBadge(it.object_class)),
        colorBadge(it),
        (it.make || it.vehicle_type)
          ? el("div", { class: "id-vehicle" },
              [it.make, it.vehicle_type].filter(Boolean).join(" · "))
          : null,
        el("div", { class: "id-dwell" }, "⏱ ", el("strong", {}, fmtDuration(it.total_seconds)), " in view"),
        el("div", { class: "id-meta" },
          `${it.num_sightings || 0} sightings · ${(it.cameras || []).length} camera${(it.cameras || []).length === 1 ? "" : "s"}`),
        el("div", { class: "id-meta subtle" }, "last " + fmtTime(it.last_seen)))
    );
  }

  function updateMergeBtn() {
    const btn = $("#identities-merge");
    if (btn) {
      btn.disabled = state.selected.size < 2;
      btn.textContent = state.selected.size >= 2
        ? `Merge ${state.selected.size} selected` : "Merge selected";
    }
    const del = $("#identities-delete");
    if (del) {
      del.disabled = state.selected.size < 1;
      del.textContent = state.selected.size >= 1
        ? `Delete ${state.selected.size} selected` : "Delete selected";
    }
  }

  async function openDetail(id) {
    let it;
    try {
      it = await api.get(`/api/identities/${id}`);
    } catch (e) {
      toast(e.message, "error");
      return;
    }
    const wrap = el("div", { class: "id-detail" });

    // Header: editable name + class.
    const nameInput = el("input", { type: "text", value: it.name || "", class: "id-name-input" });
    const header = el("div", { class: "id-detail-head" },
      nameInput,
      classBadge(it.object_class),
      el("button", {
        class: "btn btn-sm btn-primary",
        onclick: async () => {
          try {
            await api.put(`/api/identities/${id}`, { name: nameInput.value.trim(), is_named: true });
            toast("Renamed", "ok");
            render();
          } catch (e) { toast(e.message, "error"); }
        },
      }, "Save name"));
    wrap.append(header);

    const grid = el("div", { class: "detail-grid" });
    const add = (k, v) => { grid.append(el("div", { class: "k" }, k), el("div", {}, v == null || v === "" ? "—" : String(v))); };
    add("Object type", it.object_class);
    add("Dominant colour", it.color && it.color !== "unknown" ? it.color : "—");
    if (it.make || it.vehicle_type) {
      add("Vehicle make", it.make || "—");
      add("Body type", it.vehicle_type || "—");
    }
    add("Total time in view", fmtDuration(it.total_seconds));
    add("Sightings", it.num_sightings);
    add("Cameras", (it.cameras || []).join(", ") || "—");
    add("First seen", fmtTime(it.first_seen));
    add("Last seen", fmtTime(it.last_seen));
    add("Appearance samples", it.num_appearance_exemplars);
    if (it.object_class === "person") add("Face samples", it.num_face_exemplars);
    wrap.append(grid);

    // Recorded clips for this identity (click → play in the rich viewer).
    const clipsWrap = el("div", {});
    wrap.append(el("div", { class: "id-section-title" }, "Recorded video"));
    wrap.append(clipsWrap);
    clipsWrap.append(el("div", { class: "subtle" }, "Loading clips…"));
    api.get(`/api/identities/${id}/events`).then((resp) => {
      clipsWrap.innerHTML = "";
      const evs = (resp && resp.items) || [];
      if (!evs.length) {
        clipsWrap.append(el("div", { class: "empty" }, "No recorded clips for this person yet."));
        return;
      }
      const grid = el("div", { class: "clip-list" });
      for (const ev of evs) {
        grid.append(el("div", { class: "clip-tile", onclick: () => openEventDetail(ev.id) },
          ev.thumb_url ? el("img", { src: ev.thumb_url, loading: "lazy", alt: "" })
                       : el("div", { class: "no-thumb" }, "clip"),
          el("div", { class: "clip-meta" },
            el("span", {}, ev.camera_name || `Camera #${ev.camera_id}`),
            el("span", { class: "subtle" }, fmtTime(ev.ts)))));
      }
      clipsWrap.append(grid);
    }).catch(() => { clipsWrap.innerHTML = ""; });

    // Recent sightings strip (clickable → its event clip).
    const sightings = it.recent_sightings || [];
    if (sightings.length) {
      wrap.append(el("div", { class: "id-section-title" }, "Recent sightings"));
      const strip = el("div", { class: "id-sightings" });
      for (const s of sightings) {
        const img = s.thumb_url
          ? el("img", { src: s.thumb_url, alt: "sighting", loading: "lazy" })
          : el("div", { class: "no-thumb" }, "—");
        const tile = el("div", { class: "id-sighting" + (s.event_id ? " clickable" : ""), title: fmtTime(s.ts) },
          img,
          el("div", { class: "id-sighting-meta" }, fmtTime(s.ts)));
        if (s.event_id) tile.addEventListener("click", () => openEventDetail(s.event_id));
        strip.append(tile);
      }
      wrap.append(strip);
    }

    // Face samples captured for this identity.
    const facesWrap = el("div", {});
    wrap.append(facesWrap);
    api.get(`/api/identities/${id}/faces`).then((resp) => {
      const faces = (resp && resp.items) || [];
      if (!faces.length) return;
      facesWrap.append(el("div", { class: "id-section-title" }, `Face samples (${resp.total})`));
      const strip = el("div", { class: "id-sightings" });
      for (const f of faces) {
        const img = el("img", { src: f.thumb_url, alt: "face", loading: "lazy" });
        img.addEventListener("error", function () { this.style.display = "none"; });
        strip.append(el("div", { class: "id-sighting", title: fmtTime(f.ts) }, img));
      }
      facesWrap.append(strip);
    }).catch(() => {});

    wrap.append(el("div", { class: "form-actions" },
      el("button", { class: "btn btn-danger", onclick: () => confirmDelete(id) }, "Delete identity"),
      el("button", { class: "btn", onclick: () => modal.close() }, "Close")));

    modal.open(it.name || `Identity #${id}`, wrap, { wide: true });
  }

  async function confirmDelete(id) {
    if (!confirm("Delete this identity and its sightings? Recorded clips are kept.")) return;
    try {
      await api.del(`/api/identities/${id}`);
      state.selected.delete(id);
      toast("Identity deleted", "ok");
      modal.close();
      render();
    } catch (e) { toast(e.message, "error"); }
  }

  async function doMerge() {
    const ids = Array.from(state.selected);
    if (ids.length < 2) return;
    const target = ids[0];
    const sources = ids.slice(1);
    if (!confirm(`Merge ${sources.length} identit${sources.length === 1 ? "y" : "ies"} into #${target}? Their sightings and dwell time fold together.`)) return;
    try {
      await api.post("/api/identities/merge", { target_id: target, source_ids: sources });
      toast("Merged", "ok");
      state.selected.clear();
      render();
    } catch (e) { toast(e.message, "error"); }
  }

  async function deleteSelected() {
    const ids = Array.from(state.selected);
    if (!ids.length) return;
    if (!confirm(`Delete ${ids.length} selected identit${ids.length === 1 ? "y" : "ies"} (their sightings + samples)? This cannot be undone.`)) return;
    try {
      const r = await api.post("/api/identities/bulk-delete", { ids });
      toast(`Deleted ${r.deleted}`, "ok");
      state.selected.clear();
      render();
    } catch (e) { toast(e.message, "error"); }
  }

  async function clearAll() {
    if (!confirm("Delete ALL identities (every auto-discovered person/object, their sightings + samples)? This cannot be undone.")) return;
    try {
      const r = await api.post("/api/identities/clear-all", { confirm: true });
      toast(`Deleted ${r.deleted} identities`, "ok");
      state.selected.clear();
      render();
    } catch (e) { toast(e.message, "error"); }
  }

  function init() {
    const refresh = $("#identities-refresh");
    if (refresh) refresh.addEventListener("click", () => render());
    const merge = $("#identities-merge");
    if (merge) merge.addEventListener("click", () => doMerge());
    const del = $("#identities-delete");
    if (del) del.addEventListener("click", () => deleteSelected());
    const clr = $("#identities-clear");
    if (clr) clr.addEventListener("click", () => clearAll());
  }

  function teardown() {
    state.selected.clear();
  }

  // Open an identity's detail from anywhere (e.g. a face group or an event).
  function openById(id) {
    if (typeof switchView === "function") switchView("identities");
    openDetail(id);
  }

  window.Identities = { init, render, teardown, openById };
})();
