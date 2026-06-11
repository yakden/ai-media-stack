/* Face Groups view — unsupervised grouping of similar faces, with settings.
 * Loaded after app.js; shares globals (el, $, $$, api, modal, toast, fmtTime).
 * Settings (similarity / +clothing / min group size) recluster on the server. */
"use strict";

(function () {
  let _timer = null;

  // Inline placeholder shown when a face crop isn't on disk (e.g. backfilled
  // vector-only samples). Keeps the layout clean instead of a broken image.
  const PLACEHOLDER =
    "data:image/svg+xml;utf8," + encodeURIComponent(
      "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'>" +
      "<rect width='64' height='64' fill='%23111'/>" +
      "<text x='32' y='38' font-size='11' fill='%23666' text-anchor='middle'>no crop</text></svg>"
    );
  function faceImg(attrs) {
    const img = el("img", attrs);
    img.addEventListener("error", function () { this.onerror = null; this.src = PLACEHOLDER; });
    return img;
  }

  function settings() {
    return {
      face_threshold: parseFloat($("#faces-threshold").value),
      clothing_weight: parseFloat($("#faces-clothing").value),
      min_size: parseInt($("#faces-minsize").value || "1", 10),
      max_members: 24,
    };
  }

  async function render() {
    const grid = $("#faces-grid");
    grid.innerHTML = "";
    grid.append(el("div", { class: "empty" }, "Grouping…"));
    let data;
    try {
      data = await api.get("/api/face-groups?" + new URLSearchParams(settings()).toString());
    } catch (e) {
      grid.innerHTML = "";
      grid.append(el("div", { class: "empty" }, "Error: " + e.message));
      return;
    }
    $("#faces-stats").textContent =
      `${data.total_samples} faces → ${data.total_groups} groups`;
    grid.innerHTML = "";
    if (!data.groups.length) {
      grid.append(el("div", { class: "empty" },
        "No faces captured yet. They are collected automatically when a face is visible on a person; come back after the cameras have seen some faces."));
      return;
    }
    for (const g of data.groups) grid.append(groupCard(g));
  }

  function groupCard(g) {
    const head = el("div", { class: "fg-head" },
      faceImg({ class: "fg-rep", src: g.representative.thumb_url, alt: "face", loading: "lazy" }),
      el("div", { class: "fg-meta" },
        el("div", { class: "fg-title" }, g.label || `Group · ${g.size} face${g.size === 1 ? "" : "s"}`),
        el("div", { class: "fg-sub subtle" },
          `${g.size} samples · ${(g.cameras || []).length} cam${(g.cameras || []).length === 1 ? "" : "s"}`),
        el("div", { class: "fg-sub subtle" }, "last " + fmtTime(g.last_seen))));

    const strip = el("div", { class: "fg-strip" });
    for (const m of g.members) {
      strip.append(faceImg({ class: "fg-thumb", src: m.thumb_url, alt: "face",
        loading: "lazy", title: fmtTime(m.ts) }));
    }

    // Linked identities present in this group (per-member; groups can be mixed).
    const identIds = Array.from(new Set((g.members || []).map((m) => m.identity_id).filter((x) => x != null)));
    const links = el("div", { class: "fg-links" });
    for (const iid of identIds.slice(0, 6)) {
      links.append(el("button", {
        class: "btn btn-sm", title: "Open this identity",
        onclick: () => { if (window.Identities) window.Identities.openById(iid); },
      }, `Identity #${iid}`));
    }

    const actions = el("div", { class: "fg-actions" },
      el("button", { class: "btn btn-sm", onclick: () => reviewVideo(g) }, "Review video"),
      el("button", { class: "btn btn-sm", onclick: () => nameGroup(g) }, g.label ? "Rename" : "Label"),
      el("button", { class: "btn btn-sm btn-primary", onclick: () => makePerson(g) }, "Make a person"));

    return el("div", { class: "fg-card" }, head, strip, identIds.length ? links : null, actions);
  }

  async function reviewVideo(g) {
    let resp;
    try {
      resp = await api.post("/api/face-groups/clips", { sample_ids: g.member_sample_ids });
    } catch (e) { toast(e.message, "error"); return; }
    const items = resp.items || [];
    const body = el("div", {});
    const renderList = () => {
      body.innerHTML = "";
      if (!items.length) {
        body.append(el("div", { class: "empty" }, "No recorded clips for this group yet."));
        return;
      }
      const grid = el("div", { class: "clip-list" });
      for (const ev of items) {
        grid.append(el("div", { class: "clip-tile", onclick: () => playClip(ev) },
          ev.thumb_url ? el("img", { src: ev.thumb_url, loading: "lazy", alt: "" })
                       : el("div", { class: "no-thumb" }, "clip"),
          el("div", { class: "clip-meta" },
            el("span", {}, ev.camera_name || `Camera #${ev.camera_id}`),
            el("span", { class: "subtle" }, fmtTime(ev.ts)))));
      }
      body.append(grid);
    };
    const playClip = (ev) => {
      body.innerHTML = "";
      body.append(el("button", { class: "btn btn-sm", onclick: renderList }, "← Back to clips"));
      // buildVideoViewer wants clip_url (EventListItem has it).
      body.append(buildVideoViewer(ev));
    };
    renderList();
    modal.open(g.label ? `Clips · ${g.label}` : `Clips · group ${g.group_id}`, body, { wide: true });
  }

  async function nameGroup(g) {
    const name = prompt("Name for this face group" + (g.label ? ` (current: ${g.label})` : "") + ":", g.label || "");
    if (name == null) return;
    const label = name.trim();
    if (!label) return;
    try {
      const r = await api.post("/api/face-groups/label", { sample_ids: g.member_sample_ids, label });
      toast(`Labelled ${r.updated} faces as "${label}"`, "ok");
      render();
    } catch (e) { toast(e.message, "error"); }
  }

  function debouncedRender() {
    clearTimeout(_timer);
    _timer = setTimeout(render, 250);
  }

  async function makePerson(g) {
    const name = prompt("Create a known person from this face group — name:", g.label || "");
    if (name == null) return;
    const nm = name.trim();
    if (!nm) return;
    try {
      const r = await api.post("/api/face-groups/enroll", { sample_ids: g.member_sample_ids, name: nm });
      toast(`Created person "${r.name}" (${r.enrolled_faces} faces enrolled)`, "ok");
      render();
    } catch (e) { toast(e.message, "error"); }
  }

  async function clearAll() {
    if (!confirm("Delete ALL captured face samples (clears every face group)? This cannot be undone.")) return;
    try {
      const r = await api.post("/api/face-groups/samples/clear-all", { confirm: true });
      toast(`Deleted ${r.deleted} face samples`, "ok");
      render();
    } catch (e) { toast(e.message, "error"); }
  }

  function init() {
    const refresh = $("#faces-refresh");
    if (refresh) refresh.addEventListener("click", () => render());
    const clr = $("#faces-clear");
    if (clr) clr.addEventListener("click", () => clearAll());
    const th = $("#faces-threshold"), cl = $("#faces-clothing"), ms = $("#faces-minsize");
    if (th) th.addEventListener("input", () => { $("#faces-threshold-val").textContent = parseFloat(th.value).toFixed(2); debouncedRender(); });
    if (cl) cl.addEventListener("input", () => { $("#faces-clothing-val").textContent = parseFloat(cl.value).toFixed(2); debouncedRender(); });
    if (ms) ms.addEventListener("change", () => render());
  }

  function teardown() { clearTimeout(_timer); }

  window.Faces = { init, render, teardown };
})();
