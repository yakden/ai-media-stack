/* VMS People — analytical cross-camera unique-people dashboard.
 * Renders the auto person-class Identities as a face-first people database +
 * analytics. Reuses app.js globals: el, api, toast, modal, fmtTime.
 * Exposes window.People = { init, render, teardown }.
 */
"use strict";
(function () {
  const root = () => document.getElementById("view-people");
  const state = { items: [], analytics: null, q: "", filter: "all", sort: "recent", cams: {} };

  function fmtDur(sec) {
    sec = Math.round(sec || 0);
    if (sec < 60) return sec + "с";
    if (sec < 3600) return Math.floor(sec / 60) + "м " + (sec % 60) + "с";
    return Math.floor(sec / 3600) + "ч " + Math.floor((sec % 3600) / 60) + "м";
  }
  function ago(iso) {
    if (!iso) return "—";
    const d = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
    if (d < 0) return "только что";
    if (d < 60) return d + "с назад";
    if (d < 3600) return Math.floor(d / 60) + "м назад";
    if (d < 86400) return Math.floor(d / 3600) + "ч назад";
    return Math.floor(d / 86400) + "д назад";
  }
  function faceImg(url, cls) {
    if (!url) return el("div", { class: cls + " ppl-noface" }, "?");
    const img = el("img", { class: cls, src: url, loading: "lazy" });
    img.addEventListener("error", function () { this.classList.add("ppl-noface"); this.removeAttribute("src"); });
    return img;
  }

  async function render() {
    const v = root();
    if (!v) return;
    v.innerHTML = "";
    v.append(el("div", { class: "empty" }, "Загрузка…"));
    let a, list, cams;
    try {
      [a, list, cams] = await Promise.all([
        api.get("/api/identities/analytics?object_class=person"),
        api.get("/api/identities?object_class=person&limit=500"),
        api.get("/api/cameras").catch(() => ({ items: [] })),
      ]);
    } catch (e) {
      v.innerHTML = "";
      v.append(el("div", { class: "empty" }, "Ошибка: " + e.message));
      return;
    }
    state.analytics = a;
    state.items = list.items || [];
    state.cams = {};
    (cams.items || cams || []).forEach((c) => (state.cams[c.id] = c.name));
    paint();
  }

  function paint() {
    const v = root();
    v.innerHTML = "";
    const head = el("div", { class: "ppl-head" },
      el("h2", {}, "Люди — кросс-камерная база"),
      (() => { const b = el("button", { class: "btn primary" }, "+ Добавить по фото"); b.onclick = openEnroll; return b; })(),
    );
    v.append(head, kpis(state.analytics.summary), charts(state.analytics), controls());
    const g = el("div", { id: "ppl-gallery" });
    g.append(galleryInner());
    v.append(g);
  }

  function kpis(s) {
    const card = (label, val, sub) =>
      el("div", { class: "ppl-kpi" },
        el("div", { class: "ppl-kpi-v" }, String(val)),
        el("div", { class: "ppl-kpi-l" }, label),
        el("div", { class: "ppl-kpi-s" }, sub || ""));
    return el("div", { class: "ppl-kpis" },
      card("Уникальных людей", s.total_people, s.named + " названо"),
      card("За сегодня", s.created_today, "новых"),
      card("Видели сегодня", s.seen_today, "за 7д: " + s.seen_7d),
      card("Появлений", s.total_sightings, "всего"));
  }

  function charts(a) {
    const wrap = el("div", { class: "ppl-charts" });
    const maxH = Math.max(1, ...a.by_hour.map((h) => h.sightings));
    const hc = el("div", { class: "ppl-card" }, el("div", { class: "ppl-card-t" }, "Появления за 24ч"));
    const bars = el("div", { class: "ppl-hbars" });
    a.by_hour.forEach((h) => {
      const bar = el("div", { class: "ppl-hbar", title: h.hour + ":00 — " + h.sightings });
      bar.append(el("i", { style: "height:" + Math.round((h.sightings / maxH) * 100) + "%" }));
      bars.append(bar);
    });
    hc.append(bars);
    wrap.append(hc);

    const cc = el("div", { class: "ppl-card" }, el("div", { class: "ppl-card-t" }, "По камерам"));
    const maxC = Math.max(1, ...a.by_camera.map((c) => c.sightings));
    if (!a.by_camera.length) cc.append(el("div", { class: "ppl-muted" }, "нет данных"));
    a.by_camera.slice(0, 6).forEach((c) => {
      const row = el("div", { class: "ppl-crow" },
        el("span", { class: "ppl-cname" }, c.camera_name),
        (() => { const w = el("span", { class: "ppl-cbar" }); w.append(el("i", { style: "width:" + Math.round((c.sightings / maxC) * 100) + "%" })); return w; })(),
        el("span", { class: "ppl-cval" }, c.people + " чел · " + c.sightings));
      cc.append(row);
    });
    wrap.append(cc);
    return wrap;
  }

  function controls() {
    const bar = el("div", { class: "ppl-controls" });
    const search = el("input", { type: "search", placeholder: "Поиск по имени / #id…", value: state.q });
    search.oninput = (e) => { state.q = e.target.value; renderGallery(); };
    const filt = el("select", {},
      el("option", { value: "all" }, "Все"),
      el("option", { value: "named" }, "Названные"),
      el("option", { value: "unnamed" }, "Безымянные"));
    filt.value = state.filter;
    filt.onchange = (e) => { state.filter = e.target.value; renderGallery(); };
    const sort = el("select", {},
      el("option", { value: "recent" }, "Недавние"),
      el("option", { value: "most" }, "Чаще видели"),
      el("option", { value: "cameras" }, "Больше камер"));
    sort.value = state.sort;
    sort.onchange = (e) => { state.sort = e.target.value; renderGallery(); };
    bar.append(search, filt, sort);
    return bar;
  }

  function filtered() {
    let it = state.items.slice();
    if (state.filter === "named") it = it.filter((x) => x.is_named);
    if (state.filter === "unnamed") it = it.filter((x) => !x.is_named);
    if (state.q) {
      const q = state.q.toLowerCase();
      it = it.filter((x) => (x.name || "").toLowerCase().includes(q) || String(x.id).includes(q));
    }
    if (state.sort === "most") it.sort((a, b) => b.num_sightings - a.num_sightings);
    else if (state.sort === "cameras") it.sort((a, b) => (b.cameras || []).length - (a.cameras || []).length);
    else it.sort((a, b) => new Date(b.last_seen || 0) - new Date(a.last_seen || 0));
    return it;
  }

  function renderGallery() {
    const g = document.getElementById("ppl-gallery");
    if (!g) return;
    g.innerHTML = "";
    g.append(galleryInner());
  }
  function galleryInner() {
    const it = filtered();
    if (!it.length)
      return el("div", { class: "empty" },
        "Пока нет людей. Они появляются автоматически по мере детекции на камерах, либо добавь вручную по фото.");
    const grid = el("div", { class: "ppl-grid" });
    it.forEach((p) => grid.append(personCard(p)));
    return grid;
  }

  function personCard(p) {
    const name = p.is_named && p.name ? p.name : "Человек #" + p.id;
    const card = el("div", { class: "ppl-pcard" },
      faceImg(p.face_thumb_url || p.rep_thumb_url, "ppl-face"),
      el("div", { class: "ppl-pinfo" },
        el("div", { class: "ppl-pname" + (p.is_named ? " named" : "") }, name),
        el("div", { class: "ppl-pbadges" },
          el("span", { class: "badge" }, (p.cameras || []).length + " 📷"),
          el("span", { class: "badge" }, p.num_sightings + " появл.")),
        el("div", { class: "ppl-pmeta" }, ago(p.last_seen))));
    card.onclick = () => openDetail(p.id);
    return card;
  }

  async function openDetail(id) {
    let d, faces, sights;
    try {
      [d, faces, sights] = await Promise.all([
        api.get("/api/identities/" + id),
        api.get("/api/identities/" + id + "/faces").catch(() => ({ items: [] })),
        api.get("/api/identities/" + id + "/sightings?limit=60").catch(() => ({ items: [] })),
      ]);
    } catch (e) {
      toast("Ошибка: " + e.message, "error");
      return;
    }
    const body = el("div", { class: "ppl-detail" });
    const nameInput = el("input", { class: "ppl-rename", value: d.name || "", placeholder: "Имя человека" });
    const saveBtn = el("button", { class: "btn primary" }, "Сохранить имя");
    saveBtn.onclick = async () => {
      try { await api.put("/api/identities/" + id, { name: nameInput.value.trim() }); toast("Сохранено", "ok"); modal.close(); render(); }
      catch (e) { toast("Ошибка", "error"); }
    };
    body.append(el("div", { class: "ppl-drow" }, nameInput, saveBtn));

    const dg = el("div", { class: "detail-grid" });
    const kv = (k, val) => { dg.append(el("div", { class: "dg-k" }, k)); dg.append(el("div", { class: "dg-v" }, val)); };
    kv("Появлений", String(d.num_sightings));
    kv("Камер", String((d.cameras || []).length));
    kv("В кадре всего", fmtDur(d.total_seconds));
    kv("Впервые", fmtTime(d.first_seen));
    kv("Последний раз", fmtTime(d.last_seen));
    body.append(dg);

    if ((faces.items || []).length) {
      body.append(el("div", { class: "ppl-sect" }, "Лица"));
      const fg = el("div", { class: "ppl-facestrip" });
      faces.items.slice(0, 20).forEach((f) => fg.append(faceImg(f.thumb_url, "ppl-fthumb")));
      body.append(fg);
    }
    if ((sights.items || []).length) {
      body.append(el("div", { class: "ppl-sect" }, "Где и когда видели (кросс-камера)"));
      const tl = el("div", { class: "ppl-timeline" });
      sights.items.forEach((s) => {
        const cam = s.camera_name || state.cams[s.camera_id] || "#" + s.camera_id;
        tl.append(el("div", { class: "ppl-tlrow" },
          faceImg(s.thumb_url, "ppl-tlthumb"),
          el("span", { class: "ppl-tlcam" }, "📷 " + cam),
          el("span", { class: "ppl-tlkind badge" }, s.match_kind || ""),
          el("span", { class: "ppl-tltime" }, fmtTime(s.ts))));
      });
      body.append(tl);
    }
    const del = el("button", { class: "btn danger" }, "Удалить личность");
    del.onclick = async () => {
      if (!confirm("Удалить эту личность?")) return;
      try { await api.del("/api/identities/" + id); toast("Удалено"); modal.close(); render(); }
      catch (e) { toast("Ошибка", "error"); }
    };
    body.append(el("div", { class: "ppl-dactions" }, del));
    modal.open(d.is_named && d.name ? d.name : "Человек #" + id, body, { wide: true });
  }

  function openEnroll() {
    const body = el("div", { class: "ppl-enroll" });
    const nameI = el("input", { placeholder: "Имя человека" });
    const fileI = el("input", { type: "file", accept: "image/*", multiple: true });
    const status = el("div", { class: "ppl-enroll-status" });
    const go = el("button", { class: "btn primary" }, "Создать и загрузить");
    go.onclick = async () => {
      const name = nameI.value.trim();
      if (!name) { toast("Введите имя", "error"); return; }
      const files = Array.from(fileI.files || []);
      if (!files.length) { toast("Выберите фото", "error"); return; }
      try {
        const person = await api.post("/api/people", { name });
        let ok = 0;
        for (const f of files) {
          const fd = new FormData(); fd.append("file", f);
          try { await api.postForm("/api/people/" + person.id + "/faces", fd); ok++; } catch (e) {}
          status.textContent = ok + " / " + files.length + " фото загружено…";
        }
        toast(ok + " фото для «" + name + "»", "ok");
        modal.close();
      } catch (e) { toast("Ошибка: " + e.message, "error"); }
    };
    body.append(
      el("p", { class: "ppl-hint" }, "3–5 хороших фронтальных фото, лицо чётко. Так человек начнёт надёжно узнаваться на камерах."),
      nameI, fileI, go, status);
    modal.open("Добавить человека по фото", body, { wide: false });
  }

  window.People = { render, teardown: function () {}, init: function () {} };
})();
