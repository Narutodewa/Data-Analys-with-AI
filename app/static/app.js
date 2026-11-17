/* Data Analyst with AI - antarmuka (vanilla JS, tanpa build step) */
const $ = (s, r = document) => r.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const S = { list: [], cur: null, tab: "data", page: { offset: 0, sort: null, desc: false, q: "" }, chat: [], health: null };
const TABS = [["data", "Data"], ["clean", "Cleaning"], ["eda", "EDA"], ["charts", "Grafik & Rekomendasi"], ["ask", "Tanya AI"], ["pivot", "Pivot & Rumus"]];

async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) { o.method = o.method || "POST"; o.headers = { "content-type": "application/json" }; o.body = JSON.stringify(o.json); delete o.json; }
  const r = await fetch("/api" + path, o);
  let j = null; try { j = await r.json(); } catch { /* bukan JSON */ }
  if (!r.ok) throw new Error((j && (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail))) || r.statusText);
  return j;
}
function toast(msg, err) { const d = document.createElement("div"); d.textContent = msg; if (err) d.className = "err"; $("#toast").append(d); setTimeout(() => d.remove(), err ? 7000 : 3500); }
const guard = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message, true); } };
const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString("id-ID") : v.toLocaleString("id-ID", { maximumFractionDigits: 4 })) : v);
const cur = () => S.cur;

/* ---------------------------------------------------------------- sidebar */
function renderSide() {
  $("#dslist").innerHTML = S.list.map((d) => `<li data-id="${d.id}" class="${cur() === d.id ? "on" : ""}"><div>${esc(d.name)}<small>${{ table: "Tabel", nosql: "NoSQL", text: "Teks", image: "Citra" }[d.kind]} · ${d.rows.toLocaleString("id-ID")} × ${d.cols}</small></div><button class="x" data-del="${d.id}" title="Hapus">✕</button></li>`).join("") || '<li class="muted">Belum ada data</li>';
  const ai = S.health?.ai;
  $("#ai-chip").innerHTML = ai?.enabled ? `AI: <b>aktif</b> · ${esc(ai.provider)} / ${esc(ai.model)}` : "AI: nonaktif (mode heuristik offline)";
}
async function refreshList(selectId) {
  S.list = (await api("/datasets")).datasets;
  if (selectId) cur() !== selectId && (S.cur = selectId);
  if (!S.list.find((d) => d.id === S.cur)) S.cur = S.list[0]?.id || null;
  renderSide(); renderMain();
}
function addDatasets(res) {
  (res.errors || []).forEach((e) => toast(`${e.file}: ${e.error}`, true));
  if (res.datasets?.length) { toast(`${res.datasets.length} dataset ditambahkan`); S.tab = "data"; S.page = { offset: 0, sort: null, desc: false, q: "" }; S.chat = []; return refreshList(res.datasets[0].id); }
}
async function upload(files) {
  const fd = new FormData(); [...files].forEach((f) => fd.append("files", f));
  toast("Mengunggah…"); await addDatasets(await api("/upload", { method: "POST", body: fd }));
}
const drop = $("#drop");
drop.onclick = () => $("#file").click();
drop.onkeydown=(e)=>{if(e.key==="Enter")$("#file").click()};
$("#file").onchange = guard((e) => { upload(e.target.files); e.target.value = ""; });
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", guard((e) => upload(e.dataTransfer.files)));
$("#btn-sample").onclick = guard(async () => addDatasets(await api("/sample", { method: "POST" })));
$("#dslist").onclick = guard(async (e) => {
  const del = e.target.closest("[data-del]");
  if (del) { await api(`/datasets/${del.dataset.del}`, { method: "DELETE" }); if (cur() === del.dataset.del) S.cur = null; return refreshList(); }
  const li = e.target.closest("li[data-id]"); if (li) { S.cur = li.dataset.id; S.page = { offset: 0, sort: null, desc: false, q: "" }; S.chat = []; renderSide(); renderMain(); }
});
$("#btn-paste").onclick = () => {
  const d = $("#dlg");
  d.innerHTML = `<h3>Tempel data</h3><p class="muted">CSV, JSON, JSON Lines, perintah SQL (CREATE/INSERT), atau teks bebas.</p><textarea id="pt" rows="10" placeholder="nama,umur&#10;Ani,23&#10;Budi,31"></textarea><div class="row" style="margin-top:10px;justify-content:flex-end"><button id="pc">Batal</button><button class="primary" id="pg">Muat</button></div>`;
  d.showModal(); $("#pc").onclick = () => d.close();
  $("#pg").onclick = guard(async () => { const r = await api("/paste", { json: { text: $("#pt").value } }); d.close(); addDatasets(r); });
};
$("#btn-ai").onclick = () => {
  const d = $("#dlg"), a = S.health.ai;
  d.innerHTML = `<h3>Pengaturan AI</h3>
  <p class="muted">Tanpa AI aplikasi tetap berfungsi (analisis heuristik). Kunci hanya disimpan di memori server lokal, hilang saat restart. Untuk permanen, isi file <code>.env</code>.</p>
  <div class="form" style="flex-direction:column;align-items:stretch">
   <label>Penyedia<select id="a-p"><option value="anthropic">Anthropic (Claude)</option><option value="openai">OpenAI / kompatibel (Ollama, LM Studio…)</option></select></label>
   <label>Model (kosongkan untuk default)<input id="a-m" value="${esc(a.model)}"></label>
   <label>API key<input id="a-k" type="password" placeholder="${a.enabled ? "•••••• (tersimpan)" : "sk-…"}"></label>
   <label>Base URL (khusus OpenAI-kompatibel)<input id="a-u" value="${esc(a.base_url || "https://api.openai.com/v1")}"></label>
   <label style="flex-direction:row;align-items:center;gap:8px"><input id="a-s" type="checkbox" ${a.send_sample ? "checked" : ""}> Kirim 5 baris contoh ke AI (matikan untuk data sensitif; hanya skema+statistik dikirim)</label>
  </div><div class="row" style="margin-top:12px;justify-content:flex-end"><button id="ac">Tutup</button><button class="primary" id="as">Simpan</button></div>`;
  $("#a-p").value = a.provider; d.showModal(); $("#ac").onclick = () => d.close();
  $("#as").onclick = guard(async () => {
    S.health.ai = await api("/ai/config", { json: { provider: $("#a-p").value, model: $("#a-m").value, api_key: $("#a-k").value || null, base_url: $("#a-u").value, send_sample: $("#a-s").checked } });
    d.close(); renderSide(); toast("Pengaturan AI disimpan");
  });
};

/* ------------------------------------------------------------------- main */
function renderMain() {
  const ds = S.list.find((d) => d.id === cur());
  $("#empty").hidden = !!ds; $("#work").hidden = !ds;
  if (!ds) return;
  $("#head").innerHTML = `<div><h2>${esc(ds.name)}</h2><small>${ds.rows.toLocaleString("id-ID")} baris × ${ds.cols} kolom${ds.meta?.note ? " · " + esc(ds.meta.note) : ""}</small></div>
   <div class="row"><button id="h-undo" ${ds.can_undo ? "" : "disabled"}>↶ Urungkan</button><button id="h-reset" class="danger">Reset ke asli</button>
   <button data-ex="csv">⬇ CSV</button><button data-ex="xlsx">⬇ Excel</button><button data-ex="json">⬇ JSON</button></div>`;
  $("#h-undo").onclick = guard(async () => { const r = await api(`/datasets/${ds.id}/undo`, { method: "POST" }); toast("Dibatalkan: " + r.undone); refreshList(); });
  $("#h-reset").onclick = guard(async () => { if (confirm("Kembalikan data ke kondisi awal? Semua perubahan hilang.")) { await api(`/datasets/${ds.id}/reset`, { method: "POST" }); refreshList(); } });
  $("#head").querySelectorAll("[data-ex]").forEach((b) => (b.onclick = () => (location.href = `/api/datasets/${ds.id}/export?fmt=${b.dataset.ex}`)));
  $("#tabs").innerHTML = TABS.map(([k, l]) => `<button data-t="${k}" class="${S.tab === k ? "on" : ""}">${l}</button>`).join("");
  $("#tabs").onclick = (e) => { const b = e.target.closest("[data-t]"); if (b) { S.tab = b.dataset.t; renderMain(); } };
  ({ data: viewData, clean: viewClean, eda: viewEda, charts: viewCharts, ask: viewAsk, pivot: viewPivot }[S.tab])(ds);
}
const view = () => $("#view");
const loading = (t = "Memuat…") => (view().innerHTML = `<p class="muted spin">${t}</p>`);

/* ------------------------------------------------------------ plotly util */
function plot(el, fig) {
  if (typeof Plotly === "undefined") { el.innerHTML = '<p class="muted">Pustaka grafik (Plotly) belum termuat. Periksa koneksi internet, atau jalankan <code>scripts/vendor_plotly.sh</code> untuk mode offline.</p>'; return; }
  const cs = getComputedStyle(document.body), ink = cs.getPropertyValue("--ink").trim(), line = cs.getPropertyValue("--line").trim();
  const layout = JSON.parse(JSON.stringify(fig.layout));
  layout.paper_bgcolor = "rgba(0,0,0,0)"; layout.plot_bgcolor = "rgba(0,0,0,0)"; layout.font = { color: ink, size: 12 };
  for (const ax of ["xaxis", "yaxis"]) layout[ax] = { ...(layout[ax] || {}), gridcolor: line, zerolinecolor: line, automargin: true };
  layout.legend = { orientation: "h", y: -0.2 }; layout.hovermode = layout.hovermode || "closest";
  Plotly.newPlot(el, fig.data, layout, { responsive: true, displaylogo: false });
}
function chartCard(c, withSave) {
  const d = document.createElement("div"); d.className = "card";
  d.innerHTML = `${c.why ? `<p class="why">💡 ${esc(c.why)}</p>` : ""}<div class="chart"></div>`;
  setTimeout(() => plot($(".chart", d), c.figure), 0);
  return d;
}

/* ------------------------------------------------------------------- DATA */
async function viewData(ds) {
  const p = S.page, qs = new URLSearchParams({ offset: p.offset, limit: 100, desc: p.desc, ...(p.sort ? { sort: p.sort } : {}), ...(p.q ? { q: p.q } : {}) });
  loading();
  const r = await api(`/datasets/${ds.id}/rows?${qs}`).catch((e) => (toast(e.message, true), null)); if (!r) return;
  const isImg = ds.kind === "image";
  let html = "";
  if (ds.meta?.schema?.length) html += `<details class="card"><summary><b>Skema dokumen NoSQL</b> (${ds.meta.schema.length} field)</summary><div class="tablewrap" style="margin-top:8px"><table><tr><th>Field</th><th>Tipe</th><th>Cakupan</th><th>Contoh</th></tr>${ds.meta.schema.map((s) => `<tr><td>${esc(s.field)}</td><td>${esc(s.types.join(", "))}</td><td><div class="bar"><i style="width:${s.coverage_pct}%"></i></div>${s.coverage_pct}%</td><td>${esc(s.example)}</td></tr>`).join("")}</table></div></details>`;
  html += `<div class="toolbar"><input id="q" placeholder="🔎 Cari di semua kolom…" value="${esc(p.q)}" style="min-width:240px"><button id="qs">Cari</button><span class="muted">${r.total.toLocaleString("id-ID")} baris</span><span class="grow"></span><button id="del-sel" class="danger" disabled>Hapus baris terpilih</button><button id="drop-col">Hapus kolom…</button></div>`;
  html += `<div class="tablewrap"><table><thead><tr><th></th>${isImg ? "<th></th>" : ""}<th data-sort="_row">#</th>${r.columns.map((c) => `<th data-sort="${esc(c)}">${esc(c)}${p.sort === c ? (p.desc ? " ▼" : " ▲") : ""}</th>`).join("")}</tr></thead><tbody>`;
  html += r.rows.map((row) => `<tr data-row="${row._row}"><td><input type="checkbox" class="sel"></td>${isImg ? `<td><img class="thumb" loading="lazy" src="/api/datasets/${ds.id}/thumb?name=${encodeURIComponent(row.filename)}"></td>` : ""}<td class="rownum">${row._row}</td>${r.columns.map((c) => cell(row[c], c)).join("")}</tr>`).join("");
  html += `</tbody></table></div><div class="toolbar" style="margin-top:10px"><button id="pv" ${p.offset ? "" : "disabled"}>← Sebelumnya</button><span class="muted">${r.total ? p.offset + 1 : 0}–${Math.min(p.offset + 100, r.total)} dari ${r.total}</span><button id="nx" ${p.offset + 100 < r.total ? "" : "disabled"}>Berikutnya →</button><span class="muted">· klik dua kali sel untuk mengedit · klik judul kolom untuk mengurutkan</span></div>`;
  view().innerHTML = html;
  const go = () => { p.q = $("#q").value; p.offset = 0; renderMain(); };
  $("#qs").onclick = go; $("#q").onkeydown=(e)=>{if(e.key==="Enter")go()};
  $("#pv").onclick = () => { p.offset = Math.max(0, p.offset - 100); renderMain(); };
  $("#nx").onclick = () => { p.offset += 100; renderMain(); };
  view().querySelectorAll("th[data-sort]").forEach((th) => (th.onclick = () => { const c = th.dataset.sort; if (c === "_row") { p.sort = null; p.desc = false; } else { p.desc = p.sort === c ? !p.desc : false; p.sort = c; } p.offset = 0; renderMain(); }));
  const upd = () => ($("#del-sel").disabled = !view().querySelector(".sel:checked"));
  view().querySelectorAll(".sel").forEach((c) => (c.onchange = upd));
  $("#del-sel").onclick = guard(async () => { const rows = [...view().querySelectorAll(".sel:checked")].map((c) => +c.closest("tr").dataset.row); if (confirm(`Hapus ${rows.length} baris?`)) { await api(`/datasets/${ds.id}/rows/delete`, { json: { rows } }); refreshList(); } });
  $("#drop-col").onclick = guard(async () => { const c = prompt("Nama kolom yang dihapus:\n" + r.columns.join(", ")); if (c && r.columns.includes(c)) { await api(`/datasets/${ds.id}/columns/drop`, { json: { cols: [c] } }); refreshList(); } else if (c) toast("Kolom tidak ditemukan", true); });
  view().querySelectorAll("td[data-col]").forEach((td) => (td.ondblclick = () => editCell(ds, td)));
}
function cell(v, c) { const n = v === null || v === undefined; return `<td data-col="${esc(c)}" class="${n ? "null" : typeof v === "number" ? "num" : ""}" title="${esc(v)}">${n ? "kosong" : esc(typeof v === "number" ? fmt(v) : v)}</td>`; }
function editCell(ds, td) {
  const row = td.closest("tr").dataset.row, col = td.dataset.col, old = td.classList.contains("null") ? "" : td.title;
  td.classList.add("edit"); td.innerHTML = `<input value="${esc(old)}">`; const inp = $("input", td); inp.focus(); inp.select();
  let done = false;
  const save = guard(async () => { if (done) return; done = true; if (inp.value === old) return renderMain(); await api(`/datasets/${ds.id}/cell`, { method: "PUT", json: { row: +row, col, value: inp.value } }); toast("Sel diperbarui"); refreshList(); });
  inp.onkeydown = (e) => { if (e.key === "Enter") save(); if (e.key === "Escape") { done = true; renderMain(); } };
  inp.onblur = () => save().catch(() => { done = false; renderMain(); });
}

/* ---------------------------------------------------------------- CLEANING */
async function viewClean(ds) {
  loading("Memindai masalah data…");
  const r = await api(`/datasets/${ds.id}/issues`).catch((e) => (toast(e.message, true), null)); if (!r) return;
  let html = `<div class="toolbar"><b>${r.issues.length} masalah ditemukan</b>${r.ignored ? `<span class="muted">(${r.ignored} diabaikan)</span><button id="unig" class="sm">Tampilkan lagi</button>` : ""}<span class="grow"></span>
   <button id="adv">🧠 Saran AI</button><button id="auto" class="primary" ${r.issues.length ? "" : "disabled"}>✨ Terapkan semua rekomendasi</button></div><div id="advice"></div>`;
  if (!r.issues.length) html += `<div class="card">✅ Tidak ada masalah kualitas yang terdeteksi.</div>`;
  const sev = { high: "Tinggi", medium: "Sedang", low: "Rendah" };
  html += r.issues.map((i, n) => `<div class="card issue ${i.severity}" data-i="${n}"><h3>${esc(i.title)} <span class="tag">${sev[i.severity]}</span></h3><p>${esc(i.detail)}</p>
   ${i.samples?.length ? `<div class="chips">${i.samples.map((s) => `<span>baris ${s.row}: <b>${esc(s.value)}</b></span>`).join("")}${i.count > i.samples.length ? "<span>…</span>" : ""}</div>` : ""}
   ${Object.keys(i.stats || {}).filter((k) => k !== "mapping").length ? `<div class="chips">${Object.entries(i.stats).filter(([k]) => k !== "mapping").map(([k, v]) => `<span>${{ mean: "rata-rata", median: "median", mode: "modus", lower: "batas bawah", upper: "batas atas" }[k] || k}: <b>${esc(fmt(v))}</b></span>`).join("")}</div>` : ""}
   <div class="acts">${i.actions.map((a) => `<button data-a="${a.key}" class="${a.key === i.recommended ? "rec" : ""}" ${a.needs_value ? 'data-nv="1"' : ""}>${esc(a.label)}${a.key === i.recommended ? " ★" : ""}</button>`).join("")}
   ${i.rows.length && i.type !== "constant" ? `<button data-edit="1">✏ Edit manual…</button>` : ""}</div><div class="manual"></div></div>`).join("");
  view().innerHTML = html;
  $("#unig") && ($("#unig").onclick = guard(async () => { await api(`/datasets/${ds.id}/issues/unignore`, { method: "POST" }); renderMain(); }));
  $("#auto").onclick = guard(async () => { const x = await api(`/datasets/${ds.id}/clean/auto`, { method: "POST" }); toast(x.messages.length + " langkah diterapkan (bisa diurungkan satu per satu)"); refreshList(); });
  $("#adv").onclick = guard(async () => { $("#advice").innerHTML = '<div class="card spin">Menyusun saran…</div>'; const x = await api(`/datasets/${ds.id}/ai/clean-advice`); $("#advice").innerHTML = `<div class="card msg ai">${esc(x.advice)}<br><small class="muted">${x.mode === "ai" ? "Disusun oleh AI" : "Mode offline"}</small></div>`; });
  view().querySelectorAll(".issue").forEach((card) => {
    const i = r.issues[+card.dataset.i];
    card.querySelectorAll("[data-a]").forEach((b) => (b.onclick = guard(async () => {
      let value = null; if (b.dataset.nv) { value = prompt("Isi dengan nilai:"); if (value === null) return; }
      const x = await api(`/datasets/${ds.id}/clean`, { json: { issue_id: i.id, action: b.dataset.a, value } }); toast(x.message); refreshList();
    })));
    const eb = $("[data-edit]", card); if (eb) eb.onclick = guard(() => manualEditor(ds, i, $(".manual", card)));
  });
}
async function manualEditor(ds, issue, box) {
  if (box.innerHTML) { box.innerHTML = ""; return; }
  const rows = issue.rows.slice(0, 25), cols = issue.column ? [issue.column] : null;
  const r = await api(`/datasets/${ds.id}/rows?limit=100&ids=${rows.join(",")}`);
  const byRow = Object.fromEntries(r.rows.map((x) => [x._row, x]));
  const show = cols || r.columns.slice(0, 6);
  box.innerHTML = `<div class="tablewrap" style="margin-top:10px;max-height:300px"><table><tr><th>Baris</th>${show.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${rows.map((rw) => { const x = byRow[rw]; return x ? `<tr data-row="${rw}"><td>${rw}</td>${show.map((c) => `<td><input data-col="${esc(c)}" value="${esc(x[c] ?? "")}" placeholder="kosong"></td>`).join("")}</tr>` : ""; }).join("")}</table></div><p class="muted">Ubah nilai lalu tekan Enter untuk menyimpan. ${issue.count > 25 ? "Menampilkan 25 baris pertama." : ""}</p>`;
  box.querySelectorAll("input").forEach((inp) => (inp.onkeydown = guard(async (e) => { if (e.key !== "Enter") return; await api(`/datasets/${ds.id}/cell`, { method: "PUT", json: { row: +inp.closest("tr").dataset.row, col: inp.dataset.col, value: inp.value } }); toast("Tersimpan"); refreshList(); })));
}

/* --------------------------------------------------------------------- EDA */
async function viewEda(ds) {
  loading("Menghitung profil data…");
  const r = await api(`/datasets/${ds.id}/eda`).catch((e) => (toast(e.message, true), null)); if (!r) return;
  const o = r.overview;
  let html = `<div class="stats"><div class="stat"><b>${o.rows.toLocaleString("id-ID")}</b><span>baris</span></div><div class="stat"><b>${o.cols}</b><span>kolom</span></div><div class="stat"><b>${o.missing_pct}%</b><span>sel kosong (${o.missing_cells})</span></div><div class="stat"><b>${o.duplicate_rows}</b><span>baris duplikat</span></div>${Object.entries(o.kinds).map(([k, v]) => `<div class="stat"><b>${v}</b><span>kolom ${{ numeric: "numerik", categorical: "kategori", datetime: "tanggal", text: "teks", boolean: "boolean" }[k] || k}</span></div>`).join("")}</div>`;
  html += `<div class="card"><h3 style="margin-top:0">Profil kolom <small class="muted">(klik baris untuk melihat distribusi)</small></h3><div class="tablewrap"><table><tr><th>Kolom</th><th>Jenis</th><th>Kosong</th><th>Unik</th><th>Min</th><th>Rata-rata</th><th>Median</th><th>Maks</th><th>Skew</th><th>Teratas</th></tr>${r.columns.map((c) => `<tr data-c="${esc(c.name)}" style="cursor:pointer"><td><b>${esc(c.name)}</b></td><td>${c.kind}</td><td>${c.missing_pct}%</td><td>${c.unique}</td><td class="num">${c.kind === "numeric" || c.kind === "datetime" ? esc(fmt(c.min)) : ""}</td><td class="num">${c.mean !== undefined ? fmt(c.mean) : ""}</td><td class="num">${c.median !== undefined ? fmt(c.median) : ""}</td><td class="num">${c.kind === "numeric" || c.kind === "datetime" ? esc(fmt(c.max)) : ""}</td><td class="num">${c.skew !== undefined ? c.skew.toFixed(2) : ""}</td><td>${(c.top || []).slice(0, 3).map((t) => `${esc(t.value)} (${t.pct}%)`).join(", ")}</td></tr>`).join("")}</table></div></div><div id="one"></div>`;
  const cp = r.correlations.top_pairs;
  if (cp.length) html += `<div class="card"><h3 style="margin-top:0">Korelasi tertinggi</h3>${cp.slice(0, 6).map((p) => `<div class="insight">${esc(p.a)} ↔ ${esc(p.b)}: <b>${p.r.toFixed(2)}</b></div>`).join("")}</div>`;
  html += `<h3>Gambaran otomatis</h3><div class="grid" id="auto"><p class="muted spin">Membuat diagram…</p></div>`;
  view().innerHTML = html;
  view().querySelectorAll("tr[data-c]").forEach((tr) => (tr.onclick = guard(async () => {
    const c = r.columns.find((x) => x.name === tr.dataset.c);
    const spec = c.kind === "numeric" ? { type: "histogram", x: c.name } : c.kind === "datetime" ? null : { type: "bar", x: c.name, agg: "count" };
    if (!spec) return toast("Pilih kolom numerik atau kategori", false);
    const ch = await api(`/datasets/${ds.id}/chart`, { json: spec }); $("#one").replaceChildren(chartCard(ch));
  })));
  api(`/datasets/${ds.id}/auto-charts`).then((x) => { const g = $("#auto"); g.replaceChildren(...x.charts.map((c) => chartCard(c))); if (!x.charts.length) g.innerHTML = '<p class="muted">Belum ada diagram otomatis untuk data ini.</p>'; }).catch((e) => toast(e.message, true));
}

/* ------------------------------------------------------------------ CHARTS */
async function viewCharts(ds) {
  const cols = ds.columns, opt = (none) => (none ? '<option value="">—</option>' : "") + cols.map((c) => `<option>${esc(c)}</option>`).join("");
  const types = S.health.chart_types, aggs = S.health.aggs;
  view().innerHTML = `<div class="card"><h3 style="margin-top:0">🔎 Rekomendasi otomatis</h3>
   <p class="muted">Punya masalah? Ketik di bawah. Belum tahu masalahnya? Biarkan kosong dan biarkan aplikasi mencari pola/masalah dari data.</p>
   <div class="form"><label style="flex:1;min-width:260px">Masalah / pertanyaan (opsional)<input id="goal" placeholder="mis. bagaimana tren pendapatan? atau: bandingkan pendapatan per wilayah"></label><button class="primary" id="rec">Buat rekomendasi</button></div></div>
   <div id="recout"></div>
   <div class="card"><h3 style="margin-top:0">🧩 Buat grafik sendiri</h3><div class="form">
   <label>Jenis<select id="b-t">${types.map((t) => `<option>${t}</option>`).join("")}</select></label>
   <label>X<select id="b-x">${opt(true)}</select></label><label>Y<select id="b-y">${opt(true)}</select></label><label>Warna (scatter)<select id="b-c">${opt(true)}</select></label>
   <label>Agregasi<select id="b-a">${aggs.map((a) => `<option>${a}</option>`).join("")}</select></label><button class="primary" id="b-go">Gambar</button></div></div><div id="bout"></div>`;
  $("#rec").onclick = guard(async () => {
    $("#recout").innerHTML = '<p class="muted spin">Menganalisis…</p>';
    const r = await api(`/datasets/${ds.id}/recommend`, { json: { goal: $("#goal").value } });
    const out = $("#recout"); out.innerHTML = "";
    if (r.insights.length) { const c = document.createElement("div"); c.className = "card"; c.innerHTML = `<h3 style="margin-top:0">${r.goal ? "Temuan terkait" : "Temuan & potensi masalah"}</h3>` + r.insights.map((i) => `<div class="insight"><b>${esc(i.title)}</b>${i.kind === "quality" ? ' <span class="tag">kualitas data</span>' : ""}<small>${esc(i.detail)}</small></div>`).join(""); out.append(c); }
    const g = document.createElement("div"); g.className = "grid"; g.append(...r.charts.map((c) => chartCard(c))); out.append(g);
    if (!r.insights.length && !r.charts.length) out.innerHTML = '<div class="card">Belum ada temuan menonjol. Coba tulis pertanyaan yang lebih spesifik atau gunakan tab “Tanya AI”.</div>';
  });
  $("#b-go").onclick = guard(async () => {
    const ch = await api(`/datasets/${ds.id}/chart`, { json: { type: $("#b-t").value, x: $("#b-x").value || null, y: $("#b-y").value || null, color: $("#b-c").value || null, agg: $("#b-a").value } });
    $("#bout").replaceChildren(chartCard(ch));
  });
}

/* --------------------------------------------------------------------- ASK */
function viewAsk(ds) {
  const ai = S.health.ai;
  view().innerHTML = `<div class="card muted">${ai.enabled ? "Tanyakan apa saja tentang data ini. AI membuat diagram/kueri terstruktur yang dijalankan aplikasi pada data penuh." : "AI belum diaktifkan: jawaban memakai analisis heuristik. Klik ⚙ Pengaturan AI untuk mengaktifkan."}</div><div id="chat"></div>
   <div class="toolbar"><input id="qi" style="flex:1;min-width:260px" placeholder="mis. Wilayah mana dengan pendapatan tertinggi dan kenapa?"><button class="primary" id="qgo">Kirim</button></div>`;
  const chat = $("#chat");
  const draw = () => { chat.innerHTML = S.chat.map((m) => `<div class="msg ${m.role === "user" ? "user" : "ai"}">${esc(m.content)}</div>`).join(""); S.chat.forEach((m, i) => { if (m.extra) chat.children[i].append(...m.extra()); }); };
  draw();
  const send = guard(async () => {
    const q = $("#qi").value.trim(); if (!q) return; $("#qi").value = "";
    S.chat.push({ role: "user", content: q }); draw(); chat.insertAdjacentHTML("beforeend", '<div class="msg ai spin">Berpikir…</div>');
    try {
      const r = await api(`/datasets/${ds.id}/ask`, { json: { question: q, history: S.chat.slice(0, -1).map(({ role, content }) => ({ role, content })) } });
      S.chat.push({ role: "assistant", content: r.answer, extra: () => { const n = []; r.charts.forEach((c) => n.push(chartCard(c))); if (r.table) { const w = document.createElement("div"); w.className = "tablewrap"; w.style.marginTop = "8px"; w.innerHTML = `<table><tr>${r.table.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${r.table.rows.map((row) => `<tr>${row.map((v) => `<td class="${typeof v === "number" ? "num" : ""}">${esc(fmt(v))}</td>`).join("")}</tr>`).join("")}</table>`; n.push(w); } return n; } });
    } catch (e) { S.chat.pop(); S.chat.push({ role: "user", content: q }); toast(e.message, true); }
    draw();
  });
  $("#qgo").onclick = send; $("#qi").onkeydown=(e)=>{if(e.key==="Enter")send()};
}

/* ------------------------------------------------------------------- PIVOT */
function viewPivot(ds) {
  const o = (none) => (none ? '<option value="">—</option>' : "") + ds.columns.map((c) => `<option>${esc(c)}</option>`).join("");
  view().innerHTML = `<div class="card"><h3 style="margin-top:0">Pivot table</h3><div class="form"><label>Baris<select id="p-i">${o()}</select></label><label>Kolom (opsional)<select id="p-c">${o(true)}</select></label><label>Nilai<select id="p-v">${o(true)}</select></label>
   <label>Agregasi<select id="p-a">${S.health.aggs.concat(["std"]).map((a) => `<option>${a}</option>`).join("")}</select></label><button class="primary" id="p-go">Hitung</button></div><div id="pout" style="margin-top:12px"></div></div>
   <div class="card"><h3 style="margin-top:0">Kolom rumus (seperti Excel)</h3><p class="muted">Tulis ekspresi dengan nama kolom. Contoh: <code>harga * jumlah</code>, <code>\`biaya iklan\` / 1000</code>, <code>pendapatan - harga * jumlah</code></p>
   <div class="form"><label>Nama kolom baru<input id="f-n"></label><label style="flex:1;min-width:240px">Ekspresi<input id="f-e"></label><button class="primary" id="f-go">Tambah kolom</button></div></div>`;
  $("#p-go").onclick = guard(async () => {
    const r = await api(`/datasets/${ds.id}/pivot`, { json: { index: [$("#p-i").value], columns: $("#p-c").value || null, values: $("#p-v").value || null, agg: $("#p-a").value } });
    $("#pout").innerHTML = `<div class="tablewrap"><table><tr>${r.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${r.rows.map((row) => `<tr>${r.columns.map((c) => `<td class="${typeof row[c] === "number" ? "num" : ""}">${esc(fmt(row[c]))}</td>`).join("")}</tr>`).join("")}</table></div><p class="muted">${r.total} baris hasil.</p>`;
  });
  $("#f-go").onclick = guard(async () => { await api(`/datasets/${ds.id}/columns`, { json: { name: $("#f-n").value, expr: $("#f-e").value } }); toast("Kolom ditambahkan"); S.tab = "data"; refreshList(); });
}

/* -------------------------------------------------------------------- init */
(async () => { try { S.health = await api("/health"); } catch { S.health = { ai: {}, chart_types: [], aggs: [] }; toast("Tidak bisa menghubungi server", true); } await refreshList(); })();
