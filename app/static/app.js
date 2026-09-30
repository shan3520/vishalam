"use strict";
const $ = (s) => document.querySelector(s);
const thread = $("#thread"), input = $("#input"), sendBtn = $("#sendBtn"), demoBtn = $("#demoBtn");
let status = null, running = false, activeId = null;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;  // always textContent: model output is never HTML
  return e;
}
function pre(text, cls) { const p = el("pre", cls); p.textContent = text; return p; }

async function loadStatus() {
  status = await (await fetch("/api/status")).json();
  const s = $("#status"); s.innerHTML = "";
  if (status.model) s.append(el("span", "pill ok", `model: ${status.model.weights} · ${status.model.device}` + (status.model.params_m ? ` · ${status.model.params_m}M` : "")));
  else s.append(el("span", "pill warn", "no model loaded"));
  s.append(el("span", status.sandbox === "docker" ? "pill ok" : "pill warn",
    status.sandbox === "docker" ? "sandbox: docker" : "sandbox: subprocess (install Docker for isolation)"));
  if (status.demo) { s.append(el("span", "pill warn", "demo mode: scripted agent")); demoBtn.hidden = false; }
}

async function loadHistory() {
  const items = await (await fetch("/api/tasks")).json();
  const ul = $("#history"); ul.innerHTML = "";
  for (const t of items) {
    const li = el("li"); if (t.id === activeId) li.classList.add("active");
    li.append(el("span", `dot ${t.status}`), el("span", "txt", (t.mode === "demo" ? "[demo] " : "") + t.request));
    li.onclick = () => showTask(t.id);
    ul.append(li);
  }
}

function resultClass(name, text) {
  if (name !== "run_tests") return text.startsWith("error:") ? "bad" : "";
  return /\bfailed\b|\berror\b|Timed out/i.test(text) ? "bad" : (/\bpassed\b/.test(text) ? "ok" : "");
}

function renderCall(e) {
  const card = el("div", "step");
  const head = el("div", "step-head");
  head.append(el("span", "tool", e.name));
  const a = e.args || {};
  if (a.path) head.append(el("span", "arg", a.path));
  if (a.pattern) head.append(el("span", "arg", a.pattern));
  card.append(head);
  const body = el("div", "step-body");
  if (e.name === "write_file") body.append(pre(a.content || ""));
  if (e.name === "edit_file") { body.append(el("div", "label", "replace"), pre(a.old || "", "old"), el("div", "label", "with"), pre(a.new || "", "new")); }
  if (e.name === "run_python") body.append(pre(a.code || ""));
  if (body.childNodes.length) card.append(body);
  card.dataset.tool = e.name;
  return card;
}

function attachResult(card, e) {
  const cls = resultClass(e.name, e.text || "");
  if (e.name === "run_tests") {
    const b = el("span", `badge ${cls || ""}`, cls === "ok" ? "passed" : cls === "bad" ? "failed" : "ran");
    card.querySelector(".step-head").append(b);
  }
  let body = card.querySelector(".step-body");
  if (!body) { body = el("div", "step-body"); card.append(body); }
  if (["write_file", "edit_file"].includes(e.name) && !cls) { body.append(el("div", "label", e.text)); return; }
  body.append(el("div", "label", "result"), pre(e.text || "", `result ${cls}`));
}

function renderEvents(events, live) {
  let lastCard = null, thinking = null;
  const clearThinking = () => { if (thinking) { thinking.remove(); thinking = null; } };
  const handle = (e) => {
    if (e.type === "thinking") {
      if (!thinking) { thinking = el("div", "thinking"); thinking.append("Vishalam is working", el("span", "", "."), el("span", "", "."), el("span", "", ".")); thread.append(thinking); }
      return;
    }
    clearThinking();
    if (e.type === "start") return;
    if (e.type === "call" && e.name !== "done") { lastCard = renderCall(e); thread.append(lastCard); }
    else if (e.type === "result" && lastCard) attachResult(lastCard, e);
    else if (e.type === "done") thread.append(el("div", "final", e.message || "Done."));
    else if (e.type === "error") thread.append(el("div", "error-box", e.text));
    else if (e.type === "end") {
      if (e.status !== "finished") thread.append(el("div", "final stopped", `Stopped: ${e.stop_reason || e.status}`));
      renderFiles(e.files || {});
    }
    thread.scrollTop = thread.scrollHeight;
  };
  events.forEach(handle);
  return handle;
}

function renderFiles(files) {
  const names = Object.keys(files);
  if (!names.length) return;
  const d = el("details", "files");
  d.append(el("summary", "", `Final workspace (${names.length} file${names.length > 1 ? "s" : ""})`));
  for (const n of names) { d.append(el("div", "fname", n), pre(files[n])); }
  thread.append(d);
}

function startThread(request, mode) {
  thread.innerHTML = "";
  thread.append(el("div", "msg-user", request));
  if (mode === "demo") thread.append(el("div", "demo-note", "Demo mode: a scripted agent replays a generated task. Tools and tests run for real; the Vishalam model is not involved."));
}

async function submit(body) {
  if (running) return;
  running = true; sendBtn.disabled = true; demoBtn.disabled = true;
  try {
    const r = await fetch("/api/tasks", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json();
    if (!r.ok) { thread.querySelector("#empty")?.remove(); thread.append(el("div", "error-box", data.detail || "Request failed")); return; }
    activeId = data.id; input.value = "";
    startThread(data.request, data.mode);
    await loadHistory();
    const handle = renderEvents([], true);
    await new Promise((resolve) => {
      const es = new EventSource(`/api/tasks/${data.id}/events`);
      es.onmessage = (m) => handle(JSON.parse(m.data));
      es.addEventListener("close", () => { es.close(); resolve(); });
      es.onerror = () => { es.close(); resolve(); };
    });
  } finally {
    running = false; sendBtn.disabled = false; demoBtn.disabled = false;
    loadHistory();
  }
}

async function showTask(id) {
  if (running) return;
  const t = await (await fetch(`/api/tasks/${id}`)).json();
  activeId = id;
  startThread(t.request, t.mode);
  renderEvents(t.events, false);
  loadHistory();
}

$("#composer").addEventListener("submit", (ev) => { ev.preventDefault(); const v = input.value.trim(); if (v) submit({ request: v }); });
input.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("#composer").requestSubmit(); } });
demoBtn.addEventListener("click", () => submit({ demo: true }));
$("#newTask").addEventListener("click", () => { if (!running) location.reload(); });
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { input.value = c.dataset.ex; input.focus(); }));

loadStatus(); loadHistory();
