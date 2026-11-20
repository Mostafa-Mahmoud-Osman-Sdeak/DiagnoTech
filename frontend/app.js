"use strict";

const state = { conversationId: null, pending: false };

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("messageInput");
const loadingEl = document.getElementById("loading");
const loadingTextEl = document.getElementById("loadingText");
const listEl = document.getElementById("conversationList");
const kbStatusEl = document.getElementById("kbStatus");
const kbStatusTextEl = document.getElementById("kbStatusText");
const sendBtn = document.getElementById("sendBtn");
const modelInfoEl = document.getElementById("modelInfo");

const STATE_LABELS = {
  known: "معلومة موثقة من قاعدة المعرفة",
  external: "معلومة من مصدر خارجي موثوق",
  unavailable: "غير متوفرة",
  emergency: "تنبيه طارئ",
  out_of_scope: "خارج النطاق",
  greeting: "",
  error: "خطأ",
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function addMessage(role, content, stateName = "", sources = []) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant" && stateName) bubble.classList.add(`state-${stateName}`);

  if (role === "assistant" && STATE_LABELS[stateName]) {
    const badge = document.createElement("span");
    badge.className = `state-badge ${stateName}`;
    badge.textContent = STATE_LABELS[stateName];
    bubble.appendChild(badge);
    bubble.appendChild(document.createElement("br"));
  }
  bubble.appendChild(document.createTextNode(content));

  if (sources && sources.length) {
    const box = document.createElement("div");
    box.className = "sources";
    const title = document.createElement("div");
    title.className = "sources-title";
    title.textContent = "المصادر:";
    box.appendChild(title);
    sources.forEach((s) => {
      const card = document.createElement(s.source_url ? "a" : "div");
      card.className = "source-card";
      if (s.source_url) {
        card.href = s.source_url;
        card.target = "_blank";
        card.rel = "noopener";
      }
      const disease = document.createElement("div");
      disease.className = "s-disease";
      disease.textContent = s.disease_name || s.source || "مصدر";
      card.appendChild(disease);
      const meta = document.createElement("div");
      meta.className = "s-meta";
      if (s.source) {
        const src = document.createElement("span");
        src.textContent = s.source;
        meta.appendChild(src);
      }
      if (s.category) {
        const cat = document.createElement("span");
        cat.textContent = s.category;
        meta.appendChild(cat);
      }
      if (typeof s.score === "number") {
        const score = document.createElement("span");
        score.className = "s-score";
        score.textContent = `التطابق ${Math.min(100, Math.round(s.score * 100))}%`;
        meta.appendChild(score);
      }
      card.appendChild(meta);
      box.appendChild(card);
    });
    bubble.appendChild(box);
  }

  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function refreshConversations() {
  try {
    const conversations = await api("/api/conversations");
    listEl.innerHTML = "";
    conversations.forEach((c) => {
      const li = document.createElement("li");
      if (c.id === state.conversationId) li.className = "active";
      const title = document.createElement("span");
      title.textContent = c.title || "محادثة";
      title.onclick = () => openConversation(c.id);
      const del = document.createElement("button");
      del.className = "del";
      del.textContent = "🗑";
      del.title = "حذف المحادثة";
      del.onclick = async (e) => {
        e.stopPropagation();
        await api(`/api/conversations/${c.id}`, { method: "DELETE" });
        if (state.conversationId === c.id) newConversation();
        refreshConversations();
      };
      li.appendChild(title);
      li.appendChild(del);
      listEl.appendChild(li);
    });
  } catch (err) {
    console.error(err);
  }
}

async function openConversation(id) {
  state.conversationId = id;
  messagesEl.innerHTML = "";
  const msgs = await api(`/api/conversations/${id}/messages`);
  msgs.forEach((m) => addMessage(m.role, m.content, m.state, m.sources));
  refreshConversations();
  document.getElementById("layout").classList.remove("show-sidebar");
}

function newConversation() {
  state.conversationId = null;
  messagesEl.innerHTML = "";
  addMessage("assistant", "أهلاً بك. اكتب سؤالك الطبي المتعلق بالجهاز التنفسي والصدر.");
  refreshConversations();
  document.getElementById("layout").classList.remove("show-sidebar");
}

document.getElementById("newConversationBtn").onclick = newConversation;
document.getElementById("menuBtn").onclick = () =>
  document.getElementById("layout").classList.toggle("show-sidebar");

// Show a "thinking" line with elapsed seconds while the answer is generated;
// generation on CPU can take a while, so the user can see it is still working.
let timerId = null;

function setPending(on) {
  state.pending = on;
  sendBtn.disabled = on;
  inputEl.disabled = on;
  document.querySelectorAll(".chip").forEach((ch) => (ch.disabled = on));
  if (on) {
    const started = Date.now();
    const tick = () => {
      const sec = Math.floor((Date.now() - started) / 1000);
      loadingTextEl.innerHTML = "";
      loadingTextEl.appendChild(
        document.createTextNode("جارٍ البحث في قاعدة المعرفة وتوليد الإجابة… "));
      const elapsed = document.createElement("span");
      elapsed.className = "elapsed";
      elapsed.textContent = `${sec} ث`;
      loadingTextEl.appendChild(elapsed);
    };
    tick();
    timerId = setInterval(tick, 1000);
    loadingEl.classList.remove("hidden");
  } else {
    if (timerId) clearInterval(timerId);
    timerId = null;
    loadingEl.classList.add("hidden");
    inputEl.disabled = false;
    inputEl.focus();
  }
}

document.getElementById("chatForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (state.pending) return;
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  inputEl.style.height = "auto";
  addMessage("user", text);
  setPending(true);
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text, conversation_id: state.conversationId }),
    });
    state.conversationId = result.conversation_id;
    addMessage("assistant", result.answer, result.state, result.sources);
    refreshConversations();
  } catch (err) {
    addMessage("assistant", "حدث خطأ في الاتصال بالخادم. حاول مرة أخرى.", "error");
  } finally {
    setPending(false);
  }
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("chatForm").requestSubmit();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (state.pending) return;
    inputEl.value = chip.textContent.trim();
    document.getElementById("chatForm").requestSubmit();
  });
});

(async function init() {
  try {
    const kb = await api("/api/kb/status");
    kbStatusTextEl.textContent = kb.index_ready
      ? `قاعدة المعرفة جاهزة — ${kb.chunks} مقطع مفهرس`
      : "قاعدة المعرفة غير مهيأة بعد";
    if (kb.index_ready) kbStatusEl.classList.add("ready");
    const bits = [];
    if (kb.llm_provider) bits.push(`مزوّد التوليد: ${kb.llm_provider}`);
    if (kb.embedder) bits.push(`محرّك البحث: ${kb.embedder}`);
    modelInfoEl.textContent = bits.join(" · ");
  } catch (err) {
    kbStatusTextEl.textContent = "تعذر التحقق من حالة قاعدة المعرفة";
  }
  refreshConversations();
})();
