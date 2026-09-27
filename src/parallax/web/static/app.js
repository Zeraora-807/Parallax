import { $, api, node, notify, RequestOwner } from "./api.js";
import { emptyView, renderMessages, renderContext, applyTheme } from "./ui.js";
import { setupSettings } from "./settings.js";

let state = { conversations: [], settings: {}, skills: [], agents: [], memories: [] };
let current = null;
let conversation = null;
let poll = null;
let submitting = false;
let loading = false;
const drafts = new Map();
const owner = new RequestOwner();
const stateOwner = new RequestOwner();

function renderConversations() {
  $("#conversation-count").textContent = String(state.conversations.length).padStart(2, "0");
  const query = $("#chat-search").value.trim().toLocaleLowerCase();
  const entries = state.conversations.filter(item => item.title.toLocaleLowerCase().includes(query));
  const root = $("#conversations");
  root.replaceChildren();
  for (const [index, item] of entries.entries()) {
    const button = node("button", "chat-link" + (item.id === current ? " active" : ""));
    button.title = item.title;
    if (item.id === current) button.setAttribute("aria-current", "page");
    button.append(node("span", "chat-index", String(index + 1).padStart(2, "0")), node("span", "chat-name", item.title));
    button.onclick = () => openConversation(item.id).catch(error => notify(error.message));
    root.append(button);
  }
  if (!entries.length) root.append(node("p", "list-empty", query ? "没有匹配的对话" : "分析记录将保存在这里"));
}
async function refreshState() {
  const owns = stateOwner.next();
  const result = await api("/api/state");
  if (!owns()) return;
  state = result;
  renderConversations();
  renderContext(state, conversation);
}
function updateControls() {
  const running = conversation?.messages.some(item => item.status === "running") || false;
  $("#send").disabled = submitting || loading || running;
  $("#attach-button").disabled = submitting || loading || running;
  $("#run-state").textContent = running ? "分析中" : loading ? "读取中" : "就绪";
  $("#run-state").classList.toggle("running", running);
}
function messagesChanged(previous, next) {
  if (!previous || previous.length !== next.length) return true;
  const before = previous.at(-1);
  const after = next.at(-1);
  return before?.id !== after?.id || before?.status !== after?.status || before?.content !== after?.content;
}
function closeMenu() {
  $(".shell").classList.remove("menu-open");
  $("#menu-toggle").setAttribute("aria-expanded", "false");
}
async function openConversation(id) {
  clearTimeout(poll);
  drafts.set(current, $("#message").value);
  const changed = current !== id;
  current = id;
  conversation = null;
  loading = true;
  $("#message").value = drafts.get(id) || "";
  $("#chat-title").textContent = state.conversations.find(item => item.id === id)?.title || "读取中";
  $("#messages").replaceChildren(node("p", "quiet", "正在读取对话…"));
  $("#attachments").replaceChildren();
  renderConversations();
  updateControls();
  closeMenu();
  await fetchConversation(id, changed);
}
async function fetchConversation(id, forceScroll = false) {
  const owns = owner.next();
  try {
    const result = await api(`/api/conversations/${id}`);
    if (!owns() || current !== id) return;
    const previous = conversation;
    conversation = result;
    loading = false;
    $("#chat-title").textContent = result.title;
    if (messagesChanged(previous?.messages, result.messages)) renderMessages(result, forceScroll);
    $("#attachments").replaceChildren(...result.attachments.map(item => node("span", "attachment", item.name)));
    $("#attachment-count").textContent = result.attachments.length
      ? `${result.attachments.length} 个附件 · 单文件上限 64 MiB` : "单文件上限 64 MiB";
    renderContext(state, result);
    updateControls();
    clearTimeout(poll);
    if (result.messages.some(item => item.status === "running")) {
      poll = setTimeout(() => fetchConversation(id).catch(error => notify(error.message)), 1400);
    } else if (previous?.messages.some(item => item.status === "running")) {
      await refreshState();
    }
  } catch (error) {
    if (!owns() || current !== id) return;
    loading = false;
    updateControls();
    // A temporary disconnect must not strand an in-progress conversation.
    clearTimeout(poll);
    poll = setTimeout(() => fetchConversation(id).catch(() => {}), 4000);
    throw error;
  }
}
const settings = setupSettings(() => state, refreshState);
$("#messages").append(emptyView());
$("#chat-search").oninput = renderConversations;
$("#message").oninput = () => drafts.set(current, $("#message").value);
$("#message").onkeydown = event => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) {
    event.preventDefault();
    if (!$("#send").disabled) $("#composer").requestSubmit();
  }
};
$("#menu-toggle").onclick = () => {
  const expanded = $(".shell").classList.toggle("menu-open");
  $("#menu-toggle").setAttribute("aria-expanded", String(expanded));
};
document.querySelectorAll("[data-close]").forEach(button => {
  button.onclick = () => document.getElementById(button.dataset.close).close();
});
$("#new-chat").onclick = async () => {
  try { await refreshState(); } catch (error) { notify(error.message); return; }
  $("#new-title").value = "";
  const root = $("#reference-options");
  root.replaceChildren();
  for (const item of state.conversations) {
    const label = node("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = item.id;
    label.append(input, node("span", "", item.title));
    root.append(label);
  }
  if (!state.conversations.length) root.append(node("p", "quiet", "还没有可参考的对话。"));
  $("#new-dialog").showModal();
};
$("#new-form").onsubmit = async event => {
  event.preventDefault();
  $("#new-confirm").disabled = true;
  try {
    const reference_ids = [...$("#reference-options").querySelectorAll("input:checked")].map(item => item.value);
    const result = await api("/api/conversations", { method: "POST", body: { title: $("#new-title").value.trim(), reference_ids } });
    $("#new-dialog").close();
    await refreshState();
    await openConversation(result.id);
    $("#message").focus();
  } catch (error) { notify(error.message); }
  finally { $("#new-confirm").disabled = false; }
};
$("#composer").onsubmit = async event => {
  event.preventDefault();
  if (submitting || loading || conversation?.messages.some(item => item.status === "running")) return;
  const content = $("#message").value.trim();
  if (!content) return;
  if (!state.settings.model || !state.settings.base_url) {
    notify("请先连接模型。你的草稿已保留。");
    await settings.open();
    return;
  }
  submitting = true;
  updateControls();
  let target = current;
  const sourceDraft = current;
  try {
    if (!target) {
      const result = await api("/api/conversations", { method: "POST", body: { title: content.slice(0, 36), reference_ids: [] } });
      target = result.id;
      if (current === null) current = target;
    }
    await api(`/api/conversations/${target}/messages`, { method: "POST", body: {
      content, api_key: sessionStorage.apiKey || "", base_url: state.settings.base_url,
      model: state.settings.model, skills: (state.settings.skills || "").split(",").filter(Boolean),
      agents: (state.settings.agents || "").split(",").filter(Boolean),
      multi_agent: state.settings.multi_agent === "true",
      max_agents: Number(state.settings.max_agents || 4),
    }});
    if ((drafts.get(sourceDraft) || "").trim() === content) drafts.delete(sourceDraft);
    if (current === target && $("#message").value.trim() === content) $("#message").value = "";
    await refreshState();
    if (current === target) await fetchConversation(target, true);
  } catch (error) { notify(error.message); }
  finally { submitting = false; updateControls(); }
};
$("#attach-button").onclick = () => $("#file").click();
$("#file").onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size < 1 || file.size > 64 * 1024 * 1024) {
    notify("请选择 1 字节至 64 MiB 的文件。");
    event.target.value = "";
    return;
  }
  submitting = true;
  updateControls();
  let target = current;
  try {
    if (!target) {
      const result = await api("/api/conversations", { method: "POST", body: { title: file.name, reference_ids: [] } });
      target = result.id;
      if (current === null) current = target;
    }
    await api(`/api/conversations/${target}/attachments?name=${encodeURIComponent(file.name)}`, { method: "POST", body: await file.arrayBuffer() });
    await refreshState();
    if (current === target) await fetchConversation(target);
    notify("样本已添加");
  } catch (error) { notify(error.message); }
  finally { submitting = false; event.target.value = ""; updateControls(); }
};
refreshState().then(() => {
  applyTheme(localStorage.theme || state.settings.theme || "afterglow");
}).catch(error => notify(error.message));
