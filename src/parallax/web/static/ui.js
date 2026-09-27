import { $, node } from "./api.js";

export function emptyView() {
  const root = node("div", "empty");
  root.append(node("h2", "", "从一个问题开始。"));
  return root;
}

// Model output remains text; fenced code never becomes executable HTML.
export function renderContent(text) {
  const root = node("div", "message-body");
  const chunks = text.split(/(```[^\n]*\n[\s\S]*?(?:```|$))/g);
  for (const chunk of chunks) {
    if (!chunk) continue;
    if (chunk.startsWith("```")) {
      const code = chunk.replace(/^```[^\n]*\n/, "").replace(/```$/, "");
      const pre = node("pre");
      pre.append(node("code", "", code));
      root.append(pre);
    } else {
      for (const block of chunk.split(/\n\s*\n/)) {
        if (!block.trim()) continue;
        root.append(/^#{1,4}\s/.test(block)
          ? node("h3", "", block.replace(/^#{1,4}\s/, ""))
          : node("p", "", block));
      }
    }
  }
  return root;
}
export function renderMessages(conversation, forceScroll = false) {
  const root = $("#messages");
  const nearBottom = root.scrollHeight - root.scrollTop - root.clientHeight < 90;
  const previousScroll = root.scrollTop;
  const fragment = document.createDocumentFragment();
  if (!conversation.messages.length) fragment.append(emptyView());
  for (const message of conversation.messages) {
    const article = node("article", `message ${message.role} ${message.status === "error" ? "error" : ""}`);
    const header = node("div", "message-head");
    header.append(node("span", "", message.role === "user" ? "YOU" : "PARALLAX"));
    const time = node("time", "", new Date(message.created_at).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }));
    time.dateTime = message.created_at;
    header.append(time);
    article.append(header);
    article.append(message.status === "running"
      ? node("p", "thinking", "分析进行中，完成后结果会显示在这里。")
      : renderContent(message.content));
    fragment.append(article);
  }
  root.replaceChildren(fragment);
  root.scrollTop = forceScroll || nearBottom ? root.scrollHeight : previousScroll;
}
export function renderContext(state, conversation) {
  const refs = conversation?.reference_ids || [];
  $("#reference-count").textContent = refs.length;
  $("#references").replaceChildren(...(refs.length ? refs.map(id =>
    node("span", "", state.conversations.find(c => c.id === id)?.title || "参考对话"))
    : [node("p", "quiet", "新建对话时选择参考记录")]));
  const skills = (state.settings.skills || "").split(",").filter(Boolean);
  $("#skill-count").textContent = skills.length;
  $("#active-skills").replaceChildren(...(skills.length ? skills.map(name => node("span", "mono", name))
    : [node("p", "quiet", "尚未启用扩展")]));
  $("#memory-count").textContent = state.memories.length;
  $("#composer-model").textContent = state.settings.model || "先在设置中连接模型";
  $("#composer-model").title = state.settings.model || "未配置模型";
}
export function applyTheme(theme) {
  const chosen = ["afterglow", "dark", "light"].includes(theme) ? theme : "afterglow";
  document.documentElement.dataset.theme = chosen;
  localStorage.theme = chosen;
}
