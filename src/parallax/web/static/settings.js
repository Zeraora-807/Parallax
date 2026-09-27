import { $, api, node, notify, RequestOwner } from "./api.js";
import { applyTheme } from "./ui.js";

export function setupSettings(getState, refresh) {
  const discovery = new RequestOwner();
  let detectedFor = "";
  const showStatus = (message, error = false) => {
    $("#model-status").textContent = message;
    $("#model-status").classList.toggle("error", error);
  };
  function selectTab(tab) {
    document.querySelectorAll("[data-tab]").forEach(button => {
      if (button.dataset.tab === tab) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    document.querySelectorAll("[data-panel]").forEach(panel => { panel.hidden = panel.dataset.panel !== tab; });
  }
  function renderSkills(enabled) {
    const root = $("#skill-options");
    root.replaceChildren();
    for (const skill of getState().skills) {
      const label = node("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = skill.name;
      checkbox.checked = enabled.includes(skill.name);
      label.append(checkbox, node("span", "mono", skill.name));
      root.append(label);
    }
    if (!root.children.length) root.append(node("p", "quiet", "还没有 Skill，可以在下面添加。"));
  }
  function renderAgents(enabled) {
    const root = $("#agent-options");
    root.replaceChildren();
    for (const agent of getState().agents) {
      const label = node("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = agent.name;
      checkbox.checked = enabled.includes(agent.name);
      label.append(checkbox, node("span", "mono", `${agent.name} · ${agent.role}`));
      root.append(label);
    }
    if (!root.children.length) root.append(node("p", "quiet", "还没有自定义 Agent，可以在下面添加。"));
  }
  function renderMemories() {
    const root = $("#memory-list");
    root.replaceChildren();
    for (const memory of getState().memories) {
      const row = node("article", "memory");
      row.append(node("p", "", memory.content));
      const footer = node("footer");
      footer.append(node("span", "", new Date(memory.created_at).toLocaleDateString("zh-CN")));
      const remove = node("button", "", "删除");
      remove.type = "button";
      remove.onclick = async () => {
        remove.disabled = true;
        try {
          await api(`/api/memories/${memory.memory_id}`, { method: "DELETE" });
          await refresh();
          renderMemories();
        } catch (error) { $("#settings-status").textContent = error.message; remove.disabled = false; }
      };
      footer.append(remove);
      row.append(footer);
      root.append(row);
    }
    if (!root.children.length) root.append(node("p", "quiet", "暂无记忆。模型会从后续对话中选择值得保留的原文。"));
  }
  async function open(tab = "connection") {
    try { await refresh(); } catch (error) { notify(error.message); return; }
    const settings = getState().settings;
    $("#api-key").value = sessionStorage.apiKey || "";
    $("#base-url").value = settings.base_url || "https://api.deepseek.com";
    $("#model").value = settings.model || "";
    $("#multi-agent").checked = settings.multi_agent === "true";
    $("#max-agents").value = settings.max_agents || "4";
    $("#theme").value = localStorage.theme || settings.theme || "afterglow";
    $("#settings-status").textContent = "";
    $("#model-options").replaceChildren();
    renderSkills((settings.skills || "").split(","));
    renderAgents((settings.agents || "").split(","));
    renderMemories();
    selectTab(tab);
    $("#settings").showModal();
  }
  async function detectModels() {
    const base = $("#base-url").value.trim();
    const key = $("#api-key").value.trim();
    if (!base) { showStatus("请先填写接口地址。", true); return; }
    const owns = discovery.next();
    const button = $("#detect-models");
    button.disabled = true;
    showStatus("正在读取可用模型…");
    try {
      const result = await api("/api/models", { method: "POST", body: { base_url: base, api_key: key } });
      if (!owns()) return;
      $("#model-options").replaceChildren(...result.models.map(model => new Option(model, model)));
      if (result.models.length) {
        if (!result.models.includes($("#model").value)) $("#model").value = result.models[0];
        showStatus(`检测到 ${result.models.length} 个模型，可从列表选择。`);
        detectedFor = base + "\n" + key;
      } else showStatus("未检测到模型，请手动输入模型 ID。", true);
    } catch (error) {
      if (owns()) showStatus(`检测失败，可手动输入模型 ID。 ${error.message}`, true);
    } finally { if (owns()) button.disabled = false; }
  }
  for (const selector of ["#api-key", "#base-url"]) {
    $(selector).addEventListener("input", () => {
      discovery.invalidate();
      $("#detect-models").disabled = false;
      $("#model-options").replaceChildren();
      showStatus("连接信息已更改，离开输入框后将重新检测。");
    });
    $(selector).addEventListener("change", () => {
      const fingerprint = $("#base-url").value.trim() + "\n" + $("#api-key").value.trim();
      if (fingerprint !== detectedFor) void detectModels();
    });
  }
  $("#detect-models").onclick = detectModels;
  $("#theme").onchange = event => applyTheme(event.target.value);
  document.querySelectorAll("[data-tab]").forEach(button => {
    button.onclick = () => selectTab(button.dataset.tab);
  });
  $("#settings").addEventListener("close", () => {
    discovery.invalidate();
    $("#detect-models").disabled = false;
  });
  $("#settings-form").onsubmit = async event => {
    event.preventDefault();
    const button = $("#settings-save");
    button.disabled = true;
    const skills = [...$("#skill-options").querySelectorAll("input:checked")].map(item => item.value).join(",");
    const agents = [...$("#agent-options").querySelectorAll("input:checked")].map(item => item.value).join(",");
    if (agents && !$("#multi-agent").checked) {
      $("#settings-status").textContent = "启用自定义 Agent 前请先开启多 Agent 模式。";
      button.disabled = false;
      return;
    }
    try {
      await api("/api/settings", { method: "POST", body: {
        base_url: $("#base-url").value.trim(), model: $("#model").value.trim(),
        theme: $("#theme").value, skills, agents,
        multi_agent: $("#multi-agent").checked ? "true" : "false",
        max_agents: $("#max-agents").value,
      }});
      sessionStorage.apiKey = $("#api-key").value.trim();
      await refresh();
      $("#settings").close();
      notify("设置已保存");
    } catch (error) { $("#settings-status").textContent = error.message; }
    finally { button.disabled = false; }
  };
  $("#skill-save").onclick = async () => {
    const button = $("#skill-save");
    const enabled = [...$("#skill-options").querySelectorAll("input:checked")].map(item => item.value);
    button.disabled = true;
    try {
      await api("/api/skills", { method: "POST", body: {
        name: $("#skill-name").value.trim(), content: $("#skill-content").value,
      }});
      await refresh();
      renderSkills(enabled);
      $("#skill-name").value = "";
      $("#skill-content").value = "";
      $("#settings-status").textContent = "Skill 已添加。勾选后保存设置即可启用。";
    } catch (error) { $("#settings-status").textContent = error.message; }
    finally { button.disabled = false; }
  };
  $("#agent-save").onclick = async () => {
    const button = $("#agent-save");
    const enabled = [...$("#agent-options").querySelectorAll("input:checked")].map(item => item.value);
    button.disabled = true;
    try {
      await api("/api/agents", { method: "POST", body: {
        name: $("#agent-name").value.trim(), content: $("#agent-content").value,
      }});
      await refresh();
      renderAgents(enabled);
      $("#agent-name").value = "";
      $("#agent-content").value = "";
      $("#settings-status").textContent = "Agent 已添加。勾选、启用多 Agent 后保存即可使用。";
    } catch (error) { $("#settings-status").textContent = error.message; }
    finally { button.disabled = false; }
  };
  $("#settings-open").onclick = () => open();
  $("#skills-open").onclick = () => open("skills");
  $("#memory-open").onclick = () => open("memory");
  return { open };
}
