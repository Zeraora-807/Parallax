export async function api(path, { body, ...options } = {}) {
  const binary = body instanceof ArrayBuffer;
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": binary ? "application/octet-stream" : "application/json", ...options.headers },
    body: body === undefined || binary ? body : JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({ error: "服务响应异常，请检查本地服务。" }));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
export const $ = selector => document.querySelector(selector);
let noticeTimer;
export function notify(message) {
  const notice = $("#notice");
  notice.textContent = message;
  notice.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { notice.hidden = true; }, 6000);
}
export function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
export class RequestOwner {
  #generation = 0;
  next() {
    const generation = ++this.#generation;
    return () => generation === this.#generation;
  }
  invalidate() { ++this.#generation; }
}

