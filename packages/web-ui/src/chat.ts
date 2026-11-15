export type Role = "user" | "assistant" | "system";
export type ChatMessage = { role: Role; content: string; };
// Optional generation knobs mapped onto ChatRequest; omitted keys use the
// server default.
export type ChatParams = { max_tokens?: number; temperature?: number; };

const MODEL_KEY = "honkbox.chat.model";
// ONE-RELEASE compat: fall back to the old `sd-rip.*` key once, then rewrite
// under `honkbox.*`.
const LEGACY_MODEL_KEY = "sd-rip.chat.model";

function loadModel(): string {
  const current = localStorage.getItem(MODEL_KEY);
  if (current !== null) {
    return current;
  }
  const legacy = localStorage.getItem(LEGACY_MODEL_KEY);
  if (legacy === null) {
    return "";
  }
  localStorage.setItem(MODEL_KEY, legacy);
  localStorage.removeItem(LEGACY_MODEL_KEY);
  return legacy;
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  return node;
}

function chatBody(
  model: string,
  messages: ChatMessage[],
  params: ChatParams,
): string {
  return JSON.stringify({ model, messages, ...params });
}

async function postChat(
  model: string,
  messages: ChatMessage[],
  params: ChatParams = {},
): Promise<string> {
  const res = await fetch("/v1/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: chatBody(model, messages, params),
  });
  if (!res.ok) throw new Error(`chat ${res.status}`);
  const body = (await res.json()) as { text: string; };
  return body.text;
}

// Streams SSE `data: {"token":"..."}` frames, `event: error` frames and
// `data: [DONE]`; throws if unusable so the caller falls back to POST /v1/chat.
async function streamChat(
  model: string,
  messages: ChatMessage[],
  onToken: (token: string) => void,
  params: ChatParams = {},
): Promise<string> {
  const res = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: chatBody(model, messages, params),
  });
  if (!res.ok || !res.body) throw new Error(`stream ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let out = "";
  let event = "message";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
        continue;
      }
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      const name = event;
      event = "message";
      if (data === "[DONE]") return out;
      if (name === "error") throw new Error(data);
      try {
        const frame = JSON.parse(data) as { token: string; };
        out += frame.token;
        onToken(frame.token);
      } catch {
        // Skip keep-alive / partial frames.
      }
    }
  }
  if (!out) throw new Error("empty stream");
  return out;
}

export type ChatControls = {
  // Model id for ChatRequest.model; "" = server default.
  getModel: () => string;
  // Generation knobs for ChatRequest; omit to use server defaults.
  getParams?: () => ChatParams;
};

// Shared chat surface: message list + send form with SSE token streaming and
// a POST /v1/chat fallback. Callers own the host and any controls appended
// before it.
export function mountChatCore(host: HTMLElement, controls: ChatControls): void {
  const list = el("div");
  const form = el("form");
  const input = el("input");
  input.placeholder = "message";
  const send = el("button", "Send");
  form.append(input, send);
  host.append(list, form);

  const history: ChatMessage[] = [];
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    const model = controls.getModel();
    const params = controls.getParams?.() ?? {};
    history.push({ role: "user", content: text });
    list.append(el("p", `you: ${text}`));
    const reply = el("p", "assistant: ");
    list.append(reply);
    const append = (token: string): void => {
      reply.textContent += token;
    };
    const finish = (final: string): void => {
      history.push({ role: "assistant", content: final });
      reply.textContent = `assistant: ${final}`;
    };
    streamChat(model, history, append, params).then(finish).catch(() => {
      postChat(model, history, params).then(finish).catch((err: unknown) => {
        reply.textContent = `error: ${err instanceof Error ? err.message : err}`;
      });
    });
  });
}

export function mountChat(host: HTMLElement): void {
  host.innerHTML = "";
  const slug = el("input");
  slug.placeholder = "model slug";
  slug.value = loadModel();
  slug.addEventListener("change", () => localStorage.setItem(MODEL_KEY, slug.value));
  const core = el("div");
  host.append(slug, core);
  mountChatCore(core, {
    getModel: () => {
      localStorage.setItem(MODEL_KEY, slug.value);
      return slug.value;
    },
  });
}
