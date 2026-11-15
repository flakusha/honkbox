type Role = "user" | "assistant" | "system";
type ChatMessage = { role: Role; content: string };

const MODEL_KEY = "sd-rip.chat.model";

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  return node;
}

async function postChat(
  model: string,
  messages: ChatMessage[],
): Promise<string> {
  const res = await fetch("/v1/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, messages }),
  });
  if (!res.ok) throw new Error(`chat ${res.status}`);
  const body = (await res.json()) as { text: string };
  return body.text;
}

// Streams SSE `data: {"token":"..."}` frames; throws if unusable
// so the caller falls back to POST /v1/chat.
async function streamChat(
  model: string,
  messages: ChatMessage[],
  onToken: (token: string) => void,
): Promise<string> {
  const res = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, messages }),
  });
  if (!res.ok || !res.body) throw new Error(`stream ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let out = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return out;
      try {
        const frame = JSON.parse(data) as { token: string };
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

export function mountChat(host: HTMLElement): void {
  host.innerHTML = "";
  const slug = el("input");
  slug.placeholder = "model slug";
  slug.value = localStorage.getItem(MODEL_KEY) ?? "";
  slug.addEventListener("change", () =>
    localStorage.setItem(MODEL_KEY, slug.value),
  );
  const list = el("div");
  const form = el("form");
  const input = el("input");
  input.placeholder = "message";
  const send = el("button", "Send");
  form.append(input, send);
  host.append(slug, list, form);

  const history: ChatMessage[] = [];
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    localStorage.setItem(MODEL_KEY, slug.value);
    history.push({ role: "user", content: text });
    list.append(el("p", `you: ${text}`));
    const reply = el("p", "assistant: ");
    list.append(reply);
    const append = (token: string): void => {
      reply.textContent += token;
    };
    const finish = (text: string): void => {
      history.push({ role: "assistant", content: text });
      reply.textContent = `assistant: ${text}`;
    };
    streamChat(slug.value, history, append).then(finish).catch(() => {
      postChat(slug.value, history).then(finish).catch((err: unknown) => {
        reply.textContent = `error: ${err instanceof Error ? err.message : err}`;
      });
    });
  });
}
