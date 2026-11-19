import type { ContentPart, OpenAiChatCompletion, OpenAiChunk } from "./gen/api-types";

export type Role = "user" | "assistant" | "system";
// One conversation message as the UI holds it: text plus optional image
// attachments (data: URIs, REQ-8 vision chat).
export type ChatMessage = { role: Role; content: string; images?: string[]; };
// Optional generation knobs mapped onto ChatRequest; omitted keys use the
// server default.
export type ChatParams = { max_tokens?: number; temperature?: number; };

const MODEL_KEY = "honkbox.chat.model";

function loadModel(): string {
  return localStorage.getItem(MODEL_KEY) ?? "";
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  return node;
}

// One UI message → the wire ChatMessage: attachments ride as `image_url`
// content parts after the text part — the data:-URI shape the chat routes
// already parse (REQ-8); a message without images stays the plain string
// form.
function wireMessage(message: ChatMessage): {
  role: Role;
  content: string | ContentPart[];
} {
  const images = message.images ?? [];
  if (images.length === 0) {
    return { role: message.role, content: message.content };
  }
  return {
    role: message.role,
    content: [
      { type: "text", text: message.content },
      ...images.map((url) => ({ type: "image_url" as const, image_url: { url } })),
    ],
  };
}

function chatBody(
  model: string,
  messages: ChatMessage[],
  params: ChatParams,
): string {
  return JSON.stringify({ model, messages: messages.map(wireMessage), ...params });
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

// Surface the OpenAI error envelope's human message when one comes back.
function openaiErrorMessage(body: unknown): string | null {
  if (typeof body !== "object" || body === null) return null;
  const error = (body as { error?: unknown; }).error;
  if (typeof error !== "object" || error === null) return null;
  const message = (error as { message?: unknown; }).message;
  return typeof message === "string" ? message : null;
}

async function postChatOpenAI(
  model: string,
  messages: ChatMessage[],
  params: ChatParams = {},
): Promise<string> {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, messages: messages.map(wireMessage), ...params }),
  });
  if (!res.ok) {
    const message = openaiErrorMessage(await res.json().catch(() => null))
      ?? `completions ${res.status}`;
    throw new Error(message);
  }
  const body = (await res.json()) as OpenAiChatCompletion;
  const text = body.choices[0]?.message.content ?? "";
  if (!text) throw new Error("empty completion");
  return text;
}

// Streams OpenAI `chat.completion.chunk` SSE frames plus the final
// `data: [DONE]`; throws if unusable so the caller falls back to the
// non-streaming JSON path above.
async function streamChatOpenAI(
  model: string,
  messages: ChatMessage[],
  onToken: (token: string) => void,
  params: ChatParams = {},
): Promise<string> {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, messages: messages.map(wireMessage), stream: true, ...params }),
  });
  if (!res.ok || !res.body) {
    const message = openaiErrorMessage(await res.json().catch(() => null))
      ?? `stream ${res.status}`;
    throw new Error(message);
  }
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
      if (data === "[DONE]") break;
      try {
        const chunk = JSON.parse(data) as OpenAiChunk;
        const piece = chunk.choices[0]?.delta.content;
        if (piece) {
          out += piece;
          onToken(piece);
        }
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
  // Route through POST /v1/chat/completions (OpenAI shape, stream:true)
  // instead of the native /v1/chat endpoints; default stays native. The
  // OpenAI route requires a non-empty model (server 400s otherwise).
  useOpenAI?: () => boolean;
  // True when the active model is vision-capable (catalog
  // `capabilities.vision`): shows the image-attach affordance (REQ-8).
  // Absent/false hides it — the composer never offers images it cannot
  // attribute to a vision-capable model.
  supportsVision?: () => boolean;
};

// Handle to a mounted chat surface; lets callers react to state the
// composer cannot observe (e.g. a model switch).
export type ChatCoreHandle = {
  // Re-evaluate the image-attach affordance visibility; call when the
  // active model changes.
  refreshAttach: () => void;
};

// Shared chat surface: message list + send form with SSE token streaming and
// a POST /v1/chat fallback. Callers own the host and any controls appended
// before it.
export function mountChatCore(host: HTMLElement, controls: ChatControls): ChatCoreHandle {
  const list = el("div");
  const form = el("form");
  // Image attach (REQ-8 vision chat): the file picker and drag-drop feed
  // the pending list; each image renders as a removable thumbnail chip.
  // The whole affordance stays hidden unless the active model is
  // vision-capable — the composer never offers images it cannot send.
  const attach = el("button", "Attach image");
  attach.type = "button";
  const fileInput = el("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.multiple = true;
  fileInput.hidden = true;
  const chips = el("div");
  const pendingImages: string[] = [];

  function renderChips(): void {
    chips.innerHTML = "";
    pendingImages.forEach((url, index) => {
      const chip = el("span");
      const thumb = el("img");
      thumb.src = url;
      thumb.alt = "attached image";
      thumb.style.maxWidth = "48px";
      thumb.style.maxHeight = "48px";
      thumb.style.verticalAlign = "middle";
      const remove = el("button", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", "remove attached image");
      remove.addEventListener("click", () => {
        pendingImages.splice(index, 1);
        renderChips();
      });
      chip.append(thumb, remove);
      chips.append(chip);
    });
  }

  function addImageFiles(files: FileList | null): void {
    if (!files) return;
    for (let i = 0; i < files.length; i += 1) {
      const file = files.item(i);
      if (file === null || !file.type.startsWith("image/")) continue;
      const reader = new FileReader();
      reader.addEventListener("load", () => {
        if (typeof reader.result === "string") {
          pendingImages.push(reader.result);
          renderChips();
        }
      });
      reader.readAsDataURL(file);
    }
  }

  attach.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    addImageFiles(fileInput.files);
    fileInput.value = "";
  });
  form.addEventListener("dragover", (ev) => {
    if (!attach.hidden) ev.preventDefault();
  });
  form.addEventListener("drop", (ev) => {
    if (attach.hidden) return;
    ev.preventDefault();
    addImageFiles(ev.dataTransfer?.files ?? null);
  });

  function refreshAttach(): void {
    attach.hidden = controls.supportsVision?.() !== true;
  }
  refreshAttach();

  const input = el("input");
  input.placeholder = "message";
  const send = el("button", "Send");
  form.append(chips, attach, input, send);
  host.append(list, form);

  const history: ChatMessage[] = [];
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = input.value.trim();
    const images = [...pendingImages];
    // Text alone or images alone ("describe this") — never both empty.
    if (!text && images.length === 0) return;
    input.value = "";
    pendingImages.length = 0;
    renderChips();
    const model = controls.getModel();
    const params = controls.getParams?.() ?? {};
    const message: ChatMessage = { role: "user", content: text };
    if (images.length > 0) message.images = images;
    history.push(message);
    list.append(el("p", `you: ${"[image] ".repeat(images.length)}${text}`));
    const reply = el("p", "assistant: ");
    list.append(reply);
    const append = (token: string): void => {
      reply.textContent += token;
    };
    const finish = (final: string): void => {
      history.push({ role: "assistant", content: final });
      reply.textContent = `assistant: ${final}`;
    };
    const openai = controls.useOpenAI?.() === true;
    const stream = openai ? streamChatOpenAI : streamChat;
    const post = openai ? postChatOpenAI : postChat;
    stream(model, history, append, params).then(finish).catch(() => {
      post(model, history, params).then(finish).catch((err: unknown) => {
        reply.textContent = `error: ${err instanceof Error ? err.message : err}`;
      });
    });
  });
  return { refreshAttach };
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
