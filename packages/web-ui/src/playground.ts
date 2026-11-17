import { el, mountChatCore } from "./chat";
import type { ChatParams } from "./chat";
import type { CatalogList } from "./gen/api-types";

function isCatalog(value: unknown): value is CatalogList {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  const data = record["data"];
  if (!Array.isArray(data)) {
    return false;
  }
  return data.every(
    (entry) =>
      typeof entry === "object"
      && entry !== null
      && typeof (entry as Record<string, unknown>)["id"] === "string",
  );
}

async function loadCatalog(): Promise<CatalogList | null> {
  try {
    const res = await fetch("/v1/models");
    if (!res.ok) {
      return null;
    }
    const body: unknown = await res.json();
    return isCatalog(body) ? body : null;
  } catch {
    return null;
  }
}

function parseKnob(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : null;
}

export function mountPlayground(host: HTMLElement): void {
  host.innerHTML = "";
  const status = el("p", "loading catalog…");
  let catalogOk = false;
  const modelSelect = el("select");
  const serverDefault = el("option", "(server default)");
  serverDefault.value = "";
  modelSelect.append(serverDefault);
  const modelInput = el("input");
  modelInput.placeholder = "model id";
  modelInput.disabled = true;

  loadCatalog().then((catalog) => {
    if (catalog === null) {
      status.textContent = "catalog unavailable";
      modelSelect.remove();
      modelInput.disabled = false;
      return;
    }
    catalogOk = true;
    modelInput.remove();
    if (catalog.data.length === 0) {
      status.textContent = "catalog empty";
      return;
    }
    status.textContent = "";
    for (const entry of catalog.data) {
      const option = el("option", entry.id);
      option.value = entry.id;
      modelSelect.append(option);
    }
  });

  const knobs = el("div");
  const maxTokens = el("input");
  maxTokens.placeholder = "max_tokens";
  maxTokens.type = "number";
  maxTokens.min = "1";
  maxTokens.step = "1";
  const temperature = el("input");
  temperature.placeholder = "temperature";
  temperature.type = "number";
  temperature.min = "0";
  temperature.max = "2";
  temperature.step = "0.1";
  // Image generation (REQ-10): POST /v1/images/generations, render the
  // returned base64 PNG inline.
  const imagePrompt = el("input");
  imagePrompt.placeholder = "image prompt";
  const imageSize = el("input");
  imageSize.placeholder = "size";
  imageSize.value = "1024x1024";
  const image = el("button", "Generate image");
  const imageOut = el("div");
  image.addEventListener("click", () => {
    const prompt = imagePrompt.value.trim();
    if (!prompt || image.disabled) return;
    image.disabled = true;
    image.textContent = "generating…";
    fetch("/v1/images/generations", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: catalogOk ? modelSelect.value : modelInput.value,
        prompt,
        size: imageSize.value.trim() || "1024x1024",
        n: 1,
        response_format: "b64_json",
      }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const body: { error?: { message?: string; }; } | null = await res
            .json()
            .catch(() => null);
          const message = body?.error?.message ?? res.statusText;
          throw new Error(message);
        }
        return res.json() as Promise<{ data: { b64_json: string; }[]; }>;
      })
      .then((body) => {
        imageOut.innerHTML = "";
        for (const item of body.data) {
          const img = el("img");
          img.src = `data:image/png;base64,${item.b64_json}`;
          img.alt = prompt;
          imageOut.append(img);
        }
        status.textContent = "";
      })
      .catch((err: unknown) => {
        status.textContent = err instanceof Error ? err.message : "image generation failed";
      })
      .finally(() => {
        image.disabled = false;
        image.textContent = "Generate image";
      });
  });
  knobs.append(maxTokens, temperature, imagePrompt, imageSize, image, imageOut);

  // Route toggle: native /v1/chat* by default; checked sends the same chat
  // surface through POST /v1/chat/completions (OpenAI shape, stream:true).
  const openai = el("input");
  openai.type = "checkbox";
  const openaiLabel = el("label");
  openaiLabel.append(openai, " via /v1/chat/completions (OpenAI)");

  const core = el("div");
  host.append(status, modelSelect, modelInput, knobs, openaiLabel, core);
  mountChatCore(core, {
    getModel: () => (catalogOk ? modelSelect.value : modelInput.value),
    useOpenAI: () => openai.checked,
    getParams: (): ChatParams => {
      const params: ChatParams = {};
      const max = parseKnob(maxTokens.value);
      if (max !== null) params.max_tokens = Math.trunc(max);
      const temp = parseKnob(temperature.value);
      if (temp !== null) params.temperature = temp;
      return params;
    },
  });
}
