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
  // No image-generation endpoint exists server-side; disabled until one lands.
  const image = el("button", "Generate image");
  image.disabled = true;
  image.title = "not available";
  knobs.append(maxTokens, temperature, image);

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
