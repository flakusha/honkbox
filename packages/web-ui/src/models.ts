import { el } from "./chat";
import type { DownloadsPanel } from "./downloads";
import type {
  CatalogEntry,
  CatalogList,
  DownloadAccepted,
  DownloadKind,
  ModelCapabilities,
  ModelCatalog,
  ModelCatalogEntry,
  Pill,
} from "./gen/api-types";
import { errMessage, httpJson, humanBytes, jsonInit, num } from "./util";

const PILL_COLOR: Record<Pill, string> = { ok: "#1a7f37", slow: "#9a6700", fail: "#cf222e" };
const PILL_LABEL: Record<Pill, string> = { ok: "fits", slow: "tight", fail: "over" };

function baseName(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx >= 0 ? path.slice(idx + 1) : path;
}

function flagList(caps: ModelCapabilities): string {
  const flags: string[] = [];
  if (caps.chat) flags.push("chat");
  if (caps.tool_calling) flags.push("tools");
  if (caps.vision) flags.push("vision");
  if (caps.embeddings) flags.push("embed");
  return flags.join("·");
}

function localRow(entry: ModelCatalogEntry): HTMLElement {
  const row = el("div");
  const badge = el("span", entry.kind);
  badge.style.fontWeight = "600";
  const size = `${humanBytes(entry.bytes)} · ${entry.category}`;
  const meta = entry.metadata === null
    ? ""
    : entry.metadata.format === "gguf"
    ? entry.metadata.architecture ?? entry.metadata.name ?? "gguf"
    : `${entry.metadata.dtypes.length} dtypes`;
  const ctx = entry.context_length === null ? "" : ` · ctx ${num(entry.context_length)}`;
  row.append(
    badge,
    ` ${entry.id}`,
    entry.slug === entry.id ? "" : ` (${entry.slug})`,
    ` — ${meta}${meta ? " · " : ""}${size}${ctx}`,
  );
  row.title = `${entry.path} · ${flagList(entry.capabilities)}`;
  return row;
}

function curatedRow(
  entry: CatalogEntry,
  onDisk: ModelCatalogEntry | undefined,
  downloads: DownloadsPanel | null,
): HTMLElement {
  const row = el("div");
  if (entry.local?.pill != null) {
    const pill = el("span", PILL_LABEL[entry.local.pill]);
    pill.style.color = PILL_COLOR[entry.local.pill];
    pill.style.fontWeight = "600";
    row.append(pill, " ");
  }
  row.append(
    el("strong", entry.id),
    ` ${entry.kind} ${entry.quant} · ${entry.license} — ${entry.notes}`,
  );
  if (entry.local !== null) {
    const est = entry.local.est_vram_bytes === null
      ? ""
      : ` · est vram ${humanBytes(entry.local.est_vram_bytes)}`;
    row.append(` · on disk ${humanBytes(entry.local.bytes)} (${entry.local.slug})${est}`);
  } else if (onDisk !== undefined) {
    row.append(` · on disk as ${onDisk.id}`);
  }

  if (downloads === null) return row;
  const pull = el("button", "Pull");
  const note = el("span");
  // DownloadRequest.kind only admits the four root modalities; the checked-in
  // manifest test enforces that, but narrow here instead of trusting it.
  const kind =
    entry.kind === "llm" || entry.kind === "sd" || entry.kind === "stt" || entry.kind === "tts"
      ? (entry.kind as DownloadKind)
      : null;
  if (kind === null) {
    pull.disabled = true;
    pull.title = "kind is not downloadable";
  }
  pull.addEventListener("click", () => {
    if (kind === null) return;
    pull.disabled = true;
    const request = { repo: entry.repo, file: entry.file, kind };
    httpJson<DownloadAccepted>("/v1/downloads", jsonInit("POST", request))
      .then((accepted) => {
        note.textContent = ` ${accepted.id} ${accepted.state}`;
        pull.disabled = false;
        downloads.nudge();
      })
      .catch((err: unknown) => {
        note.textContent = ` error: ${errMessage(err)}`;
        pull.disabled = false;
      });
  });
  row.append(" ", pull, note);
  return row;
}

export function mountModels(host: HTMLElement, downloads?: DownloadsPanel): void {
  host.innerHTML = "";
  const rescan = el("button", "Rescan");
  const status = el("p", "loading…");
  const budget = el("p");
  const localBox = el("div");
  const curatedBox = el("div");
  host.append(el("h2", "Models"), rescan, status, budget, localBox, curatedBox);

  async function refresh(): Promise<void> {
    status.textContent = "loading…";
    const [models, curated] = await Promise.allSettled([
      httpJson<ModelCatalog>("/v1/models"),
      httpJson<CatalogList>("/v1/catalog"),
    ]);
    localBox.innerHTML = "";
    curatedBox.innerHTML = "";
    budget.textContent = "";

    const byFile = new Map<string, ModelCatalogEntry>();
    if (models.status === "rejected") {
      status.textContent = `models: ${errMessage(models.reason)}`;
    } else {
      const catalog = models.value;
      status.textContent = catalog.data.length === 0
        ? "no local models"
        : `${catalog.data.length} local models`;
      for (const entry of catalog.data) {
        byFile.set(baseName(entry.path), entry);
        localBox.append(localRow(entry));
      }
    }

    if (curated.status === "rejected") {
      curatedBox.append(el("p", `catalog: ${errMessage(curated.reason)}`));
      return;
    }
    const entries = curated.value.data;
    curatedBox.append(el("h3", "Curated catalog"));
    let pills = 0;
    for (const entry of entries) {
      if (entry.local?.pill != null) pills += 1;
      curatedBox.append(curatedRow(entry, byFile.get(baseName(entry.file)), downloads ?? null));
    }
    // The budget itself is not part of CatalogList; the server folds it into
    // each entry's pill, so its presence is inferred from the pills.
    budget.textContent = entries.length === 0
      ? "curated catalog empty"
      : pills === 0
      ? "no fit pills (no [models] vram_budget_bytes set, or nothing local)"
      : `fit pills vs vram budget: ${pills}/${entries.length} entries`;
  }

  rescan.addEventListener("click", () => {
    rescan.disabled = true;
    httpJson<ModelCatalog>("/v1/models/refresh", jsonInit("POST", {}))
      .then(() => refresh())
      .catch((err: unknown) => {
        status.textContent = `rescan: ${errMessage(err)}`;
      })
      .finally(() => {
        rescan.disabled = false;
      });
  });

  void refresh();
}
