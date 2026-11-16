import { el } from "./chat";
import type {
  DownloadAccepted,
  DownloadKind,
  DownloadList,
  DownloadRequest,
  DownloadState,
  DownloadStatus,
} from "./gen/api-types";
import { errMessage, httpJson, humanBytes, jsonInit, num } from "./util";

// Panel handle so sibling panels (curated catalog rows) can kick the poll
// loop when they enqueue a pull.
export type DownloadsPanel = { nudge: () => void; };

const ACTIVE: ReadonlySet<DownloadStatus> = new Set(["queued", "running", "resuming"]);
const KINDS: DownloadKind[] = ["llm", "sd", "stt", "tts"];
const POLL_MS = 1000;

function progress(state: DownloadState): string {
  const done = num(state.bytes_done);
  if (state.bytes_total === null) return humanBytes(done);
  const total = num(state.bytes_total);
  const pct = total > 0 ? ` (${((done / total) * 100).toFixed(1)}%)` : "";
  return `${humanBytes(done)} / ${humanBytes(total)}${pct}`;
}

function activeRow(state: DownloadState, cancel: (id: string) => void): HTMLElement {
  const row = el("div");
  const button = el("button", "Cancel");
  button.addEventListener("click", () => cancel(state.id));
  row.append(el("strong", state.id), ` ${state.state} · ${progress(state)} `, button);
  if (state.error !== null) row.append(` — ${state.error}`);
  return row;
}

// Terminal downloads stay for history but collapse into one <details>.
function collapsed(done: DownloadState[]): HTMLElement | null {
  if (done.length === 0) return null;
  const details = el("details");
  details.append(el("summary", `${done.length} finished`));
  for (const state of done) {
    const line = `${state.id} ${state.state}${state.error === null ? "" : ` — ${state.error}`}`;
    details.append(el("div", line));
  }
  return details;
}

export function mountDownloads(host: HTMLElement): DownloadsPanel {
  host.innerHTML = "";
  const status = el("p");
  const errorLine = el("p");
  errorLine.style.color = "#cf222e";
  const activeBox = el("div");
  const doneBox = el("div");

  const repo = el("input");
  repo.placeholder = "repo (owner/name)";
  repo.required = true;
  const file = el("input");
  file.placeholder = "file path in repo";
  file.required = true;
  const kind = el("select");
  for (const value of KINDS) {
    const option = el("option", value);
    option.value = value;
    kind.append(option);
  }
  const url = el("input");
  url.placeholder = "https://…";
  url.required = true;
  url.disabled = true;
  const urlMode = el("input");
  urlMode.type = "checkbox";
  const urlLabel = el("label");
  urlLabel.append(urlMode, " direct URL");
  urlMode.addEventListener("change", () => {
    repo.disabled = urlMode.checked;
    file.disabled = urlMode.checked;
    url.disabled = !urlMode.checked;
  });
  const submit = el("button", "Download");
  const form = el("form");
  form.append(repo, file, kind, url, urlLabel, submit);

  async function start(request: DownloadRequest): Promise<void> {
    errorLine.textContent = "";
    try {
      const accepted = await httpJson<DownloadAccepted>("/v1/downloads", jsonInit("POST", request));
      status.textContent = `${accepted.id} → ${accepted.state}`;
      nudge();
    } catch (err) {
      status.textContent = "";
      errorLine.textContent = `download rejected: ${errMessage(err)}`;
    }
  }

  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const body: DownloadRequest = urlMode.checked
      ? { url: url.value.trim(), kind: kind.value as DownloadKind }
      : { repo: repo.value.trim(), file: file.value.trim(), kind: kind.value as DownloadKind };
    void start(body);
  });

  async function cancel(id: string): Promise<void> {
    try {
      const res = await fetch(`/v1/downloads/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!res.ok) errorLine.textContent = `cancel ${id} → ${res.status}`;
    } catch (err) {
      errorLine.textContent = `cancel ${id}: ${errMessage(err)}`;
    }
    nudge();
  }

  // One pass over GET /v1/downloads; true while a non-terminal row remains.
  async function pollOnce(): Promise<boolean> {
    let list: DownloadList;
    try {
      list = await httpJson<DownloadList>("/v1/downloads");
    } catch (err) {
      activeBox.innerHTML = "";
      doneBox.innerHTML = "";
      errorLine.textContent = `downloads: ${errMessage(err)}`;
      return false;
    }
    errorLine.textContent = "";
    const active = list.downloads.filter((state) => ACTIVE.has(state.state));
    const done = list.downloads.filter((state) => !ACTIVE.has(state.state));
    activeBox.innerHTML = "";
    doneBox.innerHTML = "";
    for (const state of active) activeBox.append(activeRow(state, (id) => void cancel(id)));
    if (active.length === 0) activeBox.append(el("p", "no active downloads"));
    const history = collapsed(done);
    if (history !== null) doneBox.append(history);
    return active.length > 0;
  }

  // Poll every 1s while any download is non-terminal; idle otherwise.
  let polling = false;
  async function poll(): Promise<void> {
    if (polling) return;
    polling = true;
    try {
      for (;;) {
        if (!(await pollOnce())) break;
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      }
    } finally {
      polling = false;
    }
  }

  function nudge(): void {
    void poll();
  }

  host.append(el("h2", "Downloads"), form, status, errorLine, activeBox, doneBox);
  nudge();
  return { nudge };
}
