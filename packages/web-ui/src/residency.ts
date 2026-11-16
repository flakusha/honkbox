import { el } from "./chat";
import type { MemoryInfo, ResidentModel, UnloadResult } from "./gen/api-types";
import { errMessage, httpJson, num } from "./util";

const POLL_MS = 2000;

function ttlLabel(seconds: number): string {
  if (seconds <= 0) return "expired";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h${Math.round((seconds % 3600) / 60)}m`;
}

function residentRow(entry: ResidentModel, unload: (slug: string) => void): HTMLElement {
  const row = el("div");
  const flags: string[] = [];
  if (entry.pinned) flags.push("pinned");
  if (entry.in_use) flags.push("in_use");
  const ttl = entry.expires_at == null
    ? "no ttl"
    : ttlLabel(num(entry.expires_at) - Date.now() / 1000);
  const button = el("button", "Unload");
  button.addEventListener("click", () => unload(entry.slug));
  row.append(
    el("strong", entry.alias ?? entry.slug),
    entry.alias == null ? "" : ` (${entry.slug})`,
    flags.length === 0 ? "" : ` [${flags.join(", ")}]`,
    ` · ttl ${ttl} · loaded ${new Date(num(entry.loaded_at) * 1000).toLocaleString()}`,
    " ",
    button,
  );
  return row;
}

export function mountResidency(host: HTMLElement): void {
  host.innerHTML = "";
  const status = el("p", "polling…");
  const errorLine = el("p");
  errorLine.style.color = "#cf222e";
  const box = el("div");
  host.append(el("h2", "Residency"), status, box, errorLine);

  async function pollOnce(): Promise<void> {
    try {
      const info = await httpJson<MemoryInfo>("/v1/memory");
      errorLine.textContent = "";
      box.innerHTML = "";
      status.textContent =
        `${info.resident.length} resident · capacity ${info.capacity.llm_max_loaded}`
        + ` · ttl ${info.capacity.llm_ttl_secs}s · gpu slot ${
          info.gpu.slot_busy ? "busy" : "free"
        }`;
      if (info.resident.length === 0) box.append(el("p", "nothing resident"));
      for (const entry of info.resident) {
        box.append(residentRow(entry, (slug) => void unload(slug)));
      }
    } catch (err) {
      box.innerHTML = "";
      errorLine.textContent = `memory: ${errMessage(err)}`;
    }
  }

  async function unload(slug: string): Promise<void> {
    try {
      const res = await httpJson<UnloadResult>(
        `/v1/models/${encodeURIComponent(slug)}/unload`,
        { method: "POST" },
      );
      status.textContent = `${res.slug}: ${res.reason}`;
    } catch (err) {
      errorLine.textContent = `unload ${slug}: ${errMessage(err)}`;
    }
    await pollOnce();
  }

  // Poll every 2s while the tab is visible; visibilitychange pauses/resumes.
  let timer: ReturnType<typeof setTimeout> | undefined;
  let running = false;
  const tick = (): void => {
    if (running || document.hidden) return;
    running = true;
    void pollOnce().finally(() => {
      running = false;
      if (!document.hidden) timer = setTimeout(tick, POLL_MS);
      else timer = undefined;
    });
  };
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (timer !== undefined) {
        clearTimeout(timer);
        timer = undefined;
      }
    } else if (timer === undefined) {
      tick();
    }
  });
  tick();
}
