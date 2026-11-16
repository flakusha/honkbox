// Small shared helpers for the review panels: typed JSON fetch + formatting.
// Deliberately DOM-free so `bun test` can exercise the pure parts directly.

/** u64 DTO fields are typed `bigint` by ts-rs but arrive as JSON numbers. */
export function num(value: number | bigint): number {
  return typeof value === "bigint" ? Number(value) : value;
}

export function humanBytes(value: number | bigint): string {
  const bytes = num(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const shown = unit === 0 ? String(size) : size >= 100 ? size.toFixed(0) : size.toFixed(1);
  return `${shown} ${units[unit]}`;
}

export function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export class HttpError extends Error {
  constructor(
    readonly status: number,
    url: string,
    method: string,
  ) {
    super(`${method} ${url} → ${status}`);
  }
}

export function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  };
}

export async function httpJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) throw new HttpError(res.status, url, init?.method ?? "GET");
  return (await res.json()) as T;
}
