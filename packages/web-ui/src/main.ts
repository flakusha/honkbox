export {};

// Contract: GET /api/version → 200 {"name":"sd-rip","version":"0.1.0"}.
// NOTE: packages/web-ui/src/gen/api-types.ts (written by `bun run codegen:check`)
// will own this shape; this local stand-in is replaced — not widened —
// once the first codegen run lands gen/.
interface Version {
  name: string;
  version: string;
}

function isVersion(value: unknown): value is Version {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return typeof record["name"] === "string" && typeof record["version"] === "string";
}

async function loadVersion(): Promise<Version | null> {
  try {
    const response = await fetch("/api/version");
    if (!response.ok) {
      return null;
    }
    const body: unknown = await response.json();
    return isVersion(body) ? body : null;
  } catch {
    return null;
  }
}

function render(version: Version | null): void {
  const app = document.querySelector("#app");
  if (app === null) {
    return;
  }
  const status = version === null ? "unreachable" : `${version.name} ${version.version}`;
  app.innerHTML = "";
  const heading = document.createElement("h1");
  heading.textContent = "sd-rip";
  const para = document.createElement("p");
  para.textContent = `server: ${status}`;
  const nav = document.createElement("nav");
  const health = document.createElement("a");
  health.href = "/health";
  health.textContent = "health";
  const openapi = document.createElement("a");
  openapi.href = "/v1/openapi.json";
  openapi.textContent = "openapi";
  nav.append(health, " · ", openapi);
  app.append(heading, para, nav);
}

await loadVersion().then(render);
