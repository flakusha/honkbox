import { el } from "./chat";
import { mountDownloads } from "./downloads";
import { mountModels } from "./models";
import { mountResidency } from "./residency";

// Review surface: residency strip on top, then the models browser (with
// curated catalog + pull buttons) and the downloads panel it nudges.
export function mountReview(host: HTMLElement): void {
  host.innerHTML = "";
  const residency = el("section");
  const models = el("section");
  const downloads = el("section");
  const panel = mountDownloads(downloads);
  mountResidency(residency);
  mountModels(models, panel);
  host.append(residency, models, downloads);
}
