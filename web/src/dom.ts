// Small typed DOM-builder helper.

export type ElOpts = Record<string, string | ((e: Event) => void) | undefined> & {
  class?: string;
  text?: string;
};

export function el(tag: string, opts?: ElOpts, children?: Node[]): HTMLElement {
  const e = document.createElement(tag);
  if (opts) {
    for (const [k, v] of Object.entries(opts)) {
      if (v === undefined) continue;
      if (k === "class") e.className = v as string;
      else if (k === "text") e.textContent = v as string;
      else if (k.startsWith("on") && typeof v === "function") {
        e.addEventListener(k.slice(2), v as EventListener);
      } else {
        e.setAttribute(k, v as string);
      }
    }
  }
  (children || []).forEach((c) => e.appendChild(c));
  return e;
}
