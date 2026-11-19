// Per-browser conveniences (last chosen options). Storage can be unavailable
// (private mode, blocked site data), so every access is guarded and optional.
const PREFIX = "unmix:";

export function readSetting(key: string): string | null {
  try {
    return window.localStorage.getItem(PREFIX + key);
  } catch {
    return null;
  }
}

export function writeSetting(key: string, value: string): void {
  try {
    window.localStorage.setItem(PREFIX + key, value);
  } catch {
    // ignore
  }
}
