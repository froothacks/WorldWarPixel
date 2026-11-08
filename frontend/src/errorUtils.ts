import { useCallback, useRef, useState } from "react";

/** Short, human hint for a raw model/API error message, or null when nothing fits. */
export function errorHint(message: string): string | null {
  const code = /Error code: (\d{3})/.exec(message)?.[1];
  if (code === "401" || code === "403") return "check the API key";
  if (code === "404") return "model not found";
  if (code === "429") return "rate limited";
  if (code?.startsWith("5")) return "model server error";
  if (/connect|timed? ?out/i.test(message)) return "can't reach the model server";
  return null;
}

export interface Toast {
  id: number;
  title: string;
  detail?: string;
}

export function useToasts(timeout = 6000) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);
  const dismiss = useCallback((id: number) => setToasts((ts) => ts.filter((t) => t.id !== id)), []);
  const notify = useCallback(
    (title: string, detail?: string) => {
      const id = next.current++;
      // A repeat of the same error replaces the old toast instead of stacking.
      setToasts((ts) => [...ts.filter((t) => t.title !== title || t.detail !== detail).slice(-2), { id, title, detail }]);
      window.setTimeout(() => dismiss(id), timeout);
    },
    [dismiss, timeout],
  );
  return { toasts, notify, dismiss };
}
