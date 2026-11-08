import type { Toast } from "./errorUtils";

export function ErrorIcon() {
  return (
    <svg className="err-icon" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <circle cx="8" cy="8" r="6.75" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 4.75v3.75" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="8" cy="11.1" r="0.95" fill="currentColor" />
    </svg>
  );
}

export function InlineError({ message, block = false }: { message: string; block?: boolean }) {
  return (
    <span className={`inline-error ${block ? "block" : ""}`} role="alert" title={message}>
      <ErrorIcon />
      <span>{message}</span>
    </span>
  );
}

export function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div className="toasts">
      {toasts.map((t) => (
        <div key={t.id} className="toast" role="alert">
          <ErrorIcon />
          <div className="toast-body">
            <strong>{t.title}</strong>
            {t.detail && <span>{t.detail}</span>}
          </div>
          <button className="icon-close" aria-label="Dismiss" onClick={() => onDismiss(t.id)}>
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
