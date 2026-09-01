/**
 * The server contract, in one file.
 *
 * Nothing here knows the laptop's IP, on purpose -- every call is same-origin.
 *
 * In dev, vite serves this page and proxies /api to FastAPI (see
 * vite.config.ts). In a production build FastAPI serves the page itself, so
 * there is no proxy and no prefix. Getting this wrong is a 404 on every
 * request in exactly one of the two modes, which is why it is one constant.
 */
const API_BASE = import.meta.env.DEV ? "/api" : "";

export const PHASES = ["fill", "wash", "rinse", "spin", "done"] as const;
export type Phase = (typeof PHASES)[number] | "idle";

/** Wire value -> what the washer's own front panel displays. */
export const PHASE_LABEL: Record<string, string> = {
  idle: "Idle",
  fill: "Sensing Fill",
  wash: "Wash",
  rinse: "Rinse",
  spin: "Spin",
  done: "Done",
};

export type Mode = "idle" | "running" | "done" | "offline";

export interface PhaseEvent {
  at: number;
  phase: string;
}

export interface Status {
  mode: Mode;
  now: number;
  sensor_ok: boolean;
  silent_for: number | null;

  cycle_id: number | null;
  phase: string | null;
  phase_since: number | null;
  history: PhaseEvent[];

  finished_at: number | null;
  emptied_at: number | null;
  emptied_by: string | null;

  last_finished_at: number | null;
  last_emptied_at: number | null;
  last_emptied_by: string | null;

  last_known_phase: string | null;
  last_known_at: number | null;

  /** False until a model exists: `phase` is then the last human mark. */
  predicted: boolean;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { "content-type": "application/json" },
      ...init,
    });
  } catch {
    // A dead laptop and a dead WiFi link are indistinguishable from here, and
    // the distinction does not change what the person should do.
    throw new ApiError("Can't reach the server");
  }
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      // FastAPI puts a string in `detail` for HTTPException and a list of
      // field errors there for a 422.
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) detail = body.detail[0]?.msg ?? detail;
    } catch {
      /* a non-JSON error body is not worth a second failure path */
    }
    throw new ApiError(detail, res.status);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export const api = {
  status: () => request<Status>("/status"),

  startCycle: (body: { setting?: string; load_size?: string } = {}) =>
    request<{ id: number }>("/cycles", {
      method: "POST",
      body: JSON.stringify({ machine_id: "washer-01", ...body }),
    }),

  endCycle: (id: number) => request<unknown>(`/cycles/${id}/end`, { method: "POST" }),

  /**
   * `t` is sent explicitly on retry so a mark lands at the moment it was
   * tapped, not the moment the retry succeeded. On a first attempt it is
   * omitted and the server stamps now -- keeping marks on the same clock as
   * the sensor windows they will be joined against.
   */
  mark: (id: number, phase: string, t?: number) =>
    request<unknown>(`/cycles/${id}/mark`, {
      method: "POST",
      body: JSON.stringify(t === undefined ? { phase } : { phase, t }),
    }),

  empty: (id: number, by?: string) =>
    request<unknown>(`/cycles/${id}/empty`, {
      method: "POST",
      body: JSON.stringify({ by: by ?? null }),
    }),
};

/* ── formatting ───────────────────────────────────────────────────────────
   All server times are float epoch SECONDS. JS wants milliseconds; the x1000
   is the single most likely place for this to go quietly wrong, so it lives
   in exactly one function.                                                  */

const ms = (epochSeconds: number) => epochSeconds * 1000;

export function clockTime(epochSeconds: number): string {
  return new Date(ms(epochSeconds)).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

/** "Finished 47 minutes ago" / "Just now" / "Finished 2 hours 4 minutes ago" */
export function agoPhrase(epochSeconds: number, now: number, verb = "Finished"): string {
  const mins = Math.max(0, Math.floor((now - epochSeconds) / 60));
  if (mins < 1) return "Just now";
  if (mins < 60) return `${verb} ${mins} minute${mins === 1 ? "" : "s"} ago`;
  const h = Math.floor(mins / 60);
  const r = mins % 60;
  const hourPart = `${h} hour${h === 1 ? "" : "s"} `;
  const minPart = r ? `${r} minute${r === 1 ? "" : "s"} ` : "";
  return `${verb} ${hourPart}${minPart}ago`;
}

/** "25 minutes in this phase" — the running screen's subtitle. */
export function durationPhrase(sinceEpoch: number, now: number): string {
  const mins = Math.max(0, Math.floor((now - sinceEpoch) / 60));
  if (mins < 1) return "Just started";
  return `${mins} minute${mins === 1 ? "" : "s"} in this phase`;
}

export function silentPhrase(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)} seconds`;
  const mins = Math.round(seconds / 60);
  if (mins < 60) return `${mins} minutes`;
  const h = Math.floor(mins / 60);
  return `${h} hour${h === 1 ? "" : "s"}`;
}
