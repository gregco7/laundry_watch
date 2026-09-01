import { useCallback, useState } from "react";
import {
  agoPhrase,
  api,
  ApiError,
  clockTime,
  durationPhrase,
  PHASE_LABEL,
  PHASES,
  silentPhrase,
  type Status,
} from "./api";
import { useStatus } from "./useStatus";

/* A tap the network lost. Held with the time it was tapped so a retry lands
   where the phase actually changed, not where the retry succeeded. */
type FailedMark = { phase: string; t: number; cycleId: number; message: string };

export default function App() {
  const { status, now, unreachable, refresh } = useStatus();
  const [sheet, setSheet] = useState<null | "correct" | "record">(null);
  const [failed, setFailed] = useState<FailedMark | null>(null);
  const [busy, setBusy] = useState(false);

  const sendMark = useCallback(
    async (phase: string, cycleId: number, t?: number) => {
      const stamped = t ?? Date.now() / 1000;
      setBusy(true);
      try {
        await api.mark(cycleId, phase, t);
        setFailed(null);
        await refresh();
      } catch (e) {
        setFailed({
          phase,
          t: stamped,
          cycleId,
          message: e instanceof ApiError ? e.message : "Failed to send",
        });
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  if (!status) {
    return (
      <Shell>
        <div className="flex flex-1 items-center justify-center text-[17px] text-neutral-600">
          {unreachable ? "Can't reach the server" : " "}
        </div>
      </Shell>
    );
  }

  const isDone = status.mode === "done";

  return (
    <Shell done={isDone}>
      <Header sensorOk={status.sensor_ok && !unreachable} />

      {unreachable ? (
        <ServerDown />
      ) : status.mode === "idle" ? (
        <IdleView status={status} now={now} />
      ) : status.mode === "running" ? (
        <RunningView status={status} now={now} />
      ) : status.mode === "done" ? (
        <DoneView
          status={status}
          now={now}
          busy={busy}
          onAck={async () => {
            if (status.cycle_id == null) return;
            setBusy(true);
            try {
              await api.empty(status.cycle_id);
              await refresh();
            } catch (e) {
              setFailed({
                phase: "emptied",
                t: Date.now() / 1000,
                cycleId: status.cycle_id,
                message: e instanceof ApiError ? e.message : "Failed to send",
              });
            } finally {
              setBusy(false);
            }
          }}
        />
      ) : (
        <OfflineView status={status} />
      )}

      <BottomBar
        canCorrect={status.mode === "running"}
        onCorrect={() => setSheet("correct")}
        onRecord={() => setSheet("record")}
      />

      {failed && (
        <FailedBanner
          failed={failed}
          busy={busy}
          onRetry={() => void sendMark(failed.phase, failed.cycleId, failed.t)}
          onDismiss={() => setFailed(null)}
        />
      )}

      {sheet === "correct" && status.cycle_id != null && (
        <PhaseSheet
          title="What is it actually doing?"
          busy={busy}
          onPick={async (p) => {
            setSheet(null);
            await sendMark(p, status.cycle_id!);
          }}
          onClose={() => setSheet(null)}
        />
      )}

      {sheet === "record" && (
        <RecordSheet
          status={status}
          busy={busy}
          onClose={() => setSheet(null)}
          onMark={(p) => status.cycle_id != null && void sendMark(p, status.cycle_id)}
          onStart={async () => {
            setBusy(true);
            try {
              await api.startCycle();
              await refresh();
            } finally {
              setBusy(false);
            }
          }}
          onEnd={async () => {
            if (status.cycle_id == null) return;
            setBusy(true);
            try {
              await api.endCycle(status.cycle_id);
              await refresh();
              setSheet(null);
            } finally {
              setBusy(false);
            }
          }}
        />
      )}
    </Shell>
  );
}

/* ── chrome ─────────────────────────────────────────────────────────────── */

function Shell({ children, done = false }: { children: React.ReactNode; done?: boolean }) {
  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-bg">
      {/* The done state floods the whole screen rather than showing a badge:
          it should be recognisable from across a room, before reading. */}
      {done && (
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(120% 70% at 50% 18%, #3b4290 0%, #262a60 55%, #1c1f4a 100%)",
          }}
        />
      )}
      <div
        className="relative flex min-h-0 flex-1 flex-col px-[22px] pb-5"
        style={{
          paddingTop: "calc(env(safe-area-inset-top, 0px) + 14px)",
          paddingBottom: "calc(env(safe-area-inset-bottom, 0px) + 20px)",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function Header({ sensorOk }: { sensorOk: boolean }) {
  return (
    <div className="flex flex-none items-center justify-between pb-[26px]">
      <span className="text-[13px] uppercase tracking-[.16em] text-neutral-500">Washer</span>
      {sensorOk && (
        <span className="flex items-center gap-[7px] text-[12px] uppercase tracking-[.1em] text-neutral-500">
          <span className="lw-breathe size-[7px] rounded-full bg-live" />
          Listening
        </span>
      )}
    </div>
  );
}

function BottomBar({
  canCorrect,
  onCorrect,
  onRecord,
}: {
  canCorrect: boolean;
  onCorrect: () => void;
  onRecord: () => void;
}) {
  return (
    <div className="flex flex-none items-center gap-[14px] pt-4">
      {canCorrect && (
        <button
          type="button"
          onClick={onCorrect}
          className="min-h-12 cursor-pointer border-0 bg-transparent py-[14px] pr-3 text-[15px] text-neutral-500 underline underline-offset-4"
        >
          That's not right
        </button>
      )}
      <button
        type="button"
        onClick={onRecord}
        className="ml-auto min-h-12 cursor-pointer border-0 bg-transparent py-[14px] pl-3 text-[15px] text-neutral-600"
      >
        Record a cycle →
      </button>
    </div>
  );
}

/* ── the four states ────────────────────────────────────────────────────── */

function IdleView({ status, now }: { status: Status; now: number }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="font-heading text-[44px] font-medium leading-[1.05] tracking-[-.03em] text-neutral-400">
        Nothing
        <br />
        running
      </div>

      <div className="mt-[34px] border-t border-neutral-800 pt-[18px]">
        <div className="pb-[10px] text-[12px] uppercase tracking-[.14em] text-neutral-600">
          Last load
        </div>
        {status.last_finished_at ? (
          <>
            <div className="font-heading text-[24px] font-medium tracking-[-.02em]">
              Finished {clockTime(status.last_finished_at)}
            </div>
            <div className="mt-[6px] text-[16px] text-neutral-400">
              {status.last_emptied_at
                ? `${status.last_emptied_by ? `Emptied by ${status.last_emptied_by}` : "Emptied"} at ${clockTime(status.last_emptied_at)}`
                : agoPhrase(status.last_finished_at, now, "Still sitting there — finished")}
            </div>
          </>
        ) : (
          <div className="text-[16px] text-neutral-500">No loads recorded yet.</div>
        )}
      </div>
    </div>
  );
}

function RunningView({ status, now }: { status: Status; now: number }) {
  const phase = status.phase ? (PHASE_LABEL[status.phase] ?? status.phase) : "Starting";
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="text-[13px] uppercase tracking-[.16em] text-neutral-500">Right now</div>
      <div className="mt-[10px] font-heading text-[60px] font-semibold leading-none tracking-[-.035em]">
        {phase}
      </div>
      <div className="mt-4 text-[18px] text-neutral-500">
        {status.phase_since ? durationPhrase(status.phase_since, now) : " "}
      </div>

      {/* Until a model exists this is the last thing a human marked. Saying so
          costs one line and stops the screen from implying it worked it out. */}
      {!status.predicted && status.phase && (
        <div className="mt-3 text-[13px] text-neutral-600">As marked — no model yet</div>
      )}

      <div className="mt-auto flex flex-col gap-[9px] border-t border-neutral-800 pt-4">
        {status.history.slice(-5).map((h, i) => (
          <div key={`${h.at}-${i}`} className="flex items-baseline gap-[14px]">
            <span className="w-[52px] text-[15px] tabular-nums text-neutral-600">
              {clockTime(h.at)}
            </span>
            <span className="font-heading text-[17px] text-neutral-300">
              {PHASE_LABEL[h.phase] ?? h.phase}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DoneView({
  status,
  now,
  busy,
  onAck,
}: {
  status: Status;
  now: number;
  busy: boolean;
  onAck: () => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="lw-glow mt-[26px] flex size-[78px] items-center justify-center rounded-full border-[3px] border-done-edge">
        <span className="text-[40px] leading-none text-[#eceafd]">✓</span>
      </div>

      <div className="mt-6 font-heading text-[64px] font-semibold leading-[.98] tracking-[-.04em] text-done-ink">
        Laundry's
        <br />
        done
      </div>
      <div className="mt-[18px] text-[26px] text-done-soft">
        {status.finished_at ? agoPhrase(status.finished_at, now) : ""}
      </div>

      <div className="mt-auto flex flex-col gap-[10px]">
        {status.emptied_at ? (
          <div className="rounded-md border border-[#6d72b8] px-4 py-[14px] text-[16px] text-[#d8d6f7]">
            {status.emptied_by
              ? `${status.emptied_by} took it out ${agoPhrase(status.emptied_at, now, "").trim().toLowerCase()}`
              : agoPhrase(status.emptied_at, now, "Taken out")}
          </div>
        ) : (
          <button
            type="button"
            onClick={onAck}
            disabled={busy}
            className="h-[76px] cursor-pointer rounded-md border-2 border-done-edge bg-white/6 font-heading text-[21px] font-semibold text-done-ink disabled:opacity-50"
          >
            I've taken it out
          </button>
        )}
      </div>
    </div>
  );
}

function OfflineView({ status }: { status: Status }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="font-heading text-[44px] font-medium leading-[1.04] tracking-[-.03em]">
        Can't see
        <br />
        the machine
      </div>
      <div className="mt-4 max-w-[300px] text-[18px] text-neutral-400">
        No data from the sensor for {silentPhrase(status.silent_for ?? 0)}. This screen does not
        know what the washer is doing.
      </div>

      {status.last_known_phase && status.last_known_at && (
        <div className="mt-[30px] rounded-lg border border-neutral-800 bg-[#1a1c29] px-4 py-[14px]">
          <div className="text-[12px] uppercase tracking-[.12em] text-neutral-600">
            Last reading before contact dropped
          </div>
          <div className="mt-[6px] font-heading text-[22px] text-neutral-500">
            {PHASE_LABEL[status.last_known_phase] ?? status.last_known_phase}, as of{" "}
            {clockTime(status.last_known_at)}
          </div>
        </div>
      )}

      <div className="mt-auto text-[15px] text-neutral-500">
        Usually the laptop went to sleep. Wake it and this reconnects on its own.
      </div>
    </div>
  );
}

function ServerDown() {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="font-heading text-[44px] font-medium leading-[1.04] tracking-[-.03em]">
        Can't reach
        <br />
        the server
      </div>
      <div className="mt-4 max-w-[300px] text-[18px] text-neutral-400">
        The laptop isn't answering. Nothing is being recorded right now.
      </div>
      <div className="mt-auto text-[15px] text-neutral-500">
        This page retries on its own every few seconds.
      </div>
    </div>
  );
}

/* ── sheets and banners ─────────────────────────────────────────────────── */

function FailedBanner({
  failed,
  busy,
  onRetry,
  onDismiss,
}: {
  failed: FailedMark;
  busy: boolean;
  onRetry: () => void;
  onDismiss: () => void;
}) {
  return (
    <div className="lw-rise absolute inset-x-[22px] bottom-[86px] rounded-lg border border-alert/60 bg-[#2a1d1f] p-4">
      <div className="font-heading text-[17px] text-[#f0cdc7]">
        “{PHASE_LABEL[failed.phase] ?? failed.phase}” didn't send
      </div>
      <div className="mt-1 text-[14px] text-neutral-400">{failed.message}</div>
      <div className="mt-3 flex gap-2">
        <button
          type="button"
          onClick={onRetry}
          disabled={busy}
          className="h-12 flex-1 cursor-pointer rounded-md border border-done-edge bg-white/6 font-heading text-[16px] text-done-ink disabled:opacity-50"
        >
          Retry — keeps {clockTime(failed.t)}
        </button>
        <button
          type="button"
          onClick={onDismiss}
          className="h-12 cursor-pointer rounded-md border-0 bg-transparent px-4 text-[15px] text-neutral-500"
        >
          Discard
        </button>
      </div>
    </div>
  );
}

function Sheet({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 flex items-end bg-[rgba(6,7,12,.8)]">
      <div
        className="lw-rise w-full rounded-t-[24px] border-t border-neutral-700 bg-surface px-[18px] pt-5"
        style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 0px) + 24px)" }}
      >
        {children}
      </div>
    </div>
  );
}

function PhaseSheet({
  title,
  busy,
  onPick,
  onClose,
}: {
  title: string;
  busy: boolean;
  onPick: (phase: string) => void;
  onClose: () => void;
}) {
  return (
    <Sheet>
      <h2 className="mb-4 font-heading text-[22px] font-medium">{title}</h2>
      <div className="grid grid-cols-2 gap-2">
        {PHASES.map((p, i) => (
          <button
            key={p}
            type="button"
            disabled={busy}
            onClick={() => onPick(p)}
            className="h-[68px] cursor-pointer rounded-md border border-neutral-700 bg-[#20222f] font-heading text-[19px] font-semibold text-text disabled:opacity-50"
            style={i === PHASES.length - 1 ? { gridColumn: "span 2" } : undefined}
          >
            {PHASE_LABEL[p]}
          </button>
        ))}
      </div>
      <button
        type="button"
        onClick={onClose}
        className="mt-[10px] h-[52px] w-full cursor-pointer border-0 bg-transparent font-heading text-[16px] text-neutral-500"
      >
        Cancel
      </button>
    </Sheet>
  );
}

/**
 * Stopgap. The real recording UI is its own design; this exists so a cycle can
 * be opened, marked and closed today -- without it the status screen can never
 * leave `idle`, because nothing else creates a cycle.
 */
function RecordSheet({
  status,
  busy,
  onClose,
  onStart,
  onEnd,
  onMark,
}: {
  status: Status;
  busy: boolean;
  onClose: () => void;
  onStart: () => void;
  onEnd: () => void;
  onMark: (phase: string) => void;
}) {
  const open = status.cycle_id != null && status.mode === "running";
  return (
    <Sheet>
      <h2 className="mb-1 font-heading text-[22px] font-medium">
        {open ? "Recording" : "Record a cycle"}
      </h2>
      <p className="mb-4 text-[14px] text-neutral-500">
        {open
          ? "Tap the phase your washer's panel is showing."
          : "Opens a recording and captures the last five minutes already buffered."}
      </p>

      {open ? (
        <>
          <div className="grid grid-cols-2 gap-2">
            {PHASES.map((p, i) => (
              <button
                key={p}
                type="button"
                disabled={busy}
                onClick={() => onMark(p)}
                className="h-[68px] cursor-pointer rounded-md border border-neutral-700 bg-[#20222f] font-heading text-[19px] font-semibold text-text disabled:opacity-50"
                style={i === PHASES.length - 1 ? { gridColumn: "span 2" } : undefined}
              >
                {PHASE_LABEL[p]}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={onEnd}
            disabled={busy}
            className="mt-3 h-[60px] w-full cursor-pointer rounded-md border border-neutral-700 bg-transparent font-heading text-[17px] text-neutral-300 disabled:opacity-50"
          >
            End recording
          </button>
        </>
      ) : (
        <button
          type="button"
          onClick={onStart}
          disabled={busy}
          className="h-[76px] w-full cursor-pointer rounded-md border-2 border-accent bg-accent/10 font-heading text-[21px] font-semibold text-text disabled:opacity-50"
        >
          Start recording
        </button>
      )}

      <button
        type="button"
        onClick={onClose}
        className="mt-[10px] h-[52px] w-full cursor-pointer border-0 bg-transparent font-heading text-[16px] text-neutral-500"
      >
        Close
      </button>
    </Sheet>
  );
}
