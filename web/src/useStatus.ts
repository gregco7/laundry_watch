import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type Status } from "./api";

const POLL_MS = 3000;

/**
 * Polls /status and keeps a clock that stays correct between polls.
 *
 * The offset matters: every timestamp on this screen was stamped by the
 * laptop, and the phone's clock is not guaranteed to agree with it. Comparing
 * a server timestamp against Date.now() directly would make "3 minutes in this
 * phase" drift by whatever the two clocks disagree by -- which on a phone that
 * has not synced recently can be minutes. So `now` is always the server's
 * clock, advanced locally between responses.
 */
export function useStatus() {
  const [status, setStatus] = useState<Status | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const offset = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const s = await api.status();
      offset.current = s.now - Date.now() / 1000;
      setStatus(s);
      setNow(s.now);
      setUnreachable(false);
    } catch (e) {
      // The server itself is unreachable -- a different thing from the server
      // telling us the SENSOR is offline, and it needs its own message.
      if (e instanceof ApiError) setUnreachable(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const poll = setInterval(() => void refresh(), POLL_MS);
    const tick = setInterval(() => setNow(Date.now() / 1000 + offset.current), 1000);
    return () => {
      clearInterval(poll);
      clearInterval(tick);
    };
  }, [refresh]);

  // A phone screen is off most of the time. Coming back to a stale reading is
  // exactly the moment the number matters most, so refresh immediately rather
  // than waiting out the poll interval.
  useEffect(() => {
    const onWake = () => {
      if (document.visibilityState === "visible") void refresh();
    };
    document.addEventListener("visibilitychange", onWake);
    return () => document.removeEventListener("visibilitychange", onWake);
  }, [refresh]);

  return { status, now, unreachable, refresh };
}
