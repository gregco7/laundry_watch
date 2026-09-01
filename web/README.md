# web/ — the LaundryWatch dashboard

Vite + React + Tailwind v4. Phone-first, one screen, four states.

## Running it

**For real (recommended).** Build once; FastAPI serves the bundle, so there is
one process to start before a wash instead of two — and a dead dev server
can't take the marking instrument with it:

```
npm install
npm run build
cd .. && venv/bin/uvicorn server.main:app --host 0.0.0.0
```

Then open `http://<laptop-ip>:8000/` on your phone. Add it to the home screen
and it opens without browser chrome.

**While changing the UI:**

```
npm run dev          # then open http://<laptop-ip>:5173/
```

`vite.config.ts` proxies `/api` to `127.0.0.1:8000`, so the API stays
same-origin and no LAN IP is hardcoded anywhere. `src/api.ts` drops the `/api`
prefix in a production build, where FastAPI serves the page itself.

## The four states

Which one renders is decided by the server, in `GET /status` — the client
picks a layout, never a state:

| mode | when |
|---|---|
| `offline` | no window from the node for 20s. Outranks everything. |
| `done` | a `done` mark, or a closed cycle, that nobody has emptied yet |
| `running` | a cycle is open and not finished |
| `idle` | everything else |

## Known gaps

- **`phase` is not a prediction yet.** It's the last phase a human marked.
  `StatusOut.predicted` says which, and the screen says "As marked — no model
  yet" so it doesn't imply it worked it out. When `pipeline/model.py` lands,
  only `server/main.py:status()` changes.
- **`RecordSheet` is a stopgap.** The real recording UI is its own design.
  This exists so a cycle can be opened, marked and closed today — without it
  the status screen could never leave `idle`.
