"""FastAPI app -- the local server nodes POST to and the web app reads from.

Ingest path for each reading:
  reading arrives -> normalize (z-score vs machine calibration)
                  -> classifier predicts phase
                  -> HMM smooths over the recent window
                  -> machine state + cycle record updated

Runs locally; nodes POST over the LAN in plain HTTP. No cloud, no HTTPS.

Endpoints to implement (fill in one at a time):
- POST /readings      -> ingest one reading, run the pipeline, persist
- GET  /machines      -> each machine's current state + time-in-state
- GET  /health        -> liveness

Run:  uvicorn server.main:app --host 0.0.0.0 --reload
"""
