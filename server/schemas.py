"""Pydantic schemas -- validate the JSON crossing the API boundary.

Schemas to implement (fill in one at a time):
- ReadingIn      -> shape of a node's POST /readings body (bands, centroid, rms, t, node)
- MachineState   -> shape returned by GET /machines (state, time_in_state, last_cycle)
"""
