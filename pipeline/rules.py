"""Rule-based baseline state machine.

The dumb baseline you build first: thresholds on normalized band energy with
hysteresis. Not meant to be good -- meant to be the yardstick the ML is
measured against. Measure its cycle-end timing error and write it down.

Functions to implement (fill in one at a time):
- RuleState                           -> current phase + hysteresis counters
- step(state, feature_vector)         -> next phase from thresholds
- run(feature_sequence)               -> phase per window for a whole cycle
"""
