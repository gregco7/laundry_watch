"""Per-machine calibration + z-scoring.

Learns what "off" looks like for a given machine (a quiet baseline) and
expresses every later feature relative to that noise floor. This is the
normalization that lets one model cover any washer -- including this one after
a remount or a shim, which is why it matters even with a single machine.

Functions to implement (fill in one at a time):
- learn_baseline(quiet_features)      -> per-feature mean/std for one machine
- zscore(feature_vector, baseline)    -> normalized feature vector
- serialize(baseline)                 -> blob stored on the machines row
- deserialize(blob)                   -> baseline
"""
