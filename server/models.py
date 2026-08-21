"""SQLAlchemy tables.

Three tables are enough:
- machines : id, label, type, calibration blob, install_date
- readings : id, machine_id, timestamp, bands, centroid, rms
- cycles   : id, machine_id, started_at, ended_at, model_version, confirmed

Define one class per table (fill in one at a time).
"""
