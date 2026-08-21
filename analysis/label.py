"""Spectrogram labeling tool.

Loads a raw recording, renders its spectrogram, lets you click phase
boundaries and pick a label per span, then writes the label JSON. Matplotlib
event handling is enough. Build this before you have cycles to label so
labeling stays a ~10-second afterthought.

Functions to implement (fill in one at a time):
- load_recording(path)                -> raw windows from a .jsonl file
- render_spectrogram(windows)         -> matplotlib figure
- on_click(event)                     -> record a phase boundary
- write_labels(spans, out_path)       -> labels/<name>.json

Run:  python analysis/label.py analysis/recordings/<file>.jsonl
"""
