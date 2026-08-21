"""Training entry point.

Loads recordings + labels, builds normalized feature vectors, fits the
classifier, and saves it to models/clf-v1.joblib.

Steps to implement (fill in one at a time):
- load recordings + labels from analysis/
- extract features (pipeline.features) and z-score them (pipeline.calibrate)
- attach each window's label from its phase span
- fit the model (pipeline.model.train)
- save to models/clf-v1.joblib

Run:  python analysis/train.py
"""
