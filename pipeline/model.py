"""The classifier (scikit-learn HistGradientBoostingClassifier).

Predicts a phase for each normalized feature vector. Trains in seconds; kept
deliberately small because the hand-engineered features already encode the
domain knowledge.

Functions to implement (fill in one at a time):
- train(X, y)                         -> fitted classifier
- predict(clf, X)                     -> phase per window
- predict_proba(clf, X)               -> per-window class probabilities (fed to the HMM)
- save(clf, path)                     -> joblib dump to models/clf-v1.joblib
- load(path)                          -> classifier
"""
