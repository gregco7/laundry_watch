"""Temporal smoothing with a Hidden Markov Model.

Sits on top of the classifier to kill flicker. Hidden states are phases;
observations are the classifier's per-window probabilities. The transition
matrix encodes laundry domain knowledge (phases last minutes; no idle->spin
without agitate between; done follows spin, not fill).

A washer's phase order is near-fixed -- idle -> fill -> agitate -> spin -> done
-- so the 5x5 matrix is mostly zeros and Viterbi does most of the smoothing on
structure alone.

Functions to implement (fill in one at a time):
- build_transitions(phases)           -> transition matrix from domain rules
- viterbi(probs, transitions)         -> most likely phase path
- smooth(classifier_probs)            -> smoothed phase sequence
"""
