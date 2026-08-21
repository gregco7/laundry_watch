"""Evaluation: model vs baseline, held out by whole cycle.

NON-NEGOTIABLE: hold out entire cycles, never random windows -- adjacent
windows are correlated and a random split leaks the answer.

Reports three things:
- per-window accuracy + confusion matrix
- cycle-end timing error in seconds
- the same metrics for the rule-based baseline (pipeline.rules)

Steps to implement (fill in one at a time):
- split_by_cycle(cycles, test_frac)   -> train/test split that never breaks a cycle
- eval_windows(...)                   -> accuracy + confusion matrix
- eval_cycle_end(...)                 -> timing error in seconds
- compare(model_metrics, rule_metrics)

Run:  python analysis/evaluate.py
"""
