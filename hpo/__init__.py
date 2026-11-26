"""Hidden-perturbation oversight: synthetic-ledger experiments.

Modules
-------
ledger        text data generator (originals, planted views, tells, marker) and parser
model         exact finite generative model of the view summary (count, tells[, marker])
mechanism     the cost charged to a supervisor
strategies    menu of hand-written supervisor strategies
equilibrium   convex potential, L-BFGS equilibrium, gradient play, integer best response
discriminator learned "is the original dirty?" classifiers and Brier gain
supervisor    pluggable Supervisor.report(view_text) interface and LLM prompt templates
stats         confidence-interval helpers
"""
