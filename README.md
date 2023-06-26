# Matrix-DO

A New Method to Compute Transition Probabilities in Multi-Stable Stochastic Dynamical Systems: Application to the Wind-driven Ocean Circulation, JAMES.

These directories contain Python scripts for generating the DO solutions for the Burgers Equation and wind-driven ocean circulation (QG).
Run BE_main.py or QG_main.py to initiate the DO procedure.

The DO basis needs to be sorted (post-processing) when running the DO-QG configuration:

1) Run QG_main.py
2) Run QG_sorted_basis.py (one may change the time sampling output during this step)

Due to data storage limitations, we published the last 50 time steps of the original model output as an example file. We already provided the correct DO (sorted) basis for the full time series, but the time output is less frequent (due to storage limitations) compared to the presented results in the paper. This different time sampling (from post-processing) results in slightly different results.
