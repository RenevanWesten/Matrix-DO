# Matrix-DO

A New Method to Compute Transition Probabilities in Multi-Stable Stochastic Dynamical Systems: Application to the Wind-driven Ocean Circulation, submitted to JAMES (!!! DATE !!!)

These directories contain Python scripts for generating the DO solutions for the Burgers Equation and wind-driven ocean circulation (QG).
Run BE_main.py or QG_main.py, one may change the input variables within these files.

The DO basis needs to be sorted (post-processing) when running the QG configuration:

1) Run QG_main.py
2) Run QG_sorted_basis.py (change time sampling during post-processing)

Due to data storage limitations, we published the last 50 time steps of the original model output. We already provided the correct DO (sorted) basis for the full time series, but the time output is less frequent compared to the paper due to storage limitations. This different time sampling from post-processing results in slightly different figures.
