# M2 AC model-comparison corpus

These three public PCD studies use the same source, frequency, load table, and
three scenario identifiers.  Each topology has twelve exact discrete designs,
so design, external-condition, and leave-one-topology-family-out comparisons
can use whole groups without row leakage.

The values are synthetic coverage inputs for comparing a constant predictor,
ridge regression, an MLP, and a small relational GNN.  They are not chamber
measurements, a plasma-chemistry model, an optimizer benchmark, or evidence
that a learned model may replace ngspice.
