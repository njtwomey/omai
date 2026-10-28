# Learning theory (part 4, principles of learning)

Learning theory simulated: PAC bounds against observed error, uniform convergence and capacity (VC dimension, Rademacher complexity estimated), margins and norms, algorithm-dependent bounds (stability, PAC-Bayes), rates and consistency, complexity regularisation, and bias-variance and interpolation, as sessions over the models of `supervised-learning` with `datasets` that have a known Bayes rate.

## Capabilities

- Bounds against reality: the generalisation gap of a model class as n grows, with the bound drawn; options: the class, the bound, the distribution.
- Capacity: shattering of point sets by a class, empirical Rademacher complexity by simulation.
- Bias-variance: the decomposition by repeated fits over fresh samples, as a function of the model's complexity; double descent as the parameter count crosses n.
- Margins: the margin distribution of a classifier against its bound.
- Stability and PAC-Bayes: the effect of one changed point, the KL term as the prior changes.
- Rates: an estimator's error against n on a log-log axis with the theoretical slope.

## Examples to make

- Bias-variance against polynomial degree, the fits of many samples drawn faintly and their mean in bold.
- Double descent: test error against the width of a random-feature model, n as the knob.
- Shattering: four points in the plane, every labelling tried by a line.
- The generalisation gap against the VC bound as n grows, for a few classes.

## Rendering

- Curves with bands, many faint fits in one panel, points with labels being tried (a board-like step record), log-log axes.
- Missing: the many-runs overlay (the comparison panel again, with alpha).

## Shared

- `shared/`: the repeated-fit study (fresh sample, fit, measure, repeat) shared with `statistics`; the bound catalogue as functions of n and the class's capacity.

## Problems

- `theory/` moves here. The repeated-fit study is the sweep base's hardest case: many fits per point of a curve, several knobs.
