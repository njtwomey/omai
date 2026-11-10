# benchmark/funnel-centred

Neal's funnel, centred: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 2, rows 3, scalePercent 100), 9.1 s.

**Passed**: 0 method(s) agree with the analytic truth, 2 differ as expected; 0 reported

**Neal's funnel, centred** (funnel-centred): funnel, geometry, divergences, centred, density, analytic, hierarchical-prior

Neal's funnel in ten dimensions, written directly: v ~ N(0, 9) and nine coordinates x_i ~ N(0, e^v). The neck defeats a fixed step size.

Truth (analytic): v is N(0, 9) exactly; x_i | v ~ N(0, e^v) is a scale mixture with mean 0 and sd exp(2.25), its CDF the empirical one of 100000 iid draws of the exact sampler.

Methods whose own `check` says they apply to the compiled program (a target with no observed node is refused at the spec level): nuts, svi, laplace, importance, tempered-smc. Not applicable: gibbs (no discrete or conjugate conditionals (use NUTS)); em (no params to estimate); particle-filter (no single markov loop); abc-smc (no observed sites); conjugate (no conjugate structure for v, x); gaussian (the scale of x depends on latent sites); enumeration (continuous latent sites (v, x)); chain (the program needs exactly one markov loop); abc-rejection (no observed sites).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | expected to differ | converged (1001 steps) | 2 divergences; R-hat high: v, x[0], x[1], ...; ESS low: v, x[0], x[1], ... | divergent transitions in the neck and too small a spread of v: the sampler never reaches the narrowest part (the non-centred funnel has none) |
| svi | expected to differ | completed (1500 steps) | - | a mean-field normal cannot be a funnel: v and x are dependent in scale, not in location |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | v | -0.443 | 2.71 | -0.148 | -0.475 | 0.903 | 0.0944 | 0.168 | 0.043 | 8.43 |
| svi | v | 0.211 | 0.496 | 0.0705 | 26.9 | 0.165 | 0.379 | 0.667 | 3.3 | 4000 |
| svi | x[0] | 0.0246 | 1.18 | 0.0026 | 1.32 | 0.124 | 0.087 | 0.17 | 0.705 | 4000 |
