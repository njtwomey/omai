# geweke/kernels

Geweke joint-distribution tests of the handler-layer kernels on the normal-normal program (a kernel that never rejects must fail).

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (iterations 6000), 0.7 s.

**Passed**: both kernels pass

| kernel | statistic | marginal-conditional | successive-conditional | z |
| --- | --- | --- | --- | --- |
| resimulation | m | 0.019 | 0.052 | 0.93 |
| resimulation | y | 0.035 | 0.075 | 0.90 |
| random walk | m | -0.017 | 0.056 | 1.70 |
| random walk | y | 0.006 | 0.058 | 1.02 |
