# geweke/kernels

Geweke joint-distribution tests of the handler-layer kernels on the normal-normal program (a kernel that never rejects must fail).

Run on 2026-11-10 at commit 4ce8b576; seed 0, budget scale 3 (iterations 18000), 1.6 s.

**Passed**: both kernels pass

| kernel       | statistic | marginal-conditional | successive-conditional | z     |
| ------------ | --------- | -------------------- | ---------------------- | ----- |
| resimulation | m         | 0.014                | 0.017                  | 0.16  |
| resimulation | y         | 0.017                | 0.026                  | 0.36  |
| random walk  | m         | 0.005                | 0.003                  | -0.08 |
| random walk  | y         | 0.012                | 0.008                  | -0.12 |
