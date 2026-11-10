# benchmark/gaussian-hmm

Gaussian hidden Markov model: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 4, rows 67, scalePercent 100), 10.7 s.

**Passed**: 2 method(s) agree with the exact truth; 2 reported

**Gaussian hidden Markov model** (gaussian-hmm): state-space, scan, markov, hmm, discrete-latent, time-series, exact, enumeration

Three discrete states with known transitions and overlapping Gaussian emissions, written as a scan: forward-backward is the exact truth, and the number of visits to a state tests the joint posterior.

Truth (exact): forward-backward for the marginals and the evidence, and a forward pass over (state, number of visits) for the exact distribution of the visits to state 2, written out in discrete-truth.ts (independent of the chain engine). Log evidence -29.0458.

Methods whose own `check` says they apply: gibbs, importance, particle-filter, enumeration, chain, abc-rejection. Not applicable: nuts (no continuous latent nodes); hmc (no continuous latent nodes); rwm (no continuous latent nodes); mala (no continuous latent nodes); barker (no continuous latent nodes); slice (no continuous latent nodes); elliptical-slice (no continuous latent nodes); sgld (no continuous latent nodes); sghmc (no continuous latent nodes); mclmc (no continuous latent nodes); aies (no continuous latent nodes); chees (no continuous latent nodes); parallel-tempering (no continuous latent nodes); ensemble-slice (no continuous latent nodes); svi (no continuous latent nodes); ep (y[0]: a gate selects among random values: a known component has no variable (give it a prior); z[1]: a value with event axes cannot stand in a scalar expression; y[1]: depends on 'z[1]', which could not be compiled; z[2]: a value with event axes cannot stand in a scalar expression; y[2]: depends on 'z[2]', which could not be compiled; z[3]: a value with event axes cannot stand in a scalar expression; y[3]: depends on 'z[3]', which could not be compiled; z[4]: a value with event axes cannot stand in a scalar expression; y[4]: depends on 'z[4]', which could not be compiled; z[5]: a value with event axes cannot stand in a scalar expression; y[5]: depends on 'z[5]', which could not be compiled; z[6]: a value with event axes cannot stand in a scalar expression; y[6]: depends on 'z[6]', which could not be compiled; z[7]: a value with event axes cannot stand in a scalar expression; y[7]: depends on 'z[7]', which could not be compiled; z[8]: a value with event axes cannot stand in a scalar expression; y[8]: depends on 'z[8]', which could not be compiled; z[9]: a value with event axes cannot stand in a scalar expression; y[9]: depends on 'z[9]', which could not be compiled; z[10]: a value with event axes cannot stand in a scalar expression; y[10]: depends on 'z[10]', which could not be compiled; z[11]: a value with event axes cannot stand in a scalar expression; y[11]: depends on 'z[11]', which could not be compiled; z[12]: a value with event axes cannot stand in a scalar expression; y[12]: depends on 'z[12]', which could not be compiled; z[13]: a value with event axes cannot stand in a scalar expression; y[13]: depends on 'z[13]', which could not be compiled; z[14]: a value with event axes cannot stand in a scalar expression; y[14]: depends on 'z[14]', which could not be compiled; z[15]: a value with event axes cannot stand in a scalar expression; y[15]: depends on 'z[15]', which could not be compiled); vmp (y[0]: a gate selects among random values: a known component has no variable (give it a prior); z[1]: a value with event axes cannot stand in a scalar expression; y[1]: depends on 'z[1]', which could not be compiled; z[2]: a value with event axes cannot stand in a scalar expression; y[2]: depends on 'z[2]', which could not be compiled; z[3]: a value with event axes cannot stand in a scalar expression; y[3]: depends on 'z[3]', which could not be compiled; z[4]: a value with event axes cannot stand in a scalar expression; y[4]: depends on 'z[4]', which could not be compiled; z[5]: a value with event axes cannot stand in a scalar expression; y[5]: depends on 'z[5]', which could not be compiled; z[6]: a value with event axes cannot stand in a scalar expression; y[6]: depends on 'z[6]', which could not be compiled; z[7]: a value with event axes cannot stand in a scalar expression; y[7]: depends on 'z[7]', which could not be compiled; z[8]: a value with event axes cannot stand in a scalar expression; y[8]: depends on 'z[8]', which could not be compiled; z[9]: a value with event axes cannot stand in a scalar expression; y[9]: depends on 'z[9]', which could not be compiled; z[10]: a value with event axes cannot stand in a scalar expression; y[10]: depends on 'z[10]', which could not be compiled; z[11]: a value with event axes cannot stand in a scalar expression; y[11]: depends on 'z[11]', which could not be compiled; z[12]: a value with event axes cannot stand in a scalar expression; y[12]: depends on 'z[12]', which could not be compiled; z[13]: a value with event axes cannot stand in a scalar expression; y[13]: depends on 'z[13]', which could not be compiled; z[14]: a value with event axes cannot stand in a scalar expression; y[14]: depends on 'z[14]', which could not be compiled; z[15]: a value with event axes cannot stand in a scalar expression; y[15]: depends on 'z[15]', which could not be compiled); bp (y[0]: a gate selects among random values: a known component has no variable (give it a prior); z[1]: a value with event axes cannot stand in a scalar expression; y[1]: depends on 'z[1]', which could not be compiled; z[2]: a value with event axes cannot stand in a scalar expression; y[2]: depends on 'z[2]', which could not be compiled; z[3]: a value with event axes cannot stand in a scalar expression; y[3]: depends on 'z[3]', which could not be compiled; z[4]: a value with event axes cannot stand in a scalar expression; y[4]: depends on 'z[4]', which could not be compiled; z[5]: a value with event axes cannot stand in a scalar expression; y[5]: depends on 'z[5]', which could not be compiled; z[6]: a value with event axes cannot stand in a scalar expression; y[6]: depends on 'z[6]', which could not be compiled; z[7]: a value with event axes cannot stand in a scalar expression; y[7]: depends on 'z[7]', which could not be compiled; z[8]: a value with event axes cannot stand in a scalar expression; y[8]: depends on 'z[8]', which could not be compiled; z[9]: a value with event axes cannot stand in a scalar expression; y[9]: depends on 'z[9]', which could not be compiled; z[10]: a value with event axes cannot stand in a scalar expression; y[10]: depends on 'z[10]', which could not be compiled; z[11]: a value with event axes cannot stand in a scalar expression; y[11]: depends on 'z[11]', which could not be compiled; z[12]: a value with event axes cannot stand in a scalar expression; y[12]: depends on 'z[12]', which could not be compiled; z[13]: a value with event axes cannot stand in a scalar expression; y[13]: depends on 'z[13]', which could not be compiled; z[14]: a value with event axes cannot stand in a scalar expression; y[14]: depends on 'z[14]', which could not be compiled; z[15]: a value with event axes cannot stand in a scalar expression; y[15]: depends on 'z[15]', which could not be compiled); laplace (no continuous latent nodes); em (no params to estimate); tempered-smc (no continuous latent nodes); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for z); gaussian (the scan 'z': the initial state is categorical, not Gaussian); svi-trace-graph (no continuous latent nodes); svi-trace-enum (no continuous latent nodes); svi-stl (no continuous latent nodes); svi-renyi (no continuous latent nodes); svi-structured (no continuous latent nodes); svi-mixture (no continuous latent nodes); svi-natural (no continuous latent nodes); svi-conjugate-natural (no usable conjugate structure for z); pathfinder (no continuous latent nodes).

| method          | expectation | run                   | diagnostics | why                                                                                                                                              |
| --------------- | ----------- | --------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| chain           | must agree  | converged (1 steps)   | -           | forward-backward and forward filtering backward sampling: exact marginals, and joint draws that keep the visit counts                            |
| enumeration     | must agree  | converged (1 steps)   | -           | a junction tree over the unrolled chain: exact marginals (its draws are independent per state, so the visit count is not compared)               |
| gibbs           | reported    | completed (800 steps) | -           | single-site Gibbs on a sticky chain: consistent, slow to move between runs of states                                                             |
| particle-filter | reported    | completed (16 steps)  | -           | a bootstrap filter over the categorical states: unbiased evidence, filtering marginals right, early smoothing marginals thinned by the genealogy |

| method          | quantity | mean   | sd    | mean error (sd) | mean z  | sd ratio | KS    | W1 (sd) | KL   | ESS  |
| --------------- | -------- | ------ | ----- | --------------- | ------- | -------- | ----- | ------- | ---- | ---- |
| chain           | z[0]     | 0.761  | 0.491 | 0.00851         | 0.548   | 0.982    | 0.698 | 0.023   | 4.9  | 4000 |
| chain           | z[1]     | 1.11   | 0.383 | 0.00697         | 0.44    | 1        | 0.843 | 0.00327 | 5.2  | 4000 |
| chain           | z[2]     | 1.06   | 0.311 | -0.00836        | -0.546  | 0.968    | 0.895 | 0.0218  | 5.23 | 4000 |
| chain           | z[3]     | 1.06   | 0.336 | -0.0109         | -0.685  | 1        | 0.882 | 0.00671 | 5.22 | 4000 |
| chain           | z[4]     | 1.27   | 0.457 | -0.0141         | -0.893  | 1        | 0.713 | 0.0115  | 5.18 | 4000 |
| chain           | z[5]     | 1.35   | 0.502 | -0.012          | -0.753  | 1.01     | 0.625 | 0.00907 | 5.17 | 4000 |
| chain           | z[6]     | 1.71   | 0.455 | 0.00394         | 0.249   | 1        | 0.712 | 0.0066  | 5.17 | 4000 |
| chain           | z[7]     | 1.34   | 0.598 | 0.0126          | 0.801   | 0.992    | 0.529 | 0.0108  | 4.86 | 4000 |
| chain           | z[8]     | 0.43   | 0.544 | -0.00533        | -0.339  | 0.996    | 0.593 | 0.0055  | 4.9  | 4000 |
| chain           | z[9]     | 0.0973 | 0.299 | 0.0251          | 1.53    | 1.04     | 0.91  | 0.0252  | 5.12 | 4000 |
| chain           | z[10]    | 0.0463 | 0.21  | 0.0171          | 1.04    | 1.04     | 0.957 | 0.0185  | 5.25 | 4000 |
| chain           | z[11]    | 1.35   | 0.555 | -0.00202        | -0.128  | 0.998    | 0.569 | 0.00494 | 5.17 | 4000 |
| chain           | z[12]    | 1.45   | 0.504 | -0.0178         | -1.13   | 0.996    | 0.537 | 0.0232  | 5.17 | 4000 |
| chain           | z[13]    | 1.3    | 0.5   | -0.0141         | -0.9    | 0.989    | 0.654 | 0.0198  | 5.17 | 4000 |
| chain           | z[14]    | 1.55   | 0.498 | -0.0288         | -1.83   | 0.997    | 0.546 | 0.0376  | 5.16 | 4000 |
| chain           | z[15]    | 1.56   | 0.51  | -0.025          | -1.57   | 1.01     | 0.568 | 0.0276  | 5.17 | 4000 |
| chain           | visits_2 | 4.38   | 2.45  | -0.0232         | -1.47   | 0.996    | 0.144 | 0.0253  | 2.73 | 4000 |
| enumeration     | z[0]     | 0.756  | 0.506 | -0.000481       | -0.0301 | 1.01     | 0.687 | 0.011   | 4.89 | 4000 |
| enumeration     | z[1]     | 1.1    | 0.391 | -0.0211         | -1.31   | 1.02     | 0.836 | 0.0248  | 5.2  | 4000 |
| enumeration     | z[2]     | 1.07   | 0.322 | 0.00642         | 0.405   | 1        | 0.893 | 0.00855 | 5.22 | 4000 |
| enumeration     | z[3]     | 1.07   | 0.33  | 0.00479         | 0.307   | 0.986    | 0.886 | 0.0119  | 5.22 | 4000 |
| enumeration     | z[4]     | 1.28   | 0.459 | 0.0138          | 0.867   | 1        | 0.715 | 0.0164  | 5.17 | 4000 |
| enumeration     | z[5]     | 1.37   | 0.5   | 0.0247          | 1.55    | 1.01     | 0.628 | 0.0277  | 5.17 | 4000 |
| enumeration     | z[6]     | 1.72   | 0.447 | 0.032           | 2.06    | 0.983    | 0.724 | 0.0302  | 5.18 | 4000 |
| enumeration     | z[7]     | 1.34   | 0.604 | 0.00218         | 0.138   | 1        | 0.525 | 0.0029  | 4.86 | 4000 |
| enumeration     | z[8]     | 0.445  | 0.554 | 0.0217          | 1.35    | 1.01     | 0.593 | 0.0224  | 4.9  | 4000 |
| enumeration     | z[9]     | 0.0873 | 0.283 | -0.00973        | -0.624  | 0.986    | 0.91  | 0.0113  | 5.17 | 4000 |
| enumeration     | z[10]    | 0.0388 | 0.193 | -0.02           | -1.32   | 0.953    | 0.957 | 0.0185  | 5.26 | 4000 |
| enumeration     | z[11]    | 1.35   | 0.544 | -0.00157        | -0.101  | 0.977    | 0.575 | 0.0269  | 5.17 | 4000 |
| enumeration     | z[12]    | 1.45   | 0.503 | -0.0104         | -0.663  | 0.995    | 0.538 | 0.0178  | 5.16 | 4000 |
| enumeration     | z[13]    | 1.3    | 0.5   | -0.00912        | -0.583  | 0.99     | 0.654 | 0.0158  | 5.17 | 4000 |
| enumeration     | z[14]    | 1.55   | 0.5   | -0.0303         | -1.91   | 1        | 0.547 | 0.035   | 5.16 | 4000 |
| enumeration     | z[15]    | 1.56   | 0.506 | -0.019          | -1.2    | 1        | 0.569 | 0.0217  | 5.17 | 4000 |
| gibbs           | z[0]     | 0.778  | 0.491 | 0.0425          | 0.91    | 0.98     | 0.711 | 0.046   | 3.15 | 441  |
| gibbs           | z[1]     | 1.08   | 0.394 | -0.063          | -0.89   | 1.03     | 0.828 | 0.0667  | 3.35 | 212  |
| gibbs           | z[2]     | 1.06   | 0.316 | 0.00253         | 0.0327  | 0.983    | 0.895 | 0.0109  | 3.37 | 161  |
| gibbs           | z[3]     | 1.08   | 0.359 | 0.06            | 0.532   | 1.07     | 0.884 | 0.0641  | 3.36 | 90.4 |
| gibbs           | z[4]     | 1.33   | 0.479 | 0.122           | 1.48    | 1.05     | 0.715 | 0.125   | 3.32 | 161  |
| gibbs           | z[5]     | 1.38   | 0.499 | 0.0605          | 0.826   | 1        | 0.631 | 0.0635  | 3.31 | 188  |
| gibbs           | z[6]     | 1.66   | 0.475 | -0.117          | -1.37   | 1.04     | 0.656 | 0.119   | 3.32 | 150  |
| gibbs           | z[7]     | 1.28   | 0.591 | -0.0895         | -1.76   | 0.981    | 0.522 | 0.0912  | 3.1  | 371  |
| gibbs           | z[8]     | 0.47   | 0.538 | 0.068           | 1.28    | 0.985    | 0.593 | 0.087   | 3.16 | 345  |
| gibbs           | z[9]     | 0.118  | 0.323 | 0.0974          | 1.34    | 1.12     | 0.91  | 0.0975  | 3.35 | 239  |
| gibbs           | z[10]    | 0.062  | 0.241 | 0.0948          | 1.42    | 1.19     | 0.957 | 0.0963  | 3.38 | 317  |
| gibbs           | z[11]    | 1.31   | 0.557 | -0.0671         | -0.785  | 1        | 0.561 | 0.0638  | 3.31 | 137  |
| gibbs           | z[12]    | 1.42   | 0.502 | -0.0707         | -0.749  | 0.992    | 0.536 | 0.0732  | 3.31 | 110  |
| gibbs           | z[13]    | 1.29   | 0.495 | -0.0309         | -0.346  | 0.98     | 0.655 | 0.0386  | 3.32 | 120  |
| gibbs           | z[14]    | 1.57   | 0.499 | 0.0148          | 0.182   | 1        | 0.57  | 0.016   | 3.31 | 151  |
| gibbs           | z[15]    | 1.6    | 0.499 | 0.0441          | 0.631   | 0.985    | 0.6   | 0.0415  | 3.31 | 198  |
| gibbs           | visits_2 | 4.37   | 2.43  | -0.03           | -0.68   | 0.986    | 0.165 | 0.0453  | 1.72 | 500  |
| particle-filter | z[0]     | 0.715  | 0.539 | -0.0844         | -3.51   | 1.08     | 0.638 | 0.123   | 4.3  | 2000 |
| particle-filter | z[1]     | 1.12   | 0.4   | 0.0442          | 1.89    | 1.05     | 0.842 | 0.0457  | 4.58 | 2000 |
| particle-filter | z[2]     | 1.07   | 0.279 | 0.0196          | 1.01    | 0.867    | 0.908 | 0.0778  | 4.61 | 2000 |
| particle-filter | z[3]     | 1.1    | 0.342 | 0.111           | 4.86    | 1.02     | 0.897 | 0.115   | 4.59 | 2000 |
| particle-filter | z[4]     | 1.26   | 0.459 | -0.0398         | -1.77   | 1        | 0.71  | 0.0372  | 4.56 | 2000 |
| particle-filter | z[5]     | 1.3    | 0.509 | -0.119          | -5.18   | 1.03     | 0.612 | 0.116   | 4.55 | 2000 |
| particle-filter | z[6]     | 1.73   | 0.445 | 0.0424          | 1.94    | 0.978    | 0.728 | 0.0407  | 4.55 | 2000 |
| particle-filter | z[7]     | 1.31   | 0.634 | -0.0331         | -1.41   | 1.05     | 0.502 | 0.0431  | 4.26 | 2000 |
| particle-filter | z[8]     | 0.44   | 0.558 | 0.013           | 0.568   | 1.02     | 0.593 | 0.0137  | 4.31 | 2000 |
| particle-filter | z[9]     | 0.044  | 0.205 | -0.16           | -10     | 0.714    | 0.91  | 0.16    | 4.63 | 2000 |
| particle-filter | z[10]    | 0.0265 | 0.161 | -0.0804         | -4.54   | 0.793    | 0.957 | 0.079   | 4.66 | 2000 |
| particle-filter | z[11]    | 1.41   | 0.567 | 0.111           | 4.86    | 1.02     | 0.57  | 0.114   | 4.54 | 2000 |
| particle-filter | z[12]    | 1.51   | 0.505 | 0.113           | 5.07    | 0.998    | 0.538 | 0.115   | 4.54 | 2000 |
| particle-filter | z[13]    | 1.32   | 0.522 | 0.0344          | 1.49    | 1.03     | 0.647 | 0.0564  | 4.55 | 2000 |
| particle-filter | z[14]    | 1.56   | 0.498 | -0.00424        | -0.19   | 0.996    | 0.559 | 0.011   | 4.54 | 2000 |
| particle-filter | z[15]    | 1.57   | 0.506 | -0.00423        | -0.189  | 0.999    | 0.577 | 0.00691 | 4.54 | 2000 |
| particle-filter | visits_2 | 4.59   | 2.62  | 0.0598          | 2.51    | 1.07     | 0.173 | 0.066   | 2.36 | 2000 |
