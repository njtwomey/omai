# benchmark/gaussian-hmm

Gaussian hidden Markov model: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit aceb09a9; seed 0, budget scale 1 (plans 4, rows 67, scalePercent 100), 12.2 s.

**Passed**: 2 method(s) agree with the exact truth; 2 reported

**Gaussian hidden Markov model** (gaussian-hmm): state-space, scan, markov, hmm, discrete-latent, time-series, exact, enumeration

Three discrete states with known transitions and overlapping Gaussian emissions, written as a scan: forward-backward is the exact truth, and the number of visits to a state tests the joint posterior.

Truth (exact): forward-backward for the marginals and the evidence, and a forward pass over (state, number of visits) for the exact distribution of the visits to state 2, written out in discrete-truth.ts (independent of the chain engine). Log evidence -29.0458.

Methods whose own `check` says they apply: gibbs, importance, particle-filter, enumeration, chain, abc-rejection. Not applicable: nuts (no continuous latent nodes); svi (no continuous latent nodes); ep (y[0]: a gate selects among random values: a known component has no variable (give it a prior); z[1]: a value with event axes cannot stand in a scalar expression; y[1]: depends on 'z[1]', which could not be compiled; z[2]: a value with event axes cannot stand in a scalar expression; y[2]: depends on 'z[2]', which could not be compiled; z[3]: a value with event axes cannot stand in a scalar expression; y[3]: depends on 'z[3]', which could not be compiled; z[4]: a value with event axes cannot stand in a scalar expression; y[4]: depends on 'z[4]', which could not be compiled; z[5]: a value with event axes cannot stand in a scalar expression; y[5]: depends on 'z[5]', which could not be compiled; z[6]: a value with event axes cannot stand in a scalar expression; y[6]: depends on 'z[6]', which could not be compiled; z[7]: a value with event axes cannot stand in a scalar expression; y[7]: depends on 'z[7]', which could not be compiled; z[8]: a value with event axes cannot stand in a scalar expression; y[8]: depends on 'z[8]', which could not be compiled; z[9]: a value with event axes cannot stand in a scalar expression; y[9]: depends on 'z[9]', which could not be compiled; z[10]: a value with event axes cannot stand in a scalar expression; y[10]: depends on 'z[10]', which could not be compiled; z[11]: a value with event axes cannot stand in a scalar expression; y[11]: depends on 'z[11]', which could not be compiled; z[12]: a value with event axes cannot stand in a scalar expression; y[12]: depends on 'z[12]', which could not be compiled; z[13]: a value with event axes cannot stand in a scalar expression; y[13]: depends on 'z[13]', which could not be compiled; z[14]: a value with event axes cannot stand in a scalar expression; y[14]: depends on 'z[14]', which could not be compiled; z[15]: a value with event axes cannot stand in a scalar expression; y[15]: depends on 'z[15]', which could not be compiled); vmp (y[0]: a gate selects among random values: a known component has no variable (give it a prior); z[1]: a value with event axes cannot stand in a scalar expression; y[1]: depends on 'z[1]', which could not be compiled; z[2]: a value with event axes cannot stand in a scalar expression; y[2]: depends on 'z[2]', which could not be compiled; z[3]: a value with event axes cannot stand in a scalar expression; y[3]: depends on 'z[3]', which could not be compiled; z[4]: a value with event axes cannot stand in a scalar expression; y[4]: depends on 'z[4]', which could not be compiled; z[5]: a value with event axes cannot stand in a scalar expression; y[5]: depends on 'z[5]', which could not be compiled; z[6]: a value with event axes cannot stand in a scalar expression; y[6]: depends on 'z[6]', which could not be compiled; z[7]: a value with event axes cannot stand in a scalar expression; y[7]: depends on 'z[7]', which could not be compiled; z[8]: a value with event axes cannot stand in a scalar expression; y[8]: depends on 'z[8]', which could not be compiled; z[9]: a value with event axes cannot stand in a scalar expression; y[9]: depends on 'z[9]', which could not be compiled; z[10]: a value with event axes cannot stand in a scalar expression; y[10]: depends on 'z[10]', which could not be compiled; z[11]: a value with event axes cannot stand in a scalar expression; y[11]: depends on 'z[11]', which could not be compiled; z[12]: a value with event axes cannot stand in a scalar expression; y[12]: depends on 'z[12]', which could not be compiled; z[13]: a value with event axes cannot stand in a scalar expression; y[13]: depends on 'z[13]', which could not be compiled; z[14]: a value with event axes cannot stand in a scalar expression; y[14]: depends on 'z[14]', which could not be compiled; z[15]: a value with event axes cannot stand in a scalar expression; y[15]: depends on 'z[15]', which could not be compiled); bp (y[0]: a gate selects among random values: a known component has no variable (give it a prior); z[1]: a value with event axes cannot stand in a scalar expression; y[1]: depends on 'z[1]', which could not be compiled; z[2]: a value with event axes cannot stand in a scalar expression; y[2]: depends on 'z[2]', which could not be compiled; z[3]: a value with event axes cannot stand in a scalar expression; y[3]: depends on 'z[3]', which could not be compiled; z[4]: a value with event axes cannot stand in a scalar expression; y[4]: depends on 'z[4]', which could not be compiled; z[5]: a value with event axes cannot stand in a scalar expression; y[5]: depends on 'z[5]', which could not be compiled; z[6]: a value with event axes cannot stand in a scalar expression; y[6]: depends on 'z[6]', which could not be compiled; z[7]: a value with event axes cannot stand in a scalar expression; y[7]: depends on 'z[7]', which could not be compiled; z[8]: a value with event axes cannot stand in a scalar expression; y[8]: depends on 'z[8]', which could not be compiled; z[9]: a value with event axes cannot stand in a scalar expression; y[9]: depends on 'z[9]', which could not be compiled; z[10]: a value with event axes cannot stand in a scalar expression; y[10]: depends on 'z[10]', which could not be compiled; z[11]: a value with event axes cannot stand in a scalar expression; y[11]: depends on 'z[11]', which could not be compiled; z[12]: a value with event axes cannot stand in a scalar expression; y[12]: depends on 'z[12]', which could not be compiled; z[13]: a value with event axes cannot stand in a scalar expression; y[13]: depends on 'z[13]', which could not be compiled; z[14]: a value with event axes cannot stand in a scalar expression; y[14]: depends on 'z[14]', which could not be compiled; z[15]: a value with event axes cannot stand in a scalar expression; y[15]: depends on 'z[15]', which could not be compiled); laplace (no continuous latent nodes); em (no params to estimate); tempered-smc (no continuous latent nodes); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for z); gaussian (the scan 'z': the initial state is categorical, not Gaussian).

| method          | expectation | run                   | diagnostics | why                                                                                                                                              |
| --------------- | ----------- | --------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| chain           | must agree  | converged (1 steps)   | -           | forward-backward and forward filtering backward sampling: exact marginals, and joint draws that keep the visit counts                            |
| enumeration     | must agree  | converged (1 steps)   | -           | a junction tree over the unrolled chain: exact marginals (its draws are independent per state, so the visit count is not compared)               |
| gibbs           | reported    | completed (800 steps) | -           | single-site Gibbs on a sticky chain: consistent, slow to move between runs of states                                                             |
| particle-filter | reported    | completed (16 steps)  | -           | a bootstrap filter over the categorical states: unbiased evidence, filtering marginals right, early smoothing marginals thinned by the genealogy |

| method          | quantity | mean   | sd    | mean error (sd) | mean z  | sd ratio | KS    | W1 (sd) | KL   | ESS  |
| --------------- | -------- | ------ | ----- | --------------- | ------- | -------- | ----- | ------- | ---- | ---- |
| chain           | z[0]     | 0.767  | 0.505 | 0.0215          | 1.35    | 1.01     | 0.696 | 0.025   | 4.89 | 4000 |
| chain           | z[1]     | 1.1    | 0.376 | -0.0133         | -0.855  | 0.983    | 0.843 | 0.017   | 5.21 | 4000 |
| chain           | z[2]     | 1.06   | 0.324 | 0.000975        | 0.0612  | 1.01     | 0.892 | 0.00467 | 5.22 | 4000 |
| chain           | z[3]     | 1.05   | 0.326 | -0.0392         | -2.55   | 0.972    | 0.881 | 0.0351  | 5.23 | 4000 |
| chain           | z[4]     | 1.27   | 0.457 | -0.0202         | -1.27   | 1        | 0.713 | 0.0175  | 5.18 | 4000 |
| chain           | z[5]     | 1.34   | 0.493 | -0.0216         | -1.37   | 0.994    | 0.628 | 0.0227  | 5.17 | 4000 |
| chain           | z[6]     | 1.7    | 0.458 | -0.0186         | -1.17   | 1.01     | 0.701 | 0.0203  | 5.17 | 4000 |
| chain           | z[7]     | 1.33   | 0.601 | 0.00135         | 0.086   | 0.996    | 0.527 | 0.00373 | 4.86 | 4000 |
| chain           | z[8]     | 0.43   | 0.545 | -0.00442        | -0.28   | 0.999    | 0.593 | 0.00733 | 4.9  | 4000 |
| chain           | z[9]     | 0.082  | 0.275 | -0.028          | -1.85   | 0.959    | 0.91  | 0.0296  | 5.17 | 4000 |
| chain           | z[10]    | 0.0398 | 0.195 | -0.015          | -0.985  | 0.965    | 0.957 | 0.0136  | 5.26 | 4000 |
| chain           | z[11]    | 1.36   | 0.55  | 0.0231          | 1.48    | 0.988    | 0.574 | 0.0265  | 5.17 | 4000 |
| chain           | z[12]    | 1.44   | 0.504 | -0.0213         | -1.35   | 0.997    | 0.537 | 0.0257  | 5.17 | 4000 |
| chain           | z[13]    | 1.3    | 0.509 | -0.00813        | -0.511  | 1.01     | 0.65  | 0.00594 | 5.17 | 4000 |
| chain           | z[14]    | 1.57   | 0.496 | 0.0138          | 0.877   | 0.994    | 0.568 | 0.015   | 5.17 | 4000 |
| chain           | z[15]    | 1.58   | 0.504 | 0.0214          | 1.36    | 0.996    | 0.59  | 0.0217  | 5.17 | 4000 |
| chain           | visits_2 | 4.42   | 2.43  | -0.0106         | -0.68   | 0.985    | 0.154 | 0.0196  | 2.73 | 4000 |
| enumeration     | z[0]     | 0.762  | 0.504 | 0.0115          | 0.723   | 1.01     | 0.693 | 0.015   | 4.89 | 4000 |
| enumeration     | z[1]     | 1.11   | 0.38  | 0.00893         | 0.568   | 0.994    | 0.845 | 0.00523 | 5.2  | 4000 |
| enumeration     | z[2]     | 1.06   | 0.324 | 0.000197        | 0.0124  | 1.01     | 0.892 | 0.00544 | 5.22 | 4000 |
| enumeration     | z[3]     | 1.07   | 0.341 | 0.00404         | 0.252   | 1.02     | 0.882 | 0.0082  | 5.22 | 4000 |
| enumeration     | z[4]     | 1.27   | 0.46  | -0.00592        | -0.372  | 1.01     | 0.713 | 0.00328 | 5.18 | 4000 |
| enumeration     | z[5]     | 1.36   | 0.501 | 0.0126          | 0.792   | 1.01     | 0.627 | 0.0166  | 5.17 | 4000 |
| enumeration     | z[6]     | 1.71   | 0.452 | 0.0122          | 0.777   | 0.993    | 0.715 | 0.0104  | 5.17 | 4000 |
| enumeration     | z[7]     | 1.31   | 0.614 | -0.0331         | -2.05   | 1.02     | 0.515 | 0.0348  | 4.85 | 4000 |
| enumeration     | z[8]     | 0.428  | 0.534 | -0.00854        | -0.552  | 0.979    | 0.593 | 0.00962 | 4.91 | 4000 |
| enumeration     | z[9]     | 0.093  | 0.291 | 0.0103          | 0.642   | 1.01     | 0.91  | 0.0104  | 5.16 | 4000 |
| enumeration     | z[10]    | 0.0488 | 0.215 | 0.0294          | 1.75    | 1.06     | 0.957 | 0.0309  | 5.25 | 4000 |
| enumeration     | z[11]    | 1.35   | 0.559 | -0.00291        | -0.184  | 1        | 0.567 | 0.00135 | 5.17 | 4000 |
| enumeration     | z[12]    | 1.45   | 0.506 | -0.0035         | -0.221  | 1        | 0.536 | 0.00593 | 5.16 | 4000 |
| enumeration     | z[13]    | 1.31   | 0.503 | 0.0221          | 1.4     | 0.996    | 0.656 | 0.0243  | 5.17 | 4000 |
| enumeration     | z[14]    | 1.55   | 0.502 | -0.0118         | -0.74   | 1        | 0.557 | 0.0115  | 5.17 | 4000 |
| enumeration     | z[15]    | 1.57   | 0.511 | -0.0166         | -1.04   | 1.01     | 0.573 | 0.0193  | 5.17 | 4000 |
| gibbs           | z[0]     | 0.756  | 0.498 | -0.00148        | -0.0316 | 0.996    | 0.691 | 0.004   | 3.15 | 453  |
| gibbs           | z[1]     | 1.06   | 0.372 | -0.126          | -1.77   | 0.973    | 0.826 | 0.129   | 3.36 | 188  |
| gibbs           | z[2]     | 1.04   | 0.301 | -0.0721         | -0.971  | 0.935    | 0.889 | 0.07    | 3.38 | 159  |
| gibbs           | z[3]     | 1.05   | 0.346 | -0.0534         | -0.51   | 1.03     | 0.872 | 0.0492  | 3.37 | 97.2 |
| gibbs           | z[4]     | 1.24   | 0.434 | -0.088          | -0.87   | 0.95     | 0.715 | 0.0897  | 3.33 | 88.1 |
| gibbs           | z[5]     | 1.32   | 0.465 | -0.0765         | -0.883  | 0.937    | 0.637 | 0.114   | 3.32 | 117  |
| gibbs           | z[6]     | 1.69   | 0.463 | -0.0466         | -0.677  | 1.02     | 0.688 | 0.0484  | 3.32 | 219  |
| gibbs           | z[7]     | 1.33   | 0.605 | -0.00321        | -0.0618 | 1        | 0.524 | 0.00498 | 3.1  | 373  |
| gibbs           | z[8]     | 0.488  | 0.553 | 0.101           | 1.81    | 1.01     | 0.593 | 0.102   | 3.15 | 332  |
| gibbs           | z[9]     | 0.126  | 0.332 | 0.125           | 1.32    | 1.16     | 0.91  | 0.125   | 3.35 | 148  |
| gibbs           | z[10]    | 0.05   | 0.218 | 0.0356          | 0.3     | 1.08     | 0.957 | 0.037   | 3.39 | 82.2 |
| gibbs           | z[11]    | 1.36   | 0.549 | 0.0155          | 0.216   | 0.987    | 0.573 | 0.0189  | 3.31 | 190  |
| gibbs           | z[12]    | 1.45   | 0.497 | -0.0114         | -0.129  | 0.984    | 0.54  | 0.0297  | 3.31 | 123  |
| gibbs           | z[13]    | 1.29   | 0.517 | -0.019          | -0.171  | 1.02     | 0.645 | 0.0168  | 3.32 | 84.9 |
| gibbs           | z[14]    | 1.54   | 0.498 | -0.0413         | -0.4    | 0.998    | 0.54  | 0.0501  | 3.31 | 93.4 |
| gibbs           | z[15]    | 1.58   | 0.498 | 0.00466         | 0.0554  | 0.984    | 0.578 | 0.00987 | 3.31 | 137  |
| gibbs           | visits_2 | 4.24   | 2.37  | -0.0804         | -1.87   | 0.964    | 0.129 | 0.0831  | 1.73 | 500  |
| particle-filter | z[0]     | 0.657  | 0.509 | -0.2            | -8.8    | 1.02     | 0.606 | 0.197   | 4.34 | 2000 |
| particle-filter | z[1]     | 1.12   | 0.381 | 0.0481          | 2.16    | 0.995    | 0.85  | 0.0444  | 4.58 | 2000 |
| particle-filter | z[2]     | 1.09   | 0.316 | 0.0787          | 3.59    | 0.982    | 0.905 | 0.0809  | 4.6  | 2000 |
| particle-filter | z[3]     | 1.1    | 0.326 | 0.102           | 4.68    | 0.973    | 0.901 | 0.106   | 4.59 | 2000 |
| particle-filter | z[4]     | 1.25   | 0.446 | -0.053          | -2.43   | 0.975    | 0.714 | 0.0503  | 4.56 | 2000 |
| particle-filter | z[5]     | 1.39   | 0.487 | 0.0686          | 3.12    | 0.982    | 0.637 | 0.0715  | 4.55 | 2000 |
| particle-filter | z[6]     | 1.72   | 0.448 | 0.0281          | 1.28    | 0.985    | 0.722 | 0.0264  | 4.55 | 2000 |
| particle-filter | z[7]     | 1.31   | 0.552 | -0.0347         | -1.7    | 0.916    | 0.551 | 0.119   | 4.29 | 2000 |
| particle-filter | z[8]     | 0.527  | 0.568 | 0.172           | 7.41    | 1.04     | 0.593 | 0.173   | 4.3  | 2000 |
| particle-filter | z[9]     | 0.094  | 0.295 | 0.0138          | 0.599   | 1.03     | 0.91  | 0.0139  | 4.52 | 2000 |
| particle-filter | z[10]    | 0.0455 | 0.208 | 0.0134          | 0.581   | 1.03     | 0.957 | 0.0148  | 4.63 | 2000 |
| particle-filter | z[11]    | 1.36   | 0.548 | 0.0236          | 1.07    | 0.984    | 0.575 | 0.0269  | 4.55 | 2000 |
| particle-filter | z[12]    | 1.46   | 0.506 | 0.00243         | 0.109   | 1        | 0.536 | 0.00395 | 4.54 | 2000 |
| particle-filter | z[13]    | 1.32   | 0.504 | 0.0414          | 1.85    | 0.998    | 0.657 | 0.0436  | 4.55 | 2000 |
| particle-filter | z[14]    | 1.56   | 0.499 | 0.00477         | 0.213   | 0.999    | 0.564 | 0.00601 | 4.54 | 2000 |
| particle-filter | z[15]    | 1.59   | 0.509 | 0.0224          | 0.999   | 1        | 0.593 | 0.0316  | 4.54 | 2000 |
| particle-filter | visits_2 | 4.48   | 2.41  | 0.0173          | 0.794   | 0.978    | 0.171 | 0.0406  | 2.4  | 2000 |
