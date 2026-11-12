import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Bayesian linear regression',
  summary:
    'Fit a polynomial or a sparse linear model to noisy data from a known truth: choose the data, the degree, the prior (normal, spike and slab, ARD or horseshoe) and whether the noise is known, learned (sd) or learned as a gamma precision, infer the posterior with any method that applies, see the fitted curve with its band and sampled lines the pair plot of the coefficients and which coefficients the sparse priors keep, and check it against the exact posterior or a long NUTS run.',
  tags: [
    'linear regression',
    'polynomial',
    'posterior',
    'predictive band',
    'credible interval',
    'prior',
    'noise',
    'correlated coefficients',
    'conjugate',
    'nuts',
    'exact posterior',
    'sparse',
    'spike and slab',
    'ard',
    'horseshoe',
    'variable selection',
    'expectation propagation',
  ],
  added: '2026-11-11',
}
