import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Bayesian additive model',
  summary:
    'A one-dimensional smooth fitted as a Bayesian GAM: a spline whose wiggliness penalty is the precision of its prior, with the smoothing and the noise inferred too. Choose the data, the spline and one inference method; the credible band widens where data are sparse, and the exact posterior at the end says how well the method did.',
  tags: [
    'gam',
    'additive model',
    'spline',
    'p-spline',
    'smoothing parameter',
    'penalty',
    'credible band',
    'hmc',
    'nuts',
    'expectation propagation',
    'ep',
    'svi',
  ],
  added: '2026-11-11',
  group: 'bayesian-examples',
}
