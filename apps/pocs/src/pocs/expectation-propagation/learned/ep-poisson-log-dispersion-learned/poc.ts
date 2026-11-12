import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Poisson with a log link: parameters learned (overdispersed)',
  summary:
    'The Poisson regression with a log link whose counts are overdispersed: a negative binomial with an unknown dispersion alongside the unknown intercept and slope, simulated from stated true values.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'likelihoods',
    'parameters learned',
    'recovery',
    'calibration',
    'prior',
    'numeric prior',
    'negative-binomial-dispersion',
  ],
  added: '2026-11-11',
}
