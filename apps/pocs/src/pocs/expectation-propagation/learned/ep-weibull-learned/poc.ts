import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Prior: normal on a Weibull log scale, normal hyperprior on its centre: parameters learned',
  summary:
    'The Weibull time with a normal prior on the log of its scale, as a regression: the intercept and slope of the log scale are unknown (the centre and the effect of a covariate) and the times are simulated from stated true values, with the shape fixed.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'likelihoods',
    'parameters learned',
    'recovery',
    'calibration',
    'prior',
    'matmul',
    'numeric likelihood: weibull with a log link',
  ],
  added: '2026-11-11',
}
