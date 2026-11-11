import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Poisson with a log link: parameters learned',
  summary:
    'The Poisson count with a log link of the basic atom, now a regression with an unknown intercept and slope, and the counts simulated from stated true coefficients.',
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
    'numeric likelihood: poisson with an exp link',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
