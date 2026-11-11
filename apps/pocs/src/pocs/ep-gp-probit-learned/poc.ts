import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Gaussian process with a probit likelihood: parameters learned',
  summary:
    'The probit Gaussian process classifier of the basic atom with the length scale of its kernel unknown (a log-normal prior, the covariance a custom function of it) and the labels simulated from a function drawn at a stated length scale: EP runs inside a grid over the length scale, each node weighted by its EP evidence.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'kernels',
    'gaussian process',
    'probit link',
    'hyperparameter',
    'grid',
    'evidence',
    'parameters learned',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
