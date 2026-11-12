import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Prior: inverse-gamma on a Gaussian variance, gamma hyperprior on its scale',
  summary:
    'An inverse-gamma prior on a variance whose scale is gamma distributed, six Gaussian observations around a known mean: the variance is a gamma precision with a random rate (the gamma-rate hierarchy).',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
}
