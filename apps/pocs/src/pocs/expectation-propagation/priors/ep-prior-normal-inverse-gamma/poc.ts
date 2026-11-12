import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Prior: normal-inverse-gamma on a Gaussian mean and variance',
  summary:
    'An inverse-gamma prior on the variance of a Gaussian and, given it, a normal prior on its mean with variance sigma2 / kappa: the conjugate pair, one normal-gamma variable and a factor per observation.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
}
