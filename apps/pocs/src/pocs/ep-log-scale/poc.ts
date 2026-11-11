import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Gaussian with an exp(h) scale',
  summary:
    'A latent h and one observation y ~ N(0, exp(h/2)\u00b2): the observation says how large the variance is, and the likelihood of h is not Gaussian (the stochastic volatility step).',
  tags: ['expectation propagation', 'ep', 'atom', 'functions', 'prior', 'gaussian-log-scale'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
