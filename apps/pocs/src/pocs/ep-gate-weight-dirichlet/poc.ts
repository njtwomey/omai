import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Gate with unknown mixing weights (Dirichlet prior, three components)',
  summary:
    'A mixture of three known narrow Gaussians with ten observations, a categorical switch per observation and a Dirichlet prior on the three weights: the posterior of each weight from EP, against the exact one on the simplex.',
  tags: ['expectation propagation', 'ep', 'atom', 'gate', 'mixture weight', 'prior', 'hyperprior'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
