import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Beta and Bernoulli: parameters learned',
  summary:
    'The beta and Bernoulli of the basic atom for a population of groups: each group has its own probability drawn from a symmetric beta whose strength is unknown, so the groups teach each other how spread out the probabilities are.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'conjugate',
    'parameters learned',
    'recovery',
    'calibration',
    'prior',
    'dirichlet-concentration',
    'bernoulli likelihood',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
