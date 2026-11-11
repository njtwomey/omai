import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Gate between two random values',
  summary:
    'A Bernoulli switch z selects one of two Gaussian variables a and b, and y is the selected one plus noise. The exact marginal of each of a and b is a mixture of the updated and the prior Gaussian, with weights that are the posterior of z.',
  tags: ['expectation propagation', 'ep', 'atom', 'gates', 'prior', 'prior', 'select gate', 'gaussian likelihood'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
