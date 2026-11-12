import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Gate between two random values',
  summary:
    'A Bernoulli switch z selects one of two Gaussian variables a and b, and y is the selected one plus noise. EP reports the probability of z; the exact posterior of what is selected is the mixture of the two updated components, weighted by it.',
  tags: ['expectation propagation', 'ep', 'atom', 'gates', 'prior', 'prior', 'select gate', 'gaussian likelihood'],
  added: '2026-11-11',
}
