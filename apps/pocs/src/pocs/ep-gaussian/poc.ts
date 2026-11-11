import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Gaussian',
  summary:
    'A Gaussian prior on a mean and three Gaussian observations: a prior factor and a linear-Gaussian likelihood factor, closed-form posterior.',
  tags: ['expectation propagation', 'ep', 'atom', 'conjugate', 'prior', 'gaussian likelihood'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
