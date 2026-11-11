import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Ordered means',
  summary:
    'Two means with an ordered-vector constraint mu_1 < mu_2 and data that prefer the opposite order: the constraint is a greater-than factor between neighbours, and the posterior is cut along the diagonal.',
  tags: ['expectation propagation', 'ep', 'atom', 'gates', 'prior', 'ordered constraint', 'gaussian likelihood'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
