import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Half-Cauchy scale',
  summary:
    'A half-Cauchy prior on a standard deviation sigma and four zero-mean Gaussian observations with that scale: a heavy-tailed scale prior that has no message family of its own.',
  tags: ['expectation propagation', 'ep', 'atom', 'scales', 'numeric prior', 'gaussian with a random scale'],
  added: '2026-11-11',
}
