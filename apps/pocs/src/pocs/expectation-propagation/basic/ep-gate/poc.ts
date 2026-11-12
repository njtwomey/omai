import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Gate',
  summary:
    'A Bernoulli switch z selects between the known values 0 and 3, and y is that value plus Gaussian noise: the select factor, whose message to z is the ratio of the two branches\u2019 evidence.',
  tags: ['expectation propagation', 'ep', 'atom', 'gates', 'prior', 'select gate', 'gaussian likelihood'],
  added: '2026-11-11',
}
