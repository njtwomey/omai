import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'EP atom: seeds germinating in batches',
  summary:
    "A beta prior on a germination rate and a binomial count per batch of seeds: conjugate. EP against the exact beta posterior with each batch's share drawn.",
  tags: ['expectation propagation', 'ep', 'atom', 'binomial', 'beta', 'conjugate', 'batches'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
