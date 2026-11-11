import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Spike and slab',
  summary:
    'A coefficient that is a gate between a very narrow Gaussian (the spike) and a wide one (the slab), chosen by a Bernoulli switch z, observed through noise. EP reports P(slab) and each branch; the exact posterior of the coefficient is the two branches weighted by it.',
  tags: ['expectation propagation', 'ep', 'atom', 'gates', 'prior', 'prior', 'select gate', 'gaussian likelihood'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
