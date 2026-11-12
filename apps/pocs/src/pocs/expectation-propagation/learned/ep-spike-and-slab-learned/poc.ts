import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Spike and slab: parameters learned',
  summary:
    'The spike and slab gate of the basic atom on a sequence of coefficients, with the inclusion probability (a beta) and the precision of the slab (a gamma) unknown, observed through data simulated from a stated set of active coefficients: EP recovers which are active, how common they are and how wide the slab is, against the exact posterior of the share and the slab precision.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'gates',
    'parameters learned',
    'recovery',
    'calibration',
    'prior',
    'spike and slab',
    'variable selection',
    'gate',
    'gaussian-gamma-precision',
    'bernoulli over a beta',
  ],
  added: '2026-11-11',
}
