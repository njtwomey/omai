import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Gate between two random values: parameters learned',
  summary:
    'The gate between two random values of the basic atom with both means, both widths and the weight of the switch unknown, observed through data simulated from a stated truth: the switch picks the centre and precision of a component together. EP recovers the parameters and the component of each observation, against a Gibbs sampler of the same model; the separation knob shows where the recovery degrades.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'gates',
    'parameters learned',
    'recovery',
    'calibration',
    'prior',
    'mixture',
    'gate',
    'gate weight',
    'gaussian-gamma-precision',
    'bernoulli over a beta',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
