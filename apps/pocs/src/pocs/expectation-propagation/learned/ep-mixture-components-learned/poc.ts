import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Random index into random components: parameters learned',
  summary:
    'The index gate mu[z] of the basic atom with two or three components whose means (ordered), widths (gamma precisions) and weights (a Dirichlet) are all unknown, observed through data simulated from a stated truth. EP recovers the components when they are apart and, honestly, reads overlapping ones as one: the separation knob and the calibration table show where; the reference is a Gibbs sampler of the same model.',
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
    'index gate',
    'ordered',
    'dirichlet',
    'gaussian-gamma-precision',
  ],
  added: '2026-11-11',
}
