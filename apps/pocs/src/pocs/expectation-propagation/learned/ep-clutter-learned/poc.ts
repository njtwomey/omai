import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Clutter: parameters learned',
  summary:
    'The clutter problem of the basic atom with the centre, the width of the inliers, the width of the clutter and the share of clutter all unknown (priors that keep the clutter wider than the inliers), observed through data simulated from a stated truth: a gate per observation whose branches each have a gamma precision. EP recovers the four parameters and says which observations are clutter; the exact posterior by quadrature and 12 to 40 seeds of calibration show how closely.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'gates',
    'parameters learned',
    'recovery',
    'calibration',
    'prior',
    'clutter',
    'outliers',
    'gate',
    'gaussian-gamma-precision',
    'bernoulli over a beta',
  ],
  added: '2026-11-11',
}
