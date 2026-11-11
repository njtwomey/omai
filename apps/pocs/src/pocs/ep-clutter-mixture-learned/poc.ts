import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Inlier and outlier mixture, switches summed out: parameters learned',
  summary:
    "The inlier and outlier mixture likelihood of the basic atom with the centre, the width of the inliers and the share of outliers all unknown, observed through data simulated from a stated truth. EP does not run on this form yet (the mixture site takes a known width and share): the page shows the library's reason, the exact posterior EP will be held to, and points to the explicit-switch twin, which runs.",
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'gates',
    'parameters learned',
    'recovery',
    'prior',
    'clutter',
    'outliers',
    'mixture site',
    'planned',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
