import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Inlier and outlier mixture, switches summed out',
  summary:
    'An unknown mean mu observed eight times through a mixture of a narrow and a wide Gaussian around it, two observations being far from the rest: the mixture site as a likelihood. The switch per observation is summed out, so EP gives a belief about mu and no probability that an observation is an outlier.',
  tags: ['expectation propagation', 'ep', 'atom', 'gates', 'prior', 'mixture-site'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
