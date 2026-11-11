import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Bivariate normal with an unknown correlation: parameters learned',
  summary:
    'The bivariate normal of the basic atom with the mean and both standard deviations also unknown (a Gaussian prior, gamma precisions) and the data simulated from a stated mean, sds and correlation: three numeric coordinates, matched at once by EP.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'covariance',
    'parameters learned',
    'recovery',
    'calibration',
    'correlation',
    'lkj',
    'bivariate normal',
    'beta',
    'mvn-sd-corr',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
