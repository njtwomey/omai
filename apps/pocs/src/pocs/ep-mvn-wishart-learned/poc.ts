import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Multivariate normal with a Wishart precision: parameters learned',
  summary:
    'The bivariate normal of the basic atom with the mean also unknown (a Gaussian prior) and the data simulated from a stated mean and covariance: the Wishart is no longer conjugate (the mean and the precision are independent in the prior), so EP integrates the mean out and matches the matrix by quadrature.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'covariance',
    'parameters learned',
    'recovery',
    'calibration',
    'wishart',
    'precision matrix',
    'covariance',
    'multivariate normal',
    'conjugate',
    'mvn-wishart',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
