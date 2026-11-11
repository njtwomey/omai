import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Multivariate normal with a Wishart precision',
  summary:
    'Six pairs from a bivariate normal with a known mean, whose precision matrix is a Wishart latent: the conjugate update of the matrix, exact in one sweep, checked against the closed-form posterior.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'covariance',
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
