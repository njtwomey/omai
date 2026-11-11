import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Correlated random intercepts and slopes',
  summary:
    'A line per group, its intercept and slope drawn together from a bivariate normal with a Wishart precision matrix and a known mean, observed with known noise: a Gaussian vector per group read through its coordinates, the matrix learned from the groups, checked against quadrature of the matrix with the effects integrated out.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'covariance',
    'random effects',
    'random intercepts',
    'random slopes',
    'correlated effects',
    'hierarchical',
    'wishart',
    'mvn-wishart',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
