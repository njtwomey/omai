import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Bivariate normal with an unknown correlation',
  summary:
    'Eight pairs from a bivariate normal with known zero means and unit sds, whose correlation has the LKJ prior of a 2 by 2 matrix (a beta variable on (rho + 1) / 2): a one-dimensional numeric site, checked against the posterior of the correlation by quadrature.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'covariance',
    'correlation',
    'lkj',
    'bivariate normal',
    'beta',
    'mvn-sd-corr',
  ],
  added: '2026-11-11',
}
