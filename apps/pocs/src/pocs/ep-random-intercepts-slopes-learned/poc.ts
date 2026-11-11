import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Correlated random intercepts and slopes: parameters learned',
  summary:
    'The random intercepts and slopes of the basic atom with the population mean also unknown (a Gaussian prior) and the groups simulated from a stated mean, covariance and noise: the Wishart matrix, the mean and every group are learned together.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'covariance',
    'parameters learned',
    'recovery',
    'calibration',
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
