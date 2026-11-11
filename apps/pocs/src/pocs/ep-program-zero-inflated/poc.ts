import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Program: counts with excess zeros',
  summary:
    'A zero-inflated Poisson regression written as a probabilistic program: each count is a structural zero or a Poisson count with a log-linear rate on a covariate, the inflation probability has a Beta prior and the coefficients Gaussian priors. Run by EP, it recovers the inflation and the rate coefficients and says which zeros are structural, against the exact posterior with the structural zeros summed out.',
  tags: [
    'expectation propagation',
    'ep',
    'program',
    'zero inflation',
    'poisson',
    'gates',
    'counts',
    'beta prior',
    'dirichlet',
    'truth recovery',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
