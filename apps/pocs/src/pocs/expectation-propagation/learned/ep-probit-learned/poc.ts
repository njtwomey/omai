import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Probit: parameters learned',
  summary:
    'The probit classifier of the basic atom with the prior precision of the weights learned (a gamma precision shared by the three weights) and the labels simulated from stated true weights: the weights and the decision boundary are recovered from the data.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'likelihoods',
    'parameters learned',
    'recovery',
    'calibration',
    'prior',
    'gaussian-gamma-precision',
    'matmul',
  ],
  added: '2026-11-11',
}
