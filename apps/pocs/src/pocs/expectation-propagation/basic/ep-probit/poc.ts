import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Probit',
  summary:
    'Three weights (an offset and two slopes) with Gaussian priors and eight binary labels through a probit link: the probit factor, a truncated Gaussian with closed-form moments, on a linear predictor.',
  tags: ['expectation propagation', 'ep', 'atom', 'likelihoods', 'prior', 'matmul', 'probit link'],
  added: '2026-11-11',
}
