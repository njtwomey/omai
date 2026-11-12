import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Prior: normal on a Gaussian mean, normal hyperprior on its centre',
  summary:
    'A normal prior on a Gaussian mean whose centre is itself normal: a chain of three Gaussians, jointly Gaussian, so the posterior of the mean and of the centre are closed forms.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
}
