import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Gate with an unknown mixing weight (beta prior)',
  summary:
    'A mixture of two known narrow Gaussians with ten observations, each with its own switch z_i ~ Bernoulli(pi) and a beta prior on pi: the posterior of the weight and each switch from EP, against the switches summed out exactly.',
  tags: ['expectation propagation', 'ep', 'atom', 'gate', 'mixture weight', 'prior', 'hyperprior'],
  added: '2026-11-11',
}
