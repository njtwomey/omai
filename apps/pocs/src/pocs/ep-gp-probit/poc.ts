import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Gaussian process with a probit likelihood',
  summary:
    'A dense Gaussian prior on three function values and a probit Bernoulli observation of each: GP classification with closed-form tilted moments.',
  tags: ['expectation propagation', 'ep', 'atom', 'likelihoods', 'dense prior', 'probit link'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
