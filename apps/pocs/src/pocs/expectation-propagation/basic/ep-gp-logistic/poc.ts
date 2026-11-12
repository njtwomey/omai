import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Gaussian process with a logistic likelihood',
  summary:
    'A dense Gaussian prior on three function values (an RBF kernel) and a logistic Bernoulli observation of each: the dense Gaussian-vector prior factor of GP classification.',
  tags: ['expectation propagation', 'ep', 'atom', 'likelihoods', 'dense prior', 'logistic link'],
  added: '2026-11-11',
}
