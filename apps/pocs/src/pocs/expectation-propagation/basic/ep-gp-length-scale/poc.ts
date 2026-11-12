import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Gaussian process with a learned length scale',
  summary:
    'A dense Gaussian prior on six function values whose kernel has a random length scale (a custom function), and a Gaussian observation of each: EP runs inside a grid over the length scale, each node weighted by its EP evidence and the prior.',
  tags: ['expectation propagation', 'ep', 'atom', 'kernels', 'gaussian process', 'hyperparameter', 'grid', 'evidence'],
  added: '2026-11-11',
}
