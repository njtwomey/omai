import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Multiclass Gaussian process',
  summary:
    'A Gaussian process per class over two inputs, written as one dense Gaussian vector over the six (input, class) values, and a softmax of the three scores of each input: the multiclass site on the cells of a dense vector.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composed',
    'dense prior (gaussian-mv)',
    'gather of the scores of an input',
    'multiclass-site (softmax)',
  ],
  added: '2026-11-12',
}
