import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Word hyphenation with a linear-chain CRF',
  summary:
    'Hyphenation as sequence labelling: a CRF with CRFsuite-style letter templates trains live on a few hundred dictionary words (pick the window, L2 and L1 penalties and the algorithm; OWL-QN zeroes whole templates), is scored on held-out words against the TeX patterns, and shows the hyphen probability in every gap of a word you type and the features behind it.',
  tags: [
    'crf',
    'conditional random field',
    'structured prediction',
    'sequence labelling',
    'hyphenation',
    'tex patterns',
    'liang',
    'owl-qn',
    'l1',
    'sparsity',
    'crfsuite',
    'marginals',
    'perceptron',
  ],
  added: '2026-11-10',
}
