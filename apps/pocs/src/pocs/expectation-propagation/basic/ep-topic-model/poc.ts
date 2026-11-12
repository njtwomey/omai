import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Topic model (Dirichlet mixtures, a word read through its topic)',
  summary:
    'Two documents of four words, two topics: a Dirichlet mixture per document, a Dirichlet word distribution per topic, a topic per token and the word drawn from that topic: one gate per token over the topics’ conjugate discrete factors, against the enumeration of the assignments.',
  tags: ['expectation propagation', 'ep', 'atom', 'nonparametric', 'topic model', 'lda', 'dirichlet', 'gate'],
  added: '2026-11-11',
}
