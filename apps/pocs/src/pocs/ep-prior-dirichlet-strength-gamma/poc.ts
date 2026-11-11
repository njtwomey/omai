import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Prior: Dirichlet on categorical probabilities, gamma hyperprior on its concentration',
  summary:
    'A symmetric Dirichlet with a gamma-distributed concentration over three categories: the concentration says how even the probabilities are, and the data inform it through the counts.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
