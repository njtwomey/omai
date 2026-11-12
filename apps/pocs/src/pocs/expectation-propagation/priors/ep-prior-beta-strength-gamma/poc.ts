import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Prior: beta on a probability, gamma hyperprior on its strength',
  summary:
    'A symmetric Beta(strength, strength) prior on a Bernoulli probability with a gamma-distributed strength: the strength says whether the probability sits near one half or near the ends.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
}
