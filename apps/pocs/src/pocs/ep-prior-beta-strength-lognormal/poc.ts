import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Prior: beta on a probability, log-normal hyperprior on its strength',
  summary:
    'A symmetric Beta(strength, strength) prior on a Bernoulli probability with a log-normal strength: the factor takes the log strength as a Gaussian variable.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
