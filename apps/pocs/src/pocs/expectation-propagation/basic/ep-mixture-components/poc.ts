import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Random index into random components',
  summary:
    'The same switch written as an index: a categorical z picks component mu[z] of two random component means for the observation y. EP reports the probability of each component; the exact posterior of the selected value is the mixture of the updated components.',
  tags: ['expectation propagation', 'ep', 'atom', 'gates', 'prior', 'prior', 'index gate', 'gaussian likelihood'],
  added: '2026-11-11',
}
