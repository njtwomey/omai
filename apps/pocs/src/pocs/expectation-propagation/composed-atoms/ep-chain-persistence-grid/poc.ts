import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Chain with a grid over its persistence',
  summary:
    'A Gaussian chain of hidden states with an unknown level, a gamma precision of its steps and a persistence that multiplies the state, each state read through a Gaussian whose sd is exp of half the state. EP runs inside a grid over the persistence (a walk outwards from the middle, each node one exact gaussian-chain factor, the nodes weighted by their EP evidence), against a point-mass filter of the same model.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composed',
    'chain',
    'gaussian-chain',
    'grid',
    'hyperparameter',
    'persistence',
    'gamma precision',
    'non-linear read',
    'evidence',
    'truth recovery',
  ],
  added: '2026-11-12',
}
