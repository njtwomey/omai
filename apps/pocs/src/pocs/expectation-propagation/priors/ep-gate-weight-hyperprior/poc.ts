import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Gate with an unknown mixing weight, gamma hyperprior on its prior strength',
  summary:
    'The beta prior on the mixing weight is symmetric with a gamma-distributed strength: a hyperprior one level above the gate, so the data inform the strength through how unbalanced the switches are.',
  tags: ['expectation propagation', 'ep', 'atom', 'gate', 'mixture weight', 'prior', 'hyperprior'],
  added: '2026-11-11',
}
