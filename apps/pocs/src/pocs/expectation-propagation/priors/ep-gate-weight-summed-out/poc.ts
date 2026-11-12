import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Mixture with an unknown weight, switches summed out',
  summary:
    'The same mixture and beta prior with the switches summed out: one mixture-weight factor per observation on the weight, so EP reports the weight but not each observation’s responsibility.',
  tags: ['expectation propagation', 'ep', 'atom', 'gate', 'mixture weight', 'prior', 'hyperprior'],
  added: '2026-11-11',
}
