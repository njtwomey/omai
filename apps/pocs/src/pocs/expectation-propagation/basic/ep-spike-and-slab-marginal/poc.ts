import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Spike and slab, switch summed out',
  summary:
    'The same prior with the switch summed out (the mixture-site form): one variable with a two-component zero-centred prior. EP matches the mean and variance of the exact two-branch posterior with one Gaussian and has no variable for the probability of the slab.',
  tags: ['expectation propagation', 'ep', 'atom', 'gates', 'mixture-site', 'gaussian likelihood'],
  added: '2026-11-11',
}
