import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'AMP: sparse recovery and its state evolution',
  summary:
    'Approximate message passing recovers a sparse signal from fewer measurements than unknowns. Each iteration’s noise level is predicted in advance by a one-line recursion, the state evolution, and the run follows it. Choose how many measurements, how sparse the signal and how noisy, and watch the error and the predicted noise level fall together.',
  tags: [
    'amp',
    'approximate message passing',
    'compressed sensing',
    'sparse recovery',
    'state evolution',
    'bernoulli-gaussian',
    'message passing',
  ],
  added: '2026-11-10',
}
