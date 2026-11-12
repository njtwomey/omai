import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'EP, alpha and the clutter problem',
  summary:
    'Minka’s clutter problem: a mean seen through points that may be outliers, so the posterior is lumpy. Alpha-divergence message passing fits a Gaussian to it at a ladder of alphas, from expectation propagation (mass-covering, wide) toward the variational Gaussian (mode-seeking, narrow), and the exact posterior shows what each choice gets wrong.',
  tags: [
    'expectation propagation',
    'ep',
    'alpha divergence',
    'clutter problem',
    'robust',
    'outliers',
    'mass covering',
    'mode seeking',
    'message passing',
    'minka',
  ],
  added: '2026-11-10',
}
