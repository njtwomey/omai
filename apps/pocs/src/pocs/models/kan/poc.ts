import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Kolmogorov-Arnold network',
  summary:
    'A KAN learning a function from the Kolmogorov-Arnold paper: every edge carries its own learnable curve. Play the training and watch the edge functions form, change the sparsification penalties, the grid, the degree and the family of curves, and see which edges the network switches off.',
  tags: ['kan', 'kolmogorov-arnold', 'neural', 'spline', 'edge functions', 'sparsity', 'symbolic', 'training'],
  added: '2026-11-01',
}
