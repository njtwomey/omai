import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Variational guides on a banana',
  summary:
    'Stochastic variational inference on Haario’s banana with four guides at once: a mean-field normal, a full-rank normal and two normalising flows. Race their ELBOs within a time budget you choose, and see which of them can bend to the curved target and which can only fit an ellipse.',
  tags: [
    'svi',
    'variational inference',
    'guide',
    'mean field',
    'full rank',
    'normalising flow',
    'iaf',
    'coupling flow',
    'banana',
    'elbo',
  ],
  added: '2026-11-10',
  group: 'probabilistic-programming',
}
