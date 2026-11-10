import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'A mixture of Gaussians: the posterior and label switching',
  summary:
    'The Old Faithful waiting times fitted by a mixture of two to four normals, with the means free or ordered, by NUTS, Gibbs or variational inference, computed in Web Workers for a time budget. Then one playhead plays the predictive density with its components, the pair plot of the means and weights, the traces and the R-hat and ESS table.',
  tags: [
    'mixture',
    'gaussian mixture',
    'label switching',
    'old faithful',
    'posterior',
    'nuts',
    'gibbs',
    'svi',
    'pair plot',
    'r-hat',
    'web worker',
    'multimodal',
  ],
  added: '2026-11-10',
  group: 'probabilistic-programming',
}
