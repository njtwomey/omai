import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Mixture of regressions (a class picks the line)',
  summary:
    'Eight points on two lines with a categorical class per row and a coefficient vector per class: the class picks the prediction of its branch, one gate of Gaussian likelihoods per row, against the enumeration of the assignments.',
  tags: ['expectation propagation', 'ep', 'atom', 'gates', 'mixture of regressions', 'switching regression', 'pick'],
  added: '2026-11-11',
}
