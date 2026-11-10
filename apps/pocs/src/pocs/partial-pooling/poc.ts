import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Partial pooling: groups that borrow strength',
  summary:
    'The classical hierarchical model on groups of very different sizes: each group alone, all groups pooled and the hierarchical posterior by NUTS, as a shrinkage plot whose intervals settle as the draws arrive. The centred and non-centred forms run side by side, so the funnel and its divergent transitions show in a pair plot of the group mean, the spread and two group effects.',
  tags: [
    'hierarchical',
    'partial pooling',
    'shrinkage',
    'multilevel',
    'radon',
    'eight schools',
    'nuts',
    'funnel',
    'divergences',
    'non-centred',
    'pair plot',
    'web worker',
  ],
  added: '2026-11-10',
  group: 'probabilistic-programming',
}
