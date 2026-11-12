import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Hierarchical radon',
  summary:
    'Log radon in houses nested in counties of unequal size: pooled, unpooled, partial pooling (centred or non-centred) or varying slopes with a correlated prior, any inference method that runs on it, the shrinkage of each county against its size and the posterior of the hyperparameters, checked against a long NUTS run.',
  tags: [
    'hierarchical',
    'radon',
    'multilevel',
    'varying intercepts',
    'varying slopes',
    'partial pooling',
    'shrinkage',
    'counties',
    'nuts',
    'svi',
    'pair plot',
  ],
  added: '2026-11-11',
}
