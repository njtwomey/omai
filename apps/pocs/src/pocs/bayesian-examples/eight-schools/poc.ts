import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Eight schools: partial pooling',
  summary:
    "The canonical hierarchical model: choose Rubin's eight schools or generated groups, centred or non-centred partial pooling (or complete or no pooling) and any inference method that runs on it, watch the estimates shrink towards the mean and the funnel with its divergences, and check the run against a long non-centred NUTS run.",
  tags: [
    'hierarchical',
    'eight schools',
    'partial pooling',
    'shrinkage',
    'funnel',
    'divergences',
    'non-centred',
    'reparameterisation',
    'nuts',
    'svi',
    'pair plot',
  ],
  added: '2026-11-11',
}
