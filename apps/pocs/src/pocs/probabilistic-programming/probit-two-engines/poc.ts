import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'One probit regression, three engines',
  summary:
    'A probit regression written once in the model builder and run by NUTS, expectation propagation and Laplace, each within a time budget you choose, against the exact answer on a grid. Watch the marginals settle, the plate diagram carry EP’s messages and the sampler’s beliefs, and see where each engine is right.',
  tags: [
    'probit',
    'regression',
    'nuts',
    'expectation propagation',
    'ep',
    'laplace',
    'model builder',
    'plate diagram',
    'posterior comparison',
    'anytime',
  ],
  added: '2026-11-10',
}
