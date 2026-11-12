import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Probit regression lab',
  summary:
    'One Bayesian probit regression on one two-dimensional dataset, solved by the inference methods you tick: samplers, variational fits, EP, Laplace, Pathfinder and tempered SMC, each in its own Web Worker for a time budget you choose. The page only counts the clock; when the budget is spent you get the pair plots of the posterior samples, the decision boundaries, the marginals and the diagnostics, all on one playhead you can scrub or play.',
  tags: [
    'probit',
    'regression',
    'classification',
    'pair plot',
    'posterior',
    'nuts',
    'hmc',
    'svi',
    'ep',
    'laplace',
    'pathfinder',
    'smc',
    'web worker',
    'decision boundary',
    'polynomial features',
    'anytime',
  ],
  added: '2026-11-10',
}
