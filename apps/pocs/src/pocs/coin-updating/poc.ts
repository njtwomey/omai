import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Coin flips: Bayesian updating',
  summary:
    'The classical beta-Bernoulli coin (or gamma-Poisson counter) updated one observation at a time. The exact conjugate posterior and the approximations you tick (Laplace, SVI, EP, NUTS, importance, SMC) are each fitted by Web Workers at a ladder of prefixes of the data; afterwards you scrub or play the number of observations seen and watch the posterior narrow, with the credible interval, the true value and the distance to the exact answer at each step.',
  tags: [
    'coin',
    'bayesian updating',
    'beta',
    'bernoulli',
    'gamma',
    'poisson',
    'conjugate',
    'prior',
    'posterior',
    'credible interval',
    'laplace',
    'svi',
    'ep',
    'nuts',
    'importance',
    'smc',
    'web worker',
  ],
  added: '2026-11-10',
  group: 'probabilistic-programming',
}
