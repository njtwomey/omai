import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Biased coin: Beta-Binomial',
  summary:
    'Flip a coin of unknown bias (or count events at an unknown rate): choose the data and the Beta (or Gamma) prior, infer the posterior with any method that applies, watch it concentrate as flips accumulate and predict the next one, and check it against the exact conjugate posterior.',
  tags: [
    'coin',
    'beta',
    'bernoulli',
    'binomial',
    'gamma',
    'poisson',
    'conjugate',
    'prior',
    'posterior',
    'predictive',
    'credible interval',
    'exact posterior',
  ],
  added: '2026-11-11',
}
