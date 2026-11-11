import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Bayesian linear regression',
  summary:
    'Fit a polynomial to noisy data from a known truth: choose the data, the degree, the prior and whether the noise is known or learned, infer the posterior with any method that applies, see the fitted curve with its band and sampled lines and the pair plot of the coefficients, and check it against the exact posterior or a long NUTS run.',
  tags: [
    'linear regression',
    'polynomial',
    'posterior',
    'predictive band',
    'credible interval',
    'prior',
    'noise',
    'correlated coefficients',
    'conjugate',
    'nuts',
    'exact posterior',
  ],
  added: '2026-11-11',
  group: 'bayesian-examples',
}
