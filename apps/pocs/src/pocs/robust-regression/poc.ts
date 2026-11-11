import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Robust regression with outliers',
  summary:
    'Fit a line to data with outliers: choose how many, how big and where they sit, then the likelihood (Gaussian, Student-t with fixed or learned degrees of freedom, Laplace, or a mixture of inliers and outliers), infer the posterior with any method that applies, and see how far each likelihood is dragged from the true line, checked against the exact posterior or a long NUTS run.',
  tags: [
    'robust regression',
    'outliers',
    'student t',
    'laplace',
    'mixture',
    'degrees of freedom',
    'leverage',
    'heavy tails',
    'posterior',
    'nuts',
    'exact posterior',
  ],
  added: '2026-11-11',
  group: 'bayesian-examples',
}
