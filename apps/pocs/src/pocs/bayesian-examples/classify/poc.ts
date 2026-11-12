import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Bayesian logistic regression',
  summary:
    'Classify a two-dimensional data set with a Bayesian probit or logistic model of polynomial features: choose the data, the model and one inference method with its own knobs and a time budget, play the run, and check it against a long NUTS run.',
  tags: [
    'classification',
    'probit',
    'logistic',
    'posterior',
    'nuts',
    'svi',
    'laplace',
    'ep',
    'pathfinder',
    'decision boundary',
    'pair plot',
  ],
  added: '2026-11-11',
}
