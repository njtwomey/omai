import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Gaussian process classification',
  summary:
    'Classify one or two inputs with a latent Gaussian process and a logistic or probit Bernoulli observation: choose the data, the kernel and its numbers (fixed or inferred), infer the posterior with any method that applies (NUTS, EP, Laplace, variational, ...), see the predictive probability and the latent function with their bands, and check it against a long NUTS run and the Laplace and EP fits of the gp domain.',
  tags: [
    'gaussian process',
    'classification',
    'kernel',
    'length scale',
    'latent function',
    'probit',
    'logistic',
    'expectation propagation',
    'laplace',
    'nuts',
    'hyperparameters',
    'predictive uncertainty',
  ],
  added: '2026-11-11',
  group: 'bayesian-examples',
}
