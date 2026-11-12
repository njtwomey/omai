import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Prior: logit-normal on a Bernoulli probability, normal hyperprior on its centre',
  summary:
    'The log odds of a Bernoulli probability are normal around a normal centre, seven flips: two Gaussians and the logistic factor.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
}
