import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Prior: normal on a Weibull log scale, normal hyperprior on its centre',
  summary:
    'The log of a Weibull scale is normal around a normal centre, five observed lifetimes with a known shape: two Gaussians and the numeric Weibull likelihood.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
