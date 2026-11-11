import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Prior: log-normal on a Gaussian scale, normal hyperprior on its log centre',
  summary:
    'The log of a Gaussian standard deviation is normal around a normal centre, four zero-mean observations: two Gaussians and the Gaussian with an exp(h) scale.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
