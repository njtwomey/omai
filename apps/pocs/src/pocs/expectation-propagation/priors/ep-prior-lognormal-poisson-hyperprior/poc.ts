import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Prior: log-normal on a Poisson rate, normal hyperprior on its log centre',
  summary:
    'The log of a Poisson rate is normal around a centre that is itself normal: two Gaussians and the numeric Poisson likelihood with a log link.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
}
