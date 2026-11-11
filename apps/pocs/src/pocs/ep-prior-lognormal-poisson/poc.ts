import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Prior: log-normal on a Poisson rate',
  summary:
    'A log-normal prior on a Poisson rate and four counts: a Gaussian on the log rate and the numeric Poisson likelihood with a log link.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
