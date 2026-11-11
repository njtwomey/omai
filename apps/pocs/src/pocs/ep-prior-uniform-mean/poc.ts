import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Prior: uniform on a Gaussian mean',
  summary:
    'A uniform prior on a Gaussian mean with six observations: the density has no message family, so it is projected once onto a Gaussian of its own mean and variance.',
  tags: ['expectation propagation', 'ep', 'atom', 'prior', 'hyperprior'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
