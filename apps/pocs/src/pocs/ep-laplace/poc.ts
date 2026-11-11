import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Laplace likelihood',
  summary:
    'A mean mu observed five times through Laplace (double exponential) noise, one observation being far from the rest: the Laplace site, an absolute-error likelihood whose posterior is piecewise Gaussian.',
  tags: ['expectation propagation', 'ep', 'atom', 'likelihoods', 'prior', 'laplace-site'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
