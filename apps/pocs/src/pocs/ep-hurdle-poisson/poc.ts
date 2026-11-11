import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Hurdle count (a zero, or a zero-truncated Poisson)',
  summary:
    'Twelve counts that are zero with an unknown probability and otherwise zero-truncated Poisson with a log-linear rate: a conjugate Bernoulli factor on the zero probability and a zero-truncated-Poisson site on the two coefficients.',
  tags: ['expectation propagation', 'ep', 'atom', 'counts', 'hurdle', 'truncated poisson', 'zero'],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
