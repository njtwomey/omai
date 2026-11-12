import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Softmax over eight alternatives',
  summary:
    'Two coefficients with Gaussian priors and twelve choices among eight alternatives with a price and a quality each, through a softmax: the multiclass site with eight class scores per choice (an auxiliary-variable integral, not the product grid that stops at six).',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composed',
    'prior (gaussian) x2',
    'scale (known attribute times a coefficient)',
    'multiclass-site (softmax, eight classes)',
  ],
  added: '2026-11-12',
}
