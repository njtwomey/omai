import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Ordinal probit: random cut points learned',
  summary:
    'The ordinal probit regression with a slope and two ordered cut points all unknown (Gaussian priors, the order enforced), the outcomes simulated from stated values: the likelihood-triple site on the predictor and the two cuts of each outcome.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composed',
    'prior (gaussian) x3',
    'ordered constraint (comparison factors on neighbouring cuts)',
    'matmul (linear predictor)',
    'numeric likelihood: ordinal probit with random cut points (three variables)',
  ],
  added: '2026-11-12',
}
