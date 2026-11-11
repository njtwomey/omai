import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Program: a Poisson rate that changes',
  summary:
    'Counts from a Poisson process whose rate switches between three regimes along a sticky Markov chain, the three rates unknown (gamma priors). The program simulates it, writes it as a model (a scan over a categorical state, a gamma rate per regime read by the state) and runs expectation propagation; the page checks the regimes and rates recovered, against Gibbs sampling, with the program source below.',
  tags: [
    'expectation propagation',
    'ep',
    'program',
    'changepoint',
    'regime switching',
    'hidden markov',
    'poisson',
    'gate',
    'gamma',
  ],
  added: '2026-11-11',
  group: 'expectation-propagation',
}
