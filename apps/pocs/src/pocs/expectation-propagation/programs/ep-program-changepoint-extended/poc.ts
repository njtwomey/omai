import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Program: a Poisson rate that changes, parameters learned',
  summary:
    'The changepoint program with both sets of parameters unknown: the three regime rates (gamma priors) and the transition matrix of the regime chain (a Dirichlet prior on each row). The compiler turns the random rows into one gate per step, so EP learns how sticky the regimes are as well as how fast each counts. The page checks the recovery against Gibbs sampling, calibrates it over many seeds, and shows where it fails: a barely visited regime, and a relabelling of the regimes.',
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
    'dirichlet',
    'transition matrix',
    'learned',
    'calibration',
    'label switching',
  ],
  added: '2026-11-11',
}
