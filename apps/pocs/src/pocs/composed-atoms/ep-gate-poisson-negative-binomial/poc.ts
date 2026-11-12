import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'A gate between a Poisson and a negative binomial',
  summary:
    'Several pieces of the model language composed in one small setup: a switch per observation that chooses between two count likelihoods of different families (a Poisson and a negative binomial), a log-linear rate on a covariate shared by the means of both, a Beta prior on the probability of the switch and a learned dispersion. The data are simulated from stated values. EP recovers the rate, the dispersion and the switch probability, and reports for each observation the probability that it is the negative binomial one, against the grid posterior with the switches summed out.',
  tags: [
    'expectation propagation',
    'ep',
    'atom',
    'composition',
    'gates',
    'gate over families',
    'poisson',
    'negative binomial',
    'dispersion',
    'beta prior',
    'log-linear rate',
    'switch',
    'truth recovery',
  ],
  added: '2026-11-12',
}
