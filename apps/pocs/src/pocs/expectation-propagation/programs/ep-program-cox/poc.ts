import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Program: a Poisson process with a Gaussian-process intensity',
  summary:
    'A log-Gaussian Cox process: counts in bins with intensity exp(f), f a draw from a Gaussian process. The program simulates it from a seed, writes it as a model (a dense multivariate normal prior and a Poisson likelihood) and runs expectation propagation; the page checks that the true intensity is recovered, with the program source below.',
  tags: ['expectation propagation', 'ep', 'program', 'cox process', 'poisson process', 'gaussian process', 'intensity'],
  added: '2026-11-11',
}
