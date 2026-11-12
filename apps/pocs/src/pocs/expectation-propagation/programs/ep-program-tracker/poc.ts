import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Program: a tracked path with outliers',
  summary:
    'A latent AR(1) path seen through Gaussian noise with a fraction of wild readings. The program simulates it, writes it as a model (a scan with an outlier-mixture observation) and runs expectation propagation; the page checks that the path is recovered through the outliers, against Gibbs sampling and against the same model with plain Gaussian noise, with the program source below.',
  tags: [
    'expectation propagation',
    'ep',
    'program',
    'outliers',
    'robust',
    'state space',
    'ar1',
    'tracking',
    'mixture site',
  ],
  added: '2026-11-11',
}
