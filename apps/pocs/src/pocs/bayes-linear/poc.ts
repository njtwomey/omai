import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Bayesian online linear regression',
  summary:
    'Fit a line one point at a time: a prior over the intercept and slope, the likelihood of each new point, and the posterior they make, which becomes the next prior. Set the true line, the noise and the prior, then step through the points or play them back.',
  tags: [
    'bayesian',
    'linear regression',
    'online',
    'prior',
    'likelihood',
    'posterior',
    'conjugate',
    'sequential update',
  ],
  added: '2026-11-01',
}
