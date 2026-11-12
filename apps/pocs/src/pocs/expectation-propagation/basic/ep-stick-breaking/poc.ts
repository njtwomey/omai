import type { Poc } from '../../../../poc.ts'

export const poc: Poc = {
  title: 'Stick-breaking mixture (truncated Dirichlet process)',
  summary:
    'Ten observations from three known Gaussians whose weights are the stick-breaking of two Beta breaks, with a categorical assignment per observation: EP reads the weights through one stick-breaking factor per assignment, against the grid over the breaks.',
  tags: ['expectation propagation', 'ep', 'atom', 'nonparametric', 'dirichlet process', 'stick-breaking', 'mixture'],
  added: '2026-11-11',
}
