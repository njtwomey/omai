import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Is it calibrated? Simulation-based calibration',
  summary:
    'Draw a line from its prior, draw data from it, fit, and rank the true intercept and slope among the draws: if a method is calibrated the ranks are uniform. NUTS, a mean-field guide that is too narrow for the correlated posterior, and a guide stopped far too early run side by side within a time budget, and their rank histograms and ECDF differences, with exact bands, fill in as replications arrive.',
  tags: [
    'simulation-based calibration',
    'sbc',
    'calibration',
    'rank histogram',
    'ecdf',
    'nuts',
    'svi',
    'mean field',
    'validation',
  ],
  added: '2026-11-10',
  group: 'probabilistic-programming',
}
