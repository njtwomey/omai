import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Bayesian online changepoint detection',
  summary:
    'Adams and MacKay (2007) running live: pick the data and the kind of shift, the segment model with its priors and the hazard, then watch the run-length posterior grow one observation at a time and score the answer against the planted changepoints.',
  tags: ['changepoint', 'time series', 'bocpd', 'online', 'bayesian', 'run length', 'hazard', 'Adams and MacKay'],
  added: '2026-11-01',
}
