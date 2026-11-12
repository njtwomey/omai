import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Generalised additive model, classic',
  summary:
    'A penalised-spline GAM (mgcv-style) on a curve with noise: choose how the smoothing parameter is picked (REML, GCV or fixed by hand), the basis and its size, and watch the fit, its credible band and the effective degrees of freedom respond.',
  tags: ['gam', 'additive', 'spline', 'smoothing', 'penalty', 'reml', 'gcv', 'regression', 'classic'],
  added: '2026-11-01',
}
