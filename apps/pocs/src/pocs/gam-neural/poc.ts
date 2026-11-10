import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Generalised additive model, neural',
  summary:
    'The additive model trained as a neural network: a learnable spline (its knots can move) or a small net on the feature, a smoothness penalty, and an optimiser. Play the training and watch the shape function settle against the truth and the classic GAM fit.',
  tags: ['gam', 'additive', 'neural', 'spline', 'learnable knots', 'nam', 'penalty', 'training', 'regression'],
  added: '2026-11-01',
  group: 'models',
}
