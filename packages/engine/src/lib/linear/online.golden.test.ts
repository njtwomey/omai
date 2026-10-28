import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import {
  kernelPerceptron,
  type PassiveAggressiveVariant,
  type PerceptronVariant,
  passiveAggressive,
  passiveAggressiveRegressor,
  perceptron,
} from './online.ts'

learnGolden('online linear learners against scikit-learn', 'linear/online', {
  perceptron: (inputs, p) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const variant = str(p, 'variant') as PerceptronVariant
    const f = perceptron({ variant, maxEpochs: num(p, 'epochs') }).fit({ x, y })
    const out: Record<string, Tensor | number | readonly number[]> = {
      decision: f.decisionFunction(q),
      predict: f.predict(q),
    }
    if (variant !== 'voted') {
      out['coef'] = f.params.coef
      out['intercept'] = f.params.intercept
    }
    return out as never
  },
  kernelPerceptron: (inputs, p) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const f = kernelPerceptron({ kernelOptions: { gamma: num(p, 'gamma') }, maxEpochs: num(p, 'epochs') }).fit({ x, y })
    return { decision: f.decisionFunction(q), intercept: f.params.intercept, mistakes: f.params.mistakes }
  },
  pa: (inputs, p) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const f = passiveAggressive({
      variant: str(p, 'variant') as PassiveAggressiveVariant,
      c: num(p, 'c'),
      maxEpochs: num(p, 'epochs'),
    }).fit({ x, y })
    return {
      coef: f.params.coef,
      intercept: f.params.intercept,
      decision: f.decisionFunction(q),
      predict: f.predict(q),
    }
  },
  paRegressor: (inputs, p) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const f = passiveAggressiveRegressor({
      variant: str(p, 'variant') as PassiveAggressiveVariant,
      c: num(p, 'c'),
      epsilon: num(p, 'epsilon'),
      maxEpochs: num(p, 'epochs'),
    }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept, predict: f.predict(q) }
  },
})
