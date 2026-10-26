import type { Tensor } from '../../lib/tensor/index.ts'
import { tensor } from '../../lib/tensor/index.ts'
import { learnGolden } from '../../testing/learn.ts'
import { type SgdSettings, sgdClassifier, sgdRegressor } from './online.ts'

/** scikit-learn's options as SGD settings (shuffling off). */
function settings(p: Readonly<Record<string, unknown>>): Partial<SgdSettings> {
  const out: Record<string, unknown> = { shuffle: false }
  const names: Record<string, keyof SgdSettings> = {
    loss: 'loss',
    alpha: 'alpha',
    max_iter: 'maxIter',
    learning_rate: 'learningRate',
    eta0: 'eta0',
    l1_ratio: 'l1Ratio',
  }
  for (const [k, v] of Object.entries(p)) {
    if (k === 'penalty') out['penalty'] = v ?? 'none'
    else if (k === 'tol') out['tol'] = v
    else if (names[k] !== undefined) out[names[k]] = v
  }
  if (!('tol' in p)) out['tol'] = 1e-3
  if (!('penalty' in p)) out['penalty'] = 'l2'
  return out as Partial<SgdSettings>
}

learnGolden('SGD and passive-aggressive models against scikit-learn', 'methods/linear/online', {
  classifier: (inputs, p) => {
    const [x, y] = inputs as readonly [Tensor, Tensor]
    const f = sgdClassifier(settings(p)).fit({ x, y })
    return { coef: tensor(f.params.coef), intercept: tensor(f.params.intercept) }
  },
  regressor: (inputs, p) => {
    const [x, y] = inputs as readonly [Tensor, Tensor]
    const f = sgdRegressor(settings(p)).fit({ x, y })
    return { coef: tensor(f.params.coef), intercept: tensor([f.params.intercept]) }
  },
})
