import { learnGolden, num, str } from '../../../testing/learn.ts'
import type { Tensor } from '../../tensor/index.ts'
import type { Estimator, Fitted } from '../estimator.ts'
import {
  maxAbsScaler,
  minMaxScaler,
  normalizer,
  powerTransformer,
  quantileTransformer,
  robustScaler,
  standardScaler,
} from './scalers.ts'

type Scaler = Fitted & { readonly transform: (x: Tensor) => Tensor; readonly inverseTransform?: (x: Tensor) => Tensor }

function run(est: Estimator, [x, z, w]: readonly Tensor[]): Record<string, Tensor | number[]> {
  const f = est.fit({ x: x as Tensor, ...(w === undefined ? {} : { weights: w }) }) as Scaler
  const t = f.transform(z as Tensor)
  const out: Record<string, Tensor | number[]> = { transform: t, train: f.transform(x as Tensor) }
  if (f.inverseTransform !== undefined) out['inverse'] = f.inverseTransform(t)
  const lambdas = (f.params as { readonly lambdas?: number[] }).lambdas
  if (lambdas !== undefined) out['lambdas'] = lambdas
  return out
}

learnGolden('scalers against sklearn.preprocessing', 'learn/preprocessing/scalers', {
  standard: (inputs, p) =>
    run(standardScaler({ withMean: p['withMean'] === true, withStd: p['withStd'] === true }), inputs),
  minmax: (inputs, p) => run(minMaxScaler({ featureRange: [num(p, 'lo'), num(p, 'hi')] }), inputs),
  minmaxClip: (inputs, p) => run(minMaxScaler({ featureRange: [num(p, 'lo'), num(p, 'hi')], clip: true }), inputs),
  maxabs: (inputs) => run(maxAbsScaler(), inputs),
  robust: (inputs, p) =>
    run(
      robustScaler({ quantileRange: [num(p, 'lo'), num(p, 'hi')], unitVariance: p['unitVariance'] === true }),
      inputs,
    ),
  normalizer: (inputs, p) => run(normalizer({ norm: str(p, 'norm') as 'l1' | 'l2' | 'max' }), inputs),
  quantile: (inputs, p) =>
    run(
      quantileTransformer({
        nQuantiles: num(p, 'nQuantiles'),
        outputDistribution: str(p, 'distribution') as 'uniform' | 'normal',
      }),
      inputs,
    ),
  power: (inputs, p) =>
    run(
      powerTransformer({
        method: str(p, 'method') as 'yeo-johnson' | 'box-cox',
        standardize: p['standardize'] === true,
      }),
      inputs,
    ),
})
