import { learnGolden, num } from '../../testing/learn.ts'
import { nystroem, randomFourierFeatures, rbfKernel } from '../kernels/index.ts'
import { matmul, swapaxes, type Tensor, toFlat } from '../tensor/index.ts'
import { featureRidge } from './feature-ridge.ts'
import { looScore } from './kernel-selection.ts'

const lengthScale = (gamma: number): number => 1 / Math.sqrt(2 * gamma)

learnGolden(
  'feature-space ridge against scikit-learn Nystroem / RBFSampler + Ridge, and RidgeCV leave-one-out',
  'kernel_methods/feature_ridge',
  {
    nystroemRidge: ([x, y, xt], p) => {
      const map = nystroem(rbfKernel({ lengthScale: lengthScale(num(p, 'gamma')) }), x as Tensor, {
        indices: p['indices'] as unknown as number[],
      })
      const fit = featureRidge(map, x as Tensor, y as Tensor, num(p, 'alpha'), p['fitIntercept'] === true)
      return { predict: fit.predict(xt as Tensor), fitted: fit.predict(x as Tensor) }
    },
    fourierRidge: ([x, y, xt, w, b], p) => {
      const fit = featureRidge(
        randomFourierFeatures({ weights: w as Tensor, offset: b as Tensor }),
        x as Tensor,
        y as Tensor,
        num(p, 'alpha'),
      )
      return { predict: fit.predict(xt as Tensor) }
    },
    looRidgeCv: ([z, y], p) => {
      const k = matmul(z as Tensor, swapaxes(z as Tensor, 0, 1))
      const alphas = p['alphas'] as unknown as number[]
      return { mse: alphas.map((a) => toFlat(looScore(k, y as Tensor, a))[0] ?? Number.NaN) }
    },
  },
)
