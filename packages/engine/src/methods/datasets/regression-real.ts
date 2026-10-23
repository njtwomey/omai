import { type RegressionData, xy } from '../../lib/data/index.ts'
import { fromStorage } from '../../lib/tensor/index.ts'
import { DIABETES_FEATURES, DIABETES_SHA256, DIABETES_SHAPE, DIABETES_X, DIABETES_Y } from './_data/diabetes.data.ts'

/** Options of {@link diabetes}. */
export interface DiabetesOptions {
  /**
   * Centre each feature and scale it to unit sum of squares, as sklearn's default `load_diabetes()` does (default
   * false: the original units).
   */
  readonly scaled?: boolean
}

/**
 * The diabetes data of Efron, Hastie, Johnstone and Tibshirani (2004, "Least angle regression"): ten baseline variables
 * (age, sex, body mass index, blood pressure and six blood serum measurements) for 442 patients, and a measure of
 * disease progression one year later. Embedded from scikit-learn (`load_diabetes(scaled=False)`).
 *
 * @param options - Whether to scale the features as sklearn does by default.
 * @returns The dataset.
 * @concept datasets
 * @example
 * ```ts
 * import { diabetes } from 'whitebox-engine/methods/datasets'
 * diabetes().x.shape // => [442, 10]
 * ```
 */
export function diabetes(options: DiabetesOptions = {}): RegressionData {
  const [n, d] = DIABETES_SHAPE
  const x = Float64Array.from(DIABETES_X)
  if (options.scaled === true) {
    for (let j = 0; j < d; j++) {
      let mean = 0
      for (let i = 0; i < n; i++) mean += x[i * d + j] ?? 0
      mean /= n
      let ss = 0
      for (let i = 0; i < n; i++) ss += ((x[i * d + j] ?? 0) - mean) ** 2
      const scale = Math.sqrt(ss)
      for (let i = 0; i < n; i++) x[i * d + j] = ((x[i * d + j] ?? 0) - mean) / scale
    }
  }
  return xy(fromStorage(x, [n, d], 'float64'), fromStorage(Float64Array.from(DIABETES_Y), [n], 'float64'), {
    name: 'diabetes',
    description: 'Disease progression one year after baseline for 442 diabetes patients (Efron et al., 2004).',
    source: 'scikit-learn load_diabetes',
    url: 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html',
    licence: 'BSD-3-Clause (scikit-learn)',
    citation: 'Efron, Hastie, Johnstone and Tibshirani (2004), Least angle regression, Annals of Statistics 32(2).',
    sha256: DIABETES_SHA256,
    featureNames: DIABETES_FEATURES,
    targetNames: ['progression'],
  })
}
