import { componentsRecord } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  featureNamesOf,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { geometry, linalg } from '../../lib/numerics/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { toArray, toFlat } from '../../lib/tensor/index.ts'
import { project, reconstruct, rowsOf } from './_linear.ts'
import type { UncentredVerbs } from './svd.ts'

/** Hyperparameters of a random projection. */
export type RandomProjectionHyper = {
  /** Output dimension, or 'auto' for the Johnson-Lindenstrauss bound at `eps` (default 2). */
  readonly nComponents?: number | 'auto'
  /** The distortion of the Johnson-Lindenstrauss bound (default 0.1). */
  readonly eps?: number
  /** Gaussian or sparse (Achlioptas, Li) entries (default Gaussian). */
  readonly kind?: 'gaussian' | 'sparse'
  /** The density of a sparse projection (default 1 / sqrt(d)). */
  readonly density?: number
}

/** The parameters of a fitted random projection, with its distortion on the training pairs. */
export type RandomProjectionParams = {
  readonly components: number[][]
  /** Ratios of projected to original pairwise distances: min, mean and max. */
  readonly distortion: { readonly min: number; readonly mean: number; readonly max: number }
}

/** A fitted random projection. */
export type FittedRandomProjection = ExplainedFitted<RandomProjectionParams, UncentredVerbs>

/**
 * Gaussian and sparse random projections (sklearn.random_projection) from `numerics/linalg`'s random projection
 * matrices, with a distortion report: the ratios of projected to original pairwise distances on the training rows.
 * Its exposition is a `components` record (the projection matrix and the projected rows).
 *
 * @param hyper - Output dimension (or 'auto'), the distortion target, the kind and the density.
 * @returns The estimator.
 */
export function randomProjection(
  hyper: RandomProjectionHyper = {},
): Estimator<RandomProjectionHyper, FittedRandomProjection> {
  return defineExplainedEstimator<RandomProjectionHyper, RandomProjectionParams, UncentredVerbs>(
    {
      name: 'random-projection',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 100, {
            label: 'components',
            description: 'The dimension projected to.',
            default: typeof h.nComponents === 'number' ? h.nComponents : 2,
          }),
          kind: choice(['gaussian', 'sparse'], {
            label: 'projection',
            description: 'Dense Gaussian or sparse (Achlioptas) random projections.',
            default: h.kind ?? 'gaussian',
          }),
          eps: real(0.01, 0.99, {
            label: 'distortion ε',
            description: 'The tolerated distortion of pairwise distances (Johnson-Lindenstrauss).',
            default: h.eps ?? 0.1,
          }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'randomProjection', { minSamples: 2 })
        const k = h.nComponents === 'auto' ? linalg.johnsonLindenstraussDim(m.n, h.eps ?? 0.1) : (h.nComponents ?? 2)
        const key = fitRunOptions(options, 'random-projection').key as Key
        const r = linalg.randomProjectionMatrix(key, k, m.d, {
          kind: h.kind ?? 'gaussian',
          ...(h.density === undefined ? {} : { density: h.density }),
        })
        const components = toArray(r) as number[][]
        const zero = new Array<number>(m.d).fill(0)
        const projected = project(data.x, zero, components)
        const before = toFlat(geometry.pdist(data.x))
        const after = toFlat(geometry.pdist(projected))
        const ratios = before.flatMap((v, i) => (v > 0 ? [(after[i] ?? 0) / v] : []))
        return {
          params: {
            components,
            distortion: {
              min: Math.min(...ratios),
              mean: ratios.reduce((a, v) => a + v, 0) / Math.max(1, ratios.length),
              max: Math.max(...ratios),
            },
          },
          exposition: [
            componentsRecord({
              method: `${h.kind ?? 'gaussian'}RandomProjection`,
              features: featureNamesOf(data, m.d),
              components,
              scores: rowsOf(projected),
            }),
          ],
        }
      },
      methods: (p) => {
        const zero = new Array<number>(p.components[0]?.length ?? 0).fill(0)
        return {
          transform: (x) => project(x, zero, p.components),
          inverseTransform: (z) => reconstruct(z, zero, p.components),
          nFeaturesIn: zero.length,
        }
      },
    },
    hyper,
  )
}

/** A random projection as a class. */
export class RandomProjection extends EstimatorClass<RandomProjectionHyper, FittedRandomProjection> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link randomProjection}.
   */
  constructor(hyper: RandomProjectionHyper = {}) {
    super(randomProjection(hyper))
  }
}
