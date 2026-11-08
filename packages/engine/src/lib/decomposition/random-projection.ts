/**
 * Random projections (scikit-learn's `random_projection`) assembled over `numerics/linalg`'s projection matrices:
 * Gaussian and sparse random matrices that reduce dimension while approximately preserving pairwise distances, the
 * Johnson-Lindenstrauss guarantee. The fit reports the distortion, the ratios of projected to original distances on the
 * training pairs.
 */

import { maxOf, minOf } from '../base/index.ts'
import { resolveKey } from '../context/index.ts'
import { defineEstimator, type Estimator, type Fitted, matrixView } from '../learn/index.ts'
import { geometry, linalg } from '../numerics/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { type Tensor, type TensorLike, toArray, toFlat } from '../tensor/index.ts'
import { project, reconstruct } from './_linear.ts'

/** Hyperparameters of {@link randomProjection}. */
export interface RandomProjectionHyper {
  /** Output dimension, or `'auto'` for the Johnson-Lindenstrauss bound at `eps`. */
  readonly nComponents: number | 'auto'
  /** The distortion of the Johnson-Lindenstrauss bound. */
  readonly eps: number
  /** Gaussian or sparse (Achlioptas, Li) entries. */
  readonly kind: 'gaussian' | 'sparse'
  /** The density of a sparse projection (`null`: 1 / sqrt(d)). */
  readonly density: number | null
}

/** The parameters of a fitted random projection, with its distortion on the training pairs. */
export type RandomProjectionParams = {
  readonly components: number[][]
  /** Ratios of projected to original pairwise distances: min, mean and max. */
  readonly distortion: { readonly min: number; readonly mean: number; readonly max: number }
}

/** The verbs of a random projection (uncentred): the projection and its pseudo-inverse direction. */
export interface RandomProjectionVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (z: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * Gaussian and sparse random projections (sklearn.random_projection) from `numerics/linalg`'s random projection
 * matrices, with a distortion report: the ratios of projected to original pairwise distances on the training rows.
 *
 * @param hyper - Output dimension (or 'auto'), the distortion target, the kind and the density.
 * @returns The estimator.
 * @example
 * ```ts
 * import { randomProjection } from 'omai-engine/decomposition'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 10 }, (_, i) => Array.from({ length: 20 }, (_, j) => Math.sin(i * j)))
 * randomProjection({ nComponents: 5 }).fit({ x }, { key: key(0) }).params.components.length // => 5
 * ```
 */
export function randomProjection(
  hyper: Partial<RandomProjectionHyper> = {},
): Estimator<RandomProjectionHyper, Fitted<RandomProjectionParams> & RandomProjectionVerbs> {
  return defineEstimator<RandomProjectionHyper, RandomProjectionParams, RandomProjectionVerbs>(
    {
      name: 'decomposition/random-projection',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 100, {
            default: typeof h.nComponents === 'number' ? h.nComponents : 2,
            label: 'components',
          }),
          kind: choice(['gaussian', 'sparse'] as const, { default: h.kind, label: 'projection' }),
          eps: real(0.01, 0.99, { default: h.eps, label: 'distortion' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'randomProjection', { minSamples: 2 })
        const k = h.nComponents === 'auto' ? linalg.johnsonLindenstraussDim(m.n, h.eps) : h.nComponents
        const r = linalg.randomProjectionMatrix(resolveKey(options, 'random-projection'), k, m.d, {
          kind: h.kind,
          ...(h.density === null ? {} : { density: h.density }),
        })
        const components = toArray(r) as number[][]
        const projected = project(data.x, new Array<number>(m.d).fill(0), components)
        const before = toFlat(geometry.pdist(data.x))
        const after = toFlat(geometry.pdist(projected))
        const ratios = before.flatMap((v, i) => (v > 0 ? [(after[i] ?? 0) / v] : []))
        return {
          components,
          distortion: {
            min: minOf(ratios),
            mean: ratios.reduce((a, v) => a + v, 0) / Math.max(1, ratios.length),
            max: maxOf(ratios),
          },
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
    { nComponents: 2, eps: 0.1, kind: 'gaussian', density: null, ...hyper },
  )
}
