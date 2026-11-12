/**
 * A Gaussian process latent variable model (GPLVM) as a model that expectation propagation can run: the observations `y` (N
 * points, D dimensions) are smooth functions of a few hidden coordinates `x` per point, and inference is over the coordinates
 * and the functions together. The Gram matrix of a Gaussian process on random inputs is a custom function of random values, which a
 * factor graph cannot read, so the functions are written with a finite set of random Fourier features instead (Rahimi and Recht):
 * `f_d(x) = sum_m W_dm cos(omega_m . x + b_m)` with the frequencies `omega_m` and phases `b_m` drawn once from a seeded key
 * (they make the prior of `f_d` a squared-exponential Gaussian process, as the number of features grows) and the weights `W`
 * unknown with normal priors. The coordinates enter through a known linear map, a cosine and a product with the weights,
 * which are all factors of the registry. This file also holds a generator of data from a known smooth map of known coordinates
 * (the truth that the recovered coordinates are scored against) and the alignment of recovered coordinates to the true ones,
 * which is needed because a GPLVM finds its coordinates only up to a rotation, a reflection and a monotone reparametrisation.
 */

import { OmaiError } from '../../base/index.ts'
import { orthogonalProcrustes } from '../../numerics/geometry/index.ts'
import { foldIn, type Key, normal, uniform } from '../../random/index.ts'
import {
  add,
  div,
  matmul,
  mean,
  mul,
  sign,
  sin,
  sqrt,
  sub,
  sum,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, type ModelDefinition } from '../model/index.ts'

/** The options of {@link fourierFeatures} and {@link gplvmFourierModel}. */
export interface GplvmOptions {
  /** The number of random Fourier features per output dimension (more is a closer Gaussian process and a slower run). */
  readonly features: number
  /** The number of latent coordinates per point (1 or 2 are the useful values). */
  readonly coords: number
  /** The number of observed dimensions per point. */
  readonly outputs: number
  /** The known standard deviation of the observation noise (default 0.1). */
  readonly noise?: number
  /** The length scale of the squared-exponential kernel the features approximate, in units of the coordinates (default 1). */
  readonly lengthScale?: number
}

/**
 * The random frequencies and phases of the features: the frequencies are normal with standard deviation `1 / lengthScale` (the
 * spectral density of a squared-exponential kernel) and the phases uniform on a full turn, both from one seeded key.
 *
 * @param key - The seed.
 * @param options - The number of features and coordinates, and the length scale (default 1).
 * @returns The frequencies `[coords, features]` and the phases `[features]`.
 * @throws OmaiError with code `value` for a length scale that is not positive.
 * @concept model-spec
 * @example
 * ```ts
 * import { fourierFeatures } from 'omai-engine/probprog/models'
 * import { key } from 'omai-engine/random'
 * const f = fourierFeatures(key(0), { features: 5, coords: 2 })
 * [f.frequencies.shape, f.phases.shape] // => [[2, 5], [5]]
 * ```
 */
export function fourierFeatures(
  key: Key,
  options: Pick<GplvmOptions, 'features' | 'coords' | 'lengthScale'>,
): { readonly frequencies: Value; readonly phases: Value } {
  const { features, coords, lengthScale = 1 } = options
  if (!(lengthScale > 0)) throw new OmaiError('value', 'fourierFeatures: the length scale is positive')
  return {
    frequencies: div(normal(foldIn(key, 0), [coords, features]), lengthScale),
    phases: mul(uniform(foldIn(key, 1), [features]), 2 * Math.PI),
  }
}

/**
 * A GPLVM with random Fourier features: a normal prior on the coordinates `x` of every point, a normal prior on the feature
 * weights `W` whose variance makes each output a unit-amplitude function (`2 / features`), and a Gaussian observation of every
 * output of every point. It is bound to the frequencies `omega`, the phases `phase` (from {@link fourierFeatures}) and the
 * observations `y`, a matrix with a row per point. Every factor is in the registry, so `epRoute` runs it by plain EP; the graph is loopy (a
 * coordinate is read by all the features of its point, a weight by all the points), so the beliefs are narrower than the
 * posterior they approximate.
 *
 * @param options - The numbers of features, coordinates and outputs, the noise (default 0.1) and the length scale (default 1).
 * @returns The definition; bind it to `{ omega, phase, y }`.
 * @throws OmaiError with code `value` for a count that is not a whole number from 1.
 * @concept model-spec
 * @index function A GPLVM with random Fourier features: latent coordinates per point, EP-ready.
 * @example
 * ```ts
 * import { fourierFeatures, gplvmFourierModel } from 'omai-engine/probprog/models'
 * import { key, normal } from 'omai-engine/random'
 * const options = { features: 4, coords: 1, outputs: 3 }
 * const { frequencies, phases } = fourierFeatures(key(0), options)
 * const bound = gplvmFourierModel(options).bind({ omega: frequencies, phase: phases, y: normal(key(1), [6, 3]) })
 * bound.sizes // => { point: 6, feature: 4, output: 3, coord: 1 }
 * ```
 */
export function gplvmFourierModel(options: GplvmOptions): ModelDefinition {
  const { features, coords, outputs, noise = 0.1 } = options
  for (const [name, count] of [
    ['features', features],
    ['coords', coords],
    ['outputs', outputs],
  ] as const)
    if (!(Number.isInteger(count) && count >= 1))
      throw new OmaiError('value', `gplvmFourierModel: ${name} is a whole number from 1`)
  // region gplvm
  const definition = defineModel(`gplvm-fourier-${coords}d`, (m) => {
    const point = m.range('point')
    const feature = m.range('feature', { size: features })
    const output = m.range('output', { size: outputs })
    const coord = m.range('coord', { size: coords })
    // the random frequencies and phases of the features, and the observations
    const omega = m.data('omega', { dims: [coord, feature] })
    const phase = m.data('phase', { dims: [feature] })
    const y = m.data('y', { dims: [point, output], role: 'response' })
    // the hidden coordinates of each point, and the weights of each output on each feature
    const x = m.latent('x', d.normal(0, 1), { dims: [point, coord], doc: 'the coordinates of each point' })
    const w = m.latent('W', d.normal(0, Math.sqrt(2 / features)), {
      dims: [output, feature],
      doc: 'the feature weights',
    })
    // each output is a weighted sum of cosines of a linear map of the coordinates
    const phi = x.matmul(omega).add(phase).cos()
    const f = phi.mul(w).sum(feature)
    m.observe('y', d.normal(f, noise), y)
  })
  // endregion
  return definition
}

/** The data {@link simulateGplvm} draws, with the truth the recovered coordinates are scored against. */
export interface GplvmData {
  /** The observations `[points, outputs]`: the smooth map of the coordinates, with noise. */
  readonly y: Value
  /** The true coordinates `[points, coords]`, uniform on `[-1.7, 1.7]` (about unit variance, as the prior of the model). */
  readonly latent: Value
  /** The map of the true coordinates before noise `[points, outputs]`. */
  readonly clean: Value
}

/**
 * Data from a known smooth map of known coordinates: each output is `sin(a . t + p)` of a point's coordinates `t`, with rates `a`
 * between 0.4 and 1.4 in size (either sign) and a phase `p` on a full turn, plus Gaussian noise. With one coordinate the rows
 * trace a noisy Lissajous curve in the output dimensions, with two a noisy surface. The map is known so a recovery can be scored.
 *
 * @param key - The seed.
 * @param options - `points`, `coords`, `outputs` and the `noise` standard deviation (default 0.1).
 * @returns The observations, the true coordinates and the map before noise.
 * @concept model-spec
 * @example
 * ```ts
 * import { simulateGplvm } from 'omai-engine/probprog/models'
 * import { key } from 'omai-engine/random'
 * const data = simulateGplvm(key(0), { points: 8, coords: 1, outputs: 4 })
 * [data.y.shape, data.latent.shape] // => [[8, 4], [8, 1]]
 * ```
 */
export function simulateGplvm(
  key: Key,
  options: { readonly points: number; readonly coords: number; readonly outputs: number; readonly noise?: number },
): GplvmData {
  const { points, coords, outputs, noise = 0.1 } = options
  const latent = mul(sub(uniform(foldIn(key, 0), [points, coords]), 0.5), 3.4)
  const sides = sign(sub(uniform(foldIn(key, 1), [coords, outputs]), 0.5))
  const rates = mul(add(mul(uniform(foldIn(key, 2), [coords, outputs]), 0.8), 0.3), sides)
  const phases = mul(uniform(foldIn(key, 3), [outputs]), 2 * Math.PI)
  const clean = sin(add(matmul(latent, rates), phases))
  return { y: add(clean, mul(normal(foldIn(key, 4), [points, outputs]), noise)), latent, clean }
}

/** Recovered coordinates fitted to the true ones: see {@link alignLatents}. */
export interface Alignment {
  /** The recovered coordinates after the best rotation, reflection and scale, in the frame of the true ones `[points, coords]`. */
  readonly aligned: Value
  /** The correlation of each aligned coordinate with the true one `[coords]`. */
  readonly correlation: Value
  /** The root-mean-square distance of the aligned coordinates from the true ones, over the root-mean-square size of the true ones. */
  readonly error: Value
  /** The fit itself: `aligned = (estimate - from) rotation scale + to`, so a point of the true frame maps back by the inverse. */
  readonly rotation: Value
  readonly scale: Value
  readonly from: Value
  readonly to: Value
}

/**
 * The recovered coordinates fitted to the true ones by an orthogonal Procrustes fit: both are centred, the recovered ones are
 * rotated (with a reflection if that is closer) and scaled to lie as near the true ones as a rotation allows, and moved to the
 * centre of the true ones. A GPLVM cannot tell a rotation or a reflection of its coordinates from the coordinates themselves, so
 * a score is taken after this fit.
 *
 * @param estimate - The recovered coordinates `[points, coords]`.
 * @param truth - The true coordinates `[points, coords]`.
 * @returns The aligned coordinates, the correlation of each with the truth, the relative error and the fit.
 * @concept model-spec
 * @example
 * ```ts
 * import { alignLatents } from 'omai-engine/probprog/models'
 * import { item, toFlat } from 'omai-engine/tensor'
 * // The same points turned over (a reflection) are the true ones after the fit.
 * const fit = alignLatents([[1], [2], [4]], [[-1], [-2], [-4]])
 * [toFlat(fit.correlation)[0]?.toFixed(6), item(fit.error).toFixed(6)] // => ['1.000000', '0.000000']
 * ```
 */
export function alignLatents(estimate: TensorLike, truth: TensorLike): Alignment {
  const from = mean(estimate as Value, 0, true)
  const to = mean(truth as Value, 0, true)
  const a = sub(estimate as Value, from)
  const b = sub(truth as Value, to)
  const { rotation, scale: gain } = orthogonalProcrustes(a, b)
  const scale = div(gain, sum(mul(a, a)))
  const centred = mul(matmul(a, rotation), scale)
  const spread = (z: Value): Value => sqrt(mean(mul(z, z), 0))
  const correlation = div(mean(mul(centred, b), 0), mul(spread(centred), spread(b)))
  const miss = sub(centred, b)
  return {
    aligned: add(centred, to),
    correlation,
    error: div(sqrt(mean(mul(miss, miss))), sqrt(mean(mul(b, b)))),
    rotation,
    scale,
    from,
    to,
  }
}
