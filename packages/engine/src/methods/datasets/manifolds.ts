import { type DatasetGenerator, type Generated, type Knobs, type RegressionData, xy } from '../../lib/data/index.ts'
import { foldIn, type Key, normal, randint, uniform } from '../../lib/random/index.ts'
import { bool, dict, int, real } from '../../lib/space/index.ts'
import { asTensor, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { matrix, stamp, vector } from './_util.ts'

/**
 * The Swiss roll embedding of intrinsic coordinates: `(t cos t, height, t sin t)` (sklearn's make_swiss_roll).
 *
 * @param t - Positions along the roll.
 * @param height - Positions across it.
 * @returns The points `[n, 3]`.
 * @example
 * ```ts
 * import { swissRollEmbedding } from 'whitebox-engine/methods/datasets'
 * swissRollEmbedding([Math.PI], [2]).data[1] // => 2
 * ```
 */
export function swissRollEmbedding(t: TensorLike, height: TensorLike): Tensor {
  const tt = toFlat(asTensor(t))
  const hh = toFlat(asTensor(height))
  return matrix(
    tt.flatMap((v, i) => [v * Math.cos(v), hh[i] ?? 0, v * Math.sin(v)]),
    tt.length,
    3,
  )
}

/**
 * The S-curve embedding of intrinsic coordinates: `(sin t, height, sign(t) (cos t - 1))` (sklearn's make_s_curve).
 *
 * @param t - Positions along the curve, in `[-3 pi / 2, 3 pi / 2]`.
 * @param height - Positions across it.
 * @returns The points `[n, 3]`.
 * @example
 * ```ts
 * import { sCurveEmbedding } from 'whitebox-engine/methods/datasets'
 * Array.from(sCurveEmbedding([0], [1]).data) // => [0, 1, 0]
 * ```
 */
export function sCurveEmbedding(t: TensorLike, height: TensorLike): Tensor {
  const tt = toFlat(asTensor(t))
  const hh = toFlat(asTensor(height))
  return matrix(
    tt.flatMap((v, i) => [Math.sin(v), hh[i] ?? 0, Math.sign(v) * (Math.cos(v) - 1)]),
    tt.length,
    3,
  )
}

function manifoldData(
  name: string,
  description: string,
  knobs: Knobs,
  k: Key,
  clean: Tensor,
  colour: readonly number[],
  coordinates: readonly (readonly number[])[],
  latentNames: readonly string[],
  extra: Readonly<Record<string, unknown>> = {},
): Generated<RegressionData> {
  const noise = knobs['noise'] as number
  const [n = 0, d = 3] = clean.shape
  const z = normal(foldIn(k, 50), [n, d]).data
  const x = matrix(
    Float64Array.from(clean.data as ArrayLike<number>, (v, i) => v + noise * (z[i] ?? 0)),
    n,
    d,
  )
  const latents: Record<string, unknown> = {
    coordinates: matrix(coordinates.flat(), n, coordinates[0]?.length ?? 0),
    clean,
  }
  for (const [j, nm] of latentNames.entries()) latents[nm] = vector(coordinates.map((c) => c[j] ?? 0))
  return {
    data: xy(x, vector(colour), {
      ...stamp(name, description, knobs),
      featureNames: ['x', 'y', 'z'].slice(0, d),
      targetNames: [latentNames[0] ?? 't'],
    }),
    truth: { latents, noiseScale: noise, extra: { intrinsicDimension: coordinates[0]?.length ?? 0, ...extra } },
  }
}

const n = int(1, 100_000, { default: 100, label: 'samples' })
const noise = real(0, 5, { default: 0, label: 'noise standard deviation' })

/**
 * The Swiss roll (sklearn.datasets.make_swiss_roll), optionally with a hole: a rolled-up rectangle in 3-D with the
 * rectangle coordinates `t` (also the target, sklearn's colour) and `height` as `truth.latents`.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { swissRoll } from 'whitebox-engine/methods/datasets'
 * generate(swissRoll, { knobs: { n: 10 } }).data.x.shape // => [10, 3]
 * ```
 */
export const swissRoll: DatasetGenerator<RegressionData> = {
  name: 'swissRoll',
  kind: 'regression',
  description: 'A rolled-up rectangle in three dimensions (sklearn make_swiss_roll), optionally with a hole.',
  knobs: dict({ n, noise, hole: bool({ default: false, label: 'cut a hole' }) }),
  generate: (k, knobs) => {
    const m = knobs['n'] as number
    let t: number[]
    let h: number[]
    if (knobs['hole'] === true) {
      const corners: [number, number][] = []
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) if (!(i === 1 && j === 1)) corners.push([Math.PI * (1.5 + i), j * 7])
      const which = randint(foldIn(k, 0), [m], 0, 8).data
      const u = uniform(foldIn(k, 1), [2, m]).data
      t = Array.from({ length: m }, (_, i) => (corners[which[i] ?? 0]?.[0] ?? 0) + (u[i] ?? 0) * Math.PI)
      h = Array.from({ length: m }, (_, i) => (corners[which[i] ?? 0]?.[1] ?? 0) + (u[m + i] ?? 0) * 7)
    } else {
      t = Array.from(uniform(foldIn(k, 0), [m]).data, (u) => 1.5 * Math.PI * (1 + 2 * u))
      h = Array.from(uniform(foldIn(k, 1), [m]).data, (u) => 21 * u)
    }
    return manifoldData(
      'swissRoll',
      swissRoll.description,
      knobs,
      k,
      swissRollEmbedding(t, h),
      t,
      t.map((v, i) => [v, h[i] ?? 0]),
      ['t', 'height'],
    )
  },
}

/**
 * The S-curve (sklearn.datasets.make_s_curve) with its intrinsic coordinates `t` (the target) and `height`.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { sCurve } from 'whitebox-engine/methods/datasets'
 * generate(sCurve).truth.extra?.['intrinsicDimension'] // => 2
 * ```
 */
export const sCurve: DatasetGenerator<RegressionData> = {
  name: 'sCurve',
  kind: 'regression',
  description: 'An S-shaped surface in three dimensions (sklearn make_s_curve).',
  knobs: dict({ n, noise }),
  generate: (k, knobs) => {
    const m = knobs['n'] as number
    const t = Array.from(uniform(foldIn(k, 0), [m]).data, (u) => 3 * Math.PI * (u - 0.5))
    const h = Array.from(uniform(foldIn(k, 1), [m]).data, (u) => 2 * u)
    return manifoldData(
      'sCurve',
      sCurve.description,
      knobs,
      k,
      sCurveEmbedding(t, h),
      t,
      t.map((v, i) => [v, h[i] ?? 0]),
      ['t', 'height'],
    )
  },
}

/**
 * The severed sphere of scikit-learn's manifold-learning comparison: points on the unit sphere with the poles and a
 * slice removed, so it can be unrolled; latents are the polar angle `theta` and the azimuth `phi` (the target).
 * Rows are drawn until `n` survive the cuts.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { severedSphere } from 'whitebox-engine/methods/datasets'
 * generate(severedSphere, { knobs: { n: 50 } }).data.x.shape // => [50, 3]
 * ```
 */
export const severedSphere: DatasetGenerator<RegressionData> = {
  name: 'severedSphere',
  kind: 'regression',
  description: 'The unit sphere without its poles and with a slice removed.',
  knobs: dict({ n, noise }),
  generate: (k, knobs) => {
    const m = knobs['n'] as number
    const phi: number[] = []
    const theta: number[] = []
    for (let round = 0; phi.length < m; round++) {
      const p = uniform(foldIn(k, 2 * round), [2 * m]).data
      const q = uniform(foldIn(k, 2 * round + 1), [2 * m]).data
      for (let i = 0; i < 2 * m && phi.length < m; i++) {
        const a = (p[i] ?? 0) * (2 * Math.PI - 0.55)
        const b = (q[i] ?? 0) * Math.PI
        if (b < Math.PI - Math.PI / 8 && b > Math.PI / 8) {
          phi.push(a)
          theta.push(b)
        }
      }
    }
    const clean = matrix(
      theta.flatMap((t, i) => [Math.sin(t) * Math.cos(phi[i] ?? 0), Math.sin(t) * Math.sin(phi[i] ?? 0), Math.cos(t)]),
      m,
      3,
    )
    return manifoldData(
      'severedSphere',
      severedSphere.description,
      knobs,
      k,
      clean,
      phi,
      phi.map((p, i) => [p, theta[i] ?? 0]),
      ['phi', 'theta'],
    )
  },
}

/**
 * A torus with major radius `R` and minor radius `r`: angles `u` (around the hole, the target) and `v` (around the
 * tube) uniform, `((R + r cos v) cos u, (R + r cos v) sin u, r sin v)`. The angles are `truth.latents`.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { torus } from 'whitebox-engine/methods/datasets'
 * generate(torus, { knobs: { n: 20 } }).data.featureNames // => ['x', 'y', 'z']
 * ```
 */
export const torus: DatasetGenerator<RegressionData> = {
  name: 'torus',
  kind: 'regression',
  description: 'Points on a torus with uniform angles.',
  knobs: dict({
    n,
    noise,
    majorRadius: real(0.1, 100, { default: 2, label: 'R' }),
    minorRadius: real(0.01, 100, { default: 0.5, label: 'r' }),
  }),
  generate: (k, knobs) => {
    const m = knobs['n'] as number
    const big = knobs['majorRadius'] as number
    const small = knobs['minorRadius'] as number
    const u = Array.from(uniform(foldIn(k, 0), [m]).data, (a) => 2 * Math.PI * a)
    const v = Array.from(uniform(foldIn(k, 1), [m]).data, (a) => 2 * Math.PI * a)
    const clean = matrix(
      u.flatMap((a, i) => {
        const b = v[i] ?? 0
        return [
          (big + small * Math.cos(b)) * Math.cos(a),
          (big + small * Math.cos(b)) * Math.sin(a),
          small * Math.sin(b),
        ]
      }),
      m,
      3,
    )
    return manifoldData(
      'torus',
      torus.description,
      knobs,
      k,
      clean,
      u,
      u.map((a, i) => [a, v[i] ?? 0]),
      ['u', 'v'],
      {
        majorRadius: big,
        minorRadius: small,
      },
    )
  },
}
