/**
 * Signals for fitting neural fields, each with its truth as a function you can evaluate anywhere: `fieldImage`, a
 * procedural colour image with sharp edges and fine texture (so the spectral bias of a plain MLP, and the way an
 * encoding or a SIREN removes it, show up as blur against detail), and `signedDistance`, the exact signed distance of
 * a sphere, box, torus or a union in two or three dimensions (the zero level set is the shape; the distance has unit
 * gradient almost everywhere, which the eikonal loss uses). Both are coordinate-to-value regression data.
 */

import { maxOf, minOf } from '../base/index.ts'
import { type DatasetGenerator, type Knobs, type RegressionData, type Truth, xy } from '../data/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, toFlat, type Value } from '../tensor/index.ts'
import { stamp } from './_util.ts'

const norm = (v: readonly number[]): number => Math.hypot(...v)

/**
 * The signed distance from a point to a sphere (a circle in two dimensions): negative inside.
 *
 * @param p - The point (2 or 3 coordinates).
 * @param radius - The radius.
 * @param centre - The centre (default the origin).
 * @returns The distance.
 * @concept signed-distance-field
 * @example
 * ```ts
 * import { sdfSphere } from 'mdml-engine/datasets'
 * [sdfSphere([2, 0, 0], 0.5), sdfSphere([0, 0, 0], 0.5)] // => [1.5, -0.5]
 * ```
 */
export function sdfSphere(p: readonly number[], radius: number, centre: readonly number[] = []): number {
  return norm(p.map((v, i) => v - (centre[i] ?? 0))) - radius
}

/**
 * The exact signed distance from a point to an axis-aligned box (a rectangle in two dimensions), negative inside.
 *
 * @param p - The point.
 * @param half - The half extents along each axis.
 * @param centre - The centre (default the origin).
 * @returns The distance.
 * @concept signed-distance-field
 * @example
 * ```ts
 * import { sdfBox } from 'mdml-engine/datasets'
 * [sdfBox([2, 0], [1, 1]), sdfBox([2, 2], [1, 1]), sdfBox([0, 0], [1, 1])] // => [1, 1.4142135623730951, -1]
 * ```
 */
export function sdfBox(p: readonly number[], half: readonly number[], centre: readonly number[] = []): number {
  const q = p.map((v, i) => Math.abs(v - (centre[i] ?? 0)) - (half[i] ?? 0))
  return norm(q.map((v) => Math.max(v, 0))) + Math.min(maxOf(q), 0)
}

/**
 * The signed distance to a torus around the y axis in three dimensions (major radius `major`, tube radius `minor`),
 * or to a ring (the set of points within `minor` of the circle of radius `major`) in two.
 *
 * @param p - The point (2 or 3 coordinates).
 * @param major - The radius of the circle the tube follows.
 * @param minor - The tube's radius.
 * @returns The distance.
 * @concept signed-distance-field
 * @example
 * ```ts
 * import { sdfTorus } from 'mdml-engine/datasets'
 * [sdfTorus([0.5, 0, 0], 0.5, 0.2), sdfTorus([0, 0, 0], 0.5, 0.2)] // => [-0.2, 0.3]
 * ```
 */
export function sdfTorus(p: readonly number[], major: number, minor: number): number {
  if (p.length === 2) return Math.abs(Math.hypot(p[0] ?? 0, p[1] ?? 0) - major) - minor
  return Math.hypot(Math.hypot(p[0] ?? 0, p[2] ?? 0) - major, p[1] ?? 0) - minor
}

/**
 * The union of shapes: the smaller of their distances (exact outside, a lower bound inside).
 *
 * @param distances - The shapes' distances at a point.
 * @returns The union's distance.
 * @concept signed-distance-field
 * @example
 * ```ts
 * import { sdfUnion } from 'mdml-engine/datasets'
 * sdfUnion([0.3, -0.1, 2]) // => -0.1
 * ```
 */
export function sdfUnion(distances: readonly number[]): number {
  return minOf(distances)
}

/** The analytic shapes of {@link signedDistance}, as functions of a point in two or three dimensions. */
const SHAPES: Readonly<Record<string, (p: readonly number[]) => number>> = {
  sphere: (p) => sdfSphere(p, 0.55),
  box: (p) => sdfBox(p, [0.6, 0.4, 0.4]),
  torus: (p) => sdfTorus(p, 0.55, 0.2),
  union: (p) => sdfUnion([sdfSphere(p, 0.4, [-0.25]), sdfBox(p, [0.3, 0.3, 0.3], [0.3])]),
}

/** Apply a pointwise function of a coordinate row to a matrix value `[m, d]`, giving `[m, k]` (or `[m]` for k = 1 and `flat`). */
function pointwise(points: Value, k: number, f: (p: number[]) => readonly number[]): Value {
  const m = points.shape[0] ?? 0
  const d = points.shape[1] ?? 1
  const flat = toFlat(points)
  const out = new Float64Array(m * k)
  for (let i = 0; i < m; i++) out.set(f(Array.from(flat.slice(i * d, (i + 1) * d))), i * k)
  return fromStorage(out, k === 1 ? [m] : [m, k], 'float64')
}

/**
 * Samples of the signed distance field of a sphere, box, torus or a sphere-and-box union in two or three dimensions
 * (a ring in place of the torus in two), inside `[-1, 1]^d`. A share of the points is placed near the surface (the
 * nearest to it of a larger uniform draw), where fitting matters most; the targets are the exact signed distances, and
 * the truth is the distance function.
 *
 * @concept signed-distance-field
 * @index dataset Exact signed distance samples of sphere, box, torus or union shapes.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { signedDistance } from 'mdml-engine/datasets'
 * const g = generate(signedDistance, { knobs: { shape: 'sphere', dim: 2, n: 64 } })
 * [g.data.x.shape, g.data.y.shape] // => [[64, 2], [64]]
 * ```
 */
export const signedDistance: DatasetGenerator<RegressionData> = {
  name: 'signedDistance',
  kind: 'regression',
  description:
    'Points in [-1, 1]^d with the exact signed distance to a sphere, box, torus or union, for neural fields.',
  knobs: dict({
    n: int(8, 100_000, { label: 'points', description: 'The number of points.', default: 512 }),
    dim: choice([2, 3], { label: 'dimension', description: 'Two or three dimensions.', default: 3 }),
    shape: choice(Object.keys(SHAPES), {
      label: 'shape',
      description: 'The shape whose distance is sampled.',
      default: 'sphere',
    }),
    nearSurface: real(0, 1, {
      label: 'share near the surface',
      description: 'The share of points taken from the surface’s neighbourhood.',
      default: 0.5,
    }),
  }),
  generate: (k: Key, knobs: Knobs) => {
    const n = knobs['n'] as number
    const dim = knobs['dim'] as number
    const share = knobs['nearSurface'] as number
    const shape = SHAPES[knobs['shape'] as string] as (p: readonly number[]) => number
    const near = Math.round(n * share)
    const pool = Math.max(near, 1) * 8
    const draw = (key: Key, m: number): number[][] => {
      const u = uniform(key, [m, dim], { low: -1, high: 1 }).data
      return Array.from({ length: m }, (_, i) => Array.from({ length: dim }, (__, j) => u[i * dim + j] as number))
    }
    const candidates = draw(foldIn(k, 0), pool)
    const closest = candidates
      .map((p, i) => [Math.abs(shape(p)), i] as const)
      .sort((a, b) => a[0] - b[0] || a[1] - b[1])
      .slice(0, near)
      .map(([, i]) => candidates[i] as number[])
    const points = [...closest, ...draw(foldIn(k, 1), n - near)]
    const x = fromStorage(Float64Array.from(points.flat()), [n, dim], 'float64')
    const sdf = (v: Value): Value => pointwise(v, 1, (p) => [shape(p)])
    const truth: Truth = {
      regressionFunction: sdf,
      noiseScale: 0,
      bayesError: 0,
      extra: { shape: knobs['shape'] as string, dim, bounds: [-1, 1], eikonal: true },
    }
    return { data: xy(x, sdf(x), stamp('signedDistance', signedDistance.description, knobs)), truth }
  },
}

type Rgb = readonly [r: number, g: number, b: number]

const BACKGROUND: Rgb = [0.1, 0.14, 0.32]

/** The colour of the {@link fieldImage} patterns at a point `(u, v)` of `[-1, 1]^2`, v downwards. */
function paint(pattern: string, u: number, v: number, texture: number, frequency: number): Rgb {
  const disc = (cu: number, cv: number, r: number): boolean => Math.hypot(u - cu, v - cv) < r
  const cell = (k: number, a: number, b: number): number => (Math.floor(a * k) + Math.floor(b * k)) & 1
  switch (pattern) {
    case 'checker':
      return cell(frequency, u, v) === 0 ? [0.92, 0.92, 0.92] : [0.1, 0.1, 0.12]
    case 'stripes': {
      const s = Math.sin(2 * Math.PI * frequency * (u + 0.35 * v)) > 0 ? 1 : 0
      return [0.15 + 0.7 * s, 0.2 + 0.4 * s, 0.7 - 0.5 * s]
    }
    case 'discs': {
      if (disc(-0.4, -0.35, 0.38)) return [0.95, 0.45, 0.15]
      if (disc(0.45, -0.4, 0.22)) return [0.95, 0.95, 0.95]
      if (disc(0.1, 0.5, 0.3)) return [0.2, 0.75, 0.4]
      return BACKGROUND
    }
    default: {
      // Mixed: a textured background, a checkered square, a striped disc and a small white disc, painted in that order.
      const t = texture * 0.5 * Math.sin(2 * Math.PI * frequency * u) * Math.sin(2 * Math.PI * frequency * v)
      let c: Rgb = [BACKGROUND[0] + t, BACKGROUND[1] + t, BACKGROUND[2] + t + 0.12 * (u + 1)]
      if (u > 0.15 && u < 0.85 && v > 0.1 && v < 0.8)
        c = cell(5, (u - 0.15) / 0.7, (v - 0.1) / 0.7) === 0 ? [0.9, 0.9, 0.9] : [0.15, 0.6, 0.35]
      if (disc(-0.4, -0.35, 0.38)) {
        const s = Math.sin(2 * Math.PI * frequency * 1.5 * (u + v)) > 0 ? 1 : 0
        c = [0.95 - 0.25 * s, 0.45 + 0.4 * s, 0.15 + 0.1 * s]
      }
      if (disc(0.45, -0.5, 0.18)) c = [1, 1, 1]
      return c
    }
  }
}

/**
 * A procedural colour image as coordinate-to-colour regression data: the pixel centres of a `side` by `side` grid over
 * `[-1, 1]^2` (u to the right, v downwards, row by row) with the colour there, in `[0, 1]`. Its patterns have sharp
 * edges and fine texture on purpose: a plain ReLU MLP blurs them (spectral bias), a coordinate encoding or a SIREN does
 * not. The truth is the continuous image, `regressionFunction` on any `[m, 2]` coordinates, so a fit can be judged on a
 * finer grid than it saw.
 *
 * @concept spectral-bias
 * @index dataset Procedural colour image as coordinate-to-colour data, with its continuous truth.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { fieldImage } from 'mdml-engine/datasets'
 * const g = generate(fieldImage, { knobs: { side: 8 } })
 * [g.data.x.shape, g.data.y.shape] // => [[64, 2], [64, 3]]
 * ```
 */
export const fieldImage: DatasetGenerator<RegressionData> = {
  name: 'fieldImage',
  kind: 'regression',
  description: 'A procedural image with sharp edges and fine texture, as coordinates and colours, for neural fields.',
  knobs: dict({
    side: int(2, 512, { label: 'side', description: 'The image is side by side pixels.', default: 32 }),
    pattern: choice(['mixed', 'checker', 'stripes', 'discs'], {
      label: 'pattern',
      description: 'The picture: mixed, checker, stripes or discs.',
      default: 'mixed',
    }),
    texture: real(0, 1, {
      label: 'texture',
      description: 'The amplitude of the fine texture of the mixed pattern’s background.',
      default: 0.5,
    }),
    frequency: real(1, 24, {
      label: 'frequency',
      description: 'Cycles of texture, stripes or checker cells per unit length.',
      default: 6,
    }),
  }),
  generate: (_k: Key, knobs: Knobs) => {
    const side = knobs['side'] as number
    const pattern = knobs['pattern'] as string
    const texture = knobs['texture'] as number
    const frequency = knobs['frequency'] as number
    const image = (points: Value): Value =>
      pointwise(points, 3, (p) => paint(pattern, p[0] ?? 0, p[1] ?? 0, texture, frequency))
    const coords = new Float64Array(side * side * 2)
    for (let j = 0; j < side; j++)
      for (let i = 0; i < side; i++) {
        coords[2 * (j * side + i)] = -1 + (2 * i + 1) / side
        coords[2 * (j * side + i) + 1] = -1 + (2 * j + 1) / side
      }
    const x = fromStorage(coords, [side * side, 2], 'float64')
    const truth: Truth = {
      regressionFunction: image,
      noiseScale: 0,
      bayesError: 0,
      extra: { side, pattern, bounds: [-1, 1], channels: 3 },
    }
    return { data: xy(x, image(x), stamp('fieldImage', fieldImage.description, knobs)), truth }
  },
}
