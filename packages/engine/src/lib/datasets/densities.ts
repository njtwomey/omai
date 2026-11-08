/**
 * Two-dimensional target densities with exact, differentiable log densities and samplers, for testing MCMC, normalising
 * flows and diffusion models: rings, Gaussian mixtures on a circle or grid, the pinwheel, the banana, Neal's funnel and
 * the FFJORD checkerboard. `densitySamples` turns any of them into a labelled dataset.
 */

import { LOG_2PI, maxOf } from '../base/index.ts'
import { type ClassificationData, type DatasetGenerator, xyClasses } from '../data/index.ts'
import { erf } from '../numerics/special/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import {
  add,
  cos,
  div,
  equal,
  exp,
  floor,
  full,
  greaterEqual,
  item,
  lessEqual,
  log,
  logicalAnd,
  logsumexp,
  mod,
  mul,
  sin,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  type Value,
  where,
} from '../tensor/index.ts'
import { column, labels, matrix, stamp } from './_util.ts'

/**
 * A density with an exact, differentiable log density and a sampler, for testing samplers, normalising flows and
 * diffusion models. `logProb` maps rows `[n, d]` to `[n]`; `sample` returns rows and the mixture component (or spoke,
 * or ring) each came from.
 *
 * @concept densities
 */
export interface Density {
  readonly name: string
  readonly dim: number
  /** The exact log density (normalised) of each row. */
  logProb(x: Value): Value
  sample(key: Key, n: number): { readonly x: Tensor; readonly component: readonly number[] }
  /** A window that holds nearly all the mass, for plots. */
  readonly bounds: readonly (readonly [number, number])[]
  readonly nComponents: number
}

/** `log N(x; m, s^2)` elementwise. */
function logNormal(x: Value, m: Value | number, s: Value | number): Value {
  const z = div(sub(x, m), s)
  return sub(mul(add(square(z), LOG_2PI), -0.5), typeof s === 'number' ? Math.log(s) : log(s))
}

/** Options of {@link ringDensity}. */
export interface RingOptions {
  /** Radii of the rings (default `[2]`); each ring gets equal mass. */
  readonly radii?: readonly number[]
  /** Radial standard deviation (default 0.2). */
  readonly sigma?: number
}

/**
 * Rings: `p(x) ∝ exp(-(|x| - R)² / (2σ²))` around each radius, normalised exactly in the plane
 * (`Z = 2π σ (σ e^(-R²/2σ²) + R √(π/2) (1 + erf(R / (σ√2))))`), a standard target for flows and samplers.
 *
 * @param options - Radii and radial spread.
 * @returns The density.
 * @example
 * ```ts
 * import { ringDensity } from 'omai-engine/datasets'
 * ringDensity({ radii: [1, 3] }).nComponents // => 2
 * ```
 */
export function ringDensity(options: RingOptions = {}): Density {
  const radii = options.radii ?? [2]
  const s = options.sigma ?? 0.2
  const logZ = radii.map((r) => {
    const z =
      2 *
      Math.PI *
      s *
      (s * Math.exp((-r * r) / (2 * s * s)) + r * Math.sqrt(Math.PI / 2) * (1 + item(erf(r / (s * Math.SQRT2)))))
    return Math.log(z)
  })
  const k = radii.length
  const outer = maxOf(radii) + 4 * s
  return {
    name: 'rings',
    dim: 2,
    nComponents: k,
    bounds: [
      [-outer, outer],
      [-outer, outer],
    ],
    logProb: (x) => {
      const r = sqrt(sum(square(x), 1))
      const parts = radii.map((rad, c) => sub(mul(square(sub(r, rad)), -0.5 / (s * s)), (logZ[c] ?? 0) + Math.log(k)))
      return logsumexp(stack(parts, 1), 1)
    },
    sample: (key, n) => {
      const comp = Array.from(uniform(foldIn(key, 0), [n]).data, (u) => Math.min(k - 1, Math.floor(u * k)))
      const theta = uniform(foldIn(key, 1), [n], { low: 0, high: 2 * Math.PI }).data
      // The radius has density ∝ r exp(-(r - R)^2 / 2s^2) on r > 0: rejection from N(R, s^2) with acceptance
      // probability r / (R + 8 s), in vectorised rounds.
      const rad = new Float64Array(n)
      let pending = Array.from({ length: n }, (_, i) => i)
      for (let round = 0; pending.length > 0 && round < 1000; round++) {
        const z = normal(foldIn(key, 10 + 2 * round), [pending.length]).data
        const u = uniform(foldIn(key, 11 + 2 * round), [pending.length]).data
        const next: number[] = []
        for (const [j, i] of pending.entries()) {
          const centre = radii[comp[i] ?? 0] ?? 1
          const r = centre + s * (z[j] ?? 0)
          if (r > 0 && (u[j] ?? 1) * (centre + 8 * s) < r) rad[i] = r
          else next.push(i)
        }
        pending = next
      }
      const out = new Float64Array(2 * n)
      for (let i = 0; i < n; i++) {
        out[2 * i] = (rad[i] ?? 0) * Math.cos(theta[i] ?? 0)
        out[2 * i + 1] = (rad[i] ?? 0) * Math.sin(theta[i] ?? 0)
      }
      return { x: matrix(out, n, 2), component: comp }
    },
  }
}

/** Options of {@link gaussianMixtureDensity}. */
export interface MixtureDensityOptions {
  /** `circle` (default): `count` Gaussians evenly on a circle of radius `radius`; `grid`: a `count` by `count` grid
   * with spacing `radius`. */
  readonly layout?: 'circle' | 'grid'
  readonly count?: number
  readonly radius?: number
  readonly sigma?: number
}

/**
 * Equal-weight isotropic Gaussian mixtures on a circle (8 modes, the usual mode-collapse test) or a grid (5 by 5),
 * with the exact log density.
 *
 * @param options - Layout, number of modes, radius or spacing, and spread.
 * @returns The density, with `means` for reference.
 * @example
 * ```ts
 * import { gaussianMixtureDensity } from 'omai-engine/datasets'
 * gaussianMixtureDensity({ layout: 'grid', count: 3 }).nComponents // => 9
 * ```
 */
export function gaussianMixtureDensity(
  options: MixtureDensityOptions = {},
): Density & { readonly means: readonly (readonly number[])[] } {
  const layout = options.layout ?? 'circle'
  const count = options.count ?? (layout === 'circle' ? 8 : 5)
  const radius = options.radius ?? (layout === 'circle' ? 4 : 2)
  const s = options.sigma ?? (layout === 'circle' ? 0.3 : 0.2)
  const means =
    layout === 'circle'
      ? Array.from({ length: count }, (_, i) => [
          radius * Math.cos((2 * Math.PI * i) / count),
          radius * Math.sin((2 * Math.PI * i) / count),
        ])
      : Array.from({ length: count * count }, (_, i) => [
          radius * ((i % count) - (count - 1) / 2),
          radius * (Math.floor(i / count) - (count - 1) / 2),
        ])
  const k = means.length
  let extent = 0
  for (const m of means) extent = Math.max(extent, Math.abs(m[0] ?? 0), Math.abs(m[1] ?? 0))
  extent += 4 * s
  return {
    name: `mixture-${layout}`,
    dim: 2,
    nComponents: k,
    means,
    bounds: [
      [-extent, extent],
      [-extent, extent],
    ],
    logProb: (x) => {
      const a = column(x, 0)
      const b = column(x, 1)
      const parts = means.map((m) => add(logNormal(a, m[0] ?? 0, s), logNormal(b, m[1] ?? 0, s)))
      return sub(logsumexp(stack(parts, 1), 1), Math.log(k))
    },
    sample: (key, n) => {
      const comp = Array.from(uniform(foldIn(key, 0), [n]).data, (u) => Math.min(k - 1, Math.floor(u * k)))
      const z = normal(foldIn(key, 1), [n, 2]).data
      return {
        x: matrix(
          Float64Array.from(z, (v, i) => (means[comp[Math.floor(i / 2)] ?? 0]?.[i % 2] ?? 0) + s * v),
          n,
          2,
        ),
        component: comp,
      }
    },
  }
}

/** Options of {@link pinwheelDensity}. */
export interface PinwheelOptions {
  /** The number of arms (default 5). */
  readonly arms?: number
  /** Spread along an arm (default 0.3) and across it (default 0.05). */
  readonly radialStd?: number
  readonly tangentialStd?: number
  /** Swirl: the rotation in radians per unit radius (default 1). */
  readonly rate?: number
}

/**
 * A pinwheel (after Johnson et al., 2016): `arms` elongated Gaussians on evenly spaced spokes at distance 1, swirled by
 * the rotation `x -> R(rate |x|) x`. The swirl preserves area, so the log density is exact: the mixture's log density
 * at the unswirled point.
 *
 * @param options - Arms, spreads and swirl rate.
 * @returns The density.
 * @example
 * ```ts
 * import { pinwheelDensity } from 'omai-engine/datasets'
 * pinwheelDensity({ arms: 3 }).nComponents // => 3
 * ```
 */
export function pinwheelDensity(options: PinwheelOptions = {}): Density {
  const arms = options.arms ?? 5
  const sr = options.radialStd ?? 0.3
  const st = options.tangentialStd ?? 0.05
  const rate = options.rate ?? 1
  const angles = Array.from({ length: arms }, (_, i) => (2 * Math.PI * i) / arms)
  const unswirl = (x: Value): [Value, Value] => {
    const a = column(x, 0)
    const b = column(x, 1)
    const t = mul(sqrt(add(add(square(a), square(b)), 1e-300)), -rate)
    return [sub(mul(a, cos(t)), mul(b, sin(t))), add(mul(a, sin(t)), mul(b, cos(t)))]
  }
  return {
    name: 'pinwheel',
    dim: 2,
    nComponents: arms,
    bounds: [
      [-3, 3],
      [-3, 3],
    ],
    logProb: (x) => {
      const [a, b] = unswirl(x)
      const parts = angles.map((th) => {
        // Coordinates along and across the spoke.
        const u = add(mul(a, Math.cos(th)), mul(b, Math.sin(th)))
        const v = sub(mul(b, Math.cos(th)), mul(a, Math.sin(th)))
        return add(logNormal(u, 1, sr), logNormal(v, 0, st))
      })
      return sub(logsumexp(stack(parts, 1), 1), Math.log(arms))
    },
    sample: (key, n) => {
      const comp = Array.from(uniform(foldIn(key, 0), [n]).data, (u) => Math.min(arms - 1, Math.floor(u * arms)))
      const z = normal(foldIn(key, 1), [n, 2]).data
      const out = new Float64Array(2 * n)
      for (let i = 0; i < n; i++) {
        const th = angles[comp[i] ?? 0] ?? 0
        const u = 1 + sr * (z[2 * i] ?? 0)
        const v = st * (z[2 * i + 1] ?? 0)
        const px = u * Math.cos(th) - v * Math.sin(th)
        const py = u * Math.sin(th) + v * Math.cos(th)
        const t = rate * Math.hypot(px, py)
        out[2 * i] = px * Math.cos(t) - py * Math.sin(t)
        out[2 * i + 1] = px * Math.sin(t) + py * Math.cos(t)
      }
      return { x: matrix(out, n, 2), component: comp }
    },
  }
}

/** Options of {@link bananaDensity}. */
export interface BananaOptions {
  /** Standard deviation of the first coordinate (default 2). */
  readonly a?: number
  /** Curvature (default 0.5). */
  readonly b?: number
}

/**
 * The banana (Haario, Saksman and Tamminen's twisted Gaussian): `x0 ~ N(0, a²)`, `x1 | x0 ~ N(b (x0² - a²), 1)`, a
 * curved, strongly correlated target for MCMC.
 *
 * @param options - Width and curvature.
 * @returns The density.
 * @example
 * ```ts
 * import { bananaDensity } from 'omai-engine/datasets'
 * bananaDensity().dim // => 2
 * ```
 */
export function bananaDensity(options: BananaOptions = {}): Density {
  const a = options.a ?? 2
  const b = options.b ?? 0.5
  return {
    name: 'banana',
    dim: 2,
    nComponents: 1,
    bounds: [
      [-4 * a, 4 * a],
      [-b * a * a - 4, b * 15 * a * a + 4],
    ],
    logProb: (x) => {
      const x0 = column(x, 0)
      const x1 = column(x, 1)
      return add(logNormal(x0, 0, a), logNormal(sub(x1, mul(sub(square(x0), a * a), b)), 0, 1))
    },
    sample: (key, n) => {
      const z = normal(key, [n, 2]).data
      const out = new Float64Array(2 * n)
      for (let i = 0; i < n; i++) {
        const x0 = a * (z[2 * i] ?? 0)
        out[2 * i] = x0
        out[2 * i + 1] = b * (x0 * x0 - a * a) + (z[2 * i + 1] ?? 0)
      }
      return { x: matrix(out, n, 2), component: new Array<number>(n).fill(0) }
    },
  }
}

/**
 * Neal's funnel in two dimensions: `v ~ N(0, scale²)` and `x | v ~ N(0, e^v)`, whose neck defeats fixed-step-size
 * samplers. Rows are `(v, x)`.
 *
 * @param scale - The standard deviation of `v` (default 3).
 * @returns The density.
 * @example
 * ```ts
 * import { funnelDensity } from 'omai-engine/datasets'
 * funnelDensity().name // => 'funnel'
 * ```
 */
export function funnelDensity(scale = 3): Density {
  return {
    name: 'funnel',
    dim: 2,
    nComponents: 1,
    bounds: [
      [-3 * scale, 3 * scale],
      [-20, 20],
    ],
    logProb: (x) => {
      const v = column(x, 0)
      const u = column(x, 1)
      return add(logNormal(v, 0, scale), logNormal(u, 0, exp(mul(v, 0.5))))
    },
    sample: (key, n) => {
      const z = normal(key, [n, 2]).data
      const out = new Float64Array(2 * n)
      for (let i = 0; i < n; i++) {
        const v = scale * (z[2 * i] ?? 0)
        out[2 * i] = v
        out[2 * i + 1] = Math.exp(v / 2) * (z[2 * i + 1] ?? 0)
      }
      return { x: matrix(out, n, 2), component: new Array<number>(n).fill(0) }
    },
  }
}

/**
 * The checkerboard of FFJORD and related flow papers: uniform on the dark squares of a 4 by 4 board on `[-4, 4)²`
 * (density 1/32 there, 0 elsewhere, so the log density is `-log 32` or `-Infinity`).
 *
 * @returns The density.
 * @example
 * ```ts
 * import { checkerboardDensity } from 'omai-engine/datasets'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * toFlat(checkerboardDensity().logProb(tensor([[0.5, 0.5], [-0.5, 0.5]]))) // => [-3.4657359027997265, -Infinity]
 * ```
 */
export function checkerboardDensity(): Density {
  const cellOf = (v: Value): Value => floor(mul(add(v, 4), 0.5))
  return {
    name: 'checkerboard',
    dim: 2,
    nComponents: 1,
    bounds: [
      [-4, 4],
      [-4, 4],
    ],
    logProb: (x) => {
      const cx = cellOf(column(x, 0))
      const cy = cellOf(column(x, 1))
      const onBoard = logicalAnd(
        logicalAnd(greaterEqual(cx, 0), lessEqual(cx, 3)),
        logicalAnd(greaterEqual(cy, 0), lessEqual(cy, 3)),
      )
      const dark = logicalAnd(onBoard, equal(mod(add(cx, cy), 2), 0))
      const n = x.shape[0] ?? 0
      return where(dark, full([n], -Math.log(32)), full([n], Number.NEGATIVE_INFINITY))
    },
    sample: (key, n) => {
      const u = uniform(foldIn(key, 0), [n, 2]).data
      const pick = uniform(foldIn(key, 1), [n]).data
      const out = new Float64Array(2 * n)
      for (let i = 0; i < n; i++) {
        // The 8 dark cells: (col + row) even; choose one, then a uniform point inside it.
        const c = Math.min(7, Math.floor((pick[i] ?? 0) * 8))
        const row = Math.floor(c / 2)
        const col = 2 * (c % 2) + (row % 2)
        out[2 * i] = -4 + 2 * col + 2 * (u[2 * i] ?? 0)
        out[2 * i + 1] = -4 + 2 * row + 2 * (u[2 * i + 1] ?? 0)
      }
      return { x: matrix(out, n, 2), component: new Array<number>(n).fill(0) }
    },
  }
}

const DENSITIES: Readonly<Record<string, () => Density>> = {
  rings: () => ringDensity({ radii: [1, 2.5] }),
  mixture: () => gaussianMixtureDensity(),
  grid: () => gaussianMixtureDensity({ layout: 'grid' }),
  pinwheel: () => pinwheelDensity(),
  banana: () => bananaDensity(),
  funnel: () => funnelDensity(),
  checkerboard: () => checkerboardDensity(),
}

/**
 * Samples from one of the two-dimensional densities, labelled by the component (ring, mode or arm) each came from,
 * with the exact log density as `truth.logProb` and the density object in `truth.extra.density`.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { densitySamples } from 'omai-engine/datasets'
 * generate(densitySamples, { knobs: { density: 'pinwheel', n: 50 } }).data.classes.length // => 5
 * ```
 */
export const densitySamples: DatasetGenerator<ClassificationData> = {
  name: 'densitySamples',
  kind: 'classification',
  description:
    'Samples from a two-dimensional density with an exact log density (rings, mixtures, pinwheel, banana, funnel, checkerboard).',
  knobs: dict({
    n: int(1, 100_000, { label: 'samples', description: 'The number of points.', default: 500 }),
    density: choice(['rings', 'mixture', 'grid', 'pinwheel', 'banana', 'funnel', 'checkerboard'], {
      label: 'density',
      description: 'The two-dimensional density sampled.',
      default: 'mixture',
    }),
    noise: real(0, 1, {
      description: 'The standard deviation of extra isotropic Gaussian noise.',
      default: 0,
      label: 'extra isotropic noise',
    }),
  }),
  generate: (k, knobs) => {
    const make = DENSITIES[knobs['density'] as string] ?? DENSITIES['mixture']
    const density = (make as () => Density)()
    const { x, component } = density.sample(foldIn(k, 0), knobs['n'] as number)
    const noise = knobs['noise'] as number
    const z = normal(foldIn(k, 1), x.shape).data
    const noisy =
      noise > 0
        ? matrix(
            Float64Array.from(x.data, (v, i) => v + noise * (z[i] ?? 0)),
            x.shape[0] ?? 0,
            2,
          )
        : x
    const data = xyClasses(noisy, labels(component), {
      ...stamp('densitySamples', density.name, knobs),
      classes: Array.from({ length: density.nComponents }, (_, i) => String(i)),
    })
    return {
      data,
      truth: {
        ...(noise > 0 ? {} : { logProb: (v: Value) => density.logProb(v) }),
        clusters: labels(component),
        extra: { density },
      },
    }
  },
}
