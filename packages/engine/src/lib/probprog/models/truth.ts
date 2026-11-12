/**
 * Truths built from the validation kernel's oracles: `gridTruth` runs `gridPosterior` on a program and returns its marginals
 * and evidence as a `BenchmarkTruth`; `mixtureOracle` is the marginal of a finite mixture of normals (a CDF and a
 * quantile function from a list of weighted components), which is what a conditionally Gaussian quantity (a school's effect
 * given the hyperparameters, a bimodal target) has once the weights are known; `samplesOracle` is the oracle of iid draws
 * from a target whose sampler is exact.
 */

import { ndtr } from '../../numerics/special/index.ts'
import { add, arange, div, item, mul, toFlat } from '../../tensor/index.ts'
import type { Model } from '../primitives.ts'
import {
  type GridOptions,
  type GridPosterior,
  gridPosterior,
  type MarginalOracle,
  weightedOracle,
} from '../validation/index.ts'
import type { BenchmarkTruth } from './benchmark.ts'

/**
 * The truth of a small program by quadrature: its marginals on the constrained scale, and the evidence, from
 * `gridPosterior` (at most four coordinates).
 *
 * @param program - The program (a compiled spec with its data).
 * @param options - Grid options (points per axis, refinements, an explicit box) and a `quantities` filter.
 * @returns The truth, of kind `quadrature`.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { gridTruth } from 'omai-engine/probprog/models'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1.2) }
 * Math.abs((gridTruth(model).marginals['m']?.mean ?? 0) - 0.6) < 1e-6 // => true
 * ```
 */
export function gridTruth(
  program: Model,
  options: GridOptions & { readonly quantities?: readonly string[] } = {},
): BenchmarkTruth {
  const grid = gridPosterior(program, undefined, options)
  const marginals: Record<string, MarginalOracle> = {}
  for (const name of options.quantities ?? grid.names) marginals[name] = grid.marginal(name)
  return {
    kind: 'quadrature',
    source: `grid quadrature of the posterior density, ${grid.points} points per axis over ${grid.dim} coordinate(s) (${grid.evaluations} density evaluations, evidence error ${grid.evidenceError.toExponential(1)})`,
    logEvidence: grid.logEvidence,
    marginals,
  }
}

/** The standard normal CDF at a number, the argument clamped to [-40, 40] (where it is 0 or 1 to double precision). */
const phi = (z: number): number => item(ndtr(Math.max(-40, Math.min(40, z))))

/**
 * An oracle from a CDF that is costly to evaluate: the CDF is tabulated once on `points` values between `lo` and `hi`
 * (made monotone), then interpolated linearly, so the quantile function that `agree` asks for hundreds of times is an
 * inverse table lookup.
 */
function tabulatedOracle(
  name: string,
  mean: number,
  sd: number,
  cdfAt: (x: number) => number,
  lo: number,
  hi: number,
  points: number,
): MarginalOracle {
  const xs = toFlat(add(div(mul(arange(points), hi - lo), points - 1), lo))
  const F = xs.map(cdfAt)
  for (let i = 1; i < points; i++) F[i] = Math.max(F[i] ?? 0, F[i - 1] ?? 0)
  const locate = (grid: ArrayLike<number>, target: number): number => {
    let a = 0
    let b = points - 1
    while (b - a > 1) {
      const mid = (a + b) >> 1
      if ((grid[mid] ?? 0) <= target) a = mid
      else b = mid
    }
    return a
  }
  return {
    name,
    mean,
    sd,
    cdf: (x) => {
      if (x <= lo) return 0
      if (x >= hi) return 1
      const j = locate(xs, x)
      const t = (x - (xs[j] ?? 0)) / ((xs[j + 1] ?? 0) - (xs[j] ?? 0))
      return (F[j] ?? 0) + t * ((F[j + 1] ?? 0) - (F[j] ?? 0))
    },
    quantile: (p) => {
      if (p <= (F[0] ?? 0)) return lo
      if (p >= (F[points - 1] ?? 1)) return hi
      const j = locate(F, p)
      const span = (F[j + 1] ?? 0) - (F[j] ?? 0)
      const t = span === 0 ? 0 : (p - (F[j] ?? 0)) / span
      return (xs[j] ?? 0) + t * ((xs[j + 1] ?? 0) - (xs[j] ?? 0))
    },
  }
}

/**
 * An oracle under another name (the quantity a method's draws are named by, where the quadrature named it by its coordinate).
 *
 * @param oracle - The oracle.
 * @param name - The new name.
 * @returns The oracle, named.
 * @example
 * ```ts
 * import { mixtureOracle, renamed } from 'omai-engine/probprog/models'
 * renamed(mixtureOracle('a', [{ weight: 1, mean: 0, sd: 1 }]), 'b').name // => 'b'
 * ```
 */
export function renamed(oracle: MarginalOracle, name: string): MarginalOracle {
  return { ...oracle, name }
}

/** One component of a mixture of normals. */
export interface NormalComponent {
  readonly weight: number
  readonly mean: number
  readonly sd: number
}

/**
 * The marginal oracle of a finite mixture of normals: the mean and sd from the mixture's moments, the CDF as the weighted
 * sum of normal CDFs and the quantile function by bisection on it. Components with a zero weight are dropped.
 *
 * @param name - The quantity's name.
 * @param components - The weighted components (the weights are normalised).
 * @returns The oracle.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { mixtureOracle } from 'omai-engine/probprog/models'
 * const o = mixtureOracle('x', [{ weight: 1, mean: -2, sd: 1 }, { weight: 1, mean: 2, sd: 1 }])
 * [o.mean, o.cdf(0), Math.abs(o.quantile(0.5)) < 1e-9] // => [0, 0.5, true]
 * ```
 */
export function mixtureOracle(name: string, components: readonly NormalComponent[]): MarginalOracle {
  const live = components.filter((c) => c.weight > 0)
  const total = live.reduce((a, c) => a + c.weight, 0)
  const parts = live.map((c) => ({ ...c, weight: c.weight / total }))
  const mean = parts.reduce((a, c) => a + c.weight * c.mean, 0)
  const second = parts.reduce((a, c) => a + c.weight * (c.sd * c.sd + c.mean * c.mean), 0)
  const cdf = (x: number): number => parts.reduce((a, c) => a + c.weight * phi((x - c.mean) / c.sd), 0)
  const sd = Math.sqrt(Math.max(second - mean * mean, 0))
  const lo0 = Math.min(...parts.map((c) => c.mean - 10 * c.sd))
  const hi0 = Math.max(...parts.map((c) => c.mean + 10 * c.sd))
  // Many components (a quadrature's worth): tabulate the CDF once instead of summing them at every call.
  if (parts.length > 40) return tabulatedOracle(name, mean, sd, cdf, lo0, hi0, 600)
  return {
    name,
    mean,
    sd,
    cdf,
    quantile: (p) => {
      let lo = lo0
      let hi = hi0
      for (let i = 0; i < 80; i++) {
        const mid = (lo + hi) / 2
        if (cdf(mid) < p) lo = mid
        else hi = mid
      }
      return (lo + hi) / 2
    },
  }
}

/**
 * The oracle of iid draws from an exact sampler (a target known in closed form whose marginals have no simple CDF): the
 * empirical distribution of many draws, with the mean and sd replaced by the exact ones when given.
 *
 * @param name - The quantity's name.
 * @param draws - The iid draws.
 * @param exact - The exact mean and sd, where known.
 * @returns The oracle.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { samplesOracle } from 'omai-engine/probprog/models'
 * samplesOracle('x', [1, 2, 3, 4], { mean: 2.5, sd: 1.2 }).mean // => 2.5
 * ```
 */
export function samplesOracle(
  name: string,
  draws: ArrayLike<number>,
  exact?: { readonly mean: number; readonly sd: number },
): MarginalOracle {
  const empirical = weightedOracle(name, draws)
  return exact === undefined ? empirical : { ...empirical, mean: exact.mean, sd: exact.sd }
}

/**
 * The marginal oracle of a quantity that is normal given the unconstrained coordinates of a grid posterior: its mean and
 * sd follow from the law of total variance, and its CDF is the grid expectation of the conditional normal CDF, tabulated on
 * a fine grid of values once and interpolated (so the quantile function, which `agree` asks for hundreds of times, is an
 * inverse table lookup). It is the exact marginal of a group effect given the hyperparameters, to the quadrature of the
 * grid.
 *
 * @param name - The quantity's name.
 * @param grid - The grid posterior of the coordinates the quantity is conditioned on.
 * @param conditional - The mean and sd of the quantity given the unconstrained coordinates `u` of the grid.
 * @param options - Points of the table (default 400) and its half-width in standard deviations (default 10).
 * @returns The oracle.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { conditionalOracle } from 'omai-engine/probprog/models'
 * import { gridPosterior } from 'omai-engine/probprog/validation'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * // m ~ N(0, 1), y = 1.2 ~ N(m, 1); x | m ~ N(m, 0.5²): x is N(0.6, 0.5 + 0.25) = N(0.6, 0.75) given the data.
 * const grid = gridPosterior(() => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1.2) }, undefined)
 * const o = conditionalOracle('x', grid, (u) => ({ mean: u[0] ?? 0, sd: 0.5 }))
 * [o.mean.toFixed(3), o.sd.toFixed(3), o.cdf(0.6).toFixed(2)] // => ['0.600', '0.866', '0.50']
 * ```
 */
export function conditionalOracle(
  name: string,
  grid: GridPosterior,
  conditional: (u: Float64Array) => { readonly mean: number; readonly sd: number },
  options: { readonly points?: number; readonly width?: number } = {},
): MarginalOracle {
  const mean = grid.expectation((u) => conditional(u).mean)
  const second = grid.expectation((u) => {
    const c = conditional(u)
    return c.sd * c.sd + c.mean * c.mean
  })
  const sd = Math.sqrt(Math.max(second - mean * mean, 0))
  const width = options.width ?? 10
  return tabulatedOracle(
    name,
    mean,
    sd,
    (x) =>
      grid.expectation((u) => {
        const c = conditional(u)
        return phi((x - c.mean) / c.sd)
      }),
    mean - width * sd,
    mean + width * sd,
    options.points ?? 400,
  )
}

/**
 * The oracle of a weighted average of marginals (a mixture of the laws they describe): the moments from the components'
 * moments, the CDF as the weighted sum and the quantile function by bisection. It is the raw marginal of a component of a
 * mixture model whose labels may switch: half the mass where the component is the smaller mean, half where it is the
 * larger.
 *
 * @param name - The quantity's name.
 * @param parts - The component oracles with their weights (normalised here).
 * @returns The oracle.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { averageOracle, mixtureOracle } from 'omai-engine/probprog/models'
 * const lo = mixtureOracle('lo', [{ weight: 1, mean: -2, sd: 1 }])
 * const hi = mixtureOracle('hi', [{ weight: 1, mean: 2, sd: 1 }])
 * const both = averageOracle('mu', [{ weight: 1, oracle: lo }, { weight: 1, oracle: hi }])
 * [both.mean, both.cdf(0)] // => [0, 0.5]
 * ```
 */
export function averageOracle(
  name: string,
  parts: readonly { readonly weight: number; readonly oracle: MarginalOracle }[],
): MarginalOracle {
  const total = parts.reduce((a, p) => a + p.weight, 0)
  const w = parts.map((p) => p.weight / total)
  const mean = parts.reduce((a, p, i) => a + (w[i] ?? 0) * p.oracle.mean, 0)
  const second = parts.reduce((a, p, i) => a + (w[i] ?? 0) * (p.oracle.sd ** 2 + p.oracle.mean ** 2), 0)
  const cdf = (x: number): number => parts.reduce((a, p, i) => a + (w[i] ?? 0) * p.oracle.cdf(x), 0)
  const lo0 = Math.min(...parts.map((p) => p.oracle.quantile(1e-9)))
  const hi0 = Math.max(...parts.map((p) => p.oracle.quantile(1 - 1e-9)))
  return {
    name,
    mean,
    sd: Math.sqrt(Math.max(second - mean * mean, 0)),
    cdf,
    quantile: (p) => {
      let lo = lo0
      let hi = hi0
      for (let i = 0; i < 80; i++) {
        const mid = (lo + hi) / 2
        if (cdf(mid) < p) lo = mid
        else hi = mid
      }
      return (lo + hi) / 2
    },
  }
}
