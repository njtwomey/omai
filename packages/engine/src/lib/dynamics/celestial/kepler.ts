/**
 * Kepler's equation, the bridge between time and position on a conic: the anomalies (mean, eccentric, hyperbolic,
 * parabolic, true) and their conversions, the solvers of M = E - e sin E (Newton, Halley, bisection, secant and
 * fixed-point iteration as a table of modes, each a stepped algorithm whose iterate, residual and error are logged), its
 * hyperbolic form M = e sinh H - H and the parabolic Barker equation in closed form, and the series that solve it
 * without iterating (Fourier-Bessel and Lagrange).
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { jv } from '../../numerics/special/index.ts'
import { choice, dict, real } from '../../space/index.ts'
import { item } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  type ModeSpec,
  type ModeTable,
  modes,
  scalar,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'

const TWO_PI = 2 * Math.PI

/**
 * An angle wrapped into (-pi, pi].
 *
 * @param x - An angle in radians.
 * @returns The equivalent angle in (-pi, pi].
 * @example
 * ```ts
 * import { wrapAngle } from 'omai-engine/dynamics/celestial'
 * wrapAngle(3 * Math.PI) // => 3.141592653589793
 * ```
 */
export function wrapAngle(x: number): number {
  const w = x - TWO_PI * Math.round(x / TWO_PI)
  return w <= -Math.PI ? w + TWO_PI : w
}

/** Which conic an eccentricity describes. */
export type ConicKind = 'elliptic' | 'parabolic' | 'hyperbolic'

/**
 * The conic of an eccentricity: an ellipse below 1, a parabola at 1, a hyperbola above.
 *
 * @param e - The eccentricity (non-negative).
 * @param tol - How close to 1 counts as a parabola (default 1e-12).
 * @returns The kind.
 * @throws OmaiError with code `value` for a negative or non-finite eccentricity.
 * @example
 * ```ts
 * import { conicKind } from 'omai-engine/dynamics/celestial'
 * [conicKind(0.3), conicKind(1), conicKind(2)] // => ['elliptic', 'parabolic', 'hyperbolic']
 * ```
 */
export function conicKind(e: number, tol = 1e-12): ConicKind {
  if (!(e >= 0) || !Number.isFinite(e))
    throw new OmaiError('value', `conicKind: the eccentricity must be >= 0, got ${e}`)
  return Math.abs(e - 1) <= tol ? 'parabolic' : e < 1 ? 'elliptic' : 'hyperbolic'
}

/**
 * The true anomaly from the eccentric anomaly on an ellipse.
 *
 * @param E - Eccentric anomaly.
 * @param e - Eccentricity in [0, 1).
 * @returns The true anomaly nu, on the same branch as E.
 * @example
 * ```ts
 * import { trueFromEccentric } from 'omai-engine/dynamics/celestial'
 * trueFromEccentric(Math.PI, 0.5) // => 3.141592653589793
 * ```
 */
export function trueFromEccentric(E: number, e: number): number {
  const nu = 2 * Math.atan2(Math.sqrt(1 + e) * Math.sin(E / 2), Math.sqrt(1 - e) * Math.cos(E / 2))
  return nu + TWO_PI * Math.round((E - nu) / TWO_PI)
}

/**
 * The eccentric anomaly from the true anomaly on an ellipse.
 *
 * @param nu - True anomaly.
 * @param e - Eccentricity in [0, 1).
 * @returns The eccentric anomaly E, on the same branch as nu.
 * @example
 * ```ts
 * import { eccentricFromTrue } from 'omai-engine/dynamics/celestial'
 * eccentricFromTrue(Math.PI, 0.5) // => 3.141592653589793
 * ```
 */
export function eccentricFromTrue(nu: number, e: number): number {
  const E = 2 * Math.atan2(Math.sqrt(1 - e) * Math.sin(nu / 2), Math.sqrt(1 + e) * Math.cos(nu / 2))
  return E + TWO_PI * Math.round((nu - E) / TWO_PI)
}

/**
 * The true anomaly from the hyperbolic anomaly H on a hyperbola.
 *
 * @param H - Hyperbolic anomaly.
 * @param e - Eccentricity above 1.
 * @returns The true anomaly, within (-acos(-1/e), acos(-1/e)).
 * @example
 * ```ts
 * import { trueFromHyperbolic } from 'omai-engine/dynamics/celestial'
 * trueFromHyperbolic(0, 2) // => 0
 * ```
 */
export function trueFromHyperbolic(H: number, e: number): number {
  return 2 * Math.atan(Math.sqrt((e + 1) / (e - 1)) * Math.tanh(H / 2))
}

/**
 * The hyperbolic anomaly from the true anomaly.
 *
 * @param nu - True anomaly, within the asymptotes (|nu| < acos(-1/e)).
 * @param e - Eccentricity above 1.
 * @returns H.
 * @throws OmaiError with code `value` when nu is on or beyond an asymptote.
 * @example
 * ```ts
 * import { hyperbolicFromTrue } from 'omai-engine/dynamics/celestial'
 * hyperbolicFromTrue(0, 2) // => 0
 * ```
 */
export function hyperbolicFromTrue(nu: number, e: number): number {
  const arg = Math.sqrt((e - 1) / (e + 1)) * Math.tan(nu / 2)
  if (!(Math.abs(arg) < 1) || Math.abs(nu) >= Math.acos(-1 / e)) {
    throw new OmaiError('value', `hyperbolicFromTrue: nu = ${nu} is beyond the asymptote of a hyperbola with e = ${e}`)
  }
  return 2 * Math.atanh(arg)
}

/**
 * The mean anomaly from the true anomaly on any conic: M = E - e sin E (ellipse), M = e sinh H - H (hyperbola),
 * M = D + D^3 / 3 with D = tan(nu / 2) (parabola, Barker's equation).
 *
 * @param nu - True anomaly.
 * @param e - Eccentricity.
 * @returns The mean anomaly (for an ellipse on the same branch as nu).
 * @example
 * ```ts
 * import { meanFromTrue } from 'omai-engine/dynamics/celestial'
 * meanFromTrue(Math.PI, 0.5) // => 3.141592653589793
 * ```
 */
export function meanFromTrue(nu: number, e: number): number {
  const kind = conicKind(e)
  if (kind === 'elliptic') {
    const E = eccentricFromTrue(nu, e)
    return E - e * Math.sin(E)
  }
  if (kind === 'hyperbolic') {
    const H = hyperbolicFromTrue(nu, e)
    return e * Math.sinh(H) - H
  }
  const D = Math.tan(nu / 2)
  return D + (D * D * D) / 3
}

/**
 * Barker's equation solved exactly: the true anomaly of a parabola from its mean anomaly M = D + D^3 / 3,
 * D = tan(nu / 2) = 2 sinh(asinh(3M / 2) / 3).
 *
 * @param M - The parabolic mean anomaly, 2 sqrt(mu / p^3) (t - T) for semi-latus rectum p.
 * @returns The true anomaly nu in (-pi, pi).
 * @example
 * ```ts
 * import { meanFromTrue, solveKeplerParabolic } from 'omai-engine/dynamics/celestial'
 * Math.abs(solveKeplerParabolic(meanFromTrue(1.2, 1)) - 1.2) < 1e-14 // => true
 * ```
 */
export function solveKeplerParabolic(M: number): number {
  const D = 2 * Math.sinh(Math.asinh((3 * M) / 2) / 3)
  return 2 * Math.atan(D)
}

/** The Kepler problem: the mean anomaly and eccentricity, and how the solver starts. */
export interface KeplerProblem {
  /** The mean anomaly (any real; reduced modulo 2 pi on an ellipse). */
  readonly M: number
  /** The eccentricity: below 1 the elliptic equation, above 1 the hyperbolic one. */
  readonly e: number
}

/**
 * The residual of Kepler's equation: E - e sin E - M on an ellipse, e sinh H - H - M on a hyperbola.
 *
 * @param x - The eccentric (or hyperbolic) anomaly.
 * @param M - Mean anomaly.
 * @param e - Eccentricity (not 1).
 * @returns The residual, zero at the solution.
 * @example
 * ```ts
 * import { keplerResidual } from 'omai-engine/dynamics/celestial'
 * keplerResidual(Math.PI, Math.PI, 0.7) // => 0
 * ```
 */
export function keplerResidual(x: number, M: number, e: number): number {
  return e < 1 ? x - e * Math.sin(x) - M : e * Math.sinh(x) - x - M
}

/** The equation on the half-line the solver works on (M >= 0, and for an ellipse M <= pi). */
interface Form {
  f(x: number): number
  df(x: number): number
  ddf(x: number): number
  /** The fixed-point map whose fixed point is the root. */
  g(x: number): number
  bracket(): readonly [number, number]
  start(kind: KeplerStart): number
}

/** Where the iteration starts. */
export type KeplerStart = 'auto' | 'mean' | 'pi'

function formOf(M: number, e: number): Form {
  if (e < 1) {
    return {
      f: (x) => x - e * Math.sin(x) - M,
      df: (x) => 1 - e * Math.cos(x),
      ddf: (x) => e * Math.sin(x),
      g: (x) => M + e * Math.sin(x),
      bracket: () => [M, Math.min(M + e, Math.PI)],
      start: (kind) => (kind === 'pi' || (kind === 'auto' && e > 0.8) ? Math.PI : M),
    }
  }
  const f = (x: number): number => e * Math.sinh(x) - x - M
  return {
    f,
    df: (x) => e * Math.cosh(x) - 1,
    ddf: (x) => e * Math.sinh(x),
    g: (x) => Math.asinh((x + M) / e),
    bracket: () => {
      let hi = Math.max(1, Math.log((2 * M) / e + 1.8))
      while (f(hi) < 0) hi *= 2
      return [0, hi]
    },
    start: (kind) => (kind === 'mean' ? Math.asinh(M / e) : Math.log((2 * M) / e + 1.8)),
  }
}

/** Mean anomaly reduced to the half-line: its sign, the whole turns removed, and the value that is solved for. */
function reduce(M: number, e: number): { sign: 1 | -1; turns: number; m: number } {
  if (e >= 1) return { sign: M < 0 ? -1 : 1, turns: 0, m: Math.abs(M) }
  const w = wrapAngle(M)
  return { sign: w < 0 ? -1 : 1, turns: Math.round((M - w) / TWO_PI), m: Math.abs(w) }
}

/** The iteration the solvers share. */
export type KeplerMethod = 'newton' | 'halley' | 'bisection' | 'secant' | 'fixed-point'

/** The state of {@link keplerAlgorithm}. Everything is a number, so it serialises. */
export interface KeplerState {
  /** The reduced mean anomaly (the half-line problem being solved). */
  readonly m: number
  /** The eccentricity. */
  readonly e: number
  /** The sign and whole turns of the original mean anomaly, to restore. */
  readonly sign: 1 | -1
  readonly turns: number
  /** The iterate. */
  readonly x: number
  /** The previous iterate (secant) and bracket ends (bisection). */
  readonly xPrev: number
  readonly lo: number
  readonly hi: number
  /** The residual at the iterate. */
  readonly residual: number
  readonly iteration: number
  /** The root found by Newton to machine precision (the truth the error is measured from). */
  readonly truth: number
}

/** The paths of {@link keplerAlgorithm}. */
export type KeplerPath = 'kepler/anomaly' | 'kepler/residual' | 'kepler/error' | 'kepler/digits'

/** What a Kepler solver logs per iteration. */
export const keplerContract: Contract<KeplerState, KeplerPath> = defineContract({
  'kepler/anomaly': scalar({ description: 'the iterate: the eccentric (or hyperbolic) anomaly estimate, unreduced' }),
  'kepler/residual': scalar({
    description: 'E - e sin E - M at the iterate (or its hyperbolic form)',
    direction: 'lower',
  }),
  'kepler/error': scalar({ description: 'the distance of the iterate from the root', direction: 'lower' }),
  'kepler/digits': scalar({ description: 'correct decimal digits of the iterate: -log10 of the error, capped at 16' }),
})

/** Newton to the last digit, for the truth the error is measured against. */
function reference(form: Form, start: number): number {
  let x = start
  for (let i = 0; i < 60; i++) {
    const dx = form.f(x) / form.df(x)
    x -= dx
    if (Math.abs(dx) <= 1e-16 * Math.max(1, Math.abs(x))) break
  }
  return x
}

/** Options of {@link keplerAlgorithm}. */
export interface KeplerOptions {
  /** The iteration (default Newton). */
  readonly method?: KeplerMethod
  /** Where it starts (default `auto`: the mean anomaly, or pi for an eccentric ellipse). */
  readonly start?: KeplerStart
  /** Stop when the residual is below this fraction of max(1, |M|) (default 1e-14). */
  readonly tol?: number
}

const CAP = 16

const digitsOf = (error: number): number => Math.min(CAP, error <= 0 ? CAP : Math.max(0, -Math.log10(error)))

/** The pieces of one iteration, shared by the algorithm and the one-shot solver. */
interface Iteration {
  init(): KeplerState
  advance(s: KeplerState): KeplerState
  converged(s: KeplerState): boolean
  anomaly(s: KeplerState): number
}

function iteration(problem: KeplerProblem, options: KeplerOptions): Iteration & { readonly config: Json } {
  const { M, e } = problem
  if (!Number.isFinite(M)) throw new OmaiError('value', 'keplerAlgorithm: M must be finite')
  if (conicKind(e) === 'parabolic') {
    throw new OmaiError('value', 'keplerAlgorithm: e = 1 is the parabola; use solveKeplerParabolic')
  }
  const method = options.method ?? 'newton'
  const startKind = options.start ?? 'auto'
  const tol = options.tol ?? 1e-14
  const { sign, turns, m } = reduce(M, e)
  const form = formOf(m, e)
  return {
    config: { M, e, method, start: startKind, tol },
    init: () => {
      const [lo, hi] = form.bracket()
      const x0 = method === 'bisection' ? 0.5 * (lo + hi) : form.start(startKind)
      const x = method === 'secant' ? form.g(x0) : x0
      return {
        m,
        e,
        sign,
        turns,
        x,
        xPrev: x0,
        lo,
        hi,
        residual: form.f(x),
        iteration: 0,
        truth: reference(form, form.start('auto')),
      }
    },
    advance: (s) => {
      let x = s.x
      let xPrev = s.xPrev
      let lo = s.lo
      let hi = s.hi
      const fx = s.residual
      switch (method) {
        case 'newton':
          x = s.x - fx / form.df(s.x)
          break
        case 'halley': {
          const d1 = form.df(s.x)
          x = s.x - (2 * fx * d1) / (2 * d1 * d1 - fx * form.ddf(s.x))
          break
        }
        case 'bisection': {
          if (fx > 0) hi = s.x
          else lo = s.x
          x = 0.5 * (lo + hi)
          break
        }
        case 'secant': {
          const fPrev = form.f(s.xPrev)
          x = fx === fPrev ? s.x : s.x - (fx * (s.x - s.xPrev)) / (fx - fPrev)
          xPrev = s.x
          break
        }
        case 'fixed-point':
          x = form.g(s.x)
          break
      }
      return { ...s, x, xPrev, lo, hi, residual: form.f(x), iteration: s.iteration + 1 }
    },
    converged: (s) => Math.abs(s.residual) <= tol * Math.max(1, m),
    anomaly: (s) => sign * s.x + (e < 1 ? TWO_PI * turns : 0),
  }
}

/**
 * Kepler's equation solved iteratively as a stepped algorithm: one iteration per step, the iterate, residual and error
 * logged, the truth carried in the state (Newton run to the last digit). The elliptic equation M = E - e sin E (any
 * M, reduced by symmetry and periodicity to [0, pi]) and the hyperbolic M = e sinh H - H (e > 1) take the same
 * methods. Newton and Halley converge from the default starts for every e (the equation is convex on the half-line),
 * bisection halves its bracket, the secant method needs no derivative, and fixed-point iteration E = M + e sin E
 * contracts at rate e cos E, which is why it crawls as e -> 1.
 *
 * @param problem - The mean anomaly and eccentricity (e != 1; the parabola is {@link solveKeplerParabolic}).
 * @param options - Method, start and tolerance.
 * @returns The algorithm; it stops (converged) when the residual is within the tolerance.
 * @throws OmaiError with code `value` for e = 1, a negative e or a non-finite M.
 * @example
 * ```ts
 * import { keplerAlgorithm } from 'omai-engine/dynamics/celestial'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const { state, steps } = runAlgorithm(keplerAlgorithm({ M: 1, e: 0.5 }), { maxSteps: 50 })
 * [steps < 8, Math.abs(state.x - 1.4987011335178484) < 1e-12] // => [true, true]
 * ```
 */
export function keplerAlgorithm(
  problem: KeplerProblem,
  options: KeplerOptions = {},
): Algorithm<KeplerState, KeplerPath> {
  const it = iteration(problem, options)
  return {
    name: 'kepler',
    config: it.config,
    init: () => it.init(),
    step: (s, ctx) => {
      const next = it.advance(s)
      const error = Math.abs(next.x - next.truth)
      ctx.log('kepler/anomaly', it.anomaly(next))
      ctx.log('kepler/residual', next.residual)
      ctx.log('kepler/error', error)
      ctx.log('kepler/digits', digitsOf(error))
      return next
    },
    done: (s) => (it.converged(s) ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as KeplerState,
  }
}

/** What every Kepler solver yields: the anomaly reached. */
export type KeplerParams = { readonly anomaly: number }

const anomalyOf = (s: KeplerState): KeplerParams => ({ anomaly: s.sign * s.x + (s.e < 1 ? TWO_PI * s.turns : 0) })

const tolKnob = real(1e-16, 1e-2, { log: true, default: 1e-14, label: 'tolerance' })
const startKnob = choice(['auto', 'mean', 'pi'], { default: 'auto', label: 'start' })

function mode(
  method: KeplerMethod,
  description: string,
  withStart: boolean,
): ModeSpec<KeplerProblem, KeplerState, KeplerParams> {
  return {
    description,
    space: dict(withStart ? { start: startKnob, tol: tolKnob } : { tol: tolKnob }),
    make: (p, k) =>
      keplerAlgorithm(p, {
        method,
        ...(k['start'] === undefined ? {} : { start: k['start'] as KeplerStart }),
        tol: (k['tol'] as number | undefined) ?? 1e-14,
      }),
    params: anomalyOf,
  }
}

/**
 * The solvers of Kepler's equation as one slot: Newton (quadratic), Halley (cubic), bisection (a bit per step), the
 * secant method (the golden-ratio order) and fixed-point iteration (linear, at rate e cos E), each with its own knobs.
 */
export const keplerSolvers: ModeTable<KeplerProblem, KeplerParams> = modes(
  {
    newton: mode('newton', "Newton's method: quadratic convergence, global on the convex half-line", true),
    halley: mode('halley', "Halley's method: cubic convergence from the second derivative", true),
    bisection: mode('bisection', 'bisection of the bracket [M, M + e]: one bit per step, never fails', false),
    secant: mode('secant', 'the secant method: no derivative, superlinear', true),
    'fixed-point': mode(
      'fixed-point',
      'E <- M + e sin E: contraction at rate e cos E, hopeless as e approaches 1',
      true,
    ),
  },
  { tag: 'method', label: 'Kepler solver', description: 'how Kepler’s equation is solved' },
)

/** A solved Kepler equation. */
export interface KeplerSolution {
  /** The eccentric anomaly (ellipse), hyperbolic anomaly (hyperbola) or true anomaly (parabola). */
  readonly anomaly: number
  /** The true anomaly. */
  readonly trueAnomaly: number
  readonly kind: ConicKind
  /** Iterations used (0 for the closed form of the parabola). */
  readonly iterations: number
  /** The residual of the equation at the solution. */
  readonly residual: number
}

/**
 * Solve Kepler's equation on any conic. An ellipse (e < 1) and a hyperbola (e > 1) are iterated (Newton by default);
 * a parabola (e = 1) is solved exactly by Barker's formula, with `M` the parabolic mean anomaly.
 *
 * @param M - Mean anomaly.
 * @param e - Eccentricity.
 * @param options - Method, start and tolerance for the iterated cases, and the iteration cap (default 100).
 * @returns The anomaly, the true anomaly and the work done.
 * @throws OmaiError with code `convergence` when the iteration does not converge in the cap.
 * @concept keplers-equation
 * @example
 * ```ts
 * import { solveKepler } from 'omai-engine/dynamics/celestial'
 * const s = solveKepler(1, 0.5)
 * [s.kind, Math.abs(s.anomaly - 1.4987011335178484) < 1e-12] // => ['elliptic', true]
 * ```
 */
export function solveKepler(
  M: number,
  e: number,
  options: KeplerOptions & { readonly maxIterations?: number } = {},
): KeplerSolution {
  const kind = conicKind(e)
  if (kind === 'parabolic') {
    const nu = solveKeplerParabolic(M)
    return { anomaly: nu, trueAnomaly: nu, kind, iterations: 0, residual: M - meanFromTrue(nu, 1) }
  }
  const it = iteration({ M, e }, options)
  const cap = options.maxIterations ?? 100
  let s = it.init()
  while (!it.converged(s) && s.iteration < cap) s = it.advance(s)
  if (!it.converged(s)) throw new OmaiError('convergence', `solveKepler: no convergence in ${cap} iterations`)
  const anomaly = anomalyOf(s).anomaly
  const nu = kind === 'elliptic' ? trueFromEccentric(anomaly, e) : trueFromHyperbolic(anomaly, e)
  return { anomaly, trueAnomaly: nu, kind, iterations: s.iteration, residual: s.residual }
}

/**
 * The true anomaly at a mean anomaly, on any conic.
 *
 * @param M - Mean anomaly.
 * @param e - Eccentricity.
 * @returns The true anomaly (for an ellipse on the branch of M).
 * @example
 * ```ts
 * import { trueFromMean } from 'omai-engine/dynamics/celestial'
 * trueFromMean(0, 0.3) // => 0
 * ```
 */
export function trueFromMean(M: number, e: number): number {
  return solveKepler(M, e).trueAnomaly
}

/**
 * The Fourier-Bessel series of the eccentric anomaly, E = M + 2 sum_{n >= 1} J_n(n e) sin(n M) / n: no iteration, exact
 * as the terms grow (the series converges for every e < 1, but slowly as e -> 1).
 *
 * @param M - Mean anomaly.
 * @param e - Eccentricity in [0, 1).
 * @param terms - Terms to sum (default 60).
 * @returns The eccentric anomaly.
 * @throws OmaiError with code `value` for e outside [0, 1).
 * @concept keplers-equation
 * @example
 * ```ts
 * import { keplerBesselSeries } from 'omai-engine/dynamics/celestial'
 * Math.abs(keplerBesselSeries(1, 0.5, 80) - 1.4987011335178484) < 1e-12 // => true
 * ```
 */
export function keplerBesselSeries(M: number, e: number, terms = 60): number {
  if (!(e >= 0 && e < 1)) throw new OmaiError('value', 'keplerBesselSeries: e must be in [0, 1)')
  let sum = 0
  for (let n = 1; n <= terms; n++) sum += (item(jv(n, n * e)) / n) * Math.sin(n * M)
  return M + 2 * sum
}

/** The eccentricity beyond which the Lagrange series in powers of e diverges (Laplace's limit, 0.6627434...). */
export const LAPLACE_LIMIT = 0.6627434193491816

/**
 * The Lagrange series of the eccentric anomaly in powers of the eccentricity, through e^5:
 * E = M + e sin M + (e^2 / 2) sin 2M + (e^3 / 8)(3 sin 3M - sin M) + (e^4 / 6)(2 sin 4M - sin 2M) +
 * (e^5 / 384)(125 sin 5M - 81 sin 3M + 2 sin M). Each order cuts the error by a factor e while e is below
 * {@link LAPLACE_LIMIT}; beyond it the full series diverges whatever M.
 *
 * @param M - Mean anomaly.
 * @param e - Eccentricity.
 * @param order - The highest power of e kept, 1 to 5 (default 5).
 * @returns The eccentric anomaly to that order.
 * @throws OmaiError with code `value` for an order outside 1..5.
 * @concept keplers-equation
 * @example
 * ```ts
 * import { keplerLagrangeSeries } from 'omai-engine/dynamics/celestial'
 * Math.abs(keplerLagrangeSeries(1, 0.1, 5) - 1.0 - 0.1 * Math.sin(1) - 0.005 * Math.sin(2)) < 1e-4 // => true
 * ```
 */
export function keplerLagrangeSeries(M: number, e: number, order = 5): number {
  if (!Number.isInteger(order) || order < 1 || order > 5)
    throw new OmaiError('value', 'keplerLagrangeSeries: order is 1..5')
  const s = (n: number): number => Math.sin(n * M)
  const terms = [
    e * s(1),
    (e ** 2 / 2) * s(2),
    (e ** 3 / 8) * (3 * s(3) - s(1)),
    (e ** 4 / 6) * (2 * s(4) - s(2)),
    (e ** 5 / 384) * (125 * s(5) - 81 * s(3) + 2 * s(1)),
  ]
  return M + terms.slice(0, order).reduce((a, b) => a + b, 0)
}
