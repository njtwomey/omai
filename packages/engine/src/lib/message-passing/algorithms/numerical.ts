/**
 * Numerical projections: how the moments of a factor against Gaussian cavities are found when it has no closed form, as
 * rules the engine's `numeric` slot selects for every numeric operator of a run. The panelled product rule is the
 * numeric operator's own default; Gauss-Hermite and unscented sigma points are cheap rules that are exact for
 * polynomial-like factors; Laplace's method takes the mode and the curvature of the factor times the cavities (a Gaussian
 * fitted where the density peaks, the projection of Laplace propagation, not a moment match); and importance sampling
 * draws from the cavities with a fixed set of antithetic draws, so a rule is a pure function of its input and a run
 * still serialises and resumes to the same state. Each rule returns the log mass, which the evidence of the run is made
 * of (so Laplace's rule gives the Laplace approximation of the evidence).
 *
 * @index module Numerical projections: quadrature, Gauss-Hermite, Laplace, sample and sigma-point rules for numeric factors.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, key, normal } from '../random/index.ts'
import { dict, int, real, union } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Algorithm, type Config, defineMethod, type Method } from '../trace/index.ts'
import { spd } from './_dense.ts'
import { gaussianGrid, gaussianGridOn, gaussianNodes } from './_quadrature.ts'
import { type MpState, messagePassingAlgorithm } from './engine.ts'
import type { CompiledGraph } from './graph.ts'
import {
  type EpPath,
  epContract,
  epDefaults,
  type MessagePassingOptions,
  messagePassingOptions,
  messagePassingSettings,
  messagePassingSpace,
} from './named.ts'
import type { Projection } from './projections.ts'
import type { GaussianBase, TiltedProblem, TiltedRule } from './types.ts'

const isPoint = (b: GaussianBase): boolean => !(b.variance > 0)

/** The moments of the density proportional to a weighted point set, at the argument `to`, from weights given in the log scale. */
function weightedMoments(
  points: readonly (readonly number[])[],
  logWeights: readonly number[],
  to: number,
): { readonly logMass: number; readonly mean: number; readonly variance: number } {
  let top = Number.NEGATIVE_INFINITY
  for (const v of logWeights) if (v > top) top = v
  if (!Number.isFinite(top))
    throw new OmaiError('value', 'numeric rule: the factor has no mass under the incoming messages')
  let m0 = 0
  let m1 = 0
  for (const [i, p] of points.entries()) {
    const w = Math.exp((logWeights[i] ?? Number.NEGATIVE_INFINITY) - top)
    m0 += w
    m1 += w * (p[to] ?? 0)
  }
  const mean = m1 / m0
  let m2 = 0
  for (const [i, p] of points.entries())
    m2 += Math.exp((logWeights[i] ?? Number.NEGATIVE_INFINITY) - top) * ((p[to] ?? 0) - mean) ** 2
  return { logMass: top + Math.log(m0), mean, variance: m2 / m0 }
}

/**
 * The numeric operator's own rule: a panelled Gauss-Legendre product rule over the arguments (each with its cavity as the
 * weight), then zoomed onto where the tilted mass is and integrated again. Robust for narrow factors inside wide
 * messages; up to three free arguments.
 *
 * @concept numerical-projections
 * @index structure The panelled Gauss-Legendre rule of the numeric operator, zoomed onto the tilted mass.
 */
export const quadratureRule: TiltedRule = {
  name: 'quadrature',
  moments: ({ bases, to, logf }: TiltedProblem) => {
    const n = bases.length
    const integrateOver = (
      grids: readonly { x: Float64Array; w: Float64Array }[],
    ): { mass: number; m1: number[]; m2: number[]; xs: number[]; ws: number[] } => {
      const point = new Array<number>(n).fill(0)
      const m1 = new Array<number>(n).fill(0)
      const m2 = new Array<number>(n).fill(0)
      const xs: number[] = []
      const ws: number[] = []
      let mass = 0
      const walk = (d: number, w: number): void => {
        if (d === n) {
          const v = w * Math.exp(logf(point))
          if (!(v > 0)) return
          mass += v
          for (let k = 0; k < n; k++) {
            m1[k] = (m1[k] ?? 0) + v * (point[k] ?? 0)
            m2[k] = (m2[k] ?? 0) + v * (point[k] ?? 0) ** 2
          }
          xs.push(point[to] ?? 0)
          ws.push(v)
          return
        }
        const g = grids[d]
        if (g === undefined) return
        for (let i = 0; i < g.x.length; i++) {
          point[d] = g.x[i] ?? 0
          walk(d + 1, w * (g.w[i] ?? 0))
        }
      }
      walk(0, 1)
      return { mass, m1, m2, xs, ws }
    }
    const grids = bases.map((b) =>
      isPoint(b)
        ? { x: Float64Array.of(b.mean), w: Float64Array.of(1) }
        : gaussianGrid(b.mean, b.variance, n === 1 ? 64 : 24, 8),
    )
    let pass = integrateOver(grids)
    // The factor is usually much narrower than the messages: zoom each argument onto where the tilted mass is, and integrate again.
    if (pass.mass > 0) {
      const zoomed = bases.map((b, d) => {
        if (isPoint(b)) return grids[d] as { x: Float64Array; w: Float64Array }
        const mu = (pass.m1[d] ?? 0) / pass.mass
        const sd = Math.sqrt(Math.max((pass.m2[d] ?? 0) / pass.mass - mu * mu, 0))
        const spread = Math.sqrt(b.variance)
        const lo = Math.max(b.mean - 9 * spread, mu - 8 * Math.max(sd, 1e-3 * spread))
        const hi = Math.min(b.mean + 9 * spread, mu + 8 * Math.max(sd, 1e-3 * spread))
        return hi > lo
          ? gaussianGridOn(b.mean, b.variance, lo, hi, n === 1 ? 48 : n === 2 ? 14 : 6, n === 1 ? 10 : 8)
          : (grids[d] as { x: Float64Array; w: Float64Array })
      })
      const second = integrateOver(zoomed)
      if (second.mass > 0) pass = second
    }
    const mass = pass.mass
    if (!(mass > 0)) throw new OmaiError('value', 'numeric rule: the factor has no mass under the incoming messages')
    const mean = pass.ws.reduce((s, w, i) => s + w * (pass.xs[i] ?? 0), 0) / mass
    const variance = pass.ws.reduce((s, w, i) => s + w * ((pass.xs[i] ?? 0) - mean) ** 2, 0) / mass
    return { logMass: Math.log(mass), mean, variance }
  },
}

/**
 * Gauss-Hermite quadrature against the cavities: a product rule of `points` nodes per free argument (at most 40,000
 * evaluations in all). Exact for the factor times a polynomial of degree below 2 `points`, so excellent for smooth
 * factors and poor for kinked or very narrow ones (use {@link quadratureRule} for those).
 *
 * @param points - Nodes per argument (default 32).
 * @returns The rule.
 * @throws OmaiError with code `value` for fewer than 2 nodes or a product larger than 40,000 evaluations.
 * @concept numerical-projections
 * @index function Gauss-Hermite product rule against the cavities, as a numeric projection rule.
 * @example
 * ```ts
 * import { gaussHermiteRule } from 'omai-engine/message-passing'
 * // exp(-x^2 / 2) against N(0, 1): the mass is 1 / sqrt(2), the tilted variance 1 / 2.
 * const r = gaussHermiteRule(16).moments({ bases: [{ mean: 0, variance: 1 }], to: 0, logf: ([x]) => -0.5 * (x as number) ** 2 })
 * [Math.exp(r.logMass).toFixed(4), r.variance.toFixed(4)] // => ['0.7071', '0.5000']
 * ```
 */
export function gaussHermiteRule(points = 32): TiltedRule {
  if (!(Number.isInteger(points) && points >= 2))
    throw new OmaiError('value', 'gaussHermiteRule: need at least 2 nodes')
  return {
    name: `gauss-hermite(${points})`,
    moments: ({ bases, to, logf }) => {
      const grids = bases.map((b) =>
        isPoint(b) ? { x: Float64Array.of(b.mean), w: Float64Array.of(1) } : gaussianNodes(b.mean, b.variance, points),
      )
      const total = grids.reduce((a, g) => a * g.x.length, 1)
      if (total > 40_000)
        throw new OmaiError('value', `gaussHermiteRule: ${total} evaluations are more than the cap of 40000`)
      const pts: number[][] = []
      const logw: number[] = []
      const walk = (d: number, lw: number, point: number[]): void => {
        if (d === grids.length) {
          pts.push(point.slice())
          logw.push(lw + logf(point))
          return
        }
        const g = grids[d] as { x: Float64Array; w: Float64Array }
        for (let i = 0; i < g.x.length; i++) {
          point[d] = g.x[i] ?? 0
          walk(d + 1, lw + Math.log(g.w[i] ?? 0), point)
        }
      }
      walk(0, 0, new Array<number>(grids.length).fill(0))
      return weightedMoments(pts, logw, to)
    },
  }
}

/**
 * The unscented rule: the factor weighed at the sigma points of the product of the cavities (the mean, and the mean plus
 * and minus `sqrt((d + kappa) variance)` along each free argument) with the unscented weights. Three points per
 * argument at `kappa = 2` is the three-point Gauss-Hermite rule for one argument; it is cheap and exact only for factors
 * that are low-order polynomial. The weights must be positive (`kappa > 0`) to give a mass.
 *
 * @param kappa - The spread parameter (default 2).
 * @returns The rule.
 * @throws OmaiError with code `value` for a kappa that is not positive.
 * @concept numerical-projections
 * @index function Unscented sigma-point rule for numeric factors.
 * @example
 * ```ts
 * import { unscentedRule } from 'omai-engine/message-passing'
 * // exp(-x^2 / 2) against N(0, 1) at the three Gauss-Hermite points: exact would be 1 / sqrt(2) = 0.7071.
 * const r = unscentedRule(2).moments({ bases: [{ mean: 0, variance: 1 }], to: 0, logf: ([x]) => -0.5 * (x as number) ** 2 })
 * Math.exp(r.logMass).toFixed(4) // => '0.7410'
 * ```
 */
export function unscentedRule(kappa = 2): TiltedRule {
  if (!(kappa > 0)) throw new OmaiError('value', 'unscentedRule: kappa must be positive')
  return {
    name: `unscented(${kappa})`,
    moments: ({ bases, to, logf }) => {
      const free = bases.map((b, d) => (isPoint(b) ? -1 : d)).filter((d) => d >= 0)
      const d = free.length
      const centre = bases.map((b) => b.mean)
      const pts: number[][] = [centre]
      const logw: number[] = [Math.log(kappa / (d + kappa)) + logf(centre)]
      for (const j of free) {
        const spread = Math.sqrt((d + kappa) * (bases[j] as GaussianBase).variance)
        for (const sign of [1, -1]) {
          const p = centre.slice()
          p[j] = (centre[j] ?? 0) + sign * spread
          pts.push(p)
          logw.push(Math.log(1 / (2 * (d + kappa))) + logf(p))
        }
      }
      return weightedMoments(pts, logw, to)
    },
  }
}

/** The fixed standard normal draws of a sampling rule: `n` antithetic pairs of `d` coordinates, a pure function of the seed. */
const drawCache = new Map<string, readonly number[]>()
function antitheticDraws(seed: number, pairs: number, d: number): readonly number[] {
  const id = `${seed}/${pairs}/${d}`
  let draws = drawCache.get(id)
  if (draws === undefined) {
    draws = toFlat(normal(foldIn(key(seed), 0), [pairs, d]))
    drawCache.set(id, draws)
  }
  return draws
}

/**
 * Importance sampling from the cavities: `samples` points (half of them the mirror images of the other half) drawn once
 * from a fixed seed, so the rule is a deterministic function of its input, scaled to each cavity and weighed by the
 * factor. The estimate of the mass and the moments has the Monte Carlo error of the sample size (and the weights
 * degenerate when the factor sits far in the tail of the cavities); the rule serves the factors nothing else reaches.
 *
 * @param samples - The number of points (default 2048; an odd number is rounded up).
 * @param seed - The seed of the draws (default 0).
 * @returns The rule.
 * @throws OmaiError with code `value` for fewer than 2 samples.
 * @concept numerical-projections
 * @index function Importance-sampling rule with fixed antithetic draws for numeric factors.
 * @example
 * ```ts
 * import { sampleRule } from 'omai-engine/message-passing'
 * // exp(-x^2 / 2) against N(0, 1): the tilted variance is 1 / 2 (within the Monte Carlo error of 4096 points).
 * const r = sampleRule(4096, 1).moments({ bases: [{ mean: 0, variance: 1 }], to: 0, logf: ([x]) => -0.5 * (x as number) ** 2 })
 * Math.abs(r.variance - 0.5) < 0.03 // => true
 * ```
 */
export function sampleRule(samples = 2048, seed = 0): TiltedRule {
  if (!(Number.isInteger(samples) && samples >= 2)) throw new OmaiError('value', 'sampleRule: need at least 2 samples')
  const pairs = Math.ceil(samples / 2)
  return {
    name: `sample(${2 * pairs})`,
    moments: ({ bases, to, logf }) => {
      const free = bases.map((b, d) => (isPoint(b) ? -1 : d)).filter((d) => d >= 0)
      const z = antitheticDraws(seed, pairs, free.length)
      const pts: number[][] = []
      const logw: number[] = []
      for (let i = 0; i < pairs; i++) {
        for (const sign of [1, -1]) {
          const p = bases.map((b) => b.mean)
          for (const [k, j] of free.entries())
            p[j] =
              (bases[j] as GaussianBase).mean +
              sign * Math.sqrt((bases[j] as GaussianBase).variance) * (z[i * free.length + k] ?? 0)
          pts.push(p)
          logw.push(logf(p))
        }
      }
      // The weights are the factor itself (the cavities are the proposal): the mass is their mean.
      const r = weightedMoments(pts, logw, to)
      return { ...r, logMass: r.logMass - Math.log(pts.length) }
    },
  }
}

/** log of the mass-normalised Gaussian bases times the factor, with the free coordinates in `free`: the function Laplace maximises. */
function laplaceObjective(bases: readonly GaussianBase[], free: readonly number[], logf: TiltedProblem['logf']) {
  return (x: readonly number[]): number => {
    let h = logf(x)
    for (const j of free) {
      const b = bases[j] as GaussianBase
      h += -0.5 * Math.log(2 * Math.PI * b.variance) - (0.5 * ((x[j] ?? 0) - b.mean) ** 2) / b.variance
    }
    return h
  }
}

/**
 * Laplace's method: find the mode of the factor times the cavities (Newton's method from the cavities' means, with
 * finite-difference derivatives and a backtracking line search) and the curvature there, and take the Gaussian they
 * define as the marginal: its mean is the mode and its variance the inverse of the negative Hessian. The log mass is the
 * Laplace approximation `h(mode) + (d / 2) log 2 pi + log det(Sigma) / 2`. It is exact for a Gaussian factor and
 * approximate otherwise, and a local method: it finds the mode nearest the cavities' means, whatever the other modes hold.
 *
 * @returns The rule.
 * @throws OmaiError with code `value` when the density has no maximum with negative-definite curvature near the start.
 * @concept numerical-projections
 * @index structure Laplace's mode-and-curvature rule for numeric factors, with its evidence.
 * @example
 * ```ts
 * import { laplaceRule } from 'omai-engine/message-passing'
 * // exp(-x^2 / 2) against N(0, 1) is Gaussian: the Laplace rule is exact (mass 1 / sqrt 2, variance 1 / 2).
 * const r = laplaceRule.moments({ bases: [{ mean: 0, variance: 1 }], to: 0, logf: ([x]) => -0.5 * (x as number) ** 2 })
 * [Math.exp(r.logMass).toFixed(6), r.variance.toFixed(6)] // => ['0.707107', '0.500000']
 * ```
 */
export const laplaceRule: TiltedRule = {
  name: 'laplace',
  moments: ({ bases, to, logf }) => {
    const free = bases.map((b, d) => (isPoint(b) ? -1 : d)).filter((d) => d >= 0)
    const d = free.length
    const h = laplaceObjective(bases, free, logf)
    const x = bases.map((b) => b.mean)
    const steps = free.map((j) => 1e-3 * Math.sqrt((bases[j] as GaussianBase).variance))
    const derivatives = (at: readonly number[]): { grad: number[]; hess: number[] } => {
      const grad = new Array<number>(d).fill(0)
      const hess = new Array<number>(d * d).fill(0)
      const f0 = h(at)
      const shifted = (shifts: readonly [number, number][]): number => {
        const p = at.slice()
        for (const [k, s] of shifts) p[free[k] as number] = (p[free[k] as number] ?? 0) + s
        return h(p)
      }
      for (let a = 0; a < d; a++) {
        const sa = steps[a] as number
        const fp = shifted([[a, sa]])
        const fm = shifted([[a, -sa]])
        grad[a] = (fp - fm) / (2 * sa)
        hess[a * d + a] = (fp - 2 * f0 + fm) / (sa * sa)
        for (let b = a + 1; b < d; b++) {
          const sb = steps[b] as number
          const v =
            (shifted([
              [a, sa],
              [b, sb],
            ]) -
              shifted([
                [a, sa],
                [b, -sb],
              ]) -
              shifted([
                [a, -sa],
                [b, sb],
              ]) +
              shifted([
                [a, -sa],
                [b, -sb],
              ])) /
            (4 * sa * sb)
          hess[a * d + b] = v
          hess[b * d + a] = v
        }
      }
      return { grad, hess }
    }
    let value = h(x)
    if (!Number.isFinite(value))
      throw new OmaiError('value', 'laplace: the factor is not finite at the mean of the cavities')
    for (let iter = 0; iter < 100; iter++) {
      const { grad, hess } = derivatives(x)
      // Newton step on the negative Hessian, made positive definite by a ridge when it is not.
      let ridge = 0
      let fact = spd(
        hess.map((v) => -v),
        d,
      )
      while (fact === undefined && ridge < 1e12) {
        ridge = ridge === 0 ? 1e-6 * Math.max(1, ...hess.map(Math.abs)) : ridge * 10
        fact = spd(
          hess.map((v, i) => -v + (i % (d + 1) === 0 ? ridge : 0)),
          d,
        )
      }
      if (fact === undefined)
        throw new OmaiError('value', 'laplace: the curvature at the mean of the cavities is not usable')
      const step = fact.solve(grad)
      let t = 1
      let moved = false
      for (let k = 0; k < 40; k++) {
        const trial = x.slice()
        for (const [a, j] of free.entries()) trial[j] = (x[j] ?? 0) + t * (step[a] ?? 0)
        const v = h(trial)
        if (Number.isFinite(v) && v >= value - 1e-14 * Math.abs(value)) {
          for (const j of free) x[j] = trial[j] as number
          const gain = v - value
          value = v
          moved = true
          if (gain < 1e-13 * (1 + Math.abs(value)) && t * Math.max(...step.map(Math.abs)) < 1e-9) iter = 1000
          break
        }
        t /= 2
      }
      if (!moved) break
      if (Math.max(...step.map((s) => Math.abs(t * s))) < 1e-10) break
    }
    const { hess } = derivatives(x)
    const fact = spd(
      hess.map((v) => -v),
      d,
    )
    if (fact === undefined)
      throw new OmaiError('value', 'laplace: the density has no maximum with negative-definite curvature here')
    const covariance = fact.inverse()
    const k = free.indexOf(to)
    const variance = k >= 0 ? (covariance[k * d + k] ?? 0) : 0
    return { logMass: value + 0.5 * d * Math.log(2 * Math.PI) - 0.5 * fact.logDet, mean: x[to] ?? 0, variance }
  },
}

/** The rules by name, as functions of their knobs: the table of the numeric slot. */
export const numericRules: Readonly<Record<string, (knobs: Readonly<Record<string, number>>) => TiltedRule>> = {
  quadrature: () => quadratureRule,
  'gauss-hermite': (k) => gaussHermiteRule(k['points'] ?? 32),
  unscented: (k) => unscentedRule(k['kappa'] ?? 2),
  sample: (k) => sampleRule(k['samples'] ?? 2048, k['seed'] ?? 0),
  laplace: () => laplaceRule,
}

/**
 * The numerical projection: moment matching by EP's rules, with every numeric operator of the graph told to find its
 * tilted moments by the given rule (the projection of the engine's `numeric` slot).
 *
 * @param rule - The rule.
 * @returns The projection.
 * @concept numerical-projections
 * @index function The numerical projection slot: moment matching with a chosen numeric rule.
 * @example
 * ```ts
 * import { laplaceRule, numericProjection } from 'omai-engine/message-passing'
 * numericProjection(laplaceRule).name // => 'numeric(laplace)'
 * ```
 */
export function numericProjection(rule: TiltedRule): Projection {
  return {
    name: `numeric(${rule.name})`,
    hook: 'ep',
    arguments: 'cavity',
    evidence: 'ep-energy',
    numeric: rule,
    description: `moment matching with the numeric operators integrated by the ${rule.name} rule`,
  }
}

/** The knobs of numeric EP: the common ones and the rule with its own knobs. */
export interface NumericEpOptions extends MessagePassingOptions {
  /** The numeric rule (default: the panelled quadrature of the numeric operator). */
  readonly rule?: TiltedRule
}

/**
 * Expectation propagation with numeric factors integrated by a rule of your choice (`quadratureRule`, `gaussHermiteRule`,
 * `laplaceRule`, `sampleRule`, `unscentedRule`): the factors of the graph that have closed forms keep them; the numeric
 * operators (a robust likelihood, a non-conjugate link, the clutter problem) find their tilted moments, and the evidence
 * of the run, by the rule. The tape is EP's; the evidence is the EP energy with the rule's masses (so, with Laplace, the
 * Laplace approximation of each factor's mass). The rules are pure functions, so a run serialises and resumes to the same
 * state and has anytime snapshots as every engine run does.
 *
 * @param g - The compiled graph.
 * @param options - The rule, plus schedule, granularity, damping, improper policy, tolerance and sweep cap.
 * @returns The algorithm; its tape is described by {@link epContract}.
 * @concept numerical-projections
 * @reference Minka, T. (2001). Expectation propagation for approximate Bayesian inference (UAI 2001).
 * @reference Smola, A., Vishwanathan, V. and Eskin, E. (2003). Laplace propagation (NIPS 2003).
 * @index algorithm EP with numeric factors projected by quadrature, Laplace, sample or sigma-point rules.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussHermiteRule, gaussianPrior, marginalsOf, numericEpAlgorithm, numericOperator } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const bell = numericOperator({ name: 'bell', arity: 1, logDensity: ([x]) => -0.5 * (x as number) ** 2 })
 * const g = compileGraph(
 *   { variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1), { operator: 'bell', args: ['x'] }] },
 *   { ...defaultOperators(), bell },
 * )
 * const r = runAlgorithm(numericEpAlgorithm(g, { rule: gaussHermiteRule(16) }), { maxSteps: 50 })
 * Number(marginalsOf(g, r.state).x?.variance).toFixed(4) // => '0.5000'
 * ```
 */
export function numericEpAlgorithm(g: CompiledGraph, options: NumericEpOptions = {}): Algorithm<MpState, EpPath> {
  const rule = options.rule ?? quadratureRule
  const alg = messagePassingAlgorithm(g, messagePassingSettings(numericProjection(rule), epDefaults, options))
  return { ...alg, name: 'numeric-ep' } as Algorithm<MpState, EpPath>
}

/**
 * Numeric EP on a graph as a method: the common knobs and the rule (a tagged union: each rule brings its own knobs) as
 * a space, the tape as {@link epContract}.
 *
 * @param g - The compiled graph.
 * @returns The method, named `numeric-ep`.
 * @concept numerical-projections
 * @index function The numeric EP method of a graph: a rule knob (quadrature, Gauss-Hermite, Laplace, sample, unscented).
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussianPrior, numericEpMethod } from 'omai-engine/message-passing'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * numericEpMethod(g).name // => 'numeric-ep'
 * ```
 */
export function numericEpMethod(g: CompiledGraph): Method<MpState, Config, EpPath> {
  const base = messagePassingSpace(epDefaults)
  return defineMethod<MpState, Config, EpPath>({
    name: 'numeric-ep',
    config: dict({
      ...base.spaces,
      rule: union(
        {
          quadrature: dict({}),
          'gauss-hermite': dict({ points: int(2, 200, { default: 32, description: 'nodes per argument' }) }),
          laplace: dict({}),
          sample: dict({
            samples: int(2, 1_000_000, {
              default: 2048,
              log: true,
              description: 'importance samples (antithetic pairs)',
            }),
            seed: int(0, 1_000_000, { default: 0, description: 'the seed of the fixed draws' }),
          }),
          unscented: dict({ kappa: real(0.1, 10, { default: 2, description: 'the spread of the sigma points' }) }),
        },
        { default: 'quadrature', description: 'how numeric operators find their tilted moments' },
      ),
    }),
    make: (c) => {
      const r = c['rule']
      const knobs: Record<string, number> = {}
      let kind = 'quadrature'
      if (typeof r === 'object' && r !== null && !Array.isArray(r)) {
        for (const [k, v] of Object.entries(r)) {
          if (k === 'kind' && typeof v === 'string') kind = v
          else if (typeof v === 'number') knobs[k] = v
        }
      }
      const make = numericRules[kind]
      if (make === undefined) throw new OmaiError('value', `numeric EP: unknown rule ${kind}`)
      return numericEpAlgorithm(g, { ...messagePassingOptions(c), rule: make(knobs) })
    },
    contract: epContract,
  })
}
