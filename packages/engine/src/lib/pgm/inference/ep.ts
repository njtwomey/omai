/**
 * Expectation propagation (Minka 2001), approximate inference that refines one site approximation at a time by matching
 * moments with the true factor: a fully factorised version on discrete factor graphs, and the Gaussian version for
 * latent-Gaussian models (a Gaussian prior with non-Gaussian likelihood sites, as in GP classification), returning the
 * posterior and a log-evidence estimate.
 */

import { type Json, logAddExp, OmaiError } from '../../base/index.ts'
import { gaussHermite } from '../../numerics/quadrature/index.ts'
import { logNdtr } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import {
  type Algorithm,
  channels,
  type RunStatus,
  runAlgorithm,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import { dotVec, inverse, matVec, type Rows, symmetrise } from '../_dense.ts'
import type { Factor } from '../factor.ts'
import { gaussianFactor, gaussianFromMoments, gaussianLogMass, gaussianProduct } from '../gaussian-factor.ts'
import type { DiscreteModel } from '../models/factor-graph.ts'
import {
  type Compiled,
  compile,
  factorBeliefLogs,
  marginalFactor,
  normaliseLog,
  observedMarginals,
} from './_compiled.ts'

/** Options of {@link expectationPropagation}. */
export interface DiscreteEpOptions {
  readonly evidence?: Readonly<Record<string, string | number>>
  /** The power alpha in (0, 1] of power EP (default 1: ordinary EP, whose fixed points are those of loopy BP). */
  readonly power?: number
  /** Damping in [0, 1) of the site parameters (log site values). */
  readonly damping?: number
  readonly tol?: number
  readonly maxIter?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a discrete EP run: the log site approximation of every factor on every variable. */
export interface DiscreteEpState {
  /** Site messages by edge (factor, variable), log and normalised. */
  readonly sites: readonly Float64Array[]
  readonly change: number
}

/** Result of {@link expectationPropagation}. */
export interface DiscreteEpResult {
  readonly marginals: Record<string, Factor>
  /** The Bethe estimate of log Z at the fixed point (power 1 only; NaN otherwise). */
  readonly logPartition: number
  readonly converged: boolean
  readonly status: RunStatus
  readonly iterations: number
}

function qOf(c: Compiled, sites: readonly Float64Array[], i: number): Float64Array {
  const q = new Float64Array(c.card[i] ?? 1)
  for (const e of c.variableEdges[i] ?? []) {
    const m = sites[e]
    if (m !== undefined) for (let x = 0; x < q.length; x++) q[x] = (q[x] ?? 0) + (m[x] ?? 0)
  }
  normaliseLog(q)
  return q
}

/**
 * Expectation propagation on the factor graph of a discrete model with a fully factorised approximation, as an
 * {@link Algorithm}: each factor's site is refined in turn by removing (a power alpha of) it from the approximation to
 * form the cavity, multiplying in f^alpha, matching the marginals of that tilted distribution, and dividing the cavity
 * back out. With alpha = 1 the fixed points are those of loopy belief propagation; smaller alpha gives power EP
 * (alpha-divergence message passing, Minka 2004). Damping acts on the log site values.
 *
 * @param model - The model.
 * @param options - Evidence, power, damping and stopping rule.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a power outside (0, 1] or damping outside [0, 1).
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { isingModel, expectationPropagationAlgorithm } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(expectationPropagationAlgorithm(isingModel({ rows: 3, cols: 3, coupling: 0.3 })), { maxSteps: 200 }).status // => 'converged'
 * ```
 */
export function expectationPropagationAlgorithm(
  model: DiscreteModel,
  options: DiscreteEpOptions = {},
): Algorithm<DiscreteEpState> {
  const c = compile(model, options.evidence ?? {})
  const alpha = options.power ?? 1
  const damping = options.damping ?? 0
  const tol = options.tol ?? 1e-10
  if (!(alpha > 0 && alpha <= 1)) throw new OmaiError('value', 'expectationPropagation: the power must be in (0, 1]')
  if (!(damping >= 0 && damping < 1)) throw new OmaiError('value', 'expectationPropagation: damping must be in [0, 1)')
  return {
    name: 'expectationPropagation',
    config: { power: alpha, damping, tol } as Json,
    schema: channels({
      metrics: { change: { description: 'largest change of a site (log)', direction: 'lower' } },
      messages: { site: { description: 'a factor site approximation on one variable (log values)' } },
    }),
    init: () => ({
      sites: c.edges.map((e) => new Float64Array(c.card[e.variable] ?? 1).fill(-Math.log(c.card[e.variable] ?? 1))),
      change: Number.POSITIVE_INFINITY,
    }),
    step: (state) => {
      const sites = [...state.sites]
      let change = 0
      for (const [a, f] of c.factors.entries()) {
        const es = c.factorEdges[a] ?? []
        const q = f.vars.map((i) => qOf(c, sites, i))
        const cavity = es.map((e, k) => {
          const m = sites[e] as Float64Array
          const out = (q[k] as Float64Array).map((v, x) => v - alpha * (m[x] ?? 0))
          normaliseLog(out)
          return out
        })
        const tilted = factorBeliefLogs(f, cavity, alpha)
        const d = f.vars.length
        for (const [k, e] of es.entries()) {
          const card = c.card[f.vars[k] ?? 0] ?? 1
          const marg = new Float64Array(card).fill(Number.NEGATIVE_INFINITY)
          for (let entry = 0; entry < tilted.length; entry++) {
            const x = f.digits[entry * d + k] ?? 0
            const v = tilted[entry] ?? Number.NEGATIVE_INFINITY
            marg[x] = logAddExp(marg[x] ?? Number.NEGATIVE_INFINITY, v)
          }
          normaliseLog(marg)
          const old = sites[e] as Float64Array
          const qi = q[k] as Float64Array
          // m_new = m + (q_new - q) / alpha, in logs; a state impossible under the tilted distribution stays impossible.
          const next = old.map((m, x) => {
            const nq = marg[x] ?? Number.NEGATIVE_INFINITY
            if (nq === Number.NEGATIVE_INFINITY) return Number.NEGATIVE_INFINITY
            const qq = qi[x] ?? Number.NEGATIVE_INFINITY
            return qq === Number.NEGATIVE_INFINITY ? nq : m + (nq - qq) / alpha
          })
          const damped = next.map((v, x) =>
            damping > 0 && Number.isFinite(v) && Number.isFinite(old[x] ?? Number.NaN)
              ? (1 - damping) * v + damping * (old[x] ?? 0)
              : v,
          )
          normaliseLog(damped)
          for (let x = 0; x < damped.length; x++) {
            const a1 = damped[x] ?? 0
            const b1 = old[x] ?? 0
            if (Number.isFinite(a1) || Number.isFinite(b1))
              change = Math.max(change, Math.abs(Math.exp(a1) - Math.exp(b1)))
          }
          sites[e] = damped
        }
      }
      return { sites, change }
    },
    done: (state, t) => (t > 0 && state.change <= tol ? 'converged' : undefined),
    observe: (state) => ({
      metrics: { change: Number.isFinite(state.change) ? state.change : -1 },
      messages: c.edges.map((e, k) => ({
        kind: 'site',
        payload: { factor: e.factor, variable: c.names[e.variable] ?? '', values: Array.from(state.sites[k] ?? []) },
        from: `f${e.factor}`,
        to: c.names[e.variable] ?? '',
      })),
    }),
  }
}

/**
 * Run discrete expectation propagation ({@link expectationPropagationAlgorithm}) and return the approximate marginals.
 *
 * @param model - The model.
 * @param options - Evidence, power, damping, stopping rule and tracer.
 * @returns Marginals, the Bethe estimate of log Z (power 1) and convergence information.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { isingModel, expectationPropagation, beliefPropagation } from 'omai-engine/pgm'
 * const m = isingModel({ rows: 2, cols: 2, coupling: 0.3, field: 0.2 })
 * const ep = expectationPropagation(m).marginals
 * const bp = beliefPropagation(m).marginals
 * Object.keys(bp).every((v) => Math.abs(Math.exp(ep[v]?.logValues[0] ?? 0) - Math.exp(bp[v]?.logValues[0] ?? 1)) < 1e-8) // => true
 * ```
 */
export function expectationPropagation(model: DiscreteModel, options: DiscreteEpOptions = {}): DiscreteEpResult {
  const c = compile(model, options.evidence ?? {})
  const result = runAlgorithm(expectationPropagationAlgorithm(model, options), {
    maxSteps: options.maxIter ?? 500,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const sites = result.state.sites
  const marginals: Record<string, Factor> = observedMarginals(c)
  const q = c.names.map((_, i) => qOf(c, sites, i))
  for (const [i, name] of c.names.entries()) marginals[name] = marginalFactor(name, q[i] as Float64Array)
  let logPartition = Number.NaN
  if ((options.power ?? 1) === 1) {
    // Bethe free energy of the EP beliefs: factor beliefs are the tilted distributions.
    let free = 0
    for (const [a, f] of c.factors.entries()) {
      const es = c.factorEdges[a] ?? []
      const cavity = es.map((e, k) => {
        const out = (q[f.vars[k] ?? 0] as Float64Array).map((v, x) => v - (sites[e]?.[x] ?? 0))
        normaliseLog(out)
        return out
      })
      const b = factorBeliefLogs(f, cavity)
      for (let e = 0; e < b.length; e++) {
        const lb = b[e] ?? Number.NEGATIVE_INFINITY
        if (lb > Number.NEGATIVE_INFINITY) free += Math.exp(lb) * (lb - (f.logValues[e] ?? 0))
      }
    }
    for (const [i, qi] of q.entries()) {
      let neg = 0
      for (const l of qi) if (l > Number.NEGATIVE_INFINITY) neg += Math.exp(l) * l
      free -= ((c.variableEdges[i]?.length ?? 0) - 1) * neg
    }
    logPartition = c.constant - free
  }
  return {
    marginals: Object.fromEntries(c.all.map((v) => [v.name, marginals[v.name] as Factor])),
    logPartition,
    converged: result.status === 'converged',
    status: result.status,
    iterations: result.steps,
  }
}

/**
 * A site of a latent-Gaussian model: a likelihood of the projection z = weights . x. `probit` is Phi(label z),
 * `gaussian` is N(observation; z, variance) (for which EP is exact), and `custom` takes any log-likelihood of z, whose
 * tilted moments are found by Gauss-Hermite quadrature.
 */
export type EpSite =
  | { readonly kind: 'probit'; readonly weights: readonly number[]; readonly label: 1 | -1 }
  | {
      readonly kind: 'gaussian'
      readonly weights: readonly number[]
      readonly observation: number
      readonly variance: number
    }
  | { readonly kind: 'custom'; readonly weights: readonly number[]; readonly logLikelihood: (z: number) => number }

/** Options of {@link gaussianEp}. */
export interface GaussianEpOptions {
  /** The power alpha in (0, 1] of power EP (default 1). */
  readonly power?: number
  /** Damping in [0, 1) of the site natural parameters. */
  readonly damping?: number
  readonly tol?: number
  readonly maxIter?: number
  /** Gauss-Hermite points for custom (or fractional-power) sites (default 64). */
  readonly quadraturePoints?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a Gaussian EP run: the site natural parameters and the posterior. */
export interface GaussianEpState {
  /** Site precisions tau. */
  readonly tau: Float64Array
  /** Site precision-means nu. */
  readonly nu: Float64Array
  readonly mean: readonly number[]
  readonly covariance: readonly (readonly number[])[]
  /** log of each site's tilted normaliser at its last update. */
  readonly logZ: Float64Array
  /** Cavity mean and variance of each site at its last update. */
  readonly cavity: readonly (readonly [number, number])[]
  readonly change: number
}

/** Result of {@link gaussianEp}. */
export interface GaussianEpResult {
  readonly mean: number[]
  readonly covariance: number[][]
  /** The EP approximation of the log marginal likelihood (power 1; NaN otherwise). */
  readonly logMarginalLikelihood: number
  /** Site precisions and precision-means. */
  readonly sites: readonly { readonly tau: number; readonly nu: number }[]
  readonly converged: boolean
  readonly status: RunStatus
  readonly iterations: number
}

/** log, mean and variance of the tilted distribution N(z; m, v) t(z)^alpha. */
function tiltedMoments(
  site: EpSite,
  m: number,
  v: number,
  alpha: number,
  rule: { nodes: Float64Array; weights: Float64Array },
): [number, number, number] {
  if (site.kind === 'gaussian' || (site.kind === 'probit' && alpha === 1)) {
    if (site.kind === 'gaussian') {
      // N(z; m, v) N(y; z, s)^alpha: a Gaussian in z with precision 1/v + alpha/s.
      const s = site.variance / alpha
      const vh = 1 / (1 / v + 1 / s)
      const mh = vh * (m / v + site.observation / s)
      const logZ =
        -0.5 * Math.log(2 * Math.PI * (v + s)) -
        (site.observation - m) ** 2 / (2 * (v + s)) +
        alpha * (-0.5 * Math.log(2 * Math.PI * site.variance)) -
        -0.5 * Math.log(2 * Math.PI * s)
      return [logZ, mh, vh]
    }
    const y = site.label
    const z = (y * m) / Math.sqrt(1 + v)
    const logPhi = item(logNdtr(z))
    const ratio = Math.exp(-0.5 * z * z - 0.5 * Math.log(2 * Math.PI) - logPhi)
    const mh = m + (y * v * ratio) / Math.sqrt(1 + v)
    const vh = v - ((v * v * ratio) / (1 + v)) * (z + ratio)
    return [logPhi, mh, vh]
  }
  // Gauss-Hermite: E[g(Z)] = sum w_k g(m + sqrt(2 v) x_k) / sqrt(pi).
  const logT = (z: number): number =>
    site.kind === 'custom' ? site.logLikelihood(z) : item(logNdtr((site as { label: number }).label * z))
  const n = rule.nodes.length
  const logs = new Float64Array(n)
  const zs = new Float64Array(n)
  let max = Number.NEGATIVE_INFINITY
  for (let k = 0; k < n; k++) {
    const z = m + Math.sqrt(2 * v) * (rule.nodes[k] ?? 0)
    zs[k] = z
    logs[k] = Math.log(rule.weights[k] ?? 0) + alpha * logT(z)
    if ((logs[k] ?? Number.NEGATIVE_INFINITY) > max) max = logs[k] ?? max
  }
  if (max === Number.NEGATIVE_INFINITY) throw new OmaiError('value', 'gaussianEp: a tilted distribution has zero mass')
  let s0 = 0
  let s1 = 0
  let s2 = 0
  for (let k = 0; k < n; k++) {
    const w = Math.exp((logs[k] ?? 0) - max)
    s0 += w
    s1 += w * (zs[k] ?? 0)
    s2 += w * (zs[k] ?? 0) ** 2
  }
  const mh = s1 / s0
  return [max + Math.log(s0) - 0.5 * Math.log(Math.PI), mh, s2 / s0 - mh * mh]
}

/** The posterior N(mean, covariance) from the prior and the site parameters. */
function posterior(
  priorPrecision: Rows,
  priorShift: readonly number[],
  sites: readonly EpSite[],
  tau: Float64Array,
  nu: Float64Array,
): { mean: number[]; covariance: number[][] } {
  const P = priorPrecision.map((r) => [...r])
  const h = [...priorShift]
  for (const [i, s] of sites.entries()) {
    const a = s.weights
    for (let r = 0; r < a.length; r++) {
      h[r] = (h[r] ?? 0) + (nu[i] ?? 0) * (a[r] ?? 0)
      for (let q = 0; q < a.length; q++)
        (P[r] as number[])[q] = (P[r]?.[q] ?? 0) + (tau[i] ?? 0) * (a[r] ?? 0) * (a[q] ?? 0)
    }
  }
  const covariance = symmetrise(inverse(P))
  return { mean: matVec(covariance, h), covariance }
}

/**
 * Expectation propagation for a latent-Gaussian model (Minka 2001): a Gaussian prior N(mean, covariance) on x and
 * sites t_i(w_i . x), each approximated by an unnormalised Gaussian in its projection. A sweep refines every site in
 * turn: form the cavity (removing a power alpha of the site), match the mean and variance of the tilted distribution
 * cavity times t_i^alpha (in closed form for probit and Gaussian sites, by Gauss-Hermite quadrature otherwise), and
 * update the site, with optional damping. Records the largest change of a site parameter.
 *
 * @param prior - The prior mean and covariance.
 * @param sites - The sites.
 * @param options - Power, damping, stopping rule and quadrature points.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a power outside (0, 1] or mismatched site weights.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { gaussianEpAlgorithm } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const sites = [{ kind: 'probit' as const, weights: [1, 0.5], label: 1 as const }, { kind: 'probit' as const, weights: [-0.3, 1], label: -1 as const }]
 * runAlgorithm(gaussianEpAlgorithm({ mean: [0, 0], covariance: [[1, 0], [0, 1]] }, sites), { maxSteps: 100 }).status // => 'converged'
 * ```
 */
export function gaussianEpAlgorithm(
  prior: { readonly mean: readonly number[]; readonly covariance: Rows },
  sites: readonly EpSite[],
  options: GaussianEpOptions = {},
): Algorithm<GaussianEpState> {
  const d = prior.mean.length
  const alpha = options.power ?? 1
  const damping = options.damping ?? 0
  const tol = options.tol ?? 1e-10
  if (!(alpha > 0 && alpha <= 1)) throw new OmaiError('value', 'gaussianEp: the power must be in (0, 1]')
  if (!(damping >= 0 && damping < 1)) throw new OmaiError('value', 'gaussianEp: damping must be in [0, 1)')
  for (const s of sites)
    if (s.weights.length !== d) throw new OmaiError('shape', `gaussianEp: site weights need ${d} entries`)
  const P0 = inverse(prior.covariance)
  const h0 = matVec(P0, prior.mean)
  const raw = gaussHermite(options.quadraturePoints ?? 64)
  const rule = { nodes: Float64Array.from(raw.nodes.data), weights: Float64Array.from(raw.weights.data) }
  return {
    name: 'gaussianEp',
    config: { power: alpha, damping, tol, sites: sites.length } as Json,
    schema: channels({
      metrics: { change: { description: 'largest change of a site parameter', direction: 'lower' } },
    }),
    init: () => {
      const tau = new Float64Array(sites.length)
      const nu = new Float64Array(sites.length)
      return {
        tau,
        nu,
        ...posterior(P0, h0, sites, tau, nu),
        logZ: new Float64Array(sites.length),
        cavity: sites.map(() => [0, 1] as const),
        change: Number.POSITIVE_INFINITY,
      }
    },
    step: (state) => {
      const tau = Float64Array.from(state.tau)
      const nu = Float64Array.from(state.nu)
      const logZ = Float64Array.from(state.logZ)
      const cavity = [...state.cavity]
      let post: { mean: number[]; covariance: number[][] } = {
        mean: [...state.mean],
        covariance: state.covariance.map((r) => [...r]),
      }
      let change = 0
      for (const [i, site] of sites.entries()) {
        const a = site.weights
        const m = dotVec(a, post.mean)
        const v = dotVec(a, matVec(post.covariance, a))
        const tauC = 1 / v - alpha * (tau[i] ?? 0)
        const nuC = m / v - alpha * (nu[i] ?? 0)
        if (!(tauC > 0)) continue // skip an update that would give an improper cavity
        const mc = nuC / tauC
        const vc = 1 / tauC
        const [lz, mh, vh] = tiltedMoments(site, mc, vc, alpha, rule)
        const tauNew = (1 / vh - tauC) / alpha
        const nuNew = (mh / vh - nuC) / alpha
        const t1 = (1 - damping) * tauNew + damping * (tau[i] ?? 0)
        const n1 = (1 - damping) * nuNew + damping * (nu[i] ?? 0)
        change = Math.max(change, Math.abs(t1 - (tau[i] ?? 0)), Math.abs(n1 - (nu[i] ?? 0)))
        tau[i] = t1
        nu[i] = n1
        logZ[i] = lz
        cavity[i] = [mc, vc]
        post = posterior(P0, h0, sites, tau, nu)
      }
      return { tau, nu, ...post, logZ, cavity, change }
    },
    done: (state, t) => (t > 0 && state.change <= tol ? 'converged' : undefined),
    observe: (state) => ({ metrics: { change: Number.isFinite(state.change) ? state.change : -1 } }),
  }
}

/**
 * Run expectation propagation on a latent-Gaussian model ({@link gaussianEpAlgorithm}): the Gaussian approximation of
 * the posterior and the EP approximation of the log marginal likelihood,
 * sum_i log Ztilde_i + log integral of the prior times the Gaussian sites.
 *
 * @param prior - The prior mean and covariance.
 * @param sites - The sites.
 * @param options - Power, damping, stopping rule, quadrature points and tracer.
 * @returns The posterior moments, the log marginal likelihood and the site parameters.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { gaussianEp } from 'omai-engine/pgm'
 * // One probit site is matched exactly: Z = Phi(m / sqrt(1 + v)) = Phi(0) = 1/2.
 * Math.round(Math.exp(gaussianEp({ mean: [0], covariance: [[1]] }, [{ kind: 'probit', weights: [1], label: 1 }]).logMarginalLikelihood) * 1e12) / 1e12 // => 0.5
 * ```
 */
export function gaussianEp(
  prior: { readonly mean: readonly number[]; readonly covariance: Rows },
  sites: readonly EpSite[],
  options: GaussianEpOptions = {},
): GaussianEpResult {
  const result = runAlgorithm(gaussianEpAlgorithm(prior, sites, options), {
    maxSteps: options.maxIter ?? 200,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const s = result.state
  let logMarginalLikelihood = Number.NaN
  if ((options.power ?? 1) === 1) {
    const d = prior.mean.length
    const names = Array.from({ length: d }, (_, k) => `x${k}`)
    let joint = gaussianFromMoments(names, prior.mean, prior.covariance)
    let total = 0
    for (const [i, site] of sites.entries()) {
      const a = site.weights
      const t = s.tau[i] ?? 0
      const nu = s.nu[i] ?? 0
      const [mc, vc] = s.cavity[i] ?? [0, 1]
      // log of the integral of N(z; mc, vc) exp(-t z^2 / 2 + nu z): the site's normaliser.
      const logInt = -0.5 * Math.log(1 + t * vc) + (nu * nu * vc + 2 * nu * mc - t * mc * mc) / (2 * (1 + t * vc))
      total += (s.logZ[i] ?? 0) - logInt
      joint = gaussianProduct(
        joint,
        gaussianFactor(
          names,
          a.map((x) => a.map((y) => t * x * y)),
          a.map((x) => nu * x),
          0,
        ),
      )
    }
    logMarginalLikelihood = total + gaussianLogMass(joint)
  }
  return {
    mean: [...s.mean],
    covariance: s.covariance.map((r) => [...r]),
    logMarginalLikelihood,
    sites: Array.from(s.tau, (tau, i) => ({ tau, nu: s.nu[i] ?? 0 })),
    converged: result.status === 'converged',
    status: result.status,
    iterations: result.steps,
  }
}
