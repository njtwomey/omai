import { type Json, maxOf, WhiteboxError } from '../../base/index.ts'
import { eigvalsh } from '../../numerics/linalg/index.ts'
import { toArray } from '../../tensor/index.ts'
import {
  type Algorithm,
  channels,
  type RunStatus,
  runAlgorithm,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import { matVec, sub } from '../_dense.ts'
import {
  type Gmrf,
  type GmrfInformation,
  gmrfFromLinearGaussian,
  type LinearGaussianBayesNet,
} from '../models/linear-gaussian.ts'

/** Options of {@link gaussianBeliefPropagation}. */
export interface GaussianBpOptions {
  /** Observed values by variable. */
  readonly evidence?: Readonly<Record<string, number>>
  /** `flooding` (all messages from the previous iteration, default) or `sequential` (node by node, newest messages). */
  readonly schedule?: 'flooding' | 'sequential'
  /** Damping in [0, 1) of the message parameters. */
  readonly damping?: number
  /** Stop when no mean changes by more than this (default 1e-12). */
  readonly tol?: number
  /** Iteration cap (default 500). */
  readonly maxIter?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a Gaussian belief-propagation run: the information-form message on every directed edge. */
export interface GaussianBpState {
  /** Precision of each message, by directed edge. */
  readonly P: Float64Array
  /** Potential (precision times mean) of each message, by directed edge. */
  readonly H: Float64Array
  readonly means: Float64Array
  readonly change: number
  readonly diverged: boolean
}

/** Result of {@link gaussianBeliefPropagation}. */
export interface GaussianBpResult {
  /** Posterior means (exact whenever the iteration converges). */
  readonly means: Record<string, number>
  /** Approximate posterior variances (exact on trees). */
  readonly variances: Record<string, number>
  readonly converged: boolean
  readonly status: RunStatus
  readonly iterations: number
}

interface Prepared {
  readonly variables: readonly string[]
  readonly J: readonly (readonly number[])[]
  readonly h: readonly number[]
  /** Directed edges (from, to). */
  readonly edges: readonly (readonly [number, number])[]
  /** Incoming edge ids of each node. */
  readonly incoming: readonly (readonly number[])[]
  /** The id of the reverse of each edge. */
  readonly reverse: readonly number[]
}

/** Any Gaussian model Gaussian belief propagation accepts. */
export type GaussianModel = Gmrf | GmrfInformation | LinearGaussianBayesNet

/** The information form (J, h) of a model conditioned on evidence: J_AA and h_A - J_AB x_B. */
function information(
  model: GaussianModel,
  evidence: Readonly<Record<string, number>>,
): { variables: string[]; J: number[][]; h: number[] } {
  const base = model.kind === 'linearGaussianBayesNet' ? gmrfFromLinearGaussian(model) : model
  const full = base.kind === 'gmrf' ? matVec(base.precision, base.mean) : [...base.potential]
  for (const v of Object.keys(evidence)) {
    if (!base.variables.includes(v))
      throw new WhiteboxError('value', `gaussian belief propagation: unknown variable ${v}`)
  }
  const a = base.variables.map((_, i) => i).filter((i) => !((base.variables[i] ?? '') in evidence))
  const b = base.variables.map((_, i) => i).filter((i) => (base.variables[i] ?? '') in evidence)
  const xb = b.map((i) => evidence[base.variables[i] ?? ''] ?? 0)
  const shift = matVec(sub(base.precision, a, b), xb)
  return {
    variables: a.map((i) => base.variables[i] ?? ''),
    J: sub(base.precision, a, a),
    h: a.map((i, k) => (full[i] ?? 0) - (shift[k] ?? 0)),
  }
}

function prepare(model: GaussianModel, evidence: Readonly<Record<string, number>>): Prepared {
  const { variables, J, h } = information(model, evidence)
  const n = variables.length
  const edges: [number, number][] = []
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j && (J[i]?.[j] ?? 0) !== 0) edges.push([i, j])
  const index = new Map(edges.map(([i, j], e) => [`${i},${j}`, e]))
  const incoming = Array.from({ length: n }, () => [] as number[])
  for (const [e, [, j]] of edges.entries()) incoming[j]?.push(e)
  const reverse = edges.map(([i, j]) => index.get(`${j},${i}`) ?? -1)
  return { variables, J, h, edges, incoming, reverse }
}

/** The new message on edge e from the current messages. */
function message(p: Prepared, P: Float64Array, H: Float64Array, e: number): [number, number, boolean] {
  const [i, j] = p.edges[e] ?? [0, 0]
  let jhat = p.J[i]?.[i] ?? 0
  let hhat = p.h[i] ?? 0
  const back = p.reverse[e] ?? -1
  for (const k of p.incoming[i] ?? []) {
    if (k === back) continue
    jhat += P[k] ?? 0
    hhat += H[k] ?? 0
  }
  const jij = p.J[i]?.[j] ?? 0
  if (!(jhat > 0)) return [0, 0, true]
  return [(-jij * jij) / jhat, (-jij * hhat) / jhat, false]
}

function beliefs(p: Prepared, P: Float64Array, H: Float64Array): { means: Float64Array; precisions: Float64Array } {
  const n = p.variables.length
  const means = new Float64Array(n)
  const precisions = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let jhat = p.J[i]?.[i] ?? 0
    let hhat = p.h[i] ?? 0
    for (const k of p.incoming[i] ?? []) {
      jhat += P[k] ?? 0
      hhat += H[k] ?? 0
    }
    precisions[i] = jhat
    means[i] = hhat / jhat
  }
  return { means, precisions }
}

/**
 * Gaussian belief propagation in information form (Weiss and Freeman 2001; Bickson 2008) as an {@link Algorithm}: on a
 * Gaussian Markov random field (or the field of a linear-Gaussian network, conditioned on evidence), each directed edge
 * carries a message N^{-1}(H, P); messages are P_ij = -J_ij^2 / J_i\j and H_ij = -J_ij h_i\j / J_i\j. On trees means and
 * variances are exact after one sweep's worth of iterations; on loopy graphs that converge (for example walk-summable
 * ones) the means are exact and the variances approximate. Records the largest change of the means.
 *
 * @param model - A Gaussian Markov random field (moment or information form) or a linear-Gaussian network.
 * @param options - Evidence, schedule, damping and stopping rule.
 * @returns The algorithm.
 * @concept gaussian-belief-propagation
 * @example
 * ```ts
 * import { gmrf, gaussianBpAlgorithm } from 'whitebox-engine/pgm'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const f = gmrf(['a', 'b', 'c'], [1, 2, 3], [[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
 * runAlgorithm(gaussianBpAlgorithm(f), { maxSteps: 50 }).status // => 'converged'
 * ```
 */
export function gaussianBpAlgorithm(model: GaussianModel, options: GaussianBpOptions = {}): Algorithm<GaussianBpState> {
  const p = prepare(model, options.evidence ?? {})
  const damping = options.damping ?? 0
  const tol = options.tol ?? 1e-12
  if (!(damping >= 0 && damping < 1))
    throw new WhiteboxError('value', 'gaussian belief propagation: damping must be in [0, 1)')
  const sequential = options.schedule === 'sequential'
  const m = p.edges.length
  return {
    name: 'gaussianBeliefPropagation',
    config: { schedule: sequential ? 'sequential' : 'flooding', damping, tol } as Json,
    schema: channels({
      metrics: { change: { description: 'largest change of a posterior mean', direction: 'lower' } },
      tensors: { means: { role: 'state', description: 'current posterior means' } },
    }),
    init: () => {
      const P = new Float64Array(m)
      const H = new Float64Array(m)
      return { P, H, means: beliefs(p, P, H).means, change: Number.POSITIVE_INFINITY, diverged: false }
    },
    step: (s) => {
      const P = Float64Array.from(s.P)
      const H = Float64Array.from(s.H)
      let diverged = false
      const update = (e: number, srcP: Float64Array, srcH: Float64Array): void => {
        const [np, nh, bad] = message(p, srcP, srcH, e)
        if (bad) diverged = true
        P[e] = (1 - damping) * np + damping * (s.P[e] ?? 0)
        H[e] = (1 - damping) * nh + damping * (s.H[e] ?? 0)
      }
      if (sequential) {
        for (let i = 0; i < p.variables.length; i++)
          for (const [e, [from]] of p.edges.entries()) if (from === i) update(e, P, H)
      } else {
        for (let e = 0; e < m; e++) update(e, s.P, s.H)
      }
      const { means } = beliefs(p, P, H)
      const change = maxOf(Array.from(means, (v, i) => Math.abs(v - (s.means[i] ?? 0))))
      return { P, H, means, change: Number.isFinite(change) ? change : Number.POSITIVE_INFINITY, diverged }
    },
    done: (s, t) =>
      s.diverged || (!Number.isFinite(s.change) && t > 0)
        ? 'diverged'
        : t > 0 && s.change <= tol
          ? 'converged'
          : undefined,
    observe: (s) => ({
      metrics: { change: Number.isFinite(s.change) ? s.change : -1 },
      tensors: { means: { role: 'state', value: s.means } },
    }),
  }
}

/**
 * Run Gaussian belief propagation ({@link gaussianBpAlgorithm}) and return the posterior means and variances.
 *
 * @param model - A Gaussian Markov random field (moment or information form) or a linear-Gaussian network.
 * @param options - Evidence, schedule, damping, stopping rule and tracer.
 * @returns Means, variances and convergence information, by variable (observed variables are left out).
 * @concept gaussian-belief-propagation
 * @example
 * ```ts
 * import { linearGaussianBayesNet, gaussianBeliefPropagation } from 'whitebox-engine/pgm'
 * const g = linearGaussianBayesNet([
 *   { name: 'x', parents: [], weights: [], intercept: 0, variance: 1 },
 *   { name: 'y', parents: ['x'], weights: [1], intercept: 0, variance: 1 },
 * ])
 * const r = gaussianBeliefPropagation(g, { evidence: { y: 2 } })
 * [Math.round((r.means['x'] ?? 0) * 1e10) / 1e10, Math.round((r.variances['x'] ?? 0) * 1e10) / 1e10] // => [1, 0.5]
 * ```
 */
export function gaussianBeliefPropagation(model: GaussianModel, options: GaussianBpOptions = {}): GaussianBpResult {
  const p = prepare(model, options.evidence ?? {})
  const result = runAlgorithm(gaussianBpAlgorithm(model, options), {
    maxSteps: options.maxIter ?? 500,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const { means, precisions } = beliefs(p, result.state.P, result.state.H)
  const names = p.variables
  return {
    means: Object.fromEntries(names.map((v, i) => [v, means[i] ?? Number.NaN])),
    variances: Object.fromEntries(names.map((v, i) => [v, 1 / (precisions[i] ?? Number.NaN)])),
    converged: result.status === 'converged',
    status: result.status,
    iterations: result.steps,
  }
}

/**
 * Whether a Gaussian Markov random field is walk-summable (Malioutov, Johnson and Willsky 2006): the spectral radius of
 * |R|, with R the normalised off-diagonal part of the precision, is below one. Gaussian BP then converges and its means
 * are exact.
 *
 * @param field - The field.
 * @returns `true` when walk-summable.
 * @concept gaussian-belief-propagation
 * @example
 * ```ts
 * import { gmrf, isWalkSummable } from 'whitebox-engine/pgm'
 * isWalkSummable(gmrf(['a', 'b', 'c'], [0, 0, 0], [[2, -1, 0], [-1, 2, -1], [0, -1, 2]])) // => true
 * ```
 */
export function isWalkSummable(field: Gmrf | GmrfInformation): boolean {
  const d = field.precision.map((r, i) => 1 / Math.sqrt(r[i] ?? 1))
  const R = field.precision.map((r, i) => r.map((v, j) => (i === j ? 0 : Math.abs(v * (d[i] ?? 0) * (d[j] ?? 0)))))
  if (R.length === 0) return true
  const w = toArray(eigvalsh(R)) as number[]
  return maxOf(w.map(Math.abs)) < 1
}
