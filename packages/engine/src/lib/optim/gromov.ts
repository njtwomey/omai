/**
 * Gromov-Wasserstein transport, which matches two spaces by comparing their internal distances rather than points
 * across them (Memoli), so it can align point clouds, graphs or shapes in different coordinates. Exact (conditional
 * gradient) and entropic (Sinkhorn-based) solvers as traced algorithms, as in POT's `ot.gromov`.
 */

import { type Json, MdmlError } from '../base/index.ts'
import {
  add,
  fromStorage,
  log,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  square,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../trace/index.ts'
import { emd, sinkhorn } from './transport.ts'

/** The pointwise loss L(c1, c2) between structure entries: squared or Kullback-Leibler. */
export type GromovLoss = 'square' | 'kl'

/** The decomposition L(a, b) = f1(a) + f2(b) - h1(a) h2(b) behind the O(n^2 m + n m^2) GW tensor (Peyre et al. 2016). */
interface GromovTerms {
  readonly constC: Value
  readonly hC1: Value
  readonly hC2: Value
}

function terms(c1: Value, c2: Value, p: Value, q: Value, loss: GromovLoss): GromovTerms {
  const f1 = loss === 'square' ? square(c1) : sub(mul(c1, log(add(c1, 1e-18))), c1)
  const f2 = loss === 'square' ? square(c2) : c2
  const hC1 = c1
  const hC2 = loss === 'square' ? mul(c2, 2) : log(add(c2, 1e-18))
  const n = p.shape[0] ?? 0
  const m = q.shape[0] ?? 0
  const constC = add(
    matmul(matmul(f1, reshape(p, [-1, 1])), fromStorage(new Float64Array(m).fill(1), [1, m], 'float64')),
    matmul(
      fromStorage(new Float64Array(n).fill(1), [n, 1], 'float64'),
      matmul(reshape(q, [1, -1]), swapaxes(f2, 0, 1)),
    ),
  )
  return { constC, hC1, hC2 }
}

/** The GW tensor L(C1, C2) (x) T = constC - hC1 T hC2^T. */
const tensorProduct = (t: GromovTerms, plan: Value): Value =>
  sub(t.constC, matmul(matmul(t.hC1, plan), swapaxes(t.hC2, 0, 1)))

/** Options of {@link gromovWasserstein}. */
export interface GromovOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The pointwise loss (default `square`). */
  readonly loss?: GromovLoss
  /** POT's stopping thresholds on the relative and absolute change of the loss (default 1e-9 each). */
  readonly tolRel?: number
  readonly tolAbs?: number
  readonly maxIterations?: number
  /** A starting plan (default p q^T). */
  readonly g0?: TensorLike
}

/** State of the Gromov-Wasserstein solvers. */
export interface GromovState {
  readonly plan: Tensor
  readonly loss: number
  readonly step: number
  readonly change: number
}

/** Result of the Gromov-Wasserstein solvers. */
export interface GromovResult {
  readonly plan: Tensor
  /** The GW discrepancy sum_{ijkl} L(C1_ik, C2_jl) T_ij T_kl at the plan. */
  readonly distance: number
  readonly iterations: number
  readonly status: RunStatus
  readonly converged: boolean
}

function setup(c1: TensorLike, c2: TensorLike, p: TensorLike, q: TensorLike, loss: GromovLoss) {
  const [C1, C2, pv, qv] = promoteFloat(c1, c2, p, q).map((v) => primalOf(v)) as [Tensor, Tensor, Tensor, Tensor]
  const n = pv.shape[0] ?? 0
  const m = qv.shape[0] ?? 0
  if (C1.shape[0] !== n || C1.shape[1] !== n || C2.shape[0] !== m || C2.shape[1] !== m)
    throw new MdmlError('shape', 'gromovWasserstein: C1 must be [n, n] and C2 [m, m] for p [n] and q [m]')
  const symmetric = isSymmetric(C1) && isSymmetric(C2)
  const forward = terms(C1, C2, pv, qv, loss)
  const backward = symmetric ? forward : terms(swapaxes(C1, 0, 1), swapaxes(C2, 0, 1), pv, qv, loss)
  const gradient = (plan: Value): Value =>
    symmetric ? mul(tensorProduct(forward, plan), 2) : add(tensorProduct(forward, plan), tensorProduct(backward, plan))
  const lossOf = (plan: Value): number => toFlat(sum(mul(tensorProduct(forward, plan), plan)))[0] ?? 0
  const start = (g0: TensorLike | undefined): Tensor =>
    g0 === undefined
      ? (primalOf(matmul(reshape(pv, [-1, 1]), reshape(qv, [1, -1]))) as Tensor)
      : (primalOf(promoteFloat(g0)[0] as Value) as Tensor)
  return { C1, C2, pv, qv, n, m, symmetric, forward, gradient, lossOf, start }
}

function isSymmetric(c: Tensor): boolean {
  const n = c.shape[0] ?? 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < i; j++)
      if (Math.abs(((c.data[i * n + j] as number) ?? 0) - ((c.data[j * n + i] as number) ?? 0)) > 1e-10) return false
  return true
}

/** POT's exact line search for a quadratic along a direction on [0, 1]. */
function quadraticStep(a: number, b: number): number {
  if (a > 0) return Math.min(1, Math.max(0, -b / (2 * a)))
  return a + b < 0 ? 1 : 0
}

/**
 * Gromov-Wasserstein transport between two metric-measure spaces given by their structure matrices (Memoli 2011;
 * POT's ot.gromov.gromov_wasserstein) by the conditional gradient method, as a traced algorithm: the linearised
 * problem is an exact transport (emd) on the gradient 2 L(C1, C2) (x) T, and the step is the exact minimiser of the
 * quadratic along it. A nonconvex problem: the result is a stationary plan from the start p q^T.
 *
 * @param c1 - The source structure `[n, n]` (distances, similarities).
 * @param c2 - The target structure `[m, m]`.
 * @param p - Source weights `[n]`.
 * @param q - Target weights `[m]`.
 * @param options - Loss, tolerances and starting plan.
 * @returns The algorithm.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { gromovWassersteinAlgorithm } from 'mdml-engine/optim'
 * gromovWassersteinAlgorithm([[0, 1], [1, 0]], [[0, 1], [1, 0]], [0.5, 0.5], [0.5, 0.5]).name // => 'gromovWasserstein'
 * ```
 */
export function gromovWassersteinAlgorithm(
  c1: TensorLike,
  c2: TensorLike,
  p: TensorLike,
  q: TensorLike,
  options: GromovOptions = {},
): Algorithm<GromovState> {
  const loss = options.loss ?? 'square'
  const s0 = setup(c1, c2, p, q, loss)
  const tolRel = options.tolRel ?? 1e-9
  const tolAbs = options.tolAbs ?? 1e-9
  return {
    name: 'gromovWasserstein',
    config: { loss, tolRel, tolAbs, symmetric: s0.symmetric } as Json,
    schema: channels({
      metrics: {
        loss: { description: 'GW discrepancy of the plan', direction: 'lower' },
        step: { description: 'line-search step' },
        change: { description: 'absolute change of the loss' },
      },
      tensors: { plan: { role: 'param', description: 'transport plan' } },
    }),
    init: () => {
      const plan = s0.start(options.g0)
      return { plan, loss: s0.lossOf(plan), step: 0, change: Infinity }
    },
    step: (s) => {
      const grad = s0.gradient(s.plan)
      const vertex = emd(s0.pv, s0.qv, primalOf(grad)).plan
      const delta = primalOf(sub(vertex, s.plan)) as Tensor
      const dot = matmul(matmul(s0.forward.hC1, delta), swapaxes(s0.forward.hC2, 0, 1))
      const a = -(toFlat(sum(mul(dot, delta)))[0] ?? 0)
      const b = s0.symmetric
        ? -2 * (toFlat(sum(mul(dot, s.plan)))[0] ?? 0)
        : -(
            (toFlat(sum(mul(dot, s.plan)))[0] ?? 0) +
            (toFlat(sum(mul(matmul(matmul(s0.forward.hC1, s.plan), swapaxes(s0.forward.hC2, 0, 1)), delta)))[0] ?? 0)
          )
      const alpha = quadraticStep(a, b)
      const cost = s.loss + a * alpha * alpha + b * alpha
      const plan = primalOf(add(s.plan, mul(delta, alpha))) as Tensor
      return { plan, loss: cost, step: alpha, change: Math.abs(cost - s.loss) }
    },
    done: (s) =>
      s.change < tolAbs || (s.loss !== 0 && s.change / Math.abs(s.loss) < tolRel) ? 'converged' : undefined,
    observe: (s) => ({
      metrics: { loss: s.loss, step: s.step, change: s.change },
      tensors: { plan: { role: 'param', value: s.plan } },
    }),
  }
}

/**
 * The Gromov-Wasserstein plan and discrepancy between two structures.
 *
 * @param c1 - The source structure `[n, n]`.
 * @param c2 - The target structure `[m, m]`.
 * @param p - Source weights.
 * @param q - Target weights.
 * @param options - Loss, tolerances, starting plan, iteration cap (default 10000, POT's) and run options.
 * @returns The plan and the GW discrepancy.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { gromovWasserstein } from 'mdml-engine/optim'
 * // Two copies of the same three-point space: the discrepancy is zero.
 * const c = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
 * Math.abs(gromovWasserstein(c, c, [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]).distance) < 1e-12 // => true
 * ```
 */
export function gromovWasserstein(
  c1: TensorLike,
  c2: TensorLike,
  p: TensorLike,
  q: TensorLike,
  options: GromovOptions = {},
): GromovResult {
  const { loss: _l, tolRel: _r, tolAbs: _a, maxIterations, g0: _g, ...run } = options
  const result = runAlgorithm(gromovWassersteinAlgorithm(c1, c2, p, q, options), {
    ...run,
    maxSteps: maxIterations ?? 10000,
  })
  const s0 = setup(c1, c2, p, q, options.loss ?? 'square')
  return {
    plan: result.state.plan,
    distance: s0.lossOf(result.state.plan),
    iterations: result.steps,
    status: result.status,
    converged: result.status === 'converged',
  }
}

/** Options of {@link entropicGromovWasserstein}. */
export interface EntropicGromovOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  readonly loss?: GromovLoss
  /** Stop once |T - T_prev|_F falls to this (default 1e-9, POT's). */
  readonly tol?: number
  /** The inner Sinkhorn tolerance (default 1e-12). */
  readonly innerTol?: number
  readonly maxIterations?: number
  readonly g0?: TensorLike
}

/**
 * Entropic Gromov-Wasserstein (Peyre, Cuturi and Solomon 2016; POT's ot.gromov.entropic_gromov_wasserstein with the
 * projected gradient solver) as a traced algorithm: T+ = sinkhorn(p, q, 2 L(C1, C2) (x) T, epsilon) until the plan
 * stops changing.
 *
 * @param c1 - The source structure `[n, n]`.
 * @param c2 - The target structure `[m, m]`.
 * @param p - Source weights.
 * @param q - Target weights.
 * @param epsilon - The entropic regularisation.
 * @param options - Loss, tolerances and starting plan.
 * @returns The algorithm.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { entropicGromovWassersteinAlgorithm } from 'mdml-engine/optim'
 * entropicGromovWassersteinAlgorithm([[0, 1], [1, 0]], [[0, 1], [1, 0]], [0.5, 0.5], [0.5, 0.5], 0.1).name // => 'entropicGromovWasserstein'
 * ```
 */
export function entropicGromovWassersteinAlgorithm(
  c1: TensorLike,
  c2: TensorLike,
  p: TensorLike,
  q: TensorLike,
  epsilon: number,
  options: EntropicGromovOptions = {},
): Algorithm<GromovState> {
  const loss = options.loss ?? 'square'
  const s0 = setup(c1, c2, p, q, loss)
  const tol = options.tol ?? 1e-9
  const innerTol = options.innerTol ?? 1e-12
  return {
    name: 'entropicGromovWasserstein',
    config: { loss, epsilon, tol } as Json,
    schema: channels({
      metrics: {
        loss: { description: 'GW discrepancy of the plan', direction: 'lower' },
        change: { description: 'Frobenius change of the plan' },
      },
      tensors: { plan: { role: 'param', description: 'transport plan' } },
    }),
    init: () => {
      const plan = s0.start(options.g0)
      return { plan, loss: s0.lossOf(plan), step: 0, change: Infinity }
    },
    step: (s) => {
      const grad = primalOf(s0.gradient(s.plan))
      const plan = sinkhorn(s0.pv, s0.qv, grad, epsilon, { tol: innerTol, maxIterations: 100000 }).plan
      const d = toFlat(sub(plan, s.plan))
      let change = 0
      for (const v of d) change += v * v
      return { plan, loss: s0.lossOf(plan), step: 1, change: Math.sqrt(change) }
    },
    done: (s) => (s.change <= tol ? 'converged' : Number.isNaN(s.change) ? 'diverged' : undefined),
    observe: (s) => ({
      metrics: { loss: s.loss, change: s.change },
      tensors: { plan: { role: 'param', value: s.plan } },
    }),
  }
}

/**
 * The entropic Gromov-Wasserstein plan and its discrepancy.
 *
 * @param c1 - The source structure `[n, n]`.
 * @param c2 - The target structure `[m, m]`.
 * @param p - Source weights.
 * @param q - Target weights.
 * @param epsilon - The entropic regularisation.
 * @param options - Loss, tolerances, starting plan, iteration cap (default 1000, POT's) and run options.
 * @returns The plan and the (unregularised) GW discrepancy at it.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { entropicGromovWasserstein } from 'mdml-engine/optim'
 * import { sum, toArray } from 'mdml-engine/tensor'
 * const r = entropicGromovWasserstein([[0, 1], [1, 0]], [[0, 2], [2, 0]], [0.5, 0.5], [0.5, 0.5], 0.5)
 * Math.round((toArray(sum(r.plan)) as number) * 1e9) / 1e9 // => 1
 * ```
 */
export function entropicGromovWasserstein(
  c1: TensorLike,
  c2: TensorLike,
  p: TensorLike,
  q: TensorLike,
  epsilon: number,
  options: EntropicGromovOptions = {},
): GromovResult {
  const { loss: _l, tol: _t, innerTol: _i, maxIterations, g0: _g, ...run } = options
  const result = runAlgorithm(entropicGromovWassersteinAlgorithm(c1, c2, p, q, epsilon, options), {
    ...run,
    maxSteps: maxIterations ?? 1000,
  })
  return {
    plan: result.state.plan,
    distance: result.state.loss,
    iterations: result.steps,
    status: result.status,
    converged: result.status === 'converged',
  }
}
