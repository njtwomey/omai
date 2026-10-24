// The kernels, the SMO run with its history, and libsvm's rho, shared by the SVMs.
import { type Kernel, type PairwiseKernelMetric, pairwiseKernels } from '../../lib/kernels/index.ts'
import { collecting, type FitOptions, onGeometricSchedule } from '../../lib/learn/index.ts'
import { type SmoProblem, smoAlgorithm } from '../../lib/optim/index.ts'
import { asTensor, primalOf, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { runAlgorithm } from '../../lib/trace/index.ts'

/** A kernel by scikit-learn's SVM names and parameters. */
export interface KernelHyper {
  readonly kernel: 'rbf' | 'linear' | 'poly' | 'sigmoid'
  /** 'scale' is 1 / (d Var(X)), 'auto' is 1 / d (scikit-learn's). */
  readonly gamma: number | 'scale' | 'auto'
  readonly degree: number
  readonly coef0: number
}

/** The resolved kernel parameters, stored with the fitted model. */
export type KernelParams = {
  readonly kernel: KernelHyper['kernel']
  readonly gamma: number
  readonly degree: number
  readonly coef0: number
}

/** Resolve 'scale' and 'auto' against the training data. */
export function resolveKernel(h: KernelHyper, x: Float64Array, d: number): KernelParams {
  let gamma: number
  if (h.gamma === 'auto') gamma = 1 / d
  else if (h.gamma === 'scale') {
    let s = 0
    let s2 = 0
    for (const v of x) {
      s += v
      s2 += v * v
    }
    const variance = s2 / x.length - (s / x.length) ** 2
    gamma = variance > 0 ? 1 / (d * variance) : 1
  } else gamma = h.gamma
  return { kernel: h.kernel, gamma, degree: h.degree, coef0: h.coef0 }
}

/** The kernel matrix between rows, flat row-major `[m * n]`. */
export function kernelMatrix(k: KernelParams, x: TensorLike, y?: TensorLike): number[] {
  const metric: PairwiseKernelMetric | Kernel = k.kernel
  return toFlat(primalOf(pairwiseKernels(x, y, metric, { gamma: k.gamma, degree: k.degree, coef0: k.coef0 })))
}

/** A solved dual with the history of the solve. */
export interface DualRun {
  readonly alpha: Float64Array
  /** The gradient Q alpha + p at the end. */
  readonly grad: Float64Array
  readonly rho: number
  readonly iterations: number
  readonly converged: boolean
  /** The dual objective after each step (from the full gradient). */
  readonly objective: number[]
  readonly workingSets: [number, number][]
  /** The SMO step of each entry of `objective` and `workingSets`. */
  readonly steps: number[]
  /** When asked: the duals at steps on a geometric schedule (and the last), with their rho and objective. */
  readonly snapshots?: readonly DualSnapshot[]
}

/** The dual at one SMO step. */
export interface DualSnapshot {
  readonly step: number
  readonly alpha: Float64Array
  readonly rho: number
  readonly objective: number
}

/** Options of {@link runDual}. */
export interface DualOptions {
  readonly tol: number
  readonly maxIter: number
  /** Keep at most this many history entries, evenly spaced (default 2000). */
  readonly maxHistory?: number
  /** Snapshot the dual on a geometric schedule of steps, for recording the model through training (default false). */
  readonly snapshots?: boolean
}

/**
 * Solve an SMO problem with the core's traced `smoAlgorithm` (without shrinking, so the gradient, and with it the
 * recorded dual objective, is exact at every step), collecting each step's objective and working pair, and libsvm's
 * rho at the end.
 */
export function runDual(problem: SmoProblem, run: FitOptions, options: DualOptions): DualRun {
  const p = Float64Array.from(toFlat(asTensor(problem.p)))
  const n = p.length
  const algorithm = smoAlgorithm(problem, { tol: options.tol, shrinking: false })
  const y = Float64Array.from(toFlat(asTensor(problem.y)))
  const C =
    typeof problem.C === 'number' ? new Float64Array(n).fill(problem.C) : Float64Array.from(toFlat(asTensor(problem.C)))
  const snapshots: DualSnapshot[] = []
  const tracked = collecting(algorithm, (s, t) => {
    let objective = 0
    for (let i = 0; i < n; i++) objective += 0.5 * (s.alpha[i] ?? 0) * ((s.grad[i] ?? 0) + (p[i] ?? 0))
    if (options.snapshots === true && onGeometricSchedule(t))
      snapshots.push({ step: t, alpha: Float64Array.from(s.alpha), rho: rho(s.alpha, s.grad, y, C), objective })
    return { objective, pair: s.pair, t }
  })
  const r = runAlgorithm(tracked.algorithm, {
    maxSteps: options.maxIter,
    ...(run.key === undefined ? {} : { key: run.key }),
    ...(run.parent !== undefined
      ? { parent: run.parent }
      : {
          ...(run.tracer === undefined ? {} : { tracer: run.tracer }),
          ...(run.policy === undefined ? {} : { policy: run.policy }),
        }),
  })
  const history = tracked.history.slice(1).filter((h) => h.pair !== undefined)
  const limit = options.maxHistory ?? 2000
  const stride = Math.max(1, Math.ceil(history.length / limit))
  const kept = history.filter((_, i) => i % stride === 0 || i === history.length - 1)
  const finalRho = rho(r.state.alpha, r.state.grad, y, C)
  const lastT = tracked.history[tracked.history.length - 1]?.t ?? 0
  if (options.snapshots === true && snapshots[snapshots.length - 1]?.step !== lastT)
    snapshots.push({
      step: lastT,
      alpha: Float64Array.from(r.state.alpha),
      rho: finalRho,
      objective: tracked.history[tracked.history.length - 1]?.objective ?? 0,
    })
  return {
    alpha: r.state.alpha,
    grad: r.state.grad,
    rho: finalRho,
    iterations: r.state.iterations,
    converged: r.status === 'converged',
    objective: kept.map((h) => h.objective),
    workingSets: kept.map((h) => [h.pair?.[0] ?? 0, h.pair?.[1] ?? 0]),
    steps: kept.map((h) => h.t),
    ...(options.snapshots === true ? { snapshots } : {}),
  }
}

/** libsvm's rho: the mean of y_i G_i over free variables, else the middle of the feasible interval. */
function rho(alpha: Float64Array, grad: Float64Array, y: Float64Array, C: Float64Array): number {
  let ub = Number.POSITIVE_INFINITY
  let lb = Number.NEGATIVE_INFINITY
  let sum = 0
  let free = 0
  for (let i = 0; i < alpha.length; i++) {
    const yg = (y[i] ?? 1) * (grad[i] ?? 0)
    const a = alpha[i] ?? 0
    const atUpper = a >= (C[i] ?? 0)
    const atLower = a <= 0
    if (atUpper) {
      if ((y[i] ?? 1) === -1) ub = Math.min(ub, yg)
      else lb = Math.max(lb, yg)
    } else if (atLower) {
      if ((y[i] ?? 1) === 1) ub = Math.min(ub, yg)
      else lb = Math.max(lb, yg)
    } else {
      free++
      sum += yg
    }
  }
  return free > 0 ? sum / free : (ub + lb) / 2
}
