// Shared machinery of the nonlinear least-squares methods (not exported from the barrel): robust losses, the common
// state and the result shape.
import { WhiteboxError } from '../../base/index.ts'
import { dgemv, dgesvd } from '../../numerics/linalg/index.ts'
import { fromStorage } from '../../tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../trace/index.ts'
import type { LeastSquaresOptions, LeastSquaresResult, LeastSquaresState, RobustLoss } from './least-squares.ts'
import { type FlatResiduals, infNorm } from './objective.ts'

/** rho, rho' and rho'' of a loss at z. */
function lossAt(loss: RobustLoss, z: number): [number, number, number] {
  switch (loss) {
    case 'linear':
      return [z, 1, 0]
    case 'soft_l1': {
      const t = 1 + z
      return [2 * (Math.sqrt(t) - 1), 1 / Math.sqrt(t), -0.5 * t ** -1.5]
    }
    case 'huber':
      return z <= 1 ? [z, 1, 0] : [2 * Math.sqrt(z) - 1, 1 / Math.sqrt(z), -0.5 * z ** -1.5]
    case 'cauchy': {
      const t = 1 + z
      return [Math.log1p(z), 1 / t, -1 / (t * t)]
    }
    case 'arctan': {
      const t = 1 + z * z
      return [Math.atan(z), 1 / t, (-2 * z) / (t * t)]
    }
  }
}

/** The cost 0.5 sum fScale^2 rho((f / fScale)^2) (0.5 |f|^2 for the linear loss). */
export function robustCost(f: ArrayLike<number>, loss: RobustLoss, fScale: number): number {
  let s = 0
  for (let i = 0; i < f.length; i++) {
    const fi = f[i] ?? 0
    s += loss === 'linear' ? fi * fi : fScale * fScale * (lossAt(loss, (fi / fScale) ** 2)[0] ?? 0)
  }
  return 0.5 * s
}

/**
 * Scale the residuals and Jacobian so that the Gauss-Newton model of the robust cost is a least-squares problem
 * (scipy's scale_for_robust_loss_function).
 */
export function scaleForLoss(
  J: Float64Array,
  f: Float64Array,
  m: number,
  n: number,
  loss: RobustLoss,
  fScale: number,
): { J: Float64Array; f: Float64Array } {
  if (loss === 'linear') return { J, f }
  const Js = new Float64Array(m * n)
  const fs = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    const fi = f[i] ?? 0
    const [, r1, r2raw] = lossAt(loss, (fi / fScale) ** 2)
    const r2 = r2raw / (fScale * fScale)
    let scale = r1 + 2 * r2 * fi * fi
    if (scale < Number.EPSILON) scale = Number.EPSILON
    scale = Math.sqrt(scale)
    fs[i] = (fi * r1) / scale
    for (let j = 0; j < n; j++) Js[i * n + j] = (J[i * n + j] ?? 0) * scale
  }
  return { J: Js, f: fs }
}

/** J'v for an m x n Jacobian. */
export const jtv = (J: Float64Array, m: number, n: number, v: ArrayLike<number>): Float64Array =>
  dgemv(J, m, n, v, true)

/** J v for an m x n Jacobian. */
export const jv = (J: Float64Array, m: number, n: number, v: ArrayLike<number>): Float64Array => dgemv(J, m, n, v)

/** The minimum-norm least-squares solution of J p = b by the SVD (singular values below rcond s_max dropped). */
export function lstsqSvd(J: Float64Array, m: number, n: number, b: ArrayLike<number>, rcond?: number): Float64Array {
  const { u, s, v } = dgesvd(J, m, n)
  const k = s.length
  const tol = (rcond ?? Number.EPSILON * Math.max(m, n)) * (s[0] ?? 0)
  const p = new Float64Array(n)
  for (let c = 0; c < k; c++) {
    const sc = s[c] ?? 0
    if (sc <= tol) continue
    let ub = 0
    for (let i = 0; i < m; i++) ub += (u[i * k + c] ?? 0) * (b[i] ?? 0)
    const w = ub / sc
    for (let j = 0; j < n; j++) p[j] = (p[j] ?? 0) + w * (v[j * k + c] ?? 0)
  }
  return p
}

/** Run a least-squares algorithm and shape its result. */
export function runLeastSquares<S extends LeastSquaresState>(
  algorithm: Algorithm<S>,
  res: FlatResiduals,
  options: LeastSquaresOptions,
  defaultIterations: number,
): LeastSquaresResult {
  // runAlgorithm reads only its own options, so the method options can pass through.
  const { maxIterations, ...run } = options
  const out = runAlgorithm(algorithm, { ...run, maxSteps: maxIterations ?? defaultIterations })
  const s = out.state
  return {
    x: res.unravel(s.x),
    fun: s.value,
    grad: res.unravel(s.grad),
    gradNorm: infNorm(s.grad),
    steps: out.steps,
    nfev: s.nfev,
    ngev: s.ngev,
    nhev: 0,
    status: out.status,
    converged: out.status === 'converged',
    message:
      s.stop?.message ??
      (out.status === 'completed' ? 'maximum number of iterations reached' : `stopped: ${out.status}`),
    cost: s.value,
    residuals: fromStorage(Float64Array.from(s.residuals), [res.m], 'float64'),
    jacobian: fromStorage(Float64Array.from(s.jacobian), [res.m, res.n], 'float64'),
    optimality: (s as { projectedGradNorm?: number }).projectedGradNorm ?? infNorm(s.grad),
    njev: s.njev,
    activeMask: s.activeMask === undefined ? new Array<number>(res.n).fill(0) : Array.from(s.activeMask),
  }
}

/** Check that residuals have at least as many entries as parameters (MINPACK's requirement). */
export function requireOverdetermined(res: FlatResiduals, name: string): void {
  if (res.m < res.n)
    throw new WhiteboxError('shape', `${name} needs at least as many residuals (${res.m}) as parameters (${res.n})`)
}
