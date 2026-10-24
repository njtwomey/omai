import { daxpy, ddot, dgemv, dnrm2, dscal } from '../../numerics/linalg/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { qpActiveSet, qpArrays } from '../quadratic.ts'
import { configOf, observeVector, runVectorState, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import { type Constraint, constraintSet } from './constrained.ts'
import {
  type Bounds,
  boxOf,
  clipToBox,
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

/** Options of {@link slsqp}. */
export interface SlsqpOptions extends VectorMethodOptions {
  /** Equality (c(x) = 0) and inequality (c(x) >= 0) constraints. */
  readonly constraints?: readonly Constraint[]
  /** Box bounds on the flattened parameters (the start is clipped into them). */
  readonly bounds?: Bounds
  /** Stop when the objective changes by less than ftol and the constraints are satisfied to ftol (scipy's ftol, 1e-6). */
  readonly ftol?: number
  /** Line-search trials per iteration (default 10). */
  readonly maxLineSearch?: number
}

/** The state of an {@link slsqp} run. */
export interface SlsqpState extends VectorState {
  /** The BFGS approximation of the Hessian of the Lagrangian, n x n row-major. */
  readonly hessian: Float64Array
  /** Multipliers of the equalities (lambda) and inequalities (mu >= 0), in the convention grad f = J_eq' lambda + J_in' mu. */
  readonly multipliers: Float64Array
  /** The L1 merit penalties. */
  readonly rho: Float64Array
  readonly eq: Float64Array
  readonly ineq: Float64Array
  readonly jacEq: Float64Array
  readonly jacIneq: Float64Array
  /** The largest constraint violation. */
  readonly violation: number
  /** The accepted step length along the QP direction. */
  readonly alpha: number
  /** Whether the last QP had to be relaxed (inconsistent linearised constraints). */
  readonly relaxed: boolean
}

/**
 * Sequential least-squares-style quadratic programming (SLSQP, after Kraft 1988) as a traced algorithm: each step solves
 * the QP min 1/2 d'Bd + g'd subject to the linearised constraints and the bounds (by the active-set QP solver), relaxing
 * inconsistent linearisations with an elastic variable, then searches on the L1 merit function f + sum rho |violation|
 * with Powell's penalty update, and updates B by damped BFGS on the Lagrangian's gradient. Records the objective,
 * violation, step and the multipliers.
 *
 * @param f - The objective.
 * @param x0 - The start.
 * @param options - Constraints, bounds, ftol and the line-search cap.
 * @returns The algorithm.
 * @concept constrained-optimisation
 * @reference D. Kraft, A software package for sequential quadratic programming, DFVLR-FB 88-28 (1988).
 * @example
 * ```ts
 * import { slsqpAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * slsqpAlgorithm((x: Value) => sum(square(x)), [1, 2]).name // => 'slsqp'
 * ```
 */
export function slsqpAlgorithm(f: Objective, x0: Parameters, options: SlsqpOptions = {}): Algorithm<SlsqpState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  const cs = constraintSet(options.constraints ?? [], x0)
  const { me, mi } = cs
  const box = boxOf(options.bounds, n)
  const acc = options.ftol ?? 1e-6
  const maxls = options.maxLineSearch ?? 10
  const violationOf = (eq: Float64Array, ineq: Float64Array, x: Float64Array): number => {
    let v = 0
    for (const h of eq) v = Math.max(v, Math.abs(h))
    for (const c of ineq) v = Math.max(v, -c)
    for (let i = 0; i < n; i++)
      v = Math.max(v, (box.lower[i] ?? -Infinity) - (x[i] ?? 0), (x[i] ?? 0) - (box.upper[i] ?? Infinity))
    return v
  }
  const merit = (fv: number, eq: Float64Array, ineq: Float64Array, rho: Float64Array): number => {
    let m = fv
    for (let i = 0; i < me; i++) m += (rho[i] ?? 0) * Math.abs(eq[i] ?? 0)
    for (let j = 0; j < mi; j++) m += (rho[me + j] ?? 0) * Math.max(0, -(ineq[j] ?? 0))
    return m
  }
  const lagrangianGrad = (g: Float64Array, jEq: Float64Array, jIn: Float64Array, mult: Float64Array): Float64Array => {
    let out: Float64Array = Float64Array.from(g)
    if (me > 0) out = daxpy(-1, dgemv(jEq, me, n, mult.subarray(0, me), true), out)
    if (mi > 0) out = daxpy(-1, dgemv(jIn, mi, n, mult.subarray(me), true), out)
    return out
  }
  const evaluate = (x: Float64Array) => {
    const { value, grad } = obj.valueAndGrad(x)
    const vals = cs.values(x)
    const jac = cs.jacobians(x)
    return { value, grad, eq: vals.eq, ineq: vals.ineq, jacEq: jac.eq, jacIneq: jac.ineq }
  }
  /** Solve the QP subproblem, relaxing it when the linearised constraints are inconsistent. */
  const subproblem = (s: SlsqpState): { d: Float64Array; mult: Float64Array; relaxed: boolean } | undefined => {
    const toRows = (a: Float64Array, m: number, cols: number, extra?: (i: number) => number): number[][] =>
      Array.from({ length: m }, (_, i) => {
        const row = Array.from(a.subarray(i * n, (i + 1) * n))
        return extra === undefined ? row : [...row, extra(i)].slice(0, cols)
      })
    const lb = Array.from({ length: n }, (_, i) => (box.lower[i] ?? -Infinity) - (s.x[i] ?? 0))
    const ub = Array.from({ length: n }, (_, i) => (box.upper[i] ?? Infinity) - (s.x[i] ?? 0))
    const P = Array.from({ length: n }, (_, i) => Array.from(s.hessian.subarray(i * n, (i + 1) * n)))
    const tryQp = (relax: boolean) => {
      if (!relax) {
        return qpActiveSet(
          qpArrays({
            P,
            q: Array.from(s.grad),
            ...(mi > 0 ? { G: toRows(dscal(-1, s.jacIneq), mi, n), h: Array.from(s.ineq) } : {}),
            ...(me > 0 ? { A: toRows(s.jacEq, me, n), b: Array.from(dscal(-1, s.eq)) } : {}),
            lb,
            ub,
          }),
          { tol: 1e-12 },
        )
      }
      // Elastic relaxation (Kraft): an extra variable delta in [0, 1] scales the violated linearised constraints,
      // heavily penalised so that it is used only to restore consistency.
      const weight = 1e6 * Math.max(1, dnrm2(s.grad))
      const Pr = [...P.map((row) => [...row, 0]), [...new Array<number>(n).fill(0), weight]]
      const G = mi > 0 ? toRows(dscal(-1, s.jacIneq), mi, n + 1, (j) => Math.min(0, s.ineq[j] ?? 0)) : []
      const A = me > 0 ? toRows(s.jacEq, me, n + 1, (i) => -(s.eq[i] ?? 0)) : []
      // A d - delta h = -h  <=>  A d + (1 - delta) h = 0;  -J d + delta c <= c  <=>  J d + (1 - delta) c >= 0 for
      // violated c (c < 0); satisfied rows keep J d + c >= 0.
      return qpActiveSet(
        qpArrays({
          P: Pr,
          q: [...Array.from(s.grad), 0],
          ...(mi > 0 ? { G, h: Array.from(s.ineq) } : {}),
          ...(me > 0 ? { A, b: Array.from(dscal(-1, s.eq)) } : {}),
          lb: [...lb, 0],
          ub: [...ub, 1],
        }),
        { tol: 1e-12 },
      )
    }
    let sol = tryQp(false)
    let relaxed = false
    if (sol.status !== 'optimal') {
      sol = tryQp(true)
      relaxed = true
      if (sol.status !== 'optimal') return undefined
    }
    const d = sol.x.subarray(0, n)
    const mult = new Float64Array(me + mi)
    for (let i = 0; i < me; i++) mult[i] = -(sol.y[i] ?? 0)
    for (let j = 0; j < mi; j++) mult[me + j] = sol.z[j] ?? 0
    return { d: Float64Array.from(d), mult, relaxed }
  }
  return {
    name: 'slsqp',
    config: configOf(options),
    schema: vectorSchema({
      metrics: {
        violation: { description: 'largest constraint violation' },
        alpha: { description: 'step length along the QP direction' },
        merit: { description: 'L1 merit function' },
      },
      tensors: { multipliers: { role: 'state', description: 'Lagrange multipliers' } },
    }),
    init: () => {
      const x = clipToBox(obj.x0, box)
      const e = evaluate(x)
      const hessian = new Float64Array(n * n)
      for (let i = 0; i < n; i++) hessian[i * n + i] = 1
      return {
        x,
        value: e.value,
        grad: e.grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        hessian,
        multipliers: new Float64Array(me + mi),
        rho: new Float64Array(me + mi),
        eq: e.eq,
        ineq: e.ineq,
        jacEq: e.jacEq,
        jacIneq: e.jacIneq,
        violation: violationOf(e.eq, e.ineq, x),
        alpha: 0,
        relaxed: false,
      }
    },
    step: (s) => {
      const qp = subproblem(s)
      if (qp === undefined) return { ...s, stop: { status: 'failed', message: 'the QP subproblem is infeasible' } }
      const { d, mult } = qp
      // Powell's penalty update.
      const rho = new Float64Array(me + mi)
      for (let k = 0; k < me + mi; k++) {
        const l = Math.abs(mult[k] ?? 0)
        rho[k] = s.step === 0 && s.alpha === 0 ? l : Math.max(l, 0.5 * ((s.rho[k] ?? 0) + l))
      }
      const phi0 = merit(s.value, s.eq, s.ineq, rho)
      let D = ddot(s.grad, d)
      for (let i = 0; i < me; i++) D -= (rho[i] ?? 0) * Math.abs(s.eq[i] ?? 0)
      for (let j = 0; j < mi; j++) D -= (rho[me + j] ?? 0) * Math.max(0, -(s.ineq[j] ?? 0))
      if (dnrm2(d) <= Number.EPSILON * (1 + dnrm2(s.x)) && s.violation <= acc) {
        return { ...s, multipliers: mult, rho, stop: { status: 'converged', message: 'the QP step vanished' } }
      }
      let alpha = 1
      let nfev = s.nfev
      let accepted: { x: Float64Array; value: number; eq: Float64Array; ineq: Float64Array } | undefined
      // Near the solution the QP direction can fail to descend on the merit function at rounding level; the full step
      // (a Newton step on the KKT conditions) is then taken, as a watchdog would. Far from it, that is a failure.
      const small = dnrm2(d) <= 1e-6 * (1 + dnrm2(s.x))
      if (D >= 0 && !small) {
        return {
          ...s,
          multipliers: mult,
          rho,
          stop: { status: 'failed', message: 'positive directional derivative for line search' },
        }
      }
      for (let k = 0; k < (D >= 0 ? 1 : maxls); k++) {
        const x = clipToBox(daxpy(alpha, d, s.x), box)
        const fv = obj.value(x)
        const vals = cs.values(x)
        nfev += 1
        const phi = merit(fv, vals.eq, vals.ineq, rho)
        if (phi <= phi0 + 0.1 * alpha * D || D >= 0) {
          accepted = { x, value: fv, eq: vals.eq, ineq: vals.ineq }
          break
        }
        const quad = (-0.5 * alpha * alpha * D) / (phi - phi0 - alpha * D)
        alpha = Number.isFinite(quad) && quad > 0.1 * alpha && quad < alpha ? quad : Math.max(0.1 * alpha, 0.5 * alpha)
      }
      if (accepted === undefined) {
        // Take the last trial (Kraft's iteration continues with the reduced step) unless it is useless.
        const x = clipToBox(daxpy(alpha, d, s.x), box)
        const vals = cs.values(x)
        accepted = { x, value: obj.value(x), eq: vals.eq, ineq: vals.ineq }
        nfev += 1
      }
      const e = evaluate(accepted.x)
      nfev += 1
      // Damped BFGS on the gradient of the Lagrangian with the new multipliers.
      const sv = daxpy(-1, s.x, accepted.x)
      const gl0 = lagrangianGrad(s.grad, s.jacEq, s.jacIneq, mult)
      const gl1 = lagrangianGrad(e.grad, e.jacEq, e.jacIneq, mult)
      let y = daxpy(-1, gl0, gl1)
      const B = s.hessian
      const Bs = dgemv(B, n, n, sv)
      const sBs = ddot(sv, Bs)
      const sy = ddot(sv, y)
      let hessian = B
      if (sBs > 0) {
        if (sy < 0.2 * sBs) {
          const theta = (0.8 * sBs) / (sBs - sy)
          y = daxpy(1 - theta, Bs, dscal(theta, y))
        }
        const syd = ddot(sv, y)
        if (syd > 0) {
          hessian = new Float64Array(n * n)
          for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
              hessian[i * n + j] =
                (B[i * n + j] ?? 0) - ((Bs[i] ?? 0) * (Bs[j] ?? 0)) / sBs + ((y[i] ?? 0) * (y[j] ?? 0)) / syd
        }
      }
      const vio = violationOf(e.eq, e.ineq, accepted.x)
      const converged = Math.abs(e.value - s.value) < acc && vio < acc
      return {
        x: accepted.x,
        value: e.value,
        grad: e.grad,
        nfev,
        ngev: s.ngev + 1,
        nhev: 0,
        step: dnrm2(sv),
        hessian,
        multipliers: mult,
        rho,
        eq: e.eq,
        ineq: e.ineq,
        jacEq: e.jacEq,
        jacIneq: e.jacIneq,
        violation: vio,
        alpha,
        relaxed: qp.relaxed,
        ...(converged
          ? { stop: { status: 'converged' as const, message: 'optimization terminated successfully' } }
          : {}),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { violation: s.violation, alpha: s.alpha, merit: merit(s.value, s.eq, s.ineq, s.rho) },
        tensors: { multipliers: { role: 'state', value: s.multipliers } },
      }),
  }
}

/**
 * Minimise subject to equality and inequality constraints and bounds by SLSQP (scipy.optimize.minimize with method
 * 'SLSQP'; gradients and constraint Jacobians by autodiff). The result carries the Lagrange multipliers.
 *
 * @param f - The objective.
 * @param x0 - The start (any parameter tree; constraints and bounds apply to the flattened parameters).
 * @param options - Constraints, bounds, ftol (default 1e-6), line-search cap, iteration cap (default 100) and run
 * options.
 * @returns The result with the multipliers.
 * @concept constrained-optimisation
 * @example
 * ```ts
 * import { slsqp } from 'whitebox-engine/optim'
 * import { mul, square, sub, sum, tensor, toFlat, type Value } from 'whitebox-engine/tensor'
 * // The closest point to (1, 2.5) in scipy's tutorial region.
 * const r = slsqp((x: Value) => sum(square(sub(x, tensor([1, 2.5])))), [2, 0], {
 *   constraints: [{ type: 'ineq', fun: (x: Value) => sub(tensor([2, 6, 2]), sum(mul(x, tensor([[-1, 2], [1, 2], [1, -2]])), 1)) }],
 *   bounds: { lower: 0 },
 * })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1.4, 1.7]
 * ```
 */
export function slsqp(
  f: Objective,
  x0: Parameters,
  options: SlsqpOptions = {},
): MinimizeResult & { readonly multipliers: Float64Array } {
  const obj = flatObjective(f, x0)
  const { result, state } = runVectorState(slsqpAlgorithm(f, x0, options), obj, options, 100)
  return { ...result, multipliers: state.multipliers }
}
