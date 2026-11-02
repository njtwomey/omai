/**
 * Gradients through ODE solves: `odeintAdjoint` makes a whole solve one primitive whose reverse rule is the continuous
 * adjoint method (backsolve, interpolated or checkpointed), as in neural ODEs. Also forward sensitivity analysis, and a
 * comparison of the gradient methods on one problem with their cost and storage.
 */

import { grad, jacfwd, jvp, vjp } from '../../autodiff/index.ts'
import { MdmlError } from '../../base/index.ts'
import {
  add,
  arg,
  bind,
  concat,
  definePrimitive,
  matmul,
  neg,
  type Primitive,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  type Tensor,
  type TensorLike,
  type Value,
  zerosLike,
} from '../../tensor/index.ts'
import { floatState, fromNumbers, numbersOf, rms } from './_vec.ts'
import type { DenseSolution, ParametricField, Stepper } from './problem.ts'
import { type OdeMethodName, solveIvp } from './solve-ivp.ts'

/**
 * How reverse-mode gradients of an ODE solve are computed: `backsolve` integrates the state backwards together with
 * the adjoint (constant memory; can be unstable for dissipative problems), `interpolated` keeps the forward dense
 * output and integrates only the adjoint backwards, `checkpointed` stores the state at checkpoints and recomputes
 * each segment's dense output during the backward pass.
 */
export type AdjointMethod = 'backsolve' | 'interpolated' | 'checkpointed'

/** Options of {@link odeintAdjoint} and the sensitivity functions. */
export interface AdjointOptions {
  readonly method?: OdeMethodName | Stepper
  readonly rtol?: number
  readonly atol?: number
  /** The reverse-mode method (default `interpolated`). */
  readonly adjoint?: AdjointMethod
  /** Checkpoints between output times for `checkpointed` (default 4 per interval). */
  readonly checkpoints?: number
  /** Tolerances of the backward (adjoint) solve; default the forward ones. */
  readonly adjointRtol?: number
  readonly adjointAtol?: number
  /** Step budget of each solve (default 100000). */
  readonly maxSteps?: number
}

interface SolveParams {
  readonly f: ParametricField
  readonly times: readonly number[]
  readonly options: AdjointOptions
}

const flat = (v: Value): Value => reshape(v, [-1])

function forwardSolve(
  f: ParametricField,
  y0: Value,
  p: Value,
  times: readonly number[],
  options: AdjointOptions,
  dense: boolean,
): { ys: Value[]; sol: DenseSolution | undefined; nfev: number; steps: number } {
  const t0 = times[0] as number
  const t1 = times[times.length - 1] as number
  if (times.length === 1 || t0 === t1) return { ys: times.map(() => y0), sol: undefined, nfev: 0, steps: 0 }
  const sol = solveIvp((t, y) => f(t, y, p), [t0, t1], y0, {
    method: options.method ?? 'DOP853',
    rtol: options.rtol ?? 1e-8,
    atol: options.atol ?? 1e-10,
    tEval: times,
    denseOutput: dense,
    maxSteps: options.maxSteps ?? 100000,
  })
  if (!sol.success) throw new MdmlError('convergence', `odeintAdjoint: the forward solve failed: ${sol.message}`)
  return { ys: [y0, ...sol.ys.slice(1)], sol: sol.sol, nfev: sol.nfev, steps: sol.nSteps }
}

const solvePrimitive: Primitive<SolveParams> = definePrimitive<SolveParams>({
  name: 'odeSolve',
  abstractEval: ([y0], params) => [
    { shape: [params.times.length, ...(y0 as { shape: readonly number[] }).shape], dtype: 'float64' },
  ],
  impl: ([y0, p], params) => {
    const { ys } = forwardSolve(params.f, y0 as Tensor, p as Tensor, params.times, params.options, false)
    return [primalOf(stack(ys))]
  },
  jvp: (primals, tangents, params) => {
    const y0 = arg(primals, 0)
    const p = arg(primals, 1)
    const ty = arg(tangents, 0) ?? zerosLike(primalOf(y0))
    const tp = arg(tangents, 1) ?? zerosLike(primalOf(p))
    if (arg(tangents, 0) === null && arg(tangents, 1) === null) return [null]
    const n = numbersOf(y0).length
    const np = numbersOf(p).length
    const shape = y0.shape
    const pshape = p.shape
    // The tangent system: (y, u)' = (f, J_y u + J_p dp), solved by this same primitive.
    const augmented: ParametricField = (t, z, q) => {
      const y = reshape(sliceAxis(z, 0, 0, n), shape)
      const u = reshape(sliceAxis(z, 0, n, 2 * n), shape)
      const pp = reshape(sliceAxis(q, 0, 0, np), pshape)
      const dp = reshape(sliceAxis(q, 0, np, 2 * np), pshape)
      const { out, tangent } = jvp((args: Value[]) => params.f(t, args[0] as Value, args[1] as Value), [y, pp], [u, dp])
      return concat([flat(out), flat(tangent)], 0)
    }
    const z0 = concat([flat(y0), flat(ty)], 0)
    const q = concat([flat(p), flat(tp)], 0)
    const out = arg(bind(solvePrimitive, [z0, q], { ...params, f: augmented }), 0)
    return [reshape(sliceAxis(out, 1, n, 2 * n), [params.times.length, ...shape])]
  },
  vjp: (cotangents, inputs, params, outputs) => {
    const g = arg(cotangents, 0)
    if (g === null) return [null, null]
    const y0 = primalOf(arg(inputs, 0))
    const p = primalOf(arg(inputs, 1))
    const ys = primalOf(arg(outputs, 0))
    const { y0Bar, pBar } = adjointPass(params.f, y0, p, params.times, ys, primalOf(g), params.options)
    return [y0Bar, pBar]
  },
})

/** Counts and storage of one adjoint pass. */
interface AdjointPass {
  readonly y0Bar: Tensor
  readonly pBar: Tensor
  readonly nfev: number
  /** Floats kept from the forward pass. */
  readonly stored: number
}

function adjointPass(
  f: ParametricField,
  y0: Tensor,
  p: Tensor,
  times: readonly number[],
  ys: Tensor,
  g: Tensor,
  options: AdjointOptions,
): AdjointPass {
  const kind = options.adjoint ?? 'interpolated'
  const n = y0.data.length
  const np = p.data.length
  const shape = y0.shape
  const pshape = p.shape
  const nt = times.length
  const yAt = (k: number): Value => reshape(sliceAxis(ys, 0, k, k + 1), shape)
  const gAt = (k: number): Float64Array => numbersOf(reshape(sliceAxis(g, 0, k, k + 1), [n]))
  let a = Float64Array.from(gAt(nt - 1))
  let theta = new Float64Array(np)
  let nfev = 0
  let stored = n * nt
  const backTol = {
    rtol: options.adjointRtol ?? options.rtol ?? 1e-8,
    atol: options.adjointAtol ?? options.atol ?? 1e-10,
  }
  const method = options.method ?? 'DOP853'
  // Pullback of f at (t, y): a^T df/dy and a^T df/dp.
  const pull = (t: number, y: Value, av: Value): [Value, Value] => {
    nfev += 1
    const { pullback } = vjp((args: Value[]) => f(t, args[0] as Value, args[1] as Value), [y, p])
    const [yb, pb] = pullback(reshape(av, shape)) as [Value, Value]
    return [flat(yb), flat(pb)]
  }
  // Dense forward solutions for the interpolated and checkpointed variants.
  let dense: DenseSolution | undefined
  if (kind === 'interpolated') {
    const fwd = forwardSolve(f, y0, p, times, options, true)
    dense = fwd.sol
    nfev += fwd.nfev
    stored += (dense?.segments.length ?? 0) * 8 * n
  }
  for (let k = nt - 1; k >= 1; k--) {
    const tk = times[k] as number
    const tm = times[k - 1] as number
    if (tk !== tm) {
      if (kind === 'backsolve') {
        const field = (t: number, z: Value): Value => {
          const y = reshape(sliceAxis(z, 0, 0, n), shape)
          const av = sliceAxis(z, 0, n, 2 * n)
          nfev += 1
          const dy = flat(f(t, y, p))
          const [yb, pb] = pull(t, y, av)
          return concat([dy, neg(yb), neg(pb)], 0)
        }
        const z0 = fromNumbers(concatNumbers(numbersOf(yAt(k)), a, theta), [2 * n + np])
        const sol = solveIvp(field, [tk, tm], z0, { method, ...backTol, maxSteps: options.maxSteps ?? 100000 })
        if (!sol.success) throw new MdmlError('convergence', `odeintAdjoint: the backward solve failed: ${sol.message}`)
        const z = numbersOf(sol.ys[sol.ys.length - 1] as Value)
        a = z.slice(n, 2 * n)
        theta = z.slice(2 * n)
      } else {
        let segments: { t0: number; t1: number; sol: DenseSolution | undefined }[]
        if (kind === 'interpolated') segments = [{ t0: tm, t1: tk, sol: dense }]
        else {
          // Recompute the dense output of each checkpoint segment from the state stored at its start.
          const m = Math.max(1, options.checkpoints ?? 4)
          const marks = Array.from({ length: m + 1 }, (_, i) => tm + ((tk - tm) * i) / m)
          const states: Value[] = [yAt(k - 1)]
          for (let i = 1; i < m; i++) {
            const seg = forwardSolve(
              f,
              states[i - 1] as Value,
              p,
              [marks[i - 1] as number, marks[i] as number],
              options,
              false,
            )
            nfev += seg.nfev
            states.push(seg.ys[1] as Value)
          }
          stored += m * n
          segments = []
          for (let i = m; i >= 1; i--) {
            const seg = forwardSolve(
              f,
              states[i - 1] as Value,
              p,
              [marks[i - 1] as number, marks[i] as number],
              options,
              true,
            )
            nfev += seg.nfev
            segments.push({ t0: marks[i - 1] as number, t1: marks[i] as number, sol: seg.sol })
          }
        }
        for (const seg of segments) {
          if (seg.t0 === seg.t1) continue
          const sol0 = seg.sol
          const yOf = (t: number): Value => (sol0 === undefined ? yAt(k) : sol0.at(t))
          const field = (t: number, z: Value): Value => {
            const [yb, pb] = pull(t, yOf(t), sliceAxis(z, 0, 0, n))
            return concat([neg(yb), neg(pb)], 0)
          }
          const z0 = fromNumbers(concatNumbers(a, theta), [n + np])
          const sol = solveIvp(field, [seg.t1, seg.t0], z0, {
            method,
            ...backTol,
            maxSteps: options.maxSteps ?? 100000,
          })
          if (!sol.success)
            throw new MdmlError('convergence', `odeintAdjoint: the backward solve failed: ${sol.message}`)
          const z = numbersOf(sol.ys[sol.ys.length - 1] as Value)
          a = z.slice(0, n)
          theta = z.slice(n)
        }
      }
    }
    const gk = gAt(k - 1)
    a = Float64Array.from(a, (v, i) => v + (gk[i] ?? 0))
  }
  return { y0Bar: fromNumbers(a, shape), pBar: fromNumbers(theta, pshape), nfev, stored }
}

function concatNumbers(...parts: readonly ArrayLike<number>[]): Float64Array {
  const total = parts.reduce((s, p) => s + p.length, 0)
  const out = new Float64Array(total)
  let o = 0
  for (const p of parts) {
    out.set(p, o)
    o += p.length
  }
  return out
}

/**
 * The solution of dy/dt = f(t, y, p) at the given times, differentiable in y0 and p by the method of section 3.1:
 * the solve is one primitive whose reverse rule is the continuous adjoint method (Pontryagin; Chen et al. 2018) and
 * whose forward rule integrates the forward sensitivities (the tangent system, solved by the same primitive, so
 * higher-order forward derivatives compose). Under `vmap` each example is solved separately. Memory in reverse mode
 * does not grow with the number of steps.
 *
 * @param f - The field f(t, y, p).
 * @param y0 - The initial state (any shape).
 * @param times - Output times; times[0] is the initial time.
 * @param params - The parameters (any shape).
 * @param options - Method (default DOP853), tolerances (default rtol 1e-8, atol 1e-10), adjoint variant and budget.
 * @returns The states `[nt, ...stateShape]`.
 * @throws MdmlError with code `convergence` when a solve fails.
 * @concept adjoint-method
 * @reference Chen, Rubanova, Bettencourt and Duvenaud (2018), Neural ordinary differential equations.
 * @index function ODE solve differentiable in initial state and parameters by the adjoint method.
 * @example
 * ```ts
 * import { odeintAdjoint } from 'mdml-engine/dynamics/ode'
 * import { grad } from 'mdml-engine/autodiff'
 * import { item, mul, neg, sum, sliceAxis, tensor, type Value } from 'mdml-engine/tensor'
 * const loss = (k: Value): Value => sum(sliceAxis(odeintAdjoint((_t, y, p) => neg(mul(p, y)), tensor([1]), [0, 1], k), 0, 1, 2))
 * Math.abs(item(grad(loss)(tensor(0.5))) + Math.exp(-0.5)) < 1e-7 // => true
 * ```
 */
export function odeintAdjoint(
  f: ParametricField,
  y0: TensorLike,
  times: readonly number[],
  params: TensorLike,
  options: AdjointOptions = {},
): Value {
  if (times.length === 0) throw new MdmlError('value', 'odeintAdjoint: give at least one time')
  const y = floatState(y0)
  const p = floatState(params)
  return arg(bind(solvePrimitive, [y, p], { f, times: [...times], options }), 0)
}

/** Forward sensitivities of a solve. */
export interface ForwardSensitivities {
  /** States `[nt, n]` (flattened). */
  readonly y: Tensor
  /** dy(t_k)/dy0 as `[nt, n, n]`. */
  readonly dy0: Tensor
  /** dy(t_k)/dp as `[nt, n, np]`. */
  readonly dp: Tensor
  readonly nfev: number
}

/**
 * Forward sensitivity analysis: integrate the state together with S = dy/d(y0, p), S' = J_y S + [0 | J_p],
 * S(t0) = [I | 0], with the Jacobians by forward-mode autodiff. Cost grows with the number of parameters; memory does
 * not depend on the number of steps.
 *
 * @param f - The field f(t, y, p).
 * @param y0 - The initial state.
 * @param times - Output times (times[0] initial).
 * @param params - The parameters.
 * @param options - Method and tolerances.
 * @returns States and sensitivities at the times.
 * @throws MdmlError with code `convergence` when the solve fails.
 * @concept sensitivity-analysis
 * @example
 * ```ts
 * import { forwardSensitivities } from 'mdml-engine/dynamics/ode'
 * import { mul, neg } from 'mdml-engine/tensor'
 * const s = forwardSensitivities((_t, y, p) => neg(mul(p, y)), [1], [0, 1], [0.5])
 * Math.abs((s.dp.data[1] ?? 0) + Math.exp(-0.5)) < 1e-7 // => true
 * ```
 */
export function forwardSensitivities(
  f: ParametricField,
  y0: TensorLike,
  times: readonly number[],
  params: TensorLike,
  options: AdjointOptions = {},
): ForwardSensitivities {
  const y = floatState(y0)
  const p = floatState(params)
  const n = numbersOf(y).length
  const np = numbersOf(p).length
  const m = n + np
  const shape = y.shape
  const pshape = p.shape
  let nfev = 0
  const field = (t: number, z: Value): Value => {
    nfev += 1
    const yy = reshape(sliceAxis(z, 0, 0, n), shape)
    const S = reshape(sliceAxis(z, 0, n, n + n * m), [n, m])
    const g = (w: Value): Value =>
      flat(f(t, reshape(sliceAxis(w, 0, 0, n), shape), reshape(sliceAxis(w, 0, n, m), pshape)))
    const J = reshape(jacfwd(g)(concat([flat(yy), flat(p)], 0)), [n, m])
    const Jy = sliceAxis(J, 1, 0, n)
    const Jp = sliceAxis(J, 1, n, m)
    const dS = concat([matmul(Jy, sliceAxis(S, 1, 0, n)), add(matmul(Jy, sliceAxis(S, 1, n, m)), Jp)], 1)
    return concat([flat(f(t, yy, p)), flat(dS)], 0)
  }
  const s0 = new Float64Array(n * m)
  for (let i = 0; i < n; i++) s0[i * m + i] = 1
  const z0 = fromNumbers(concatNumbers(numbersOf(y), s0), [n + n * m])
  const t0 = times[0] as number
  const t1 = times[times.length - 1] as number
  const zs =
    times.length === 1 || t0 === t1
      ? times.map(() => z0)
      : (() => {
          const sol = solveIvp(field, [t0, t1], z0, {
            method: options.method ?? 'DOP853',
            rtol: options.rtol ?? 1e-8,
            atol: options.atol ?? 1e-10,
            tEval: times,
            maxSteps: options.maxSteps ?? 100000,
          })
          if (!sol.success) throw new MdmlError('convergence', `forwardSensitivities: ${sol.message}`)
          return [z0, ...sol.ys.slice(1)]
        })()
  const nt = times.length
  const yOut = new Float64Array(nt * n)
  const d0 = new Float64Array(nt * n * n)
  const dpOut = new Float64Array(nt * n * np)
  zs.forEach((z, k) => {
    const v = numbersOf(z)
    yOut.set(v.subarray(0, n), k * n)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const s = v[n + i * m + j] ?? 0
        if (j < n) d0[k * n * n + i * n + j] = s
        else dpOut[k * n * np + i * np + (j - n)] = s
      }
    }
  })
  return {
    y: fromNumbers(yOut, [nt, n]),
    dy0: fromNumbers(d0, [nt, n, n]),
    dp: fromNumbers(dpOut, [nt, n, np]),
    nfev,
  }
}

/** The gradients of a scalar loss of the states with respect to y0 and p by one method, with cost and storage. */
export interface GradientEstimate {
  readonly method: 'direct' | 'forward' | AdjointMethod
  readonly y0: Tensor
  readonly params: Tensor
  /** Vector-field (and vjp) evaluations. */
  readonly nfev: number
  /** Floats kept between the forward and backward passes (an estimate of memory). */
  readonly stored: number
  /** RMS difference from the reference gradient (the forward-sensitivity one at tight tolerance). */
  readonly error: number
}

/**
 * Compare ways of differentiating an ODE solve on one problem: through the solver steps (`direct`), forward
 * sensitivities, and the backsolve, interpolated and checkpointed adjoints. Each reports its gradient, its work, the
 * memory it keeps and its error against a reference computed by forward sensitivities at tight tolerance.
 *
 * @param f - The field f(t, y, p).
 * @param y0 - The initial state.
 * @param times - Output times.
 * @param params - The parameters.
 * @param loss - A scalar loss of the states `[nt, ...stateShape]`.
 * @param options - Method and tolerances for all methods.
 * @returns One estimate per method.
 * @concept adjoint-method
 * @example
 * ```ts
 * import { compareGradientMethods } from 'mdml-engine/dynamics/ode'
 * import { mul, neg, sum } from 'mdml-engine/tensor'
 * const r = compareGradientMethods((_t, y, p) => neg(mul(p, y)), [1], [0, 1], [0.5], (ys) => sum(ys))
 * r.map((e) => e.method) // => ['direct', 'forward', 'backsolve', 'interpolated', 'checkpointed']
 * ```
 */
export function compareGradientMethods(
  f: ParametricField,
  y0: TensorLike,
  times: readonly number[],
  params: TensorLike,
  loss: (ys: Value) => Value,
  options: AdjointOptions = {},
): GradientEstimate[] {
  const y = floatState(y0)
  const p = floatState(params)
  const gradOf = (solve: (yy: Value, pp: Value) => Value): [Tensor, Tensor] => {
    const g = grad((args: Value[]) => loss(solve(args[0] as Value, args[1] as Value)))([y, p]) as Value[]
    return [primalOf(g[0] as Value), primalOf(g[1] as Value)]
  }
  // The reference: forward sensitivities at tight tolerance, contracted with dloss/dys.
  const sensitivity = (opts: AdjointOptions): { y0: Tensor; p: Tensor; nfev: number } => {
    const s = forwardSensitivities(f, y, times, p, opts)
    const ys = reshape(s.y, [times.length, ...y.shape])
    const dl = numbersOf(primalOf(grad((v: Value) => loss(v))(ys)))
    const n = numbersOf(y).length
    const np = numbersOf(p).length
    const gy = new Float64Array(n)
    const gp = new Float64Array(np)
    for (let k = 0; k < times.length; k++) {
      for (let i = 0; i < n; i++) {
        const w = dl[k * n + i] ?? 0
        for (let j = 0; j < n; j++) gy[j] = (gy[j] ?? 0) + w * (s.dy0.data[k * n * n + i * n + j] ?? 0)
        for (let j = 0; j < np; j++) gp[j] = (gp[j] ?? 0) + w * (s.dp.data[k * n * np + i * np + j] ?? 0)
      }
    }
    return { y0: fromNumbers(gy, y.shape), p: fromNumbers(gp, p.shape), nfev: s.nfev }
  }
  const reference = sensitivity({ ...options, rtol: 1e-12, atol: 1e-14 })
  const err = (a: Tensor, b: Tensor): number => {
    const x = Float64Array.from([...a.data, ...b.data])
    const r = Float64Array.from([...reference.y0.data, ...reference.p.data])
    return rms(Float64Array.from(x, (v, i) => v - (r[i] ?? 0)))
  }
  const out: GradientEstimate[] = []
  const n = numbersOf(y).length
  const np = numbersOf(p).length
  {
    let calls = 0
    const counted: ParametricField = (t, yy, pp) => {
      calls += 1
      return f(t, yy, pp)
    }
    const fwd = forwardSolve(f, y, p, times, options, false)
    const [gy, gp] = gradOf((yy, pp) => {
      const sol = solveIvp((t, s) => counted(t, s, pp), [times[0] as number, times[times.length - 1] as number], yy, {
        method: options.method ?? 'DOP853',
        rtol: options.rtol ?? 1e-8,
        atol: options.atol ?? 1e-10,
        tEval: times,
      })
      return stack([yy, ...sol.ys.slice(1)])
    })
    out.push({ method: 'direct', y0: gy, params: gp, nfev: calls, stored: fwd.steps * 13 * n, error: err(gy, gp) })
  }
  {
    const s = sensitivity(options)
    out.push({ method: 'forward', y0: s.y0, params: s.p, nfev: s.nfev, stored: n * (n + np), error: err(s.y0, s.p) })
  }
  for (const adjoint of ['backsolve', 'interpolated', 'checkpointed'] as const) {
    const ys = primalOf(odeintAdjoint(f, y, times, p, { ...options, adjoint }))
    const dl = primalOf(grad((v: Value) => loss(v))(ys))
    const pass = adjointPass(f, primalOf(y), primalOf(p), times, ys, dl, { ...options, adjoint })
    out.push({
      method: adjoint,
      y0: pass.y0Bar,
      params: pass.pBar,
      nfev: pass.nfev,
      stored: pass.stored,
      error: err(pass.y0Bar, pass.pBar),
    })
  }
  return out
}
