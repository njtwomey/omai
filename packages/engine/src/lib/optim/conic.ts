import { WhiteboxError } from '../base/index.ts'
import { cholesky, choSolve, ddot, dnrm2, eigvalsh } from '../numerics/linalg/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { type Dense, matrixFrom, matVec, solveDense, solveSpd, vectorFrom, vectorTensor } from './_programming.ts'

/** A second-order cone constraint |A x + b|_2 <= c^T x + d. */
export interface SecondOrderCone {
  readonly A: TensorLike
  readonly b: TensorLike
  readonly c: TensorLike
  readonly d: number
}

/** A linear matrix inequality F0 + sum_k x_k F_k >= 0 (positive semidefinite), every F symmetric p x p. */
export interface LinearMatrixInequality {
  readonly F0: TensorLike
  /** One matrix per variable. */
  readonly F: readonly TensorLike[]
}

/**
 * A conic program: minimise c^T x subject to G x <= h, A x = b, second-order cone constraints and linear matrix
 * inequalities.
 */
export interface ConicProgram {
  readonly c: TensorLike
  readonly G?: TensorLike
  readonly h?: TensorLike
  readonly A?: TensorLike
  readonly b?: TensorLike
  readonly soc?: readonly SecondOrderCone[]
  readonly lmi?: readonly LinearMatrixInequality[]
}

interface Blocks {
  readonly n: number
  readonly c: Float64Array
  readonly G: Dense
  readonly h: Float64Array
  readonly A: Dense
  readonly b: Float64Array
  readonly soc: readonly { readonly A: Dense; readonly b: Float64Array; readonly c: Float64Array; readonly d: number }[]
  readonly lmi: readonly { readonly p: number; readonly F0: Float64Array; readonly F: readonly Float64Array[] }[]
}

function blocksOf(problem: ConicProgram): Blocks {
  const c = vectorFrom(problem.c)
  const n = c.length
  const G = matrixFrom(problem.G, n, 'G')
  const A = matrixFrom(problem.A, n, 'A')
  const soc = (problem.soc ?? []).map((k) => {
    const Ak = matrixFrom(k.A, n, 'soc A')
    return { A: Ak, b: vectorFrom(k.b), c: vectorFrom(k.c), d: k.d }
  })
  const lmi = (problem.lmi ?? []).map((l) => {
    const F0 = vectorFrom(l.F0)
    const p = Math.round(Math.sqrt(F0.length))
    if (l.F.length !== n) throw new WhiteboxError('shape', `each LMI needs one matrix per variable (${n})`)
    return { p, F0, F: l.F.map((f) => vectorFrom(f)) }
  })
  return { n, c, G, h: vectorFrom(problem.h), A, b: vectorFrom(problem.b), soc, lmi }
}

/** The barrier parameter: 1 per linear inequality, 2 per second-order cone, p per p x p LMI. */
function degree(blocks: Blocks): number {
  return blocks.G.rows + 2 * blocks.soc.length + blocks.lmi.reduce((s, l) => s + l.p, 0)
}

function lmiMatrix(l: Blocks['lmi'][number], x: ArrayLike<number>): Float64Array {
  const S = Float64Array.from(l.F0)
  for (const [k, F] of l.F.entries()) {
    const xk = x[k] ?? 0
    if (xk === 0) continue
    for (let i = 0; i < S.length; i++) S[i] = (S[i] ?? 0) + xk * (F[i] ?? 0)
  }
  return S
}

/** The barrier, its gradient and Hessian at x (value Infinity outside the domain). */
function barrier(
  blocks: Blocks,
  x: Float64Array,
  withDerivatives: boolean,
): { value: number; grad: Float64Array; hess: Float64Array } {
  const n = blocks.n
  const grad = new Float64Array(n)
  const hess = new Float64Array(withDerivatives ? n * n : 0)
  const out = (): { value: number; grad: Float64Array; hess: Float64Array } => ({ value: Infinity, grad, hess })
  let value = 0
  const gx = matVec(blocks.G, x)
  for (let i = 0; i < blocks.G.rows; i++) {
    const r = (blocks.h[i] ?? 0) - (gx[i] ?? 0)
    if (!(r > 0)) return out()
    value -= Math.log(r)
    if (!withDerivatives) continue
    for (let j = 0; j < n; j++) {
      const gj = (blocks.G.data[i * n + j] ?? 0) / r
      grad[j] = (grad[j] ?? 0) + gj
      for (let k = 0; k < n; k++) hess[j * n + k] = (hess[j * n + k] ?? 0) + gj * ((blocks.G.data[i * n + k] ?? 0) / r)
    }
  }
  for (const k of blocks.soc) {
    const u = matVec(k.A, x)
    for (let i = 0; i < u.length; i++) u[i] = (u[i] ?? 0) + (k.b[i] ?? 0)
    const v = ddot(k.c, x) + k.d
    const w = v * v - ddot(u, u)
    if (!(v > 0 && w > 0)) return out()
    value -= Math.log(w)
    if (!withDerivatives) continue
    const atu = matVec(k.A, u, true)
    const dw = Float64Array.from(k.c, (cj, j) => 2 * v * cj - 2 * (atu[j] ?? 0))
    for (let j = 0; j < n; j++) grad[j] = (grad[j] ?? 0) - (dw[j] ?? 0) / w
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        let ata = 0
        for (let r = 0; r < k.A.rows; r++) ata += (k.A.data[r * n + a] ?? 0) * (k.A.data[r * n + b] ?? 0)
        const d2w = 2 * (k.c[a] ?? 0) * (k.c[b] ?? 0) - 2 * ata
        hess[a * n + b] = (hess[a * n + b] ?? 0) + ((dw[a] ?? 0) * (dw[b] ?? 0)) / (w * w) - d2w / w
      }
    }
  }
  for (const l of blocks.lmi) {
    const p = l.p
    const S = fromStorage(lmiMatrix(l, x), [p, p], 'float64')
    let L: Tensor
    try {
      L = primalOf(cholesky(S)) as Tensor
    } catch {
      return out()
    }
    for (let i = 0; i < p; i++) value -= 2 * Math.log(L.data[i * p + i] ?? 1)
    if (!withDerivatives) continue
    // M_k = S^{-1} F_k; gradient -tr(M_k), Hessian tr(M_k M_l).
    const M = l.F.map((F) => Float64Array.from(toFlat(primalOf(choSolve(L, fromStorage(F, [p, p], 'float64'))))))
    for (let a = 0; a < n; a++) {
      const Ma = M[a] as Float64Array
      let tr = 0
      for (let i = 0; i < p; i++) tr += Ma[i * p + i] ?? 0
      grad[a] = (grad[a] ?? 0) - tr
      for (let b = a; b < n; b++) {
        const Mb = M[b] as Float64Array
        let t = 0
        for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) t += (Ma[i * p + j] ?? 0) * (Mb[j * p + i] ?? 0)
        hess[a * n + b] = (hess[a * n + b] ?? 0) + t
        if (b !== a) hess[b * n + a] = (hess[b * n + a] ?? 0) + t
      }
    }
  }
  return { value, grad, hess }
}

/**
 * Phase I: an extra variable s relaxes every constraint (h + s, d + s, F0 + s I); -s <= 1 and the ball
 * |x - centre| <= radius keep the barrier problem bounded.
 */
function phaseOne(blocks: Blocks, centre: Float64Array, radius: number): Blocks {
  const n = blocks.n + 1
  const widen = (m: Dense, extra: (i: number) => number): Dense => {
    const data = new Float64Array(m.rows * n)
    for (let i = 0; i < m.rows; i++) {
      data.set(m.data.subarray(i * m.cols, (i + 1) * m.cols), i * n)
      data[i * n + n - 1] = extra(i)
    }
    return { data, rows: m.rows, cols: n }
  }
  const G = widen(blocks.G, () => -1)
  const bound = new Float64Array(n)
  bound[n - 1] = -1
  const c = new Float64Array(n)
  c[n - 1] = 1
  return {
    n,
    c,
    G: { data: Float64Array.from([...G.data, ...bound]), rows: G.rows + 1, cols: n },
    h: Float64Array.from([...blocks.h, 1]),
    A: widen(blocks.A, () => 0),
    b: blocks.b,
    soc: [
      ...blocks.soc.map((k) => ({ A: widen(k.A, () => 0), b: k.b, c: Float64Array.from([...k.c, 1]), d: k.d })),
      {
        A: {
          data: Float64Array.from({ length: blocks.n * n }, (_, k) => (k % n === Math.floor(k / n) ? 1 : 0)),
          rows: blocks.n,
          cols: n,
        },
        b: Float64Array.from(centre, (v) => -v),
        c: new Float64Array(n),
        d: radius,
      },
    ],
    lmi: blocks.lmi.map((l) => {
      const eye = new Float64Array(l.p * l.p)
      for (let i = 0; i < l.p; i++) eye[i * l.p + i] = 1
      return { p: l.p, F0: l.F0, F: [...l.F, eye] }
    }),
  }
}

/** The Newton step of t c^T x + phi(x) subject to A x = b, its multiplier w, the decrement and the barrier value. */
function newtonStep(
  bl: Blocks,
  x: Float64Array,
  t: number,
): { dx: Float64Array; w: Float64Array; decrement: number; value: number } {
  const m = bl.n
  const size = m + bl.A.rows
  const bar = barrier(bl, x, true)
  const g = Float64Array.from(bar.grad, (v, j) => v + t * (bl.c[j] ?? 0))
  const K = new Float64Array(size * size)
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) K[i * size + j] = bar.hess[i * m + j] ?? 0
    // A relative nudge of the diagonal keeps directions the barrier does not see (unbounded ones) solvable.
    K[i * size + i] = (K[i * size + i] ?? 0) * (1 + 1e-14) + 1e-300
  }
  for (let r = 0; r < bl.A.rows; r++)
    for (let j = 0; j < m; j++) {
      K[(m + r) * size + j] = bl.A.data[r * m + j] ?? 0
      K[j * size + m + r] = bl.A.data[r * m + j] ?? 0
    }
  const rhs = new Float64Array(size)
  for (let j = 0; j < m; j++) rhs[j] = -(g[j] ?? 0)
  const sol = solveDense(K, rhs, size)
  const dx = sol.slice(0, m)
  return { dx, w: sol.slice(m), decrement: -ddot(g, dx), value: bar.value }
}

/** The state of {@link conicBarrierAlgorithm}. */
export interface ConicState {
  /** 1 while finding a strictly feasible point, 2 while minimising. */
  readonly phase: 1 | 2
  readonly x: Float64Array
  /** The barrier weight t: the duality gap of the centred point is (barrier parameter) / t. */
  readonly t: number
  readonly gap: number
  readonly objective: number
  readonly newtonSteps: number
  readonly status: 'running' | 'optimal' | 'infeasible' | 'unbounded'
}

/** Options of {@link conicBarrierAlgorithm} and {@link conicProgram}. */
export interface ConicOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Stop when the duality gap (barrier parameter / t) falls below this (default 1e-9). */
  readonly tol?: number
  /** The factor t grows by per centring (default 10). */
  readonly mu?: number
  readonly t0?: number
  /** Phase 1 searches the ball of this radius around the least-norm solution of A x = b (default 1e4 (1 + |x0|)). */
  readonly radius?: number
  readonly maxIterations?: number
}

/**
 * The barrier method for conic programs (Boyd and Vandenberghe, chapter 11) as a traced algorithm: each step centres
 * (Newton's method with equality constraints and a backtracking line search that stays inside the cones) on
 * t c^T x + phi(x), where phi is the logarithmic barrier -sum log(h - G x) - sum log((c^T x + d)^2 - |A x + b|^2) -
 * sum log det F(x), then multiplies t by mu. Phase 1 first finds a strictly feasible point by minimising the largest
 * violation s (stopping as soon as s < 0), or proves infeasibility. Records the phase, objective, gap, t and Newton
 * steps.
 *
 * @param problem - The program.
 * @param options - Tolerance, mu and t0.
 * @returns The algorithm.
 * @concept interior-point-methods
 * @reference S. Boyd and L. Vandenberghe, Convex Optimization (2004), chapter 11.
 * @example
 * ```ts
 * import { conicBarrierAlgorithm } from 'whitebox-engine/optim'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * // minimise x + y over the disc |(x, y)| <= 1
 * const alg = conicBarrierAlgorithm({ c: [1, 1], soc: [{ A: [[1, 0], [0, 1]], b: [0, 0], c: [0, 0], d: 1 }] })
 * const s = runAlgorithm(alg, { maxSteps: 100 }).state
 * Array.from(s.x, (v) => Math.round(v * 1e6) / 1e6) // => [-0.707107, -0.707107]
 * ```
 */
export function conicBarrierAlgorithm(problem: ConicProgram, options: ConicOptions = {}): Algorithm<ConicState> {
  const blocks = blocksOf(problem)
  const tol = options.tol ?? 1e-9
  const mu = options.mu ?? 10
  const t0 = options.t0 ?? 1
  const n = blocks.n
  // A least-norm point of A x = b: x = A^T (A A^T)^{-1} b.
  const start = (): Float64Array => {
    const { A, b } = blocks
    if (A.rows === 0) return new Float64Array(n)
    const aat = new Float64Array(A.rows * A.rows)
    for (let i = 0; i < A.rows; i++)
      for (let k = 0; k < A.rows; k++) {
        let s = 0
        for (let j = 0; j < n; j++) s += (A.data[i * n + j] ?? 0) * (A.data[k * n + j] ?? 0)
        aat[i * A.rows + k] = s
      }
    return matVec(A, solveSpd(aat, b, A.rows), true)
  }
  const x0 = start()
  const phase1 = phaseOne(blocks, x0, options.radius ?? 1e4 * (1 + x0.reduce((m, v) => Math.max(m, Math.abs(v)), 0)))
  const violation = (x: Float64Array): number => {
    let v = -Infinity
    const gx = matVec(blocks.G, x)
    for (let i = 0; i < blocks.G.rows; i++) v = Math.max(v, (gx[i] ?? 0) - (blocks.h[i] ?? 0))
    for (const k of blocks.soc) {
      const u = matVec(k.A, x)
      for (let i = 0; i < u.length; i++) u[i] = (u[i] ?? 0) + (k.b[i] ?? 0)
      v = Math.max(v, dnrm2(u) - (ddot(k.c, x) + k.d))
    }
    for (const l of blocks.lmi) {
      const eig = toFlat(primalOf(eigvalsh(fromStorage(lmiMatrix(l, x), [l.p, l.p], 'float64'))))
      v = Math.max(v, -(eig[0] ?? 0))
    }
    return v
  }
  // Centre: minimise t c^T x + phi(x) subject to A x = b from a strictly feasible x; stop early (phase 1) when s < 0.
  const centre = (
    bl: Blocks,
    x0: Float64Array,
    t: number,
    early: boolean,
  ): { x: Float64Array; steps: number; unbounded: boolean } => {
    const m = bl.n
    let x = x0
    let steps = 0
    let previous = Infinity
    for (let iter = 0; iter < 100; iter++) {
      const { dx, decrement, value } = newtonStep(bl, x, t)
      steps += 1
      if (!dx.every(Number.isFinite)) return { x, steps, unbounded: true }
      // Stop at a tiny decrement, or when rounding stops it from shrinking in the quadratically convergent region.
      if (decrement / 2 <= 1e-18 || (decrement < 1e-6 && decrement >= previous)) break
      previous = decrement
      const f0 = t * ddot(bl.c, x) + value
      // In the quadratically convergent region (decrement < 0.1) take the full step, backtracking only to stay inside
      // the cones; elsewhere backtrack on the Armijo condition.
      const pure = decrement < 0.1
      let alpha = 1
      let next = x
      for (let k = 0; k < 200; k++) {
        const trial = Float64Array.from(x, (v, j) => v + alpha * (dx[j] ?? 0))
        const val = barrier(bl, trial, false).value
        if (Number.isFinite(val) && (pure || t * ddot(bl.c, trial) + val <= f0 - 0.25 * alpha * decrement)) {
          next = trial
          break
        }
        alpha *= 0.5
      }
      if (next === x) break
      x = next
      if (early && (x[m - 1] ?? 0) < 0) break
      if (!early && ddot(bl.c, x) < -1e12) return { x, steps, unbounded: true }
    }
    return { x, steps, unbounded: false }
  }
  const objective = (x: Float64Array): number => ddot(blocks.c, x.subarray(0, n))
  return {
    name: 'conicBarrier',
    config: { tol, mu, t0 },
    schema: channels({
      metrics: {
        phase: { description: '1 while finding a strictly feasible point, 2 while minimising' },
        objective: { description: 'c^T x', direction: 'lower' },
        gap: { description: 'duality gap bound: barrier parameter / t' },
        t: { description: 'barrier weight' },
        newtonSteps: { description: 'Newton steps in the last centring' },
      },
      tensors: { x: { role: 'param', description: 'current point' } },
    }),
    init: () => {
      const x = x0
      if (violation(x) < 0) {
        return {
          phase: 2,
          x,
          t: t0,
          gap: degree(blocks) / t0,
          objective: objective(x),
          newtonSteps: 0,
          status: 'running',
        }
      }
      const s0 = Math.max(1, 1 + violation(x))
      const z = Float64Array.from([...x, s0])
      return { phase: 1, x: z, t: t0, gap: Infinity, objective: s0, newtonSteps: 0, status: 'running' }
    },
    step: (s) => {
      if (s.status !== 'running') return s
      if (s.phase === 1) {
        const r = centre(phase1, s.x, s.t, true)
        const sv = r.x[n] ?? 0
        if (sv < 0) {
          const x = r.x.slice(0, n)
          return {
            phase: 2,
            x,
            t: t0,
            gap: degree(blocks) / t0,
            objective: objective(x),
            newtonSteps: r.steps,
            status: 'running',
          }
        }
        const gap = degree(phase1) / s.t
        if (gap < tol || sv - gap > 0) {
          // The phase-1 optimum is positive: no strictly feasible point.
          return {
            ...s,
            x: r.x,
            objective: sv,
            newtonSteps: r.steps,
            gap,
            status: sv > tol ? 'infeasible' : 'running',
            t: s.t * mu,
          }
        }
        return { ...s, x: r.x, objective: sv, newtonSteps: r.steps, gap, t: s.t * mu }
      }
      const r = centre(blocks, s.x, s.t, false)
      if (r.unbounded) return { ...s, x: r.x, objective: objective(r.x), newtonSteps: r.steps, status: 'unbounded' }
      const gap = degree(blocks) / s.t
      return {
        ...s,
        x: r.x,
        objective: objective(r.x),
        newtonSteps: r.steps,
        gap,
        t: s.t * mu,
        status: gap < tol ? 'optimal' : 'running',
      }
    },
    done: (s) => (s.status === 'optimal' ? 'converged' : s.status === 'running' ? undefined : 'failed'),
    observe: (s) => ({
      metrics: {
        phase: s.phase,
        objective: s.objective,
        ...(Number.isFinite(s.gap) ? { gap: s.gap } : {}),
        t: s.t,
        newtonSteps: s.newtonSteps,
      },
      tensors: { x: { role: 'param', value: s.x } },
    }),
  }
}

/** Result of {@link conicProgram}. */
export interface ConicResult {
  readonly x: Tensor
  readonly fun: number
  readonly status: 'optimal' | 'infeasible' | 'unbounded' | 'iteration limit'
  readonly success: boolean
  /** The duality gap bound at the end. */
  readonly gap: number
  readonly iterations: number
  /**
   * Dual variables of the final centred point: lambda >= 0 for G x <= h, (z_k, mu_k) in the second-order cone for each
   * cone, Z_l >= 0 for each LMI and nu for the equalities, with c + G^T lambda + A^T nu = sum_k (A_k^T z_k + c_k mu_k) +
   * sum_l F_l^*(Z_l).
   */
  readonly duals: {
    readonly linear: Tensor
    readonly soc: readonly { readonly z: Tensor; readonly mu: number }[]
    readonly lmi: readonly Tensor[]
    readonly equality: Tensor
  }
}

/**
 * Solve a small conic program (linear inequalities, second-order cones, linear matrix inequalities and equalities) by
 * the barrier method, with duals from the central path.
 *
 * @param problem - The program.
 * @param options - Tolerance, mu, t0, iteration cap and run options.
 * @returns The solution, its status, gap and duals.
 * @concept conic-programming
 * @example
 * ```ts
 * import { conicProgram } from 'whitebox-engine/optim'
 * // minimise t subject to |(x - 3, x + 1)| <= t (the best x is 1)
 * const r = conicProgram({ c: [0, 1], soc: [{ A: [[1, 0], [1, 0]], b: [-3, 1], c: [0, 1], d: 0 }] })
 * Math.round(r.fun * 1e6) / 1e6 // => 2.828427
 * ```
 */
export function conicProgram(problem: ConicProgram, options: ConicOptions = {}): ConicResult {
  const { tol: _t, mu: _m, t0: _0, radius: _r, maxIterations, ...run } = options
  const r = runAlgorithm(conicBarrierAlgorithm(problem, options), { ...run, maxSteps: maxIterations ?? 200 })
  const s = r.state
  const blocks = blocksOf(problem)
  const x = s.x.subarray(0, blocks.n)
  const status: ConicResult['status'] = s.status === 'running' ? 'iteration limit' : s.status
  if (s.phase !== 2) {
    return {
      x: vectorTensor(x),
      fun: Number.NaN,
      status,
      success: false,
      gap: s.gap,
      iterations: r.steps,
      duals: {
        linear: vectorTensor(new Float64Array(blocks.G.rows)),
        soc: blocks.soc.map((k) => ({ z: vectorTensor(new Float64Array(k.A.rows)), mu: 0 })),
        lmi: blocks.lmi.map((l) => fromStorage(new Float64Array(l.p * l.p), [l.p, l.p], 'float64') as Tensor),
        equality: vectorTensor(new Float64Array(blocks.A.rows)),
      },
    }
  }
  // Duals at the end of one more Newton step from the last centre (t / mu): stationarity then holds to second order.
  const t = s.t / (options.mu ?? 10)
  const { dx, w } = newtonStep(blocks, Float64Array.from(x), t)
  let xd = Float64Array.from(x, (v, j) => v + (dx[j] ?? 0))
  if (!Number.isFinite(barrier(blocks, xd, false).value)) xd = Float64Array.from(x)
  const gx = matVec(blocks.G, xd)
  const linear = Float64Array.from({ length: blocks.G.rows }, (_, i) => 1 / (t * ((blocks.h[i] ?? 0) - (gx[i] ?? 0))))
  const soc = blocks.soc.map((k) => {
    const u = matVec(k.A, xd)
    for (let i = 0; i < u.length; i++) u[i] = (u[i] ?? 0) + (k.b[i] ?? 0)
    const v = ddot(k.c, xd) + k.d
    const q = v * v - ddot(u, u)
    return { z: vectorTensor(Float64Array.from(u, (ui) => (-2 * ui) / (t * q))), mu: (2 * v) / (t * q) }
  })
  const lmi = blocks.lmi.map((l) => {
    const S = fromStorage(lmiMatrix(l, xd), [l.p, l.p], 'float64')
    const eye = new Float64Array(l.p * l.p)
    for (let i = 0; i < l.p; i++) eye[i * l.p + i] = 1
    const inv = toFlat(primalOf(choSolve(cholesky(S), fromStorage(eye, [l.p, l.p], 'float64'))))
    return fromStorage(
      Float64Array.from(inv, (v) => v / t),
      [l.p, l.p],
      'float64',
    ) as Tensor
  })
  const equality = Float64Array.from(w, (v) => v / t)
  return {
    x: vectorTensor(x),
    fun: s.status === 'unbounded' ? -Infinity : ddot(blocks.c, x),
    status,
    success: status === 'optimal',
    gap: s.gap,
    iterations: r.steps,
    duals: { linear: vectorTensor(linear), soc, lmi, equality: vectorTensor(equality) },
  }
}

/** Result of {@link sdp}. */
export interface SdpResult {
  /** The optimal matrix X. */
  readonly X: Tensor
  readonly fun: number
  /** Dual variables y of max b^T y subject to C - sum_i y_i A_i >= 0. */
  readonly y: Tensor
  /** The dual slack C - sum_i y_i A_i. */
  readonly Z: Tensor
  readonly status: ConicResult['status']
  readonly gap: number
  readonly iterations: number
}

/**
 * A semidefinite program in standard form: minimise tr(C X) subject to tr(A_i X) = b_i and X positive semidefinite,
 * solved over the upper triangle of X by {@link conicProgram}.
 *
 * @param C - Symmetric p x p cost.
 * @param A - Symmetric p x p constraint matrices.
 * @param b - Right-hand sides.
 * @param options - Conic options.
 * @returns X, the optimal value, the dual y and slack Z.
 * @concept conic-programming
 * @example
 * ```ts
 * import { sdp } from 'whitebox-engine/optim'
 * // The smallest eigenvalue of C as min tr(C X) with tr(X) = 1.
 * const r = sdp([[2, 1], [1, 2]], [[[1, 0], [0, 1]]], [1])
 * Math.round(r.fun * 1e6) / 1e6 // => 1
 * ```
 */
export function sdp(C: TensorLike, A: readonly TensorLike[], b: TensorLike, options: ConicOptions = {}): SdpResult {
  const c = vectorFrom(C)
  const p = Math.round(Math.sqrt(c.length))
  const pairs: [number, number][] = []
  for (let i = 0; i < p; i++) for (let j = i; j < p; j++) pairs.push([i, j])
  // X = sum_{i <= j} x_ij E_ij with E_ii = e_i e_i^T and E_ij = e_i e_j^T + e_j e_i^T.
  const basis = pairs.map(([i, j]) => {
    const E = new Float64Array(p * p)
    E[i * p + j] = 1
    E[j * p + i] = 1
    return E
  })
  const coeff = (M: Float64Array): number[] => basis.map((E) => ddot(M, E))
  const as = A.map((a) => vectorFrom(a))
  const r = conicProgram(
    {
      c: coeff(c),
      A: as.map((a) => coeff(a)),
      b,
      lmi: [{ F0: new Float64Array(p * p), F: basis }],
    },
    options,
  )
  const xv = r.x.data
  const X = new Float64Array(p * p)
  for (const [k, [i, j]] of pairs.entries()) {
    X[i * p + j] = xv[k] ?? 0
    X[j * p + i] = xv[k] ?? 0
  }
  const y = Float64Array.from(r.duals.equality.data, (v) => -v)
  const Z = Float64Array.from(c)
  for (const [k, a] of as.entries()) for (let i = 0; i < Z.length; i++) Z[i] = (Z[i] ?? 0) - (y[k] ?? 0) * (a[i] ?? 0)
  return {
    X: fromStorage(X, [p, p], 'float64') as Tensor,
    fun: r.fun,
    y: vectorTensor(y),
    Z: fromStorage(Z, [p, p], 'float64') as Tensor,
    status: r.status,
    gap: r.gap,
    iterations: r.iterations,
  }
}
