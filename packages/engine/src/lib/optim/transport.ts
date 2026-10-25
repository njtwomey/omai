/**
 * Optimal transport between histograms, the cheapest way to move one distribution of mass onto another under a cost
 * matrix: exact transport (the earth mover's distance) by the transportation simplex, entropic transport by Sinkhorn's
 * iterations in the log domain, a differentiable entropic transport loss, and unbalanced transport, as in POT.
 */

import { customVjp } from '../autodiff/index.ts'
import { type Json, logSumExpOf, WhiteboxError } from '../base/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  mul,
  primalOf,
  promoteFloat,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../trace/index.ts'

type Vec = Float64Array

/** A transport problem's data as plain arrays. */
interface Problem {
  readonly a: Vec
  readonly b: Vec
  readonly m: Vec
  readonly n: number
  readonly k: number
}

function problem(a: TensorLike, b: TensorLike, cost: TensorLike, name: string, balanced = true): Problem {
  const av = Float64Array.from(toFlat(primalOf(asTensor(a))))
  const bv = Float64Array.from(toFlat(primalOf(asTensor(b))))
  const c = primalOf(promoteFloat(cost)[0] as Value) as Tensor
  if (c.shape.length !== 2 || c.shape[0] !== av.length || c.shape[1] !== bv.length)
    throw new WhiteboxError('shape', `${name}: the cost must be [len(a), len(b)]`)
  if (av.some((v) => v < 0) || bv.some((v) => v < 0))
    throw new WhiteboxError('value', `${name}: the marginals must be nonnegative`)
  if (balanced) {
    let sa = 0
    let sb = 0
    for (const v of av) sa += v
    for (const v of bv) sb += v
    if (Math.abs(sa - sb) > 1e-9 * Math.max(1, sa))
      throw new WhiteboxError('value', `${name}: the marginals must have equal mass (${sa} and ${sb})`)
  }
  return { a: av, b: bv, m: Float64Array.from(c.data), n: av.length, k: bv.length }
}

/** Result of {@link emd}. */
export interface EmdResult {
  /** The optimal plan `[n, k]`. */
  readonly plan: Tensor
  /** The optimal cost sum P_ij M_ij (the earth mover's distance for a metric cost). */
  readonly cost: number
  /** Dual potentials with u_i + v_j <= M_ij and equality on the plan's support (u_0 = 0). */
  readonly u: Tensor
  readonly v: Tensor
  /** Simplex pivots taken. */
  readonly iterations: number
}

/** State of {@link emdAlgorithm}: the basic cells (a spanning tree of the bipartite graph) with their flows. */
export interface EmdState {
  readonly basis: readonly (readonly [number, number])[]
  readonly flow: Vec
  readonly u: Vec
  readonly v: Vec
  readonly cost: number
  readonly entering?: readonly [number, number]
  readonly reducedCost: number
  readonly optimal: boolean
}

/** The north-west corner rule: a basic feasible solution with exactly n + k - 1 basic cells. */
function northWest(p: Problem): { basis: [number, number][]; flow: Vec } {
  const a = Float64Array.from(p.a)
  const b = Float64Array.from(p.b)
  const flow = new Float64Array(p.n * p.k)
  const basis: [number, number][] = []
  let i = 0
  let j = 0
  while (i < p.n && j < p.k) {
    const x = Math.min(a[i] ?? 0, b[j] ?? 0)
    flow[i * p.k + j] = x
    basis.push([i, j])
    a[i] = (a[i] ?? 0) - x
    b[j] = (b[j] ?? 0) - x
    if (i === p.n - 1 && j === p.k - 1) break
    // Advance the exhausted index; on a tie move down a row (keeping the column, a degenerate zero cell next).
    if ((a[i] ?? 0) <= (b[j] ?? 0) && i < p.n - 1) i += 1
    else j += 1
  }
  return { basis, flow }
}

/** Potentials u, v with u_i + v_j = M_ij on the basis (a tree), u_0 = 0. */
function potentials(p: Problem, basis: readonly (readonly [number, number])[]): { u: Vec; v: Vec } {
  const rows: number[][] = Array.from({ length: p.n }, () => [])
  const cols: number[][] = Array.from({ length: p.k }, () => [])
  for (const [i, j] of basis) {
    rows[i]?.push(j)
    cols[j]?.push(i)
  }
  const u = new Float64Array(p.n).fill(Number.NaN)
  const v = new Float64Array(p.k).fill(Number.NaN)
  u[0] = 0
  const queue: [boolean, number][] = [[true, 0]]
  while (queue.length > 0) {
    const [isRow, idx] = queue.pop() as [boolean, number]
    if (isRow) {
      for (const j of rows[idx] ?? [])
        if (Number.isNaN(v[j] ?? 0)) {
          v[j] = (p.m[idx * p.k + j] ?? 0) - (u[idx] ?? 0)
          queue.push([false, j])
        }
    } else {
      for (const i of cols[idx] ?? [])
        if (Number.isNaN(u[i] ?? 0)) {
          u[i] = (p.m[i * p.k + idx] ?? 0) - (v[idx] ?? 0)
          queue.push([true, i])
        }
    }
  }
  return { u, v }
}

/** The cycle created by adding cell (i, j) to the basis tree: the tree path from column j back to row i. */
function cycle(p: Problem, basis: readonly (readonly [number, number])[], i: number, j: number): number[] {
  // Nodes: rows 0..n-1, columns n..n+k-1. Edges are basis indices.
  const adjacency: number[][] = Array.from({ length: p.n + p.k }, () => [])
  for (const [e, [r, c]] of basis.entries()) {
    adjacency[r]?.push(e)
    adjacency[p.n + c]?.push(e)
  }
  const parentEdge = new Int32Array(p.n + p.k).fill(-2)
  const start = p.n + j
  parentEdge[start] = -1
  const stack = [start]
  while (stack.length > 0) {
    const node = stack.pop() as number
    if (node === i) break
    for (const e of adjacency[node] ?? []) {
      const [r, c] = basis[e] as readonly [number, number]
      const other = node === r ? p.n + c : r
      if (parentEdge[other] === -2) {
        parentEdge[other] = e
        stack.push(other)
      }
    }
  }
  // Walk back from row i to column j, collecting basis edges (alternately - and + starting with -).
  const path: number[] = []
  let node = i
  while (node !== start) {
    const e = parentEdge[node] ?? -1
    if (e < 0) throw new WhiteboxError('value', 'emd: the basis is not a spanning tree')
    path.push(e)
    const [r, c] = basis[e] as readonly [number, number]
    node = node === r ? p.n + c : r
  }
  return path
}

function emdCost(p: Problem, flow: Vec): number {
  let s = 0
  for (let q = 0; q < flow.length; q++) s += (flow[q] ?? 0) * (p.m[q] ?? 0)
  return s
}

/**
 * Exact optimal transport between histograms by the transportation simplex (the u-v method), as a traced algorithm:
 * a north-west-corner start, potentials on the basis tree, the most negative reduced cost entering (Bland's rule after
 * a run of degenerate pivots), and the ratio test on the tree cycle. Records the cost and the entering reduced cost.
 *
 * @param a - Source histogram `[n]`.
 * @param b - Target histogram `[k]` with the same mass.
 * @param cost - The cost matrix `[n, k]`.
 * @returns The algorithm.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { emdAlgorithm } from 'whitebox-engine/optim'
 * emdAlgorithm([0.5, 0.5], [0.5, 0.5], [[0, 1], [1, 0]]).name // => 'emd'
 * ```
 */
export function emdAlgorithm(a: TensorLike, b: TensorLike, cost: TensorLike): Algorithm<EmdState> {
  const p = problem(a, b, cost, 'emd')
  const scale = p.m.reduce((s, v) => Math.max(s, Math.abs(v)), 0)
  const tol = 1e-12 * Math.max(scale, 1)
  const evaluate = (basis: readonly (readonly [number, number])[], flow: Vec, degenerate: number): EmdState => {
    const { u, v } = potentials(p, basis)
    const inBasis = new Uint8Array(p.n * p.k)
    for (const [i, j] of basis) inBasis[i * p.k + j] = 1
    let best = -tol
    let entering: [number, number] | undefined
    for (let i = 0; i < p.n; i++)
      for (let j = 0; j < p.k; j++) {
        if (inBasis[i * p.k + j]) continue
        const r = (p.m[i * p.k + j] ?? 0) - (u[i] ?? 0) - (v[j] ?? 0)
        // Dantzig's rule, or Bland's (the first improving cell) to break a degenerate cycle.
        if (r < best) {
          best = r
          entering = [i, j]
          if (degenerate > 50) break
        }
      }
    return {
      basis,
      flow,
      u,
      v,
      cost: emdCost(p, flow),
      reducedCost: entering === undefined ? 0 : best,
      optimal: entering === undefined,
      ...(entering === undefined ? {} : { entering }),
    }
  }
  let degenerate = 0
  return {
    name: 'emd',
    config: { n: p.n, k: p.k } as Json,
    schema: channels({
      metrics: {
        cost: { description: 'transport cost of the current basic solution', direction: 'lower' },
        reducedCost: { description: 'reduced cost of the entering cell' },
      },
    }),
    init: () => {
      const { basis, flow } = northWest(p)
      return evaluate(basis, flow, 0)
    },
    step: (s) => {
      if (s.entering === undefined) return s
      const [i, j] = s.entering
      const path = cycle(p, s.basis, i, j)
      // Cells on the path alternate: the first (touching row i) loses flow, the next gains, and so on.
      let theta = Infinity
      let leave = -1
      for (const [t, e] of path.entries()) {
        if (t % 2 !== 0) continue
        const [r, c] = s.basis[e] as readonly [number, number]
        const f = s.flow[r * p.k + c] ?? 0
        if (f < theta) {
          theta = f
          leave = e
        }
      }
      const flow = Float64Array.from(s.flow)
      flow[i * p.k + j] = theta
      for (const [t, e] of path.entries()) {
        const [r, c] = s.basis[e] as readonly [number, number]
        flow[r * p.k + c] = (flow[r * p.k + c] ?? 0) + (t % 2 === 0 ? -theta : theta)
      }
      const [lr, lc] = s.basis[leave] as readonly [number, number]
      flow[lr * p.k + lc] = 0
      const basis = s.basis.map((cell, e) => (e === leave ? ([i, j] as const) : cell))
      degenerate = theta === 0 ? degenerate + 1 : 0
      return evaluate(basis, flow, degenerate)
    },
    done: (s) => (s.optimal ? 'converged' : undefined),
    observe: (s) => ({ metrics: { cost: s.cost, reducedCost: s.reducedCost } }),
  }
}

/**
 * Exact optimal transport (the earth mover's distance) between two histograms with a cost matrix (POT's ot.emd): the
 * plan minimising sum P_ij M_ij with row sums a and column sums b, with its dual potentials.
 *
 * @param a - Source histogram `[n]`.
 * @param b - Target histogram `[k]` with the same mass.
 * @param cost - The cost matrix `[n, k]`.
 * @param options - Pivot cap (default 100 n k) and run options.
 * @returns The plan, cost and potentials.
 * @throws WhiteboxError with code `value` when the masses differ or a marginal is negative.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { emd } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * const r = emd([0.5, 0.5], [0.5, 0.5], [[0, 1], [1, 0]])
 * toArray(r.plan) // => [[0.5, 0], [0, 0.5]]
 * r.cost // => 0
 * ```
 */
export function emd(
  a: TensorLike,
  b: TensorLike,
  cost: TensorLike,
  options: Partial<Omit<RunOptions, 'maxSteps' | 'start'>> & { readonly maxIterations?: number } = {},
): EmdResult {
  const { maxIterations, ...run } = options
  const p = problem(a, b, cost, 'emd')
  const result = runAlgorithm(emdAlgorithm(a, b, cost), { ...run, maxSteps: maxIterations ?? 100 * p.n * p.k })
  const s = result.state
  if (!s.optimal) throw new WhiteboxError('convergence', 'emd: the pivot limit was reached')
  return {
    plan: fromStorage(Float64Array.from(s.flow), [p.n, p.k], 'float64'),
    cost: s.cost,
    u: fromStorage(Float64Array.from(s.u), [p.n], 'float64'),
    v: fromStorage(Float64Array.from(s.v), [p.k], 'float64'),
    iterations: result.steps,
  }
}

/** Options of {@link sinkhorn}. */
export interface SinkhornOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** `log` (default; stable for small reg) or `standard` (the scaling iterations u = a / K v, v = b / K^T u). */
  readonly method?: 'log' | 'standard'
  /** Stop once |P^T 1 - b|_2 falls to this (default 1e-9, POT's). */
  readonly tol?: number
  readonly maxIterations?: number
}

/** State of {@link sinkhornAlgorithm}: the dual potentials f, g (log-scalings times reg). */
export interface SinkhornState {
  readonly f: Vec
  readonly g: Vec
  readonly marginalError: number
}

/** Result of {@link sinkhorn} and {@link sinkhornUnbalanced}. */
export interface SinkhornResult {
  readonly plan: Tensor
  /** The linear cost sum P_ij M_ij (POT's sinkhorn2). */
  readonly cost: number
  /** The entropic objective sum P (M + reg (log P - 1)) (with the reference measure for unbalanced transport). */
  readonly objective: number
  /** The dual potentials, P_ij = exp((f_i + g_j - M_ij) / reg) (times the reference for unbalanced transport). */
  readonly f: Tensor
  readonly g: Tensor
  readonly marginalError: number
  readonly iterations: number
  readonly status: RunStatus
  readonly converged: boolean
}

function planOf(p: Problem, f: Vec, g: Vec, reg: number, logRef?: Vec): Vec {
  const out = new Float64Array(p.n * p.k)
  for (let i = 0; i < p.n; i++)
    for (let j = 0; j < p.k; j++)
      out[i * p.k + j] = Math.exp(
        ((f[i] ?? 0) + (g[j] ?? 0) - (p.m[i * p.k + j] ?? 0)) / reg + (logRef?.[i * p.k + j] ?? 0),
      )
  return out
}

function columnError(p: Problem, plan: Vec): number {
  const col = new Float64Array(p.k)
  for (let i = 0; i < p.n; i++) for (let j = 0; j < p.k; j++) col[j] = (col[j] ?? 0) + (plan[i * p.k + j] ?? 0)
  return dnrm2(Float64Array.from(col, (v, j) => v - (p.b[j] ?? 0)))
}

/** One Sinkhorn half-step in the log domain: the potential of one side given the other (with exponent fi). */
function softUpdate(p: Problem, other: Vec, reg: number, rows: boolean, fi: number, logRef?: Vec): Vec {
  const size = rows ? p.n : p.k
  const len = rows ? p.k : p.n
  const marg = rows ? p.a : p.b
  const out = new Float64Array(size)
  const terms = new Float64Array(len)
  for (let s = 0; s < size; s++) {
    for (let t = 0; t < len; t++) {
      const q = rows ? s * p.k + t : t * p.k + s
      terms[t] = ((other[t] ?? 0) - (p.m[q] ?? 0)) / reg + (logRef?.[q] ?? 0)
    }
    const ms = marg[s] ?? 0
    out[s] = ms > 0 ? fi * reg * (Math.log(ms) - logSumExpOf(terms)) : -Infinity
  }
  return out
}

/**
 * Entropic optimal transport by Sinkhorn's iterations (Cuturi 2013), in the log domain by default, as a traced
 * algorithm recording the marginal violation: g = reg (log b - LSE_i((f_i - M_ij) / reg)), then the same for f.
 *
 * @param a - Source histogram `[n]`.
 * @param b - Target histogram `[k]`.
 * @param cost - The cost `[n, k]`.
 * @param reg - The entropic regularisation (> 0).
 * @param options - Method and tolerance.
 * @returns The algorithm.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { sinkhornAlgorithm } from 'whitebox-engine/optim'
 * sinkhornAlgorithm([0.5, 0.5], [0.5, 0.5], [[0, 1], [1, 0]], 0.1).name // => 'sinkhorn'
 * ```
 */
export function sinkhornAlgorithm(
  a: TensorLike,
  b: TensorLike,
  cost: TensorLike,
  reg: number,
  options: SinkhornOptions = {},
): Algorithm<SinkhornState> {
  if (!(reg > 0)) throw new WhiteboxError('value', `sinkhorn: reg must be positive, got ${reg}`)
  const p = problem(a, b, cost, 'sinkhorn')
  const method = options.method ?? 'log'
  const tol = options.tol ?? 1e-9
  const kernel = Float64Array.from(p.m, (v) => Math.exp(-v / reg))
  return {
    name: 'sinkhorn',
    config: { reg, method, tol } as Json,
    schema: channels({
      metrics: { marginalError: { description: '|P^T 1 - b|_2', direction: 'lower' } },
      tensors: {
        f: { role: 'state', description: 'source potential' },
        g: { role: 'state', description: 'target potential' },
      },
    }),
    init: () => ({ f: new Float64Array(p.n), g: new Float64Array(p.k), marginalError: Infinity }),
    step: (s) => {
      let f: Vec
      let g: Vec
      if (method === 'log') {
        g = softUpdate(p, s.f, reg, false, 1)
        f = softUpdate(p, g, reg, true, 1)
      } else {
        // Scalings u = exp(f / reg), v = exp(g / reg).
        const u = Float64Array.from(s.f, (v) => Math.exp(v / reg))
        const v = new Float64Array(p.k)
        for (let j = 0; j < p.k; j++) {
          let ktu = 0
          for (let i = 0; i < p.n; i++) ktu += (kernel[i * p.k + j] ?? 0) * (u[i] ?? 0)
          v[j] = (p.b[j] ?? 0) / ktu
        }
        for (let i = 0; i < p.n; i++) {
          let kv = 0
          for (let j = 0; j < p.k; j++) kv += (kernel[i * p.k + j] ?? 0) * (v[j] ?? 0)
          u[i] = (p.a[i] ?? 0) / kv
        }
        f = Float64Array.from(u, (x) => reg * Math.log(x))
        g = Float64Array.from(v, (x) => reg * Math.log(x))
      }
      const marginalError = columnError(p, planOf(p, f, g, reg))
      return { f, g, marginalError }
    },
    done: (s) => (s.marginalError <= tol ? 'converged' : Number.isNaN(s.marginalError) ? 'diverged' : undefined),
    observe: (s) => ({
      metrics: { marginalError: s.marginalError },
      tensors: { f: { role: 'state', value: s.f }, g: { role: 'state', value: s.g } },
    }),
  }
}

function entropicObjective(plan: Vec, m: Vec, reg: number, logRef?: Vec): number {
  let s = 0
  for (let q = 0; q < plan.length; q++) {
    const x = plan[q] ?? 0
    if (x > 0) s += x * (m[q] ?? 0) + reg * (x * (Math.log(x) - (logRef?.[q] ?? 0)) - x)
  }
  return s
}

/**
 * Entropic optimal transport (POT's ot.sinkhorn with `method='sinkhorn_log'` by default): the plan minimising
 * sum P M + reg sum P (log P - 1) with marginals a and b.
 *
 * @param a - Source histogram `[n]`.
 * @param b - Target histogram `[k]` with the same mass.
 * @param cost - The cost `[n, k]`.
 * @param reg - The regularisation.
 * @param options - Method, tolerance, iteration cap (default 10000) and run options.
 * @returns The plan, costs, potentials and convergence.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { sinkhorn } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * const r = sinkhorn([0.5, 0.5], [0.5, 0.5], [[0, 1], [1, 0]], 0.01)
 * toArray(r.plan).flat().map((v) => Math.round(v * 1e6) / 1e6) // => [0.5, 0, 0, 0.5]
 * ```
 */
export function sinkhorn(
  a: TensorLike,
  b: TensorLike,
  cost: TensorLike,
  reg: number,
  options: SinkhornOptions = {},
): SinkhornResult {
  const { method: _m, tol: _t, maxIterations, ...run } = options
  const p = problem(a, b, cost, 'sinkhorn')
  const result = runAlgorithm(sinkhornAlgorithm(a, b, cost, reg, options), { ...run, maxSteps: maxIterations ?? 10000 })
  const s = result.state
  const plan = planOf(p, s.f, s.g, reg)
  return {
    plan: fromStorage(plan, [p.n, p.k], 'float64'),
    cost: emdCost(p, plan),
    objective: entropicObjective(plan, p.m, reg),
    f: fromStorage(Float64Array.from(s.f), [p.n], 'float64'),
    g: fromStorage(Float64Array.from(s.g), [p.k], 'float64'),
    marginalError: s.marginalError,
    iterations: result.steps,
    status: result.status,
    converged: result.status === 'converged',
  }
}

/**
 * The entropic optimal transport value OT_reg(a, b, M) = min_P sum P M + reg sum P (log P - 1) as a differentiable
 * function of the cost (and the marginals), with the envelope-theorem derivatives: d/dM = P, d/da = f, d/db = g. Its
 * gradient in M is the plan, so it is a smooth loss between point clouds when M depends on their positions.
 *
 * @param a - Source histogram `[n]`.
 * @param b - Target histogram `[k]` with the same mass.
 * @param cost - The cost `[n, k]` (a value: differentiate through it).
 * @param reg - The regularisation.
 * @param options - Tolerance and iteration cap of the inner Sinkhorn solve.
 * @returns The value (0-d).
 * @concept optimal-transport
 * @example
 * ```ts
 * import { entropicTransportLoss } from 'whitebox-engine/optim'
 * import { grad } from 'whitebox-engine/autodiff'
 * import { tensor, toArray, type Value } from 'whitebox-engine/tensor'
 * const g = grad((m: Value) => entropicTransportLoss([0.5, 0.5], [0.5, 0.5], m, 0.01))(tensor([[0, 1], [1, 0]]))
 * toArray(g).flat().map((v) => Math.round(v * 1e6) / 1e6) // => [0.5, 0, 0, 0.5]
 * ```
 */
export function entropicTransportLoss(
  a: TensorLike,
  b: TensorLike,
  cost: TensorLike,
  reg: number,
  options: { readonly tol?: number; readonly maxIterations?: number } = {},
): Value {
  const solve = (av: Value, bv: Value, mv: Value): SinkhornResult =>
    sinkhorn(primalOf(av), primalOf(bv), primalOf(mv), reg, {
      tol: options.tol ?? 1e-12,
      maxIterations: options.maxIterations ?? 100000,
    })
  const loss = customVjp({
    name: 'entropicTransportLoss',
    forward: (av, bv, mv) =>
      fromStorage(new Float64Array([solve(av as Value, bv as Value, mv as Value).objective]), [], 'float64'),
    vjp: (g, [av, bv, mv]) => {
      const r = solve(av as Value, bv as Value, mv as Value)
      return [mul(r.f, g), mul(r.g, g), mul(r.plan, g)]
    },
    jvp: ([av, bv, mv], [ta, tb, tm]) => {
      const r = solve(av as Value, bv as Value, mv as Value)
      const parts: Value[] = []
      if (ta !== null && ta !== undefined) parts.push(sum(mul(r.f, ta)))
      if (tb !== null && tb !== undefined) parts.push(sum(mul(r.g, tb)))
      if (tm !== null && tm !== undefined) parts.push(sum(mul(r.plan, tm)))
      return parts.length === 0 ? null : parts.reduce((x, y) => add(x, y))
    },
  })
  const [av, bv, mv] = promoteFloat(a, b, cost) as [Value, Value, Value]
  return loss(av, bv, mv)
}

/** Options of {@link sinkhornUnbalanced}. */
export interface UnbalancedOptions extends SinkhornOptions {
  /** The reference measure of the entropy: `product` a b^T (POT's default `kl`) or `uniform` (POT's `entropy`). */
  readonly reference?: 'product' | 'uniform'
}

/**
 * Unbalanced entropic optimal transport (Chizat et al. 2018; POT's ot.unbalanced.sinkhorn_unbalanced): the marginal
 * constraints are relaxed to KL penalties, min_P sum P M + reg KL(P | ref) + regM (KL(P 1 | a) + KL(P^T 1 | b)), by
 * the damped scaling iterations u = (a / K v)^fi with fi = regM / (regM + reg), in the log domain, as a traced
 * algorithm.
 *
 * @param a - Source measure `[n]` (any mass).
 * @param b - Target measure `[k]`.
 * @param cost - The cost `[n, k]`.
 * @param reg - The entropic regularisation.
 * @param regM - The marginal relaxation (Infinity recovers balanced Sinkhorn).
 * @param options - Reference measure, tolerance on the scalings' relative change (default 1e-9), iteration cap.
 * @returns The algorithm.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { sinkhornUnbalancedAlgorithm } from 'whitebox-engine/optim'
 * sinkhornUnbalancedAlgorithm([1, 1], [0.5, 0.5], [[0, 1], [1, 0]], 0.1, 1).name // => 'sinkhornUnbalanced'
 * ```
 */
export function sinkhornUnbalancedAlgorithm(
  a: TensorLike,
  b: TensorLike,
  cost: TensorLike,
  reg: number,
  regM: number,
  options: UnbalancedOptions = {},
): Algorithm<SinkhornState> {
  const p = problem(a, b, cost, 'sinkhornUnbalanced', false)
  const fi = Number.isFinite(regM) ? regM / (regM + reg) : 1
  const tol = options.tol ?? 1e-9
  const logRef =
    (options.reference ?? 'product') === 'product'
      ? Float64Array.from(
          { length: p.n * p.k },
          (_, q) => Math.log(p.a[Math.floor(q / p.k)] ?? 0) + Math.log(p.b[q % p.k] ?? 0),
        )
      : undefined
  const relChange = (x: Vec, y: Vec): number => {
    // POT's test on the scalings u = exp(f / reg).
    let d = 0
    let m = 1
    for (let i = 0; i < x.length; i++) {
      const ux = Math.exp((x[i] ?? 0) / reg)
      const uy = Math.exp((y[i] ?? 0) / reg)
      d = Math.max(d, Math.abs(ux - uy))
      m = Math.max(m, Math.abs(ux), Math.abs(uy))
    }
    return d / m
  }
  return {
    name: 'sinkhornUnbalanced',
    config: { reg, regM, tol, reference: options.reference ?? 'product' } as Json,
    schema: channels({
      metrics: { marginalError: { description: 'relative change of the scalings', direction: 'lower' } },
      tensors: {
        f: { role: 'state', description: 'source potential' },
        g: { role: 'state', description: 'target potential' },
      },
    }),
    init: () => ({ f: new Float64Array(p.n), g: new Float64Array(p.k), marginalError: Infinity }),
    step: (s) => {
      const f = softUpdate(p, s.g, reg, true, fi, logRef)
      const g = softUpdate(p, f, reg, false, fi, logRef)
      return { f, g, marginalError: 0.5 * (relChange(f, s.f) + relChange(g, s.g)) }
    },
    done: (s) => (s.marginalError < tol ? 'converged' : Number.isNaN(s.marginalError) ? 'diverged' : undefined),
    observe: (s) => ({
      metrics: { marginalError: s.marginalError },
      tensors: { f: { role: 'state', value: s.f }, g: { role: 'state', value: s.g } },
    }),
  }
}

/**
 * Unbalanced entropic optimal transport.
 *
 * @param a - Source measure.
 * @param b - Target measure.
 * @param cost - The cost.
 * @param reg - The entropic regularisation.
 * @param regM - The marginal relaxation.
 * @param options - Reference measure, tolerance, iteration cap (default 10000) and run options.
 * @returns The plan, costs and potentials (`objective` is the full unbalanced objective).
 * @concept optimal-transport
 * @example
 * ```ts
 * import { sinkhornUnbalanced } from 'whitebox-engine/optim'
 * import { sum, toArray } from 'whitebox-engine/tensor'
 * // Unequal masses: the plan's mass lies between them.
 * const m = toArray(sum(sinkhornUnbalanced([1, 1], [0.5, 0.5], [[0, 1], [1, 0]], 0.1, 1).plan)) as number
 * m > 1 && m < 2 // => true
 * ```
 */
export function sinkhornUnbalanced(
  a: TensorLike,
  b: TensorLike,
  cost: TensorLike,
  reg: number,
  regM: number,
  options: UnbalancedOptions = {},
): SinkhornResult {
  const { method: _m, tol: _t, maxIterations, reference, ...run } = options
  const p = problem(a, b, cost, 'sinkhornUnbalanced', false)
  const result = runAlgorithm(sinkhornUnbalancedAlgorithm(a, b, cost, reg, regM, options), {
    ...run,
    maxSteps: maxIterations ?? 10000,
  })
  const s = result.state
  const logRef =
    (reference ?? 'product') === 'product'
      ? Float64Array.from(
          { length: p.n * p.k },
          (_, q) => Math.log(p.a[Math.floor(q / p.k)] ?? 0) + Math.log(p.b[q % p.k] ?? 0),
        )
      : undefined
  const plan = planOf(p, s.f, s.g, reg, logRef)
  const kl = (x: Vec, y: Vec): number => {
    let t = 0
    for (let i = 0; i < x.length; i++) {
      const xi = x[i] ?? 0
      const yi = y[i] ?? 0
      t += (xi > 0 ? xi * Math.log(xi / yi) : 0) - xi + yi
    }
    return t
  }
  const rows = new Float64Array(p.n)
  const cols = new Float64Array(p.k)
  for (let i = 0; i < p.n; i++)
    for (let j = 0; j < p.k; j++) {
      rows[i] = (rows[i] ?? 0) + (plan[i * p.k + j] ?? 0)
      cols[j] = (cols[j] ?? 0) + (plan[i * p.k + j] ?? 0)
    }
  const ref = logRef === undefined ? new Float64Array(plan.length).fill(1) : Float64Array.from(logRef, Math.exp)
  const objective =
    emdCost(p, plan) + reg * kl(plan, ref) + (Number.isFinite(regM) ? regM * (kl(rows, p.a) + kl(cols, p.b)) : 0)
  return {
    plan: fromStorage(plan, [p.n, p.k], 'float64'),
    cost: emdCost(p, plan),
    objective,
    f: fromStorage(Float64Array.from(s.f), [p.n], 'float64'),
    g: fromStorage(Float64Array.from(s.g), [p.k], 'float64'),
    marginalError: s.marginalError,
    iterations: result.steps,
    status: result.status,
    converged: result.status === 'converged',
  }
}
