import { heap, WhiteboxError } from '../base/index.ts'
import { ddot } from '../numerics/linalg/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type StepContext, subRun } from '../trace/index.ts'
import { vectorFrom, vectorTensor } from './_programming.ts'
import { simplexAlgorithm } from './lp-simplex.ts'
import { type Bound, fromStandard, type StandardForm, type StandardSolution, standardForm } from './lp-standard-form.ts'

/** A mixed-integer linear program: a {@link LinearProgram} plus which variables must take integer values. */
export interface MixedIntegerProgram {
  readonly c: TensorLike
  readonly aUb?: TensorLike
  readonly bUb?: TensorLike
  readonly aEq?: TensorLike
  readonly bEq?: TensorLike
  readonly bounds?: Bound | readonly Bound[]
  /** 1 (or true) for an integer variable, 0 for a continuous one; one entry, or one per variable (default all). */
  readonly integrality?: number | boolean | readonly (number | boolean)[]
}

/** One node of the branch-and-bound tree, as recorded for figures. */
export interface BranchNode {
  readonly id: number
  readonly parent: number
  readonly depth: number
  /** The branching that created the node: variable, side and bound (absent at the root). */
  readonly branch?: { readonly variable: number; readonly side: 'down' | 'up'; readonly bound: number }
  /** The relaxation's optimal value (NaN when infeasible). */
  readonly value: number
  readonly outcome: 'branched' | 'integral' | 'infeasible' | 'pruned' | 'unbounded'
}

/** Options of {@link milp}. */
export interface MilpOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Node selection: the open node with the best bound (default) or the deepest. */
  readonly strategy?: 'best-first' | 'depth-first'
  /** Rounds of Gomory mixed-integer cuts at the root (default 0). */
  readonly cutRounds?: number
  /** Cuts per round (default 10). */
  readonly cutsPerRound?: number
  /** Stop when (incumbent - bound) <= mipGap max(1, |incumbent|) (default 1e-9). */
  readonly mipGap?: number
  /** Integrality tolerance (default 1e-7). */
  readonly integerTol?: number
  readonly maxNodes?: number
}

/** Result of {@link milp}. */
export interface MilpResult {
  readonly x: Tensor
  readonly fun: number
  readonly status: 'optimal' | 'infeasible' | 'unbounded' | 'node limit'
  readonly success: boolean
  readonly message: string
  /** The best bound on the optimum (equal to fun when optimal). */
  readonly bound: number
  readonly gap: number
  readonly nodes: number
  /** Gomory cuts added at the root, as rows of A_ub x <= b_ub. */
  readonly cuts: readonly { readonly a: readonly number[]; readonly b: number }[]
  readonly tree: readonly BranchNode[]
}

interface OpenNode {
  readonly id: number
  readonly parent: number
  readonly depth: number
  readonly lower: Float64Array
  readonly upper: Float64Array
  readonly bound: number
  readonly branch?: BranchNode['branch']
}

/** The state of {@link milpAlgorithm}. */
export interface MilpState {
  readonly open: readonly OpenNode[]
  readonly incumbent?: { readonly x: Float64Array; readonly value: number }
  readonly tree: readonly BranchNode[]
  readonly cuts: readonly { readonly a: readonly number[]; readonly b: number }[]
  readonly nextId: number
  readonly unbounded: boolean
}

interface Relaxation {
  readonly status: StandardSolution['status']
  readonly x: Float64Array
  readonly value: number
  readonly form: StandardForm
  readonly solution: StandardSolution
}

function integralityOf(spec: MixedIntegerProgram['integrality'], n: number): Uint8Array {
  const out = new Uint8Array(n)
  for (let j = 0; j < n; j++) {
    const v = spec === undefined ? 1 : Array.isArray(spec) ? spec[j] : spec
    if (v === undefined) throw new WhiteboxError('shape', `milp: integrality must have ${n} entries`)
    out[j] = v === true || v === 1 ? 1 : 0
  }
  return out
}

/**
 * Gomory mixed-integer cuts from the optimal tableau of a relaxation, in the original variables: for each basic
 * integer variable with a fractional value, the GMI inequality on the non-basic columns, mapped back through the
 * standard form (rows involving split free variables are skipped).
 */
function gomoryCuts(
  relaxation: Relaxation,
  integer: Uint8Array,
  integerData: boolean[],
  limit: number,
): { a: number[]; b: number }[] {
  const { form, solution } = relaxation
  const { tableau, basis } = solution
  if (tableau === undefined || basis === undefined) return []
  const N = form.a.cols
  const width = N + 1
  const n = form.n
  const o = form.original
  // Is standard column k integer-valued at every integer-feasible point, and its affine map z_k = p^T x + r.
  const colInteger = new Uint8Array(N)
  const maps: ({ p: Float64Array; r: number } | undefined)[] = []
  for (const [k, col] of form.columns.entries()) {
    const p = new Float64Array(n)
    if (col.kind === 'shift') {
      p[col.index] = 1
      maps.push({ p, r: -(o.lower[col.index] ?? 0) })
      colInteger[k] = integer[col.index] === 1 && Number.isInteger(o.lower[col.index] ?? 0) ? 1 : 0
    } else if (col.kind === 'reflect') {
      p[col.index] = -1
      maps.push({ p, r: o.upper[col.index] ?? 0 })
      colInteger[k] = integer[col.index] === 1 && Number.isInteger(o.upper[col.index] ?? 0) ? 1 : 0
    } else if (col.kind === 'slack') {
      const row = form.rows[col.index]
      if (row?.kind === 'ub') {
        for (let j = 0; j < n; j++) p[j] = -(o.aUb.data[row.index * n + j] ?? 0)
        maps.push({ p, r: o.bUb[row.index] ?? 0 })
        colInteger[k] = integerData[row.index] === true ? 1 : 0
      } else {
        const j = row?.index ?? 0
        p[j] = -1
        maps.push({ p, r: o.upper[j] ?? 0 })
        colInteger[k] = integer[j] === 1 && Number.isInteger(o.upper[j] ?? 0) ? 1 : 0
      }
    } else maps.push(undefined)
  }
  const isBasic = new Uint8Array(N)
  for (const k of basis) isBasic[k] = 1
  const cuts: { a: number[]; b: number }[] = []
  for (let i = 0; i < basis.length && cuts.length < limit; i++) {
    const k0 = basis[i] ?? 0
    if (colInteger[k0] !== 1) continue
    const rhs = tableau[i * width + N] ?? 0
    const f0 = rhs - Math.floor(rhs)
    if (f0 < 1e-4 || f0 > 1 - 1e-4) continue
    const alpha = new Float64Array(N)
    let usable = true
    for (let k = 0; k < N; k++) {
      if (isBasic[k] === 1) continue
      const ak = tableau[i * width + k] ?? 0
      if (Math.abs(ak) < 1e-12) continue
      if (colInteger[k] === 1) {
        const fk = ak - Math.floor(ak)
        alpha[k] = fk <= f0 ? fk / f0 : (1 - fk) / (1 - f0)
      } else alpha[k] = ak > 0 ? ak / f0 : -ak / (1 - f0)
      if ((alpha[k] ?? 0) !== 0 && maps[k] === undefined) usable = false
    }
    if (!usable) continue
    // sum alpha_k (p_k^T x + r_k) >= 1  <=>  -(sum alpha_k p_k)^T x <= sum alpha_k r_k - 1.
    const a = new Array<number>(n).fill(0)
    let b = -1
    for (let k = 0; k < N; k++) {
      const ak = alpha[k] ?? 0
      const map = maps[k]
      if (ak === 0 || map === undefined) continue
      for (let j = 0; j < n; j++) a[j] = (a[j] ?? 0) - ak * (map.p[j] ?? 0)
      b += ak * map.r
    }
    if (a.every((v) => Math.abs(v) < 1e-12)) continue
    cuts.push({ a, b })
  }
  return cuts
}

/**
 * Branch and bound for mixed-integer linear programs on simplex relaxations, as a traced algorithm: each step takes one
 * open node (best bound first, or deepest first), solves its relaxation as a child run, and prunes it (infeasible, or
 * no better than the incumbent), accepts it (integral: a new incumbent) or branches on the most fractional variable.
 * Optional rounds of Gomory mixed-integer cuts tighten the root relaxation. Records the incumbent, the best bound, the
 * gap and the open nodes, and every node of the search tree as a message.
 *
 * @param problem - The program.
 * @param options - Node selection, cuts and tolerances.
 * @returns The algorithm.
 * @concept integer-programming
 * @example
 * ```ts
 * import { milpAlgorithm } from 'whitebox-engine/optim'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const alg = milpAlgorithm({ c: [-1, -1], aUb: [[2, 2]], bUb: [3] })
 * runAlgorithm(alg, { maxSteps: 100 }).state.incumbent?.value // => -1
 * ```
 */
export function milpAlgorithm(problem: MixedIntegerProgram, options: MilpOptions = {}): Algorithm<MilpState> {
  const root = standardForm(problem)
  const n = root.n
  const integer = integralityOf(problem.integrality, n)
  const tolInt = options.integerTol ?? 1e-7
  const gapTol = options.mipGap ?? 1e-9
  const depthFirst = options.strategy === 'depth-first'
  const aUb = root.original.aUb
  const integerData = Array.from({ length: aUb.rows }, (_, i) => {
    if (!Number.isInteger(root.original.bUb[i] ?? 0)) return false
    for (let j = 0; j < n; j++) {
      const v = aUb.data[i * n + j] ?? 0
      if (v !== 0 && (integer[j] !== 1 || !Number.isInteger(v))) return false
    }
    return true
  })
  const order = (a: OpenNode, b: OpenNode): number =>
    depthFirst
      ? b.depth - a.depth || a.bound - b.bound || a.id - b.id
      : a.bound - b.bound || b.depth - a.depth || a.id - b.id
  const relax = (
    lower: Float64Array,
    upper: Float64Array,
    cuts: MilpState['cuts'],
    ctx: StepContext | undefined,
    index: number,
  ): Relaxation => {
    const aRows = [
      ...Array.from({ length: aUb.rows }, (_, i) => Array.from(aUb.data.subarray(i * n, (i + 1) * n))),
      ...cuts.map((c) => [...c.a]),
    ]
    const bRows = [...Array.from(root.original.bUb), ...cuts.map((c) => c.b)]
    const bounds: Bound[] = Array.from({ length: n }, (_, j) => [
      Number.isFinite(lower[j] ?? 0) ? (lower[j] ?? 0) : null,
      Number.isFinite(upper[j] ?? 0) ? (upper[j] ?? 0) : null,
    ])
    const form = standardForm({
      c: root.original.c,
      ...(aRows.length === 0 ? {} : { aUb: aRows, bUb: bRows }),
      ...(root.original.aEq.rows === 0
        ? {}
        : {
            aEq: Array.from({ length: root.original.aEq.rows }, (_, i) =>
              Array.from(root.original.aEq.data.subarray(i * n, (i + 1) * n)),
            ),
            bEq: root.original.bEq,
          }),
      bounds,
    })
    const alg = simplexAlgorithm(form)
    const maxSteps = 50 * (form.a.rows + form.a.cols + 1)
    const result = ctx === undefined ? runAlgorithm(alg, { maxSteps }) : subRun(ctx, alg, { maxSteps, index })
    const solution = result.state.solution
    const x = fromStandard(form, solution.z)
    return { status: solution.status, x, value: ddot(root.original.c, x), form, solution }
  }
  const fractional = (x: Float64Array): number => {
    let best = -1
    let score = tolInt
    for (let j = 0; j < n; j++) {
      if (integer[j] !== 1) continue
      const v = x[j] ?? 0
      const f = Math.abs(v - Math.round(v))
      if (f > score) {
        score = f
        best = j
      }
    }
    return best
  }
  const bestBound = (s: MilpState): number => {
    let b = s.incumbent?.value ?? Infinity
    for (const node of s.open) b = Math.min(b, node.bound)
    return b
  }
  const gapOf = (s: MilpState): number =>
    s.incumbent === undefined ? Infinity : (s.incumbent.value - bestBound(s)) / Math.max(1, Math.abs(s.incumbent.value))

  return {
    name: 'milp',
    config: {
      strategy: options.strategy ?? 'best-first',
      cutRounds: options.cutRounds ?? 0,
      mipGap: gapTol,
    },
    schema: channels({
      metrics: {
        incumbent: { description: 'best integer objective found', direction: 'lower' },
        bound: { description: 'best bound on the optimum over open nodes' },
        gap: { description: 'relative gap between incumbent and bound' },
        open: { description: 'open nodes' },
      },
      messages: { node: { description: 'a processed node of the branch-and-bound tree' } },
    }),
    init: () => {
      let cuts: MilpState['cuts'] = []
      const lower = Float64Array.from(root.original.lower)
      const upper = Float64Array.from(root.original.upper)
      for (let round = 0; round < (options.cutRounds ?? 0); round++) {
        const r = relax(lower, upper, cuts, undefined, 0)
        if (r.status !== 'optimal' || fractional(r.x) < 0) break
        const fresh = gomoryCuts(r, integer, [...integerData, ...cuts.map(() => false)], options.cutsPerRound ?? 10)
        if (fresh.length === 0) break
        cuts = [...cuts, ...fresh]
      }
      return {
        open: [{ id: 0, parent: -1, depth: 0, lower, upper, bound: -Infinity }],
        tree: [],
        cuts,
        nextId: 1,
        unbounded: false,
      }
    },
    step: (s, ctx) => {
      const queue = heap(order, s.open)
      const node = queue.pop() as OpenNode
      const base = {
        id: node.id,
        parent: node.parent,
        depth: node.depth,
        ...(node.branch === undefined ? {} : { branch: node.branch }),
      }
      const rest = queue.toArray()
      const cutoff = (v: number): boolean =>
        s.incumbent !== undefined && v >= s.incumbent.value - gapTol * Math.max(1, Math.abs(s.incumbent.value))
      if (cutoff(node.bound)) {
        return { ...s, open: rest, tree: [...s.tree, { ...base, value: node.bound, outcome: 'pruned' }] }
      }
      const r = relax(node.lower, node.upper, s.cuts, ctx, node.id)
      if (r.status === 'infeasible' || r.status === 'iteration limit')
        return { ...s, open: rest, tree: [...s.tree, { ...base, value: Number.NaN, outcome: 'infeasible' }] }
      if (r.status === 'unbounded')
        return {
          ...s,
          open: [],
          unbounded: true,
          tree: [...s.tree, { ...base, value: -Infinity, outcome: 'unbounded' }],
        }
      if (cutoff(r.value))
        return { ...s, open: rest, tree: [...s.tree, { ...base, value: r.value, outcome: 'pruned' }] }
      const j = fractional(r.x)
      if (j < 0) {
        const x = Float64Array.from(r.x, (v, k) => (integer[k] === 1 ? Math.round(v) : v))
        return {
          ...s,
          open: rest,
          incumbent: { x, value: ddot(root.original.c, x) },
          tree: [...s.tree, { ...base, value: r.value, outcome: 'integral' }],
        }
      }
      const v = r.x[j] ?? 0
      const downUpper = Float64Array.from(node.upper)
      downUpper[j] = Math.floor(v)
      const upLower = Float64Array.from(node.lower)
      upLower[j] = Math.ceil(v)
      const children: OpenNode[] = [
        {
          id: s.nextId,
          parent: node.id,
          depth: node.depth + 1,
          lower: node.lower,
          upper: downUpper,
          bound: r.value,
          branch: { variable: j, side: 'down', bound: Math.floor(v) },
        },
        {
          id: s.nextId + 1,
          parent: node.id,
          depth: node.depth + 1,
          lower: upLower,
          upper: node.upper,
          bound: r.value,
          branch: { variable: j, side: 'up', bound: Math.ceil(v) },
        },
      ]
      return {
        ...s,
        open: [...rest, ...children],
        nextId: s.nextId + 2,
        tree: [...s.tree, { ...base, value: r.value, outcome: 'branched' }],
      }
    },
    done: (s) => {
      if (s.unbounded) return 'failed'
      if (s.open.length === 0) return s.incumbent === undefined ? 'failed' : 'converged'
      return s.incumbent !== undefined && gapOf(s) <= gapTol ? 'converged' : undefined
    },
    observe: (s) => {
      const last = s.tree[s.tree.length - 1]
      return {
        metrics: {
          ...(s.incumbent === undefined ? {} : { incumbent: s.incumbent.value, gap: gapOf(s) }),
          ...(Number.isFinite(bestBound(s)) ? { bound: bestBound(s) } : {}),
          open: s.open.length,
        },
        ...(last === undefined
          ? {}
          : {
              messages: [
                {
                  kind: 'node',
                  payload: {
                    id: last.id,
                    parent: last.parent,
                    depth: last.depth,
                    value: Number.isFinite(last.value) ? last.value : null,
                    outcome: last.outcome,
                    ...(last.branch === undefined ? {} : { branch: { ...last.branch } }),
                  },
                },
              ],
            }),
      }
    },
  }
}

/**
 * Solve a mixed-integer linear program (scipy.optimize.milp, in linprog's form plus `integrality`) by branch and bound
 * with optional Gomory cuts.
 *
 * @param problem - The program.
 * @param options - Node selection, cuts, tolerances, node limit and run options.
 * @returns The solution, its bound and gap, the cuts and the search tree.
 * @concept integer-programming
 * @example
 * ```ts
 * import { milp } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * const r = milp({ c: [-5, -4], aUb: [[6, 4], [1, 2]], bUb: [24, 6], integrality: [1, 1] })
 * [r.fun, toArray(r.x)] // => [-20, [4, 0]]
 * ```
 */
export function milp(problem: MixedIntegerProgram, options: MilpOptions = {}): MilpResult {
  const { strategy: _s, cutRounds: _c, cutsPerRound: _p, mipGap: _g, integerTol: _i, maxNodes, ...run } = options
  const result = runAlgorithm(milpAlgorithm(problem, options), { ...run, maxSteps: maxNodes ?? 10_000 })
  const s = result.state
  const n = vectorFrom(problem.c).length
  let bound = s.incumbent?.value ?? Number.NaN
  for (const o of s.open) bound = Math.min(bound, o.bound)
  const status: MilpResult['status'] = s.unbounded
    ? 'unbounded'
    : s.incumbent === undefined
      ? s.open.length === 0
        ? 'infeasible'
        : 'node limit'
      : result.status === 'converged'
        ? 'optimal'
        : 'node limit'
  const messages: Record<MilpResult['status'], string> = {
    optimal: 'optimal solution found',
    infeasible: 'the problem is infeasible',
    unbounded: 'the relaxation is unbounded',
    'node limit': 'node limit reached',
  }
  const fun = s.incumbent?.value ?? (s.unbounded ? -Infinity : Number.NaN)
  return {
    x: vectorTensor(s.incumbent?.x ?? new Float64Array(n).fill(Number.NaN)),
    fun,
    status,
    success: status === 'optimal',
    message: messages[status],
    bound,
    gap: s.incumbent === undefined ? Infinity : (fun - bound) / Math.max(1, Math.abs(fun)),
    nodes: s.tree.length,
    cuts: s.cuts,
    tree: s.tree,
  }
}

/** Result of {@link knapsack}. */
export interface KnapsackResult {
  readonly value: number
  readonly weight: number
  /** Indices of the chosen items, increasing. */
  readonly items: readonly number[]
  /** Branch-and-bound nodes explored (0 for dynamic programming). */
  readonly nodes: number
}

/**
 * The 0-1 knapsack problem: choose items maximising total value with total weight at most the capacity. By dynamic
 * programming over capacities (exact, O(n W), integer weights) or by depth-first branch and bound with Dantzig's
 * fractional bound (items in decreasing value density).
 *
 * @param values - Item values (non-negative).
 * @param weights - Item weights (non-negative; integers for dynamic programming).
 * @param capacity - The capacity.
 * @param options - `method`: 'dynamic-programming' (default when weights are integers) or 'branch-and-bound'.
 * @returns The best value, its weight, the chosen items and the nodes explored.
 * @throws WhiteboxError with code `value` for dynamic programming on fractional weights.
 * @concept integer-programming
 * @example
 * ```ts
 * import { knapsack } from 'whitebox-engine/optim'
 * knapsack([60, 100, 120], [10, 20, 30], 50).items // => [1, 2]
 * knapsack([60, 100, 120], [10, 20, 30], 50, { method: 'branch-and-bound' }).value // => 220
 * ```
 */
export function knapsack(
  values: ArrayLike<number>,
  weights: ArrayLike<number>,
  capacity: number,
  options: { readonly method?: 'dynamic-programming' | 'branch-and-bound' } = {},
): KnapsackResult {
  const n = values.length
  if (weights.length !== n) throw new WhiteboxError('shape', 'knapsack: values and weights differ in length')
  const integral = Array.from(weights).every(Number.isInteger) && Number.isInteger(capacity)
  const method = options.method ?? (integral ? 'dynamic-programming' : 'branch-and-bound')
  const pick = (items: number[]): KnapsackResult & { nodes: number } => ({
    value: items.reduce((s, i) => s + (values[i] ?? 0), 0),
    weight: items.reduce((s, i) => s + (weights[i] ?? 0), 0),
    items: items.sort((a, b) => a - b),
    nodes: 0,
  })
  if (method === 'dynamic-programming') {
    if (!integral) throw new WhiteboxError('value', 'knapsack: dynamic programming needs integer weights and capacity')
    const W = Math.max(0, capacity)
    // best[i][w]: the best value of the first i items within capacity w.
    const best = new Float64Array((n + 1) * (W + 1))
    for (let i = 1; i <= n; i++) {
      const wi = weights[i - 1] ?? 0
      const vi = values[i - 1] ?? 0
      for (let w = 0; w <= W; w++) {
        const skip = best[(i - 1) * (W + 1) + w] ?? 0
        best[i * (W + 1) + w] = wi <= w ? Math.max(skip, (best[(i - 1) * (W + 1) + w - wi] ?? 0) + vi) : skip
      }
    }
    const items: number[] = []
    let w = W
    for (let i = n; i >= 1; i--) {
      if ((best[i * (W + 1) + w] ?? 0) !== (best[(i - 1) * (W + 1) + w] ?? 0)) {
        items.push(i - 1)
        w -= weights[i - 1] ?? 0
      }
    }
    return pick(items)
  }
  const { state } = runAlgorithm(knapsackAlgorithm(values, weights, capacity), { maxSteps: 2 ** 31 - 1 })
  return { ...pick([...state.bestItems]), nodes: state.nodes }
}

/** One open node of the knapsack search: the next item position, value and room so far, and the items taken. */
interface KnapsackNode {
  readonly k: number
  readonly value: number
  readonly room: number
  readonly items: readonly number[]
}

/** The state of {@link knapsackAlgorithm}. */
export interface KnapsackState {
  readonly stack: readonly KnapsackNode[]
  readonly bestValue: number
  readonly bestItems: readonly number[]
  readonly nodes: number
  /** The last node: its depth, value and fractional bound, and whether it was pruned. */
  readonly last?: { readonly depth: number; readonly value: number; readonly bound: number; readonly pruned: boolean }
}

/**
 * Depth-first branch and bound for the 0-1 knapsack, as a traced algorithm: items in decreasing value density, each
 * step takes one node, updates the incumbent, prunes it when the fractional (Dantzig) bound cannot beat the incumbent,
 * and otherwise branches on taking the next item or not. Records the incumbent, the open nodes and each node.
 *
 * @param values - Item values.
 * @param weights - Item weights.
 * @param capacity - The capacity.
 * @returns The algorithm.
 * @concept integer-programming
 * @example
 * ```ts
 * import { knapsackAlgorithm } from 'whitebox-engine/optim'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * runAlgorithm(knapsackAlgorithm([60, 100, 120], [10, 20, 30], 50), { maxSteps: 100 }).state.bestValue // => 220
 * ```
 */
export function knapsackAlgorithm(
  values: ArrayLike<number>,
  weights: ArrayLike<number>,
  capacity: number,
): Algorithm<KnapsackState> {
  const n = values.length
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (values[b] ?? 0) / (weights[b] || 1e-300) - (values[a] ?? 0) / (weights[a] || 1e-300),
  )
  const bound = (k: number, value: number, room: number): number => {
    let v = value
    let r = room
    for (let t = k; t < n; t++) {
      const i = order[t] ?? 0
      const wi = weights[i] ?? 0
      if (wi <= r) {
        r -= wi
        v += values[i] ?? 0
      } else return v + ((values[i] ?? 0) * r) / wi
    }
    return v
  }
  return {
    name: 'knapsackBranchAndBound',
    config: { items: n, capacity },
    schema: channels({
      metrics: {
        incumbent: { description: 'best value found', direction: 'higher' },
        open: { description: 'open nodes' },
        bound: { description: 'fractional (Dantzig) bound at the last node' },
      },
      messages: { node: { description: 'a processed node: depth, value, bound and whether it was pruned' } },
    }),
    init: () => ({ stack: [{ k: 0, value: 0, room: capacity, items: [] }], bestValue: 0, bestItems: [], nodes: 0 }),
    step: (s) => {
      const stack = [...s.stack]
      const node = stack.pop() as KnapsackNode
      const better = node.value > s.bestValue
      const bestValue = better ? node.value : s.bestValue
      const bestItems = better ? node.items : s.bestItems
      const b = node.k >= n ? node.value : bound(node.k, node.value, node.room)
      const pruned = node.k >= n || b <= bestValue
      if (!pruned) {
        const i = order[node.k] ?? 0
        stack.push({ k: node.k + 1, value: node.value, room: node.room, items: node.items })
        if ((weights[i] ?? 0) <= node.room)
          stack.push({
            k: node.k + 1,
            value: node.value + (values[i] ?? 0),
            room: node.room - (weights[i] ?? 0),
            items: [...node.items, i],
          })
      }
      return {
        stack,
        bestValue,
        bestItems,
        nodes: s.nodes + 1,
        last: { depth: node.k, value: node.value, bound: b, pruned },
      }
    },
    done: (s) => (s.stack.length === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        incumbent: s.bestValue,
        open: s.stack.length,
        ...(s.last === undefined ? {} : { bound: s.last.bound }),
      },
      ...(s.last === undefined ? {} : { messages: [{ kind: 'node', payload: { ...s.last } }] }),
    }),
  }
}

/** Result of {@link setCover}. */
export interface SetCoverResult {
  /** Indices of the chosen sets, increasing. */
  readonly chosen: readonly number[]
  readonly cost: number
}

/**
 * The (weighted) set-cover problem: choose sets covering every element of the universe at least cost. Greedily
 * (Chvatal: repeatedly the set with the least cost per newly covered element, within a factor H(max set size) of
 * optimal) or exactly as a 0-1 program solved by {@link milp}.
 *
 * @param sets - Each set as a list of elements (integers from 0).
 * @param options - Costs (default 1 each), universe size (default the largest element + 1) and the method.
 * @returns The chosen sets and their cost.
 * @throws WhiteboxError with code `value` when some element is in no set.
 * @concept integer-programming
 * @example
 * ```ts
 * import { setCover } from 'whitebox-engine/optim'
 * const sets = [[0, 1, 2], [0, 3], [1, 4], [2, 5], [3, 4, 5]]
 * setCover(sets).chosen // => [0, 4]
 * setCover(sets, { method: 'exact' }).cost // => 2
 * ```
 */
export function setCover(
  sets: readonly (readonly number[])[],
  options: {
    readonly costs?: ArrayLike<number>
    readonly universe?: number
    readonly method?: 'greedy' | 'exact'
  } = {},
): SetCoverResult {
  const m = sets.length
  const universe = options.universe ?? 1 + sets.reduce((mx, s) => s.reduce((a, e) => Math.max(a, e), mx), -1)
  const cost = (i: number): number => options.costs?.[i] ?? 1
  const coverable = new Uint8Array(universe)
  for (const s of sets) for (const e of s) coverable[e] = 1
  if (coverable.includes(0)) throw new WhiteboxError('value', 'setCover: some element is in no set')
  if (options.method === 'exact') {
    const a = Array.from({ length: universe }, (_, e) => sets.map((s) => (s.includes(e) ? -1 : 0)))
    const r = milp({
      c: Array.from({ length: m }, (_, i) => cost(i)),
      aUb: a,
      bUb: new Array<number>(universe).fill(-1),
      bounds: [0, 1],
    })
    const x = Array.from(r.x.data)
    const chosen = x.flatMap((v, i) => (v > 0.5 ? [i] : []))
    return { chosen, cost: chosen.reduce((s, i) => s + cost(i), 0) }
  }
  const covered = new Uint8Array(universe)
  let left = universe
  const chosen: number[] = []
  while (left > 0) {
    let best = -1
    let ratio = Infinity
    for (let i = 0; i < m; i++) {
      let fresh = 0
      for (const e of new Set(sets[i] ?? [])) if (covered[e] === 0) fresh += 1
      if (fresh === 0) continue
      const q = cost(i) / fresh
      if (q < ratio) {
        ratio = q
        best = i
      }
    }
    chosen.push(best)
    for (const e of sets[best] ?? []) {
      if (covered[e] === 0) {
        covered[e] = 1
        left -= 1
      }
    }
  }
  chosen.sort((a, b) => a - b)
  return { chosen, cost: chosen.reduce((s, i) => s + cost(i), 0) }
}
