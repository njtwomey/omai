import { BinaryHeap } from '../../graph/index.ts'
import type { PlanState, StripsAction, StripsProblem } from './strips.ts'

/** A STRIPS problem compiled for relaxation: fact ids, action pre and add lists, and the actions each fact enables. */

export interface Compiled {
  readonly index: Map<string, number>
  readonly actions: readonly StripsAction[]
  readonly pre: readonly Int32Array[]
  readonly add: readonly Int32Array[]
  /** The actions each fact is a precondition of. */
  readonly preOf: readonly number[][]
  /** Actions without preconditions. */
  readonly free: readonly number[]
}

/** Compile a problem for relaxation. */
export function compile(problem: StripsProblem): Compiled {
  const index = new Map<string, number>()
  const id = (f: string): number => {
    const at = index.get(f)
    if (at !== undefined) return at
    index.set(f, index.size)
    return index.size - 1
  }
  for (const f of [...problem.initial, ...problem.goal]) id(f)
  const pre = problem.actions.map((a) => Int32Array.from(a.pre, id))
  const add = problem.actions.map((a) => Int32Array.from(a.add, id))
  for (const a of problem.actions) for (const f of a.del) id(f)
  const preOf: number[][] = Array.from({ length: index.size }, () => [])
  pre.forEach((p, a) => {
    for (const f of p) preOf[f]?.push(a)
  })
  return { index, actions: problem.actions, pre, add, preOf, free: pre.flatMap((p, a) => (p.length === 0 ? [a] : [])) }
}

export interface Costs {
  readonly dist: Float64Array
  /** The cheapest achiever of each fact (-1 for facts of the state and unreached ones). */
  readonly supporter: Int32Array
}

/**
 * Relaxed reachability by generalised Dijkstra over facts (the forward pass of HSP and FF, as pyperplan computes it):
 * an action fires once all its preconditions are reached, at the sum (h-add) or maximum (h-max) of their costs plus
 * its own.
 */
export function relaxedCosts(c: Compiled, state: PlanState, combine: 'sum' | 'max'): Costs {
  const n = c.index.size
  const dist = new Float64Array(n).fill(Number.POSITIVE_INFINITY)
  const supporter = new Int32Array(n).fill(-1)
  const done = new Uint8Array(n)
  const waiting = Int32Array.from(c.pre, (p) => p.length)
  let seq = 0
  const heap = new BinaryHeap<[number, number, number]>((x, y) => x[0] - y[0] || x[1] - y[1])
  for (const f of state) {
    const i = c.index.get(f)
    if (i === undefined) continue
    dist[i] = 0
    heap.push([0, seq++, i])
  }
  const fire = (a: number): void => {
    let v = 0
    for (const f of c.pre[a] ?? []) v = combine === 'sum' ? v + (dist[f] ?? 0) : Math.max(v, dist[f] ?? 0)
    const total = v + (c.actions[a]?.cost ?? 1)
    for (const g of c.add[a] ?? []) {
      if (total < (dist[g] ?? Number.POSITIVE_INFINITY)) {
        dist[g] = total
        supporter[g] = a
        heap.push([total, seq++, g])
      }
    }
  }
  for (const a of c.free) fire(a)
  for (let top = heap.pop(); top !== undefined; top = heap.pop()) {
    const f = top[2]
    if (done[f] === 1) continue
    done[f] = 1
    for (const a of c.preOf[f] ?? []) {
      waiting[a] = (waiting[a] ?? 1) - 1
      if (waiting[a] === 0) fire(a)
    }
  }
  return { dist, supporter }
}

/** The ids of goal facts, or null when one is unknown. */
export function goalIds(c: Compiled, goal: PlanState): number[] | null {
  const out: number[] = []
  for (const g of goal) {
    const i = c.index.get(g)
    if (i === undefined) return null
    out.push(i)
  }
  return out
}

/** FF's relaxed plan (action ids) for a goal, from h-add costs and cheapest achievers already computed. */
export function extractPlan(c: Compiled, costs: Costs, goal: PlanState): number[] | null {
  const { dist, supporter } = costs
  const ids = goalIds(c, goal)
  if (ids === null || ids.some((g) => !Number.isFinite(dist[g] ?? Number.POSITIVE_INFINITY))) return null
  const chosen = new Set<number>()
  const seen = new Set<number>()
  const stack = [...ids]
  while (stack.length > 0) {
    const f = stack.pop() as number
    if (seen.has(f)) continue
    seen.add(f)
    const a = supporter[f] ?? -1
    if (a < 0 || chosen.has(a)) continue
    chosen.add(a)
    for (const p of c.pre[a] ?? []) stack.push(p)
  }
  return [...chosen].sort((x, y) => x - y)
}

/** The relaxation heuristics as one function: h-add, h-max or h-FF of a goal from a state. */
export function relaxation(
  problem: StripsProblem,
): (state: PlanState, goal: PlanState, kind: 'h-add' | 'h-max' | 'h-ff') => number {
  const c = compile(problem)
  return (state, goal, kind) => measure(c, relaxedCosts(c, state, kind === 'h-max' ? 'max' : 'sum'), goal, kind)
}

/** A heuristic value from fact costs computed once (for regression, where every subgoal set is measured from the initial state). */
export function measure(c: Compiled, costs: Costs, goal: PlanState, kind: 'h-add' | 'h-max' | 'h-ff'): number {
  if (kind === 'h-ff') {
    const plan = extractPlan(c, costs, goal)
    return plan === null ? Number.POSITIVE_INFINITY : plan.reduce((s, a) => s + (c.actions[a]?.cost ?? 1), 0)
  }
  const ids = goalIds(c, goal)
  if (ids === null) return Number.POSITIVE_INFINITY
  let v = 0
  for (const g of ids) v = kind === 'h-add' ? v + (costs.dist[g] ?? 0) : Math.max(v, costs.dist[g] ?? 0)
  return v
}

/** Every subgoal set measured from one state: fact costs (h-max and h-add) are computed once. */
export function fromState(
  problem: StripsProblem,
  state: PlanState,
): (goal: PlanState, kind: 'h-add' | 'h-max' | 'h-ff') => number {
  const c = compile(problem)
  const sum = relaxedCosts(c, state, 'sum')
  const max = relaxedCosts(c, state, 'max')
  return (goal, kind) => measure(c, kind === 'h-max' ? max : sum, goal, kind)
}
