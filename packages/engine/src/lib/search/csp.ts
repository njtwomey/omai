/**
 * Constraint satisfaction problems (AIMA chapter 6): variables with domains and constraints of any arity, with helpers
 * such as all-different. Arc consistency by AC-3 (generalised to n-ary constraints), backtracking search with MRV,
 * degree and least-constraining-value heuristics plus forward checking or maintained arc consistency, min-conflicts
 * local search, and a CSP as an incremental search problem, all traceable.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { Generator } from '../random/index.ts'
import { type Algorithm, channels, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { check, compile, conflictsOf, type Model } from './_csp-model.ts'
import type { SearchProblem, SearchRunOptions } from './problem.ts'

/** A constraint of a CSP: a scope of variables and a test of their values (given in scope order). */
export interface CspConstraint<D> {
  readonly scope: readonly string[]
  readonly test: (values: readonly D[]) => boolean
  readonly name?: string
}

/**
 * A constraint satisfaction problem (AIMA chapter 6): named variables, a domain of values for each, and constraints of
 * any arity. Binary constraints are the common case; n-ary ones (sums of a cryptarithm, all-different over a row) are
 * handled by generalised arc consistency.
 *
 * @concept constraint-satisfaction
 */
export interface Csp<D> {
  readonly variables: readonly string[]
  readonly domains: Readonly<Record<string, readonly D[]>>
  readonly constraints: readonly CspConstraint<D>[]
}

/** A (partial) assignment of values to variables. */
export type Assignment<D> = Readonly<Record<string, D>>

/**
 * Build a CSP, checking that every variable has a domain and every constraint names known variables.
 *
 * @param variables - The variable names, in order (the order of static variable selection).
 * @param domains - The values of each variable, in order (the order of static value selection).
 * @param constraints - The constraints.
 * @returns The CSP.
 * @throws OmaiError with code `value` when a variable is repeated or lacks a domain, or a constraint names an
 * unknown variable.
 * @concept constraint-satisfaction
 * @example
 * ```ts
 * import { binaryConstraint, csp } from 'omai-engine/search'
 * const p = csp(['a', 'b'], { a: [1, 2], b: [1, 2] }, [binaryConstraint('a', 'b', (x, y) => x < y)])
 * p.constraints.length // => 1
 * ```
 */
export function csp<D>(
  variables: readonly string[],
  domains: Readonly<Record<string, readonly D[]>>,
  constraints: readonly CspConstraint<D>[],
): Csp<D> {
  const problem = { variables, domains, constraints }
  compile(problem, 'csp')
  return problem
}

/**
 * A binary constraint.
 *
 * @param a - The first variable.
 * @param b - The second variable.
 * @param test - Whether the pair of values is allowed.
 * @param name - An optional name.
 * @returns The constraint.
 * @example
 * ```ts
 * import { binaryConstraint } from 'omai-engine/search'
 * binaryConstraint('x', 'y', (x: number, y: number) => x !== y).test([1, 2]) // => true
 * ```
 */
export function binaryConstraint<D>(
  a: string,
  b: string,
  test: (x: D, y: D) => boolean,
  name?: string,
): CspConstraint<D> {
  return { scope: [a, b], test: (v) => test(v[0] as D, v[1] as D), ...(name === undefined ? {} : { name }) }
}

/**
 * A constraint of any arity.
 *
 * @param scope - The variables.
 * @param test - Whether the values (in scope order) are allowed.
 * @param name - An optional name.
 * @returns The constraint.
 * @example
 * ```ts
 * import { constraint } from 'omai-engine/search'
 * constraint(['x', 'y', 'z'], (v: readonly number[]) => (v[0] ?? 0) + (v[1] ?? 0) === v[2]).test([1, 2, 3]) // => true
 * ```
 */
export function constraint<D>(
  scope: readonly string[],
  test: (values: readonly D[]) => boolean,
  name?: string,
): CspConstraint<D> {
  return { scope, test, ...(name === undefined ? {} : { name }) }
}

/**
 * All-different as pairwise inequality constraints (so arc consistency and forward checking apply to each pair).
 *
 * @param variables - The variables that must take distinct values.
 * @returns The binary constraints, one per pair.
 * @example
 * ```ts
 * import { allDifferent } from 'omai-engine/search'
 * allDifferent(['a', 'b', 'c']).length // => 3
 * ```
 */
export function allDifferent<D>(variables: readonly string[]): CspConstraint<D>[] {
  const out: CspConstraint<D>[] = []
  for (let i = 0; i < variables.length; i++) {
    for (let j = i + 1; j < variables.length; j++) {
      out.push(binaryConstraint<D>(variables[i] as string, variables[j] as string, (x, y) => x !== y, 'all-different'))
    }
  }
  return out
}

/**
 * The constraints an assignment violates (only constraints whose whole scope is assigned are tested).
 *
 * @param problem - The CSP.
 * @param assignment - A partial or complete assignment.
 * @returns The violated constraints.
 * @example
 * ```ts
 * import { allDifferent, csp, violations } from 'omai-engine/search'
 * const p = csp(['a', 'b', 'c'], { a: [1, 2], b: [1, 2], c: [1, 2] }, allDifferent(['a', 'b', 'c']))
 * violations(p, { a: 1, b: 1 }).length // => 1
 * ```
 */
export function violations<D>(problem: Csp<D>, assignment: Assignment<D>): CspConstraint<D>[] {
  return problem.constraints.filter(
    (c) => c.scope.every((v) => v in assignment) && !c.test(c.scope.map((v) => assignment[v] as D)),
  )
}

/**
 * Whether an assignment is a solution: complete, within the domains, and violating no constraint.
 *
 * @param problem - The CSP.
 * @param assignment - The assignment.
 * @returns True for a solution.
 * @example
 * ```ts
 * import { allDifferent, csp, isSolution } from 'omai-engine/search'
 * const p = csp(['a', 'b'], { a: [1, 2], b: [1, 2] }, allDifferent(['a', 'b']))
 * [isSolution(p, { a: 1, b: 2 }), isSolution(p, { a: 1 }), isSolution(p, { a: 2, b: 2 })] // => [true, false, false]
 * ```
 */
export function isSolution<D>(problem: Csp<D>, assignment: Assignment<D>): boolean {
  return (
    problem.variables.every((v) => v in assignment && (problem.domains[v] ?? []).includes(assignment[v] as D)) &&
    violations(problem, assignment).length === 0
  )
}

// ------------------------------------------------------------------------------------------------ arc consistency

/** Current domains: which value indices are alive per variable, with counts and an undo trail. */
interface Domains {
  readonly alive: Uint8Array[]
  readonly size: Int32Array
  readonly trail: number[]
}

function fullDomains<D>(m: Model<D>): Domains {
  return {
    alive: m.values.map((d) => new Uint8Array(d.length).fill(1)),
    size: Int32Array.from(m.values, (d) => d.length),
    trail: [],
  }
}

function prune(d: Domains, v: number, a: number): void {
  const row = d.alive[v]
  if (row === undefined || row[a] !== 1) return
  row[a] = 0
  d.size[v] = (d.size[v] ?? 1) - 1
  d.trail.push(v, a)
}

function restore(d: Domains, mark: number): void {
  while (d.trail.length > mark) {
    const a = d.trail.pop() as number
    const v = d.trail.pop() as number
    const row = d.alive[v]
    if (row !== undefined && row[a] === 0) {
      row[a] = 1
      d.size[v] = (d.size[v] ?? 0) + 1
    }
  }
}

function aliveValues(d: Domains, v: number): number[] {
  const out: number[] = []
  const row = d.alive[v] ?? new Uint8Array()
  for (let a = 0; a < row.length; a++) if (row[a] === 1) out.push(a)
  return out
}

/**
 * Revise variable `u` against constraint `c` (generalised arc consistency): remove each value of `u` that no tuple of
 * the other scope variables' current values (assigned variables fixed) supports. Returns whether `u` lost a value.
 */
function revise<D>(
  m: Model<D>,
  d: Domains,
  assigned: Int32Array,
  u: number,
  c: Model<D>['constraints'][number],
): boolean {
  const others = [...new Set(c.scope)].filter((v) => v !== u)
  const position = new Map(others.map((v, i) => [v, i]))
  const choices = others.map((v) => ((assigned[v] ?? -1) >= 0 ? [assigned[v] as number] : aliveValues(d, v)))
  const supported = (a: number): boolean => {
    if (choices.some((ch) => ch.length === 0)) return false
    const pick = new Int32Array(others.length)
    const at = (v: number): number => {
      if (v === u) return a
      const i = position.get(v) ?? 0
      return choices[i]?.[pick[i] ?? 0] ?? -1
    }
    for (;;) {
      if (check(m, c, at)) return true
      // The next tuple of the other variables' values, odometer style.
      let k = 0
      while (k < others.length) {
        pick[k] = (pick[k] ?? 0) + 1
        if ((pick[k] ?? 0) < (choices[k]?.length ?? 0)) break
        pick[k] = 0
        k += 1
      }
      if (k === others.length) return false
    }
  }
  let revised = false
  for (const a of aliveValues(d, u)) {
    if (supported(a)) continue
    prune(d, u, a)
    revised = true
  }
  return revised
}

/** A FIFO queue of (variable, constraint) arcs without duplicates. */
interface ArcQueue {
  readonly arcs: [number, number][]
  head: number
  readonly queued: Set<number>
}

function arcQueue<D>(m: Model<D>, arcs: [number, number][]): ArcQueue {
  return { arcs, head: 0, queued: new Set(arcs.map(([u, c]) => u * m.constraints.length + c)) }
}

/** One revision: the next arc's variable, its constraint, the values removed, and whether its domain emptied. */
interface Revision {
  readonly variable: number
  readonly constraint: number
  readonly removed: readonly number[]
  readonly wipeout: boolean
}

/** Revise the next queued arc and queue the arcs into its variable when its domain shrank; null when none is left. */
function reviseNext<D>(m: Model<D>, d: Domains, assigned: Int32Array, q: ArcQueue): Revision | null {
  for (;;) {
    const arc = q.arcs[q.head]
    if (arc === undefined) return null
    q.head += 1
    const [u, ci] = arc
    q.queued.delete(u * m.constraints.length + ci)
    const c = m.constraints[ci]
    if (c === undefined) continue
    const mark = d.trail.length
    const revised = revise(m, d, assigned, u, c)
    const removed = d.trail.slice(mark).filter((_, i) => i % 2 === 1)
    if (!revised) return { variable: u, constraint: ci, removed, wipeout: false }
    if ((d.size[u] ?? 0) === 0) return { variable: u, constraint: ci, removed, wipeout: true }
    for (const c2 of m.on[u] ?? []) {
      if (c2.index === ci) continue
      for (const w of new Set(c2.scope)) {
        if (w === u || (assigned[w] ?? -1) >= 0) continue
        const id = w * m.constraints.length + c2.index
        if (!q.queued.has(id)) {
          q.queued.add(id)
          q.arcs.push([w, c2.index])
        }
      }
    }
    return { variable: u, constraint: ci, removed, wipeout: false }
  }
}

/** Propagate to the fixed point from a queue of (variable, constraint) arcs; false on a domain wipeout. */
function propagate<D>(
  m: Model<D>,
  d: Domains,
  assigned: Int32Array,
  queue: [number, number][],
  counter: { revisions: number },
): boolean {
  const q = arcQueue(m, queue)
  for (let r = reviseNext(m, d, assigned, q); r !== null; r = reviseNext(m, d, assigned, q)) {
    counter.revisions += 1
    if (r.wipeout) return false
  }
  return true
}

function allArcs<D>(m: Model<D>, assigned: Int32Array): [number, number][] {
  const arcs: [number, number][] = []
  for (const c of m.constraints) {
    for (const u of new Set(c.scope)) if ((assigned[u] ?? -1) < 0) arcs.push([u, c.index])
  }
  return arcs
}

function domainsOut<D>(m: Model<D>, d: Domains): Record<string, D[]> {
  const out: Record<string, D[]> = {}
  m.names.forEach((name, v) => {
    out[name] = aliveValues(d, v).map((a) => m.values[v]?.[a] as D)
  })
  return out
}

/** The result of {@link ac3}. */
export interface ArcConsistency<D> {
  /** False when some domain was emptied (the CSP has no solution). */
  readonly consistent: boolean
  /** The arc-consistent domains (values in their original order). */
  readonly domains: Record<string, D[]>
  readonly revisions: number
  readonly checks: number
}

/**
 * AC-3 (Mackworth 1977), generalised to n-ary constraints (GAC-3): revise every (variable, constraint) arc, re-queueing
 * the arcs into a variable whose domain shrank, until no domain changes. The result is the unique largest
 * arc-consistent set of domains, so it does not depend on the queue order.
 *
 * @param problem - The CSP.
 * @param options - Starting domains (default the CSP's) and tracing.
 * @returns Whether the CSP is arc consistent, the domains, and the revisions and constraint checks made.
 * @concept constraint-satisfaction
 * @example
 * ```ts
 * import { ac3, binaryConstraint, csp } from 'omai-engine/search'
 * const p = csp(['x', 'y'], { x: [1, 2, 3], y: [1, 2, 3] }, [binaryConstraint('x', 'y', (a: number, b: number) => a + 1 === b)])
 * ac3(p).domains // => { x: [1, 2], y: [2, 3] }
 * ```
 */
export function ac3<D>(
  problem: Csp<D>,
  options: SearchRunOptions & { readonly domains?: Readonly<Record<string, readonly D[]>> } = {},
): ArcConsistency<D> {
  const { domains, ...run } = options
  const { state } = runAlgorithm(ac3Engine(problem, domains), { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  return {
    consistent: state.consistent,
    domains: domainsOut(state.model, state.domains),
    revisions: state.revisions,
    checks: state.model.checks,
  }
}

/** The state of a traced AC-3 run. */
export interface Ac3State {
  readonly consistent: boolean
  readonly revisions: number
  /** Arcs waiting in the queue. */
  readonly queued: number
  /** The last revision: the variable, the constraint's index, and the values it lost. */
  readonly last: { readonly variable: string; readonly constraint: number; readonly removed: readonly Json[] } | null
}

interface Ac3Internal<D> extends Ac3State {
  consistent: boolean
  revisions: number
  queued: number
  last: { variable: string; constraint: number; removed: Json[] } | null
  done: boolean
  readonly model: Model<D>
  readonly domains: Domains
  readonly queue: ArcQueue
}

function ac3Engine<D>(problem: Csp<D>, domains?: Readonly<Record<string, readonly D[]>>): Algorithm<Ac3Internal<D>> {
  const start = domains === undefined ? problem : { ...problem, domains: { ...problem.domains, ...domains } }
  return {
    name: 'ac3',
    config: {},
    schema: channels({
      metrics: { revisions: { description: 'arc revisions so far' }, queued: { description: 'arcs in the queue' } },
      messages: {
        revise: {
          description: 'an arc revised',
          payload: { variable: 'name', constraint: 'constraint index', removed: 'values removed' },
        },
      },
    }),
    init: () => {
      const model = compile(start, 'ac3')
      const d = fullDomains(model)
      const none = new Int32Array(model.n).fill(-1)
      const consistent = model.values.every((v) => v.length > 0)
      const queue = arcQueue(model, consistent ? allArcs(model, none) : [])
      return {
        consistent,
        revisions: 0,
        queued: queue.arcs.length,
        last: null,
        done: !consistent,
        model,
        domains: d,
        queue,
      }
    },
    step: (s) => {
      const r = reviseNext(s.model, s.domains, new Int32Array(s.model.n).fill(-1), s.queue)
      s.queued = s.queue.arcs.length - s.queue.head
      if (r === null) {
        s.done = true
        s.last = null
        return s
      }
      s.revisions += 1
      s.last = {
        variable: s.model.names[r.variable] ?? '',
        constraint: r.constraint,
        removed: r.removed.map((a) => s.model.values[r.variable]?.[a] as Json),
      }
      if (r.wipeout) {
        s.consistent = false
        s.done = true
      }
      return s
    },
    done: (s) => (s.done ? (s.consistent ? 'converged' : 'stopped') : undefined),
    observe: (s) => ({
      metrics: { revisions: s.revisions, queued: s.queued },
      messages: s.last === null ? [] : [{ kind: 'revise', payload: s.last }],
    }),
  }
}

/**
 * AC-3 as a traced algorithm: each step revises one arc, recording the values it removes, so arc consistency can be
 * animated domain by domain.
 *
 * @param problem - The CSP.
 * @param options - Starting domains (default the CSP's).
 * @returns The algorithm; it ends `converged` when consistent and `stopped` on a domain wipeout.
 * @concept constraint-satisfaction
 * @example
 * ```ts
 * import { ac3Algorithm, csp } from 'omai-engine/search'
 * ac3Algorithm(csp(['a'], { a: [1] }, [])).name // => 'ac3'
 * ```
 */
export function ac3Algorithm<D>(
  problem: Csp<D>,
  options: { readonly domains?: Readonly<Record<string, readonly D[]>> } = {},
): Algorithm<Ac3State> {
  return ac3Engine(problem, options.domains)
}

// ------------------------------------------------------------------------------------------------ backtracking

/** How backtracking chooses the next variable. */
export type VariableOrder = 'static' | 'mrv' | 'degree' | 'mrv-degree'

/** Options of {@link backtrackingSearch}. */
export interface BacktrackingOptions extends SearchRunOptions {
  /** Static order, minimum remaining values, degree, or MRV with degree tie-breaks (default static). */
  readonly variableOrder?: VariableOrder
  /** Domain order, or least constraining value first (default static). */
  readonly valueOrder?: 'static' | 'lcv'
  /** Inference after each assignment: none, forward checking, or maintaining arc consistency (default none). */
  readonly inference?: 'none' | 'forward-checking' | 'mac'
  /** Find every solution (default false: stop at the first). */
  readonly all?: boolean
  /** Stop after this many solutions when `all` is set (default unlimited). */
  readonly maxSolutions?: number
  /** Stop with status `budget` after this many value attempts (default 1000000). */
  readonly maxAttempts?: number
}

/** The outcome of a CSP solver. */
export interface CspResult<D> {
  readonly status: 'solved' | 'failure' | 'budget'
  /** The first solution found, or null. */
  readonly assignment: Assignment<D> | null
  /** Every solution found (one unless `all` was set). */
  readonly solutions: readonly Assignment<D>[]
  /** Values assigned (aima-python's `nassigns`): attempts consistent with the assignment so far. */
  readonly assignments: number
  readonly backtracks: number
  /** Constraint checks made. */
  readonly checks: number
  /** Values removed by inference. */
  readonly pruned: number
}

/** The state of a traced backtracking search, as seen by observers. */
export interface BacktrackingState {
  readonly status: 'running' | 'solved' | 'failure'
  readonly depth: number
  readonly assignments: number
  readonly backtracks: number
  readonly solutions: number
  /** What the last step did. */
  readonly event: { readonly kind: string; readonly payload: Json } | null
}

interface Frame {
  readonly variable: number
  readonly order: readonly number[]
  next: number
  readonly mark: number
}

interface BtState<D> extends BacktrackingState {
  status: 'running' | 'solved' | 'failure'
  depth: number
  assignments: number
  backtracks: number
  solutions: number
  event: { kind: string; payload: Json } | null
  readonly model: Model<D>
  readonly domains: Domains
  readonly assigned: Int32Array
  readonly stack: Frame[]
  readonly found: Assignment<D>[]
  pruned: number
  attempts: number
}

const CSP_SCHEMA: TraceSchema = channels({
  metrics: {
    assignments: { description: 'values assigned so far' },
    backtracks: { description: 'backtracks so far' },
    depth: { description: 'variables assigned' },
    solutions: { description: 'solutions found' },
  },
  messages: {
    assign: {
      description: 'a value assigned',
      payload: { variable: 'name', value: 'value', pruned: 'removed values' },
    },
    conflict: { description: 'a value rejected by a constraint', payload: { variable: 'name', value: 'value' } },
    wipeout: { description: 'inference emptied a domain', payload: { variable: 'name', value: 'value' } },
    backtrack: { description: 'a variable ran out of values', payload: { variable: 'name' } },
    solution: { description: 'a solution', payload: { assignment: 'variable to value' } },
  },
})

function backtracking<D>(problem: Csp<D>, options: BacktrackingOptions): Algorithm<BtState<D>> {
  const variableOrder = options.variableOrder ?? 'static'
  const valueOrder = options.valueOrder ?? 'static'
  const inference = options.inference ?? 'none'
  const all = options.all ?? false
  const maxSolutions = options.maxSolutions ?? Number.POSITIVE_INFINITY
  const maxAttempts = options.maxAttempts ?? 1_000_000
  const legal = (s: BtState<D>, v: number): number =>
    aliveValues(s.domains, v).filter((a) => conflictsOf(s.model, s.assigned, v, a) === 0).length
  const degree = (s: BtState<D>, v: number): number =>
    (s.model.on[v] ?? []).filter((c) => c.scope.some((w) => w !== v && (s.assigned[w] ?? -1) < 0)).length
  const select = (s: BtState<D>): number => {
    let best = -1
    let bestKey: [number, number] = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]
    for (let v = 0; v < s.model.n; v++) {
      if ((s.assigned[v] ?? -1) >= 0) continue
      if (variableOrder === 'static') return v
      const k: [number, number] =
        variableOrder === 'mrv'
          ? [legal(s, v), 0]
          : variableOrder === 'degree'
            ? [-degree(s, v), 0]
            : [legal(s, v), -degree(s, v)]
      if (k[0] < bestKey[0] || (k[0] === bestKey[0] && k[1] < bestKey[1])) {
        best = v
        bestKey = k
      }
    }
    return best
  }
  // Least constraining value: the values of unassigned neighbours each value would rule out (binary-reachable ones).
  const ruledOut = (s: BtState<D>, v: number, a: number): number => {
    let count = 0
    for (const c of s.model.on[v] ?? []) {
      const open = [...new Set(c.scope)].filter((w) => w !== v && (s.assigned[w] ?? -1) < 0)
      const u = open[0]
      if (open.length !== 1 || u === undefined) continue
      for (const b of aliveValues(s.domains, u)) {
        const at = (w: number): number => (w === v ? a : w === u ? b : (s.assigned[w] ?? -1))
        if (!check(s.model, c, at)) count += 1
      }
    }
    return count
  }
  const frame = (s: BtState<D>, v: number): Frame => {
    let order = aliveValues(s.domains, v)
    if (valueOrder === 'lcv') {
      order = order
        .map((a, i) => ({ a, i, r: ruledOut(s, v, a) }))
        .sort((x, y) => x.r - y.r || x.i - y.i)
        .map((x) => x.a)
    }
    return { variable: v, order, next: 0, mark: s.domains.trail.length }
  }
  // Inference after v = a: forward checking prunes each constraint's last open variable; MAC propagates GAC.
  const infer = (s: BtState<D>, v: number): boolean => {
    if (inference === 'none') return true
    if (inference === 'mac') {
      const arcs: [number, number][] = []
      for (const c of s.model.on[v] ?? []) {
        for (const u of new Set(c.scope)) if (u !== v && (s.assigned[u] ?? -1) < 0) arcs.push([u, c.index])
      }
      return propagate(s.model, s.domains, s.assigned, arcs, { revisions: 0 })
    }
    for (const c of s.model.on[v] ?? []) {
      const open = [...new Set(c.scope)].filter((w) => w !== v && (s.assigned[w] ?? -1) < 0)
      const u = open[0]
      if (open.length !== 1 || u === undefined) continue
      for (const b of aliveValues(s.domains, u)) {
        const at = (w: number): number => (w === u ? b : (s.assigned[w] ?? -1))
        if (!check(s.model, c, at)) prune(s.domains, u, b)
      }
      if ((s.domains.size[u] ?? 0) === 0) return false
    }
    return true
  }
  const snapshot = (s: BtState<D>): Assignment<D> => {
    const out: Record<string, D> = {}
    s.model.names.forEach((name, v) => {
      out[name] = s.model.values[v]?.[s.assigned[v] ?? 0] as D
    })
    return out
  }
  return {
    name: 'backtracking',
    config: { variableOrder, valueOrder, inference, all },
    schema: CSP_SCHEMA,
    init: () => {
      const model = compile(problem, 'backtrackingSearch')
      const s: BtState<D> = {
        status: 'running',
        depth: 0,
        assignments: 0,
        backtracks: 0,
        solutions: 0,
        event: null,
        model,
        domains: fullDomains(model),
        assigned: new Int32Array(model.n).fill(-1),
        stack: [],
        found: [],
        pruned: 0,
        attempts: 0,
      }
      if (model.n === 0) {
        s.found.push({})
        s.solutions = 1
        s.status = 'solved'
      } else s.stack.push(frame(s, select(s)))
      return s
    },
    step: (s) => {
      s.event = null
      for (;;) {
        const top = s.stack[s.stack.length - 1]
        if (top === undefined) {
          s.status = s.found.length > 0 ? 'solved' : 'failure'
          return s
        }
        restore(s.domains, top.mark)
        s.assigned[top.variable] = -1
        s.depth = s.stack.length - 1
        const a = top.order[top.next]
        const name = s.model.names[top.variable] ?? ''
        if (a === undefined) {
          s.stack.pop()
          s.backtracks += 1
          s.event = { kind: 'backtrack', payload: { variable: name } }
          return s
        }
        top.next += 1
        s.attempts += 1
        const value = s.model.values[top.variable]?.[a] as D
        if (conflictsOf(s.model, s.assigned, top.variable, a) > 0) {
          s.event = { kind: 'conflict', payload: { variable: name, value: value as Json } }
          return s
        }
        s.assigned[top.variable] = a
        s.assignments += 1
        s.depth = s.stack.length
        for (const b of aliveValues(s.domains, top.variable)) if (b !== a) prune(s.domains, top.variable, b)
        const before = s.domains.trail.length
        const ok = infer(s, top.variable)
        s.pruned += (s.domains.trail.length - before) / 2
        if (!ok) {
          s.event = { kind: 'wipeout', payload: { variable: name, value: value as Json } }
          return s
        }
        if (s.stack.length === s.model.n) {
          const solution = snapshot(s)
          s.found.push(solution)
          s.solutions += 1
          s.event = { kind: 'solution', payload: { assignment: solution as Json } }
          if (!all || s.found.length >= maxSolutions) s.status = 'solved'
          return s
        }
        s.event = {
          kind: 'assign',
          payload: { variable: name, value: value as Json, pruned: (s.domains.trail.length - before) / 2 },
        }
        s.stack.push(frame(s, select(s)))
        return s
      }
    },
    done: (s) =>
      s.status === 'solved'
        ? 'converged'
        : s.status === 'failure'
          ? 'stopped'
          : s.attempts >= maxAttempts
            ? 'stopped'
            : undefined,
    observe: (s) => ({
      metrics: { assignments: s.assignments, backtracks: s.backtracks, depth: s.depth, solutions: s.solutions },
      messages: s.event === null ? [] : [s.event],
    }),
  }
}

/**
 * Backtracking search as a traced algorithm (AIMA figure 6.5, aima-python's `backtracking_search`): assign one
 * variable at a time, try its values in order, skip values that conflict with the assignment, infer, recurse, and
 * undo on failure. Each step makes one value attempt or one backtrack.
 *
 * Ties in variable ordering go to the earlier variable (aima-python breaks them at random); MRV counts the current
 * values consistent with the assignment; LCV counts the values it would remove from neighbours that one constraint
 * leaves open.
 *
 * @param problem - The CSP.
 * @param options - Variable and value ordering, inference, whether to enumerate all solutions, and budgets.
 * @returns The algorithm.
 * @concept constraint-satisfaction
 * @example
 * ```ts
 * import { allDifferent, backtrackingAlgorithm, csp } from 'omai-engine/search'
 * backtrackingAlgorithm(csp(['a'], { a: [1] }, [])).name // => 'backtracking'
 * ```
 */
export function backtrackingAlgorithm<D>(
  problem: Csp<D>,
  options: BacktrackingOptions = {},
): Algorithm<BacktrackingState> {
  return backtracking(problem, options)
}

/**
 * Backtracking search with MRV, degree and LCV heuristics, forward checking or maintained arc consistency.
 *
 * @param problem - The CSP.
 * @param options - Ordering, inference, all solutions, budgets and tracing.
 * @returns The status, the first solution, every solution found, and the effort.
 * @concept constraint-satisfaction
 * @index function Backtracking search for constraint satisfaction problems.
 * @example
 * ```ts
 * import { allDifferent, backtrackingSearch, binaryConstraint, csp } from 'omai-engine/search'
 * // Colour Australia with three colours (AIMA figure 6.1).
 * const borders = [['WA', 'NT'], ['WA', 'SA'], ['NT', 'SA'], ['NT', 'Q'], ['SA', 'Q'], ['SA', 'NSW'], ['SA', 'V'], ['Q', 'NSW'], ['NSW', 'V']]
 * const regions = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
 * const colours = Object.fromEntries(regions.map((r) => [r, ['red', 'green', 'blue']]))
 * const australia = csp(regions, colours, borders.map(([a, b]) => binaryConstraint(a ?? '', b ?? '', (x, y) => x !== y)))
 * backtrackingSearch(australia, { variableOrder: 'mrv', inference: 'forward-checking' }).assignment // => { WA: 'red', NT: 'green', SA: 'blue', Q: 'red', NSW: 'green', V: 'red', T: 'red' }
 * backtrackingSearch(australia, { all: true }).solutions.length // => 18
 * ```
 */
export function backtrackingSearch<D>(problem: Csp<D>, options: BacktrackingOptions = {}): CspResult<D> {
  const {
    variableOrder: _v,
    valueOrder: _o,
    inference: _i,
    all: _a,
    maxSolutions: _m,
    maxAttempts = 1_000_000,
    ...run
  } = options
  const { state } = runAlgorithm(backtracking(problem, options), { ...run, maxSteps: 2 * maxAttempts + 2 })
  const status = state.status === 'running' ? 'budget' : state.status
  return {
    status: state.found.length > 0 && status !== 'budget' ? 'solved' : status,
    assignment: state.found[0] ?? null,
    solutions: state.found,
    assignments: state.assignments,
    backtracks: state.backtracks,
    checks: state.model.checks,
    pruned: state.pruned,
  }
}

// ------------------------------------------------------------------------------------------------ min-conflicts

/** The state of a min-conflicts search. */
export interface MinConflictsState<D> {
  readonly assignment: Assignment<D>
  /** The number of violated constraints. */
  readonly conflicts: number
  /** The variable changed in the last step. */
  readonly changed: string | null
}

interface McState<D> extends MinConflictsState<D> {
  assignment: Assignment<D>
  conflicts: number
  changed: string | null
  readonly model: Model<D>
  readonly values: Int32Array
}

function totalConflicts<D>(m: Model<D>, values: Int32Array): number {
  return m.constraints.filter((c) => !check(m, c, (v) => values[v] ?? 0)).length
}

function minConflictsEngine<D>(problem: Csp<D>, options: { readonly initial?: Assignment<D> }): Algorithm<McState<D>> {
  const toAssignment = (m: Model<D>, values: Int32Array): Assignment<D> =>
    Object.fromEntries(m.names.map((name, v) => [name, m.values[v]?.[values[v] ?? 0] as D]))
  const leastConflicted = (m: Model<D>, v: number, rng: Generator, assigned: Int32Array): number => {
    let best: number[] = []
    let bestCount = Number.POSITIVE_INFINITY
    const domain = m.values[v] ?? []
    for (let a = 0; a < domain.length; a++) {
      const count = conflictsOf(m, assigned, v, a)
      if (count < bestCount) {
        bestCount = count
        best = [a]
      } else if (count === bestCount) best.push(a)
    }
    return best[Math.min(best.length - 1, Math.floor((rng.uniform([1]).data[0] ?? 0) * best.length))] ?? 0
  }
  return {
    name: 'min-conflicts',
    config: {},
    schema: channels({
      metrics: { conflicts: { description: 'violated constraints' } },
      messages: { change: { description: 'a variable reassigned', payload: { variable: 'name', value: 'value' } } },
    }),
    init: (key) => {
      const m = compile(problem, 'minConflicts')
      if (m.values.some((d) => d.length === 0)) throw new OmaiError('value', 'minConflicts: a domain is empty')
      const rng = new Generator(key)
      const values = new Int32Array(m.n).fill(-1)
      // aima-python's start: each variable in turn takes a value with fewest conflicts with those already set.
      m.names.forEach((name, v) => {
        const given = options.initial?.[name]
        const at = given === undefined ? -1 : (m.values[v] ?? []).indexOf(given)
        values[v] = at >= 0 ? at : leastConflicted(m, v, rng, values)
      })
      return {
        model: m,
        values,
        assignment: toAssignment(m, values),
        conflicts: totalConflicts(m, values),
        changed: null,
      }
    },
    step: (s, ctx) => {
      const rng = new Generator(ctx.key)
      const m = s.model
      const conflicted: number[] = []
      for (let v = 0; v < m.n; v++) if (conflictsOf(m, s.values, v, s.values[v] ?? 0) > 0) conflicted.push(v)
      const v =
        conflicted[Math.min(conflicted.length - 1, Math.floor((rng.uniform([1]).data[0] ?? 0) * conflicted.length))]
      if (v === undefined) return s
      s.values[v] = leastConflicted(m, v, rng, s.values)
      s.assignment = toAssignment(m, s.values)
      s.conflicts = totalConflicts(m, s.values)
      s.changed = m.names[v] ?? null
      return s
    },
    done: (s) => (s.conflicts === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { conflicts: s.conflicts },
      messages:
        s.changed === null
          ? []
          : [{ kind: 'change', payload: { variable: s.changed, value: s.assignment[s.changed] as Json } }],
    }),
  }
}

/**
 * Min-conflicts local search as a traced algorithm (AIMA figure 6.8, aima-python's `min_conflicts`): start from a
 * greedy complete assignment, then repeatedly pick a conflicted variable at random and give it the value with the
 * fewest conflicts (ties at random). Each step changes one variable.
 *
 * @param problem - The CSP.
 * @param options - A starting assignment (missing variables are set greedily).
 * @returns The algorithm.
 * @throws OmaiError with code `value` when a domain is empty.
 * @concept constraint-satisfaction
 * @example
 * ```ts
 * import { csp, minConflictsAlgorithm } from 'omai-engine/search'
 * minConflictsAlgorithm(csp(['a'], { a: [1] }, [])).name // => 'min-conflicts'
 * ```
 */
export function minConflictsAlgorithm<D>(
  problem: Csp<D>,
  options: { readonly initial?: Assignment<D> } = {},
): Algorithm<MinConflictsState<D>> {
  return minConflictsEngine(problem, options)
}

/**
 * Min-conflicts search: fast on large loosely constrained problems such as n-queens, incomplete in general.
 *
 * @param problem - The CSP.
 * @param options - The step budget (default 100000), a starting assignment, a seed or key, and tracing.
 * @returns `solved` with the assignment, or `budget` with the last assignment tried in `solutions`.
 * @concept constraint-satisfaction
 * @example
 * ```ts
 * import { binaryConstraint, csp, isSolution, minConflicts } from 'omai-engine/search'
 * // Eight queens: one variable per column, valued by row.
 * const cols = Array.from({ length: 8 }, (_, i) => `q${i}`)
 * const rows = Object.fromEntries(cols.map((c) => [c, [0, 1, 2, 3, 4, 5, 6, 7]]))
 * const safe = cols.flatMap((a, i) => cols.slice(i + 1).map((b, k) => binaryConstraint(a, b, (r: number, s: number) => r !== s && Math.abs(r - s) !== k + 1)))
 * const queens = csp(cols, rows, safe)
 * const r = minConflicts(queens, { seed: 0 })
 * [r.status, isSolution(queens, r.assignment ?? {})] // => ['solved', true]
 * ```
 */
export function minConflicts<D>(
  problem: Csp<D>,
  options: SearchRunOptions & { readonly maxSteps?: number; readonly initial?: Assignment<D> } = {},
): CspResult<D> {
  const { maxSteps = 100_000, initial, ...run } = options
  const algorithm = minConflictsEngine(problem, initial === undefined ? {} : { initial })
  const { state, steps } = runAlgorithm(algorithm, { ...run, maxSteps })
  const solved = state.conflicts === 0
  return {
    status: solved ? 'solved' : 'budget',
    assignment: solved ? state.assignment : null,
    solutions: solved ? [state.assignment] : [],
    assignments: steps + state.model.n,
    backtracks: 0,
    checks: state.model.checks,
    pruned: 0,
  }
}

/**
 * A CSP as an incremental search problem (AIMA 6.3): a state is a partial assignment, the actions assign the first
 * unassigned variable (in order) a value consistent with the assignment, and the goal is a complete assignment. Any
 * path search applies; depth-first search on it is chronological backtracking.
 *
 * @param problem - The CSP.
 * @returns The search problem; actions are `[variable, value]` pairs.
 * @concept constraint-satisfaction
 * @example
 * ```ts
 * import { allDifferent, cspProblem, csp, depthFirstSearch } from 'omai-engine/search'
 * const p = csp(['a', 'b'], { a: [1, 2], b: [1, 2] }, allDifferent(['a', 'b']))
 * depthFirstSearch(cspProblem(p)).actions // => [['a', 2], ['b', 1]]
 * ```
 */
export function cspProblem<D>(problem: Csp<D>): SearchProblem<Assignment<D>, readonly [string, D]> {
  compile(problem, 'cspProblem')
  return {
    initial: {},
    actions: (s) => {
      const v = problem.variables.find((x) => !(x in s))
      if (v === undefined) return []
      return (problem.domains[v] ?? [])
        .filter((value) => violations(problem, { ...s, [v]: value }).length === 0)
        .map((value) => [v, value] as const)
    },
    result: (s, [v, value]) => ({ ...s, [v]: value }),
    isGoal: (s) => problem.variables.every((v) => v in s),
    key: (s) => JSON.stringify(problem.variables.map((v) => (v in s ? (s[v] as Json) : null))),
  }
}
