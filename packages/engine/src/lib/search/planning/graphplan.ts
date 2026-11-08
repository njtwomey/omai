/**
 * GraphPlan (Blum and Furst): a planner that grows a planning graph of alternating fact and action levels with
 * mutual-exclusion relations, then extracts a shortest parallel plan by backward search with memoised failures. Also
 * the invariant mutex pairs read from the levelled-off graph.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import type { SearchRunOptions } from '../problem.ts'
import { type StripsProblem, validatePlan } from './strips.ts'

/** One level of a planning graph: the facts, the actions that follow them, and their mutex pairs. */
export interface PlanningGraphLevel {
  readonly facts: readonly string[]
  /** Mutex fact pairs, each `[a, b]` with a < b. */
  readonly factMutexes: readonly (readonly [string, string])[]
  /** The applicable actions, no-ops (named `noop <fact>`) included. */
  readonly actions: readonly string[]
  readonly actionMutexes: readonly (readonly [string, string])[]
}

/** The result of GraphPlan: the plan as parallel steps, and the planning graph built. */
export interface GraphPlanResult {
  readonly status: 'solved' | 'failure' | 'budget'
  /** The parallel steps (no-ops removed); the actions within a step can run in any order. */
  readonly steps: readonly (readonly string[])[]
  /** The steps serialised into a sequential plan. */
  readonly plan: readonly string[]
  /** The number of action levels in the graph when the plan was found. */
  readonly levels: number
  readonly graph: readonly PlanningGraphLevel[]
}

/** The state of GraphPlan: the planning graph so far and the memoised failures. */
export interface GraphPlanState {
  readonly status: 'running' | 'solved' | 'failure' | 'budget'
  readonly levels: number
  readonly steps: readonly (readonly string[])[]
  readonly nogoods: number
}

interface Op {
  readonly name: string
  readonly pre: readonly number[]
  readonly add: readonly number[]
  readonly del: readonly number[]
  readonly noop: boolean
}

interface Layer {
  /** Facts present (fact ids). */
  readonly facts: Set<number>
  readonly factMutex: Set<number>
  /** Actions applicable from this layer's facts, and their mutexes. */
  actions: number[]
  actionMutex: Set<number>
}

interface Internal extends GraphPlanState {
  status: 'running' | 'solved' | 'failure' | 'budget'
  /** Backward-extraction choices made (the extraction budget). */
  work: number
  levels: number
  steps: (readonly string[])[]
  nogoods: number
  readonly layers: Layer[]
  readonly memo: Map<number, Set<string>>
  /** The first layer at which the graph levelled off (-1 before), and its nogood count after the last attempt. */
  levelledAt: number
  previous: number
  readonly ops: readonly Op[]
  readonly names: readonly string[]
}

const pair = (a: number, b: number): number => (a < b ? a * 1_000_003 + b : b * 1_000_003 + a)

/** The planning-graph machinery shared by GraphPlan and {@link mutexPairs}: fact and action ids and level expansion. */
function builder(problem: StripsProblem): {
  readonly names: string[]
  readonly ops: Op[]
  readonly id: (f: string) => number
  readonly expand: (layers: Layer[]) => void
  readonly levelledOff: (layers: readonly Layer[]) => boolean
} {
  const index = new Map<string, number>()
  const names: string[] = []
  const id = (f: string): number => {
    const at = index.get(f)
    if (at !== undefined) return at
    index.set(f, names.length)
    names.push(f)
    return names.length - 1
  }
  for (const f of [...problem.initial, ...problem.goal]) id(f)
  const ops: Op[] = problem.actions.map((a) => ({
    name: a.name,
    pre: a.pre.map(id),
    add: a.add.map(id),
    del: a.del.map(id),
    noop: false,
  }))
  const noopOf = new Map<number, number>()
  const noop = (f: number): number => {
    const at = noopOf.get(f)
    if (at !== undefined) return at
    ops.push({ name: `noop ${names[f]}`, pre: [f], add: [f], del: [], noop: true })
    noopOf.set(f, ops.length - 1)
    return ops.length - 1
  }
  const interferes = (a: Op, b: Op): boolean =>
    a.del.some((f) => b.pre.includes(f) || b.add.includes(f)) ||
    b.del.some((f) => a.pre.includes(f) || a.add.includes(f))

  /** Expand: the actions of the last layer and the next fact layer with mutexes. */
  const expand = (layers: Layer[]): void => {
    const last = layers[layers.length - 1] as Layer
    const actions: number[] = []
    for (const [i, op] of ops.entries()) {
      if (op.noop) continue
      if (!op.pre.every((f) => last.facts.has(f))) continue
      if (op.pre.some((f, k) => op.pre.slice(k + 1).some((g) => last.factMutex.has(pair(f, g))))) continue
      actions.push(i)
    }
    for (const f of last.facts) actions.push(noop(f))
    const actionMutex = new Set<number>()
    for (let x = 0; x < actions.length; x++) {
      for (let y = x + 1; y < actions.length; y++) {
        const a = ops[actions[x] as number] as Op
        const b = ops[actions[y] as number] as Op
        const competing = a.pre.some((f) => b.pre.some((g) => f !== g && last.factMutex.has(pair(f, g))))
        if (interferes(a, b) || competing) actionMutex.add(pair(actions[x] as number, actions[y] as number))
      }
    }
    last.actions = actions
    last.actionMutex = actionMutex
    const facts = new Set<number>()
    const achievers = new Map<number, number[]>()
    for (const a of actions) {
      for (const f of (ops[a] as Op).add) {
        facts.add(f)
        const list = achievers.get(f)
        if (list === undefined) achievers.set(f, [a])
        else list.push(a)
      }
    }
    const factMutex = new Set<number>()
    const fs = [...facts]
    for (let x = 0; x < fs.length; x++) {
      for (let y = x + 1; y < fs.length; y++) {
        const p = achievers.get(fs[x] as number) ?? []
        const q = achievers.get(fs[y] as number) ?? []
        // Inconsistent support: every way of achieving one is mutex with every way of achieving the other.
        if (p.every((a) => q.every((b) => a !== b && actionMutex.has(pair(a, b)))))
          factMutex.add(pair(fs[x] as number, fs[y] as number))
      }
    }
    layers.push({ facts, factMutex, actions: [], actionMutex: new Set() })
  }
  const levelledOff = (layers: readonly Layer[]): boolean => {
    const a = layers[layers.length - 1] as Layer
    const b = layers[layers.length - 2]
    return b !== undefined && a.facts.size === b.facts.size && a.factMutex.size === b.factMutex.size
  }
  return { names, ops, id, expand, levelledOff }
}

function graphplanEngine(
  problem: StripsProblem,
  options: { readonly maxLevels?: number; readonly maxNodes?: number },
): Algorithm<Internal> {
  const { names, ops, id, expand, levelledOff } = builder(problem)
  const goal = problem.goal.map(id)
  const maxLevels = options.maxLevels ?? 100
  const maxNodes = options.maxNodes ?? 1_000_000

  const goalsPossible = (layer: Layer): boolean =>
    goal.every((g) => layer.facts.has(g)) &&
    goal.every((g, k) => goal.slice(k + 1).every((h) => !layer.factMutex.has(pair(g, h))))

  /** Backward extraction from layer `level` (memoised failures per level). */
  const extract = (s: Internal, goals: readonly number[], level: number): number[][] | null => {
    if (level === 0) return goals.every((g) => (s.layers[0] as Layer).facts.has(g)) ? [] : null
    const keyOf = [...new Set(goals)].sort((a, b) => a - b).join(',')
    const failed = s.memo.get(level)
    if (failed?.has(keyOf)) return null
    const below = s.layers[level - 1] as Layer
    const sortedGoals = [...new Set(goals)]
    // Choose a non-mutex set of actions covering the goals, one goal at a time (no-ops tried first).
    const choose = (k: number, chosen: number[]): number[][] | null => {
      s.work += 1
      if (s.work > maxNodes) s.status = 'budget'
      if (s.status === 'budget') return null
      if (k === sortedGoals.length) {
        const pre = [...new Set(chosen.flatMap((a) => (ops[a] as Op).pre))]
        const rest = extract(s, pre, level - 1)
        return rest === null ? null : [...rest, chosen.filter((a) => !(ops[a] as Op).noop)]
      }
      const g = sortedGoals[k] as number
      if (chosen.some((a) => (ops[a] as Op).add.includes(g))) return choose(k + 1, chosen)
      const candidates = below.actions
        .filter((a) => (ops[a] as Op).add.includes(g))
        .sort((a, b) => Number((ops[b] as Op).noop) - Number((ops[a] as Op).noop) || a - b)
      for (const a of candidates) {
        if (chosen.some((c) => below.actionMutex.has(pair(a, c)))) continue
        const r = choose(k + 1, [...chosen, a])
        if (r !== null) return r
      }
      return null
    }
    const r = choose(0, [])
    if (r === null && s.status !== 'budget') {
      const set = s.memo.get(level) ?? new Set<string>()
      set.add(keyOf)
      s.memo.set(level, set)
      s.nogoods += 1
    }
    return r
  }

  return {
    name: 'graphplan',
    config: { maxLevels } as Json,
    schema: channels({
      metrics: {
        levels: { description: 'action levels in the planning graph' },
        facts: { description: 'facts in the last level' },
        mutexes: { description: 'fact mutexes in the last level' },
        nogoods: { description: 'memoised failed goal sets' },
      },
    }),
    init: () => {
      const s: Internal = {
        status: 'running',
        work: 0,
        levels: 0,
        steps: [],
        nogoods: 0,
        layers: [
          { facts: new Set(problem.initial.map(id)), factMutex: new Set(), actions: [], actionMutex: new Set() },
        ],
        memo: new Map(),
        levelledAt: -1,
        previous: -1,
        ops,
        names,
      }
      if (goal.every((g) => s.layers[0]?.facts.has(g))) s.status = 'solved'
      return s
    },
    step: (s) => {
      expand(s.layers)
      s.levels += 1
      const top = s.layers[s.layers.length - 1] as Layer
      if (goalsPossible(top)) {
        const steps = extract(s, goal, s.layers.length - 1)
        if (steps !== null) {
          s.steps = steps.map((step) => step.map((a) => (ops[a] as Op).name).sort())
          s.status = 'solved'
          return s
        }
        if (s.status === 'budget') return s
      }
      if (s.levelledAt < 0 && levelledOff(s.layers)) s.levelledAt = s.layers.length - 1
      if (s.levelledAt >= 0) {
        // Blum and Furst: once the graph has levelled off at layer n, a failed attempt that leaves the number of
        // memoised goal sets at layer n unchanged proves there is no plan.
        const count = s.memo.get(s.levelledAt)?.size ?? 0
        if (!goalsPossible(top) || count === s.previous) s.status = 'failure'
        s.previous = count
      }
      if (s.levels >= maxLevels && s.status === 'running') s.status = 'budget'
      return s
    },
    done: (s) => (s.status === 'solved' ? 'converged' : s.status === 'running' ? undefined : 'stopped'),
    observe: (s) => {
      const top = s.layers[s.layers.length - 1] as Layer
      return { metrics: { levels: s.levels, facts: top.facts.size, mutexes: top.factMutex.size, nogoods: s.nogoods } }
    },
  }
}

/** The planning graph as named levels. */
function levelsOf(s: Internal): PlanningGraphLevel[] {
  const fact = (f: number): string => s.names[f] ?? ''
  const action = (a: number): string => s.ops[a]?.name ?? ''
  const unpair = (code: number): [number, number] => [Math.floor(code / 1_000_003), code % 1_000_003]
  const named = (codes: Set<number>, name: (i: number) => string): [string, string][] =>
    [...codes]
      .map((c) => {
        const [a, b] = unpair(c)
        const x = name(a)
        const y = name(b)
        return (x < y ? [x, y] : [y, x]) as [string, string]
      })
      .sort((p, q) => (p[0] < q[0] ? -1 : p[0] > q[0] ? 1 : p[1] < q[1] ? -1 : p[1] > q[1] ? 1 : 0))
  return s.layers.map((l) => ({
    facts: [...l.facts].map(fact).sort(),
    factMutexes: named(l.factMutex, fact),
    actions: l.actions.map(action),
    actionMutexes: named(l.actionMutex, action),
  }))
}

/**
 * The invariant mutex pairs of a problem: fact pairs that never hold together in a state reachable from the initial
 * state, read from the planning graph once it has levelled off (sound, not complete: some unreachable pairs may be
 * missed). Regression planning prunes subgoal sets that contain one.
 *
 * @param problem - The STRIPS problem.
 * @param options - The level budget (default 1000).
 * @returns The facts reachable ignoring mutexes' absence, and the mutex pairs `[a, b]` with a < b, sorted.
 * @concept graphplan
 * @example
 * ```ts
 * import { mutexPairs, strips } from 'omai-engine/search'
 * const p = strips(['home'], ['shop'], [
 *   { name: 'go', pre: ['home'], add: ['shop'], del: ['home'] },
 *   { name: 'back', pre: ['shop'], add: ['home'], del: ['shop'] },
 * ])
 * mutexPairs(p).mutexes // => [['home', 'shop']]
 * ```
 */
export function mutexPairs(
  problem: StripsProblem,
  options: { readonly maxLevels?: number } = {},
): { readonly reachable: readonly string[]; readonly mutexes: readonly (readonly [string, string])[] } {
  const b = builder(problem)
  const layers: Layer[] = [
    { facts: new Set(problem.initial.map(b.id)), factMutex: new Set(), actions: [], actionMutex: new Set() },
  ]
  const maxLevels = options.maxLevels ?? 1000
  for (let i = 0; i < maxLevels; i++) {
    b.expand(layers)
    if (b.levelledOff(layers)) break
  }
  const top = layers[layers.length - 1] as Layer
  const name = (f: number): string => b.names[f] ?? ''
  const mutexes = [...top.factMutex]
    .map((code) => {
      const x = name(Math.floor(code / 1_000_003))
      const y = name(code % 1_000_003)
      return (x < y ? [x, y] : [y, x]) as [string, string]
    })
    .sort((p, q) => (p[0] < q[0] ? -1 : p[0] > q[0] ? 1 : p[1] < q[1] ? -1 : p[1] > q[1] ? 1 : 0))
  return { reachable: [...top.facts].map(name).sort(), mutexes }
}

/**
 * GraphPlan as a traced algorithm (Blum and Furst 1997): grow a planning graph of alternating fact and action levels
 * with mutual exclusions (inconsistent effects, interference, competing needs; inconsistent support), and whenever
 * the goals appear pairwise non-mutex, search backwards for a set of non-mutex actions per level, memoising failed
 * goal sets. Each step adds one level. Stops with failure once the graph has levelled off and the memoised failures
 * stop changing.
 *
 * @param problem - The STRIPS problem.
 * @param options - The level budget (default 100) and the extraction budget in choices (default 1000000).
 * @returns The algorithm.
 * @concept graphplan
 * @example
 * ```ts
 * import { graphplanAlgorithm, strips } from 'omai-engine/search'
 * graphplanAlgorithm(strips(['a'], ['a'], [])).name // => 'graphplan'
 * ```
 */
export function graphplanAlgorithm(
  problem: StripsProblem,
  options: { readonly maxLevels?: number; readonly maxNodes?: number } = {},
): Algorithm<GraphPlanState> {
  return graphplanEngine(problem, options)
}

/**
 * GraphPlan: a shortest parallel plan (fewest steps when independent actions may share a step).
 *
 * @param problem - The STRIPS problem.
 * @param options - The level budget (default 100), the extraction budget (default 1000000 choices) and tracing.
 * @returns The parallel steps, a serialised plan, the number of levels, and the graph's levels.
 * @concept graphplan
 * @example
 * ```ts
 * import { graphplan, strips } from 'omai-engine/search'
 * // The spare tire problem (AIMA figure 10.2), without the overnight action.
 * const tire = strips(['tire(flat)', 'tire(spare)', 'at(flat,axle)', 'at(spare,trunk)'], ['at(spare,axle)'], [
 *   { name: 'remove(spare,trunk)', pre: ['at(spare,trunk)'], add: ['at(spare,ground)'], del: ['at(spare,trunk)'] },
 *   { name: 'remove(flat,axle)', pre: ['at(flat,axle)'], add: ['at(flat,ground)'], del: ['at(flat,axle)'] },
 *   { name: 'puton(spare,axle)', pre: ['tire(spare)', 'at(spare,ground)', 'free(axle)'], add: ['at(spare,axle)'], del: ['at(spare,ground)'] },
 *   { name: 'free-axle', pre: ['at(flat,ground)'], add: ['free(axle)'], del: [] },
 * ])
 * graphplan(tire).steps // => [['remove(flat,axle)', 'remove(spare,trunk)'], ['free-axle'], ['puton(spare,axle)']]
 * ```
 */
export function graphplan(
  problem: StripsProblem,
  options: SearchRunOptions & { readonly maxLevels?: number; readonly maxNodes?: number } = {},
): GraphPlanResult {
  const { maxLevels = 100, maxNodes, ...run } = options
  const algorithm = graphplanEngine(problem, { maxLevels, ...(maxNodes === undefined ? {} : { maxNodes }) })
  const { state } = runAlgorithm(algorithm, { ...run, maxSteps: maxLevels + 1 })
  const plan = state.steps.flat()
  if (state.status === 'solved' && !validatePlan(problem, plan).valid) {
    throw new OmaiError('convergence', 'graphplan: the extracted plan does not serialise (an internal error)')
  }
  return {
    status: state.status === 'running' ? 'budget' : state.status,
    steps: state.steps,
    plan,
    levels: state.levels,
    graph: levelsOf(state),
  }
}
