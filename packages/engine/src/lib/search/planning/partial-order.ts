/**
 * Partial-order planning (AIMA 10.4.4, POP): a least-commitment planner that searches over partial plans of steps,
 * ordering constraints and causal links, adding steps for open preconditions and resolving threats by promotion or
 * demotion. The planning problem over plans, best-first search on it, and linearisation of the result.
 */

import { digraph, topologicalSort } from '../../graph/index.ts'
import { bestFirstSearch } from '../informed.ts'
import type { SearchOptions, SearchProblem, SearchResult } from '../problem.ts'
import type { StripsAction, StripsProblem } from './strips.ts'

/** A causal link: step `from` achieves `fact` for step `to`. */
export interface CausalLink {
  readonly from: number
  readonly fact: string
  readonly to: number
}

/**
 * A partial-order plan (AIMA 10.4.4): steps (0 is Start, whose effects are the initial state; 1 is Finish, whose
 * preconditions are the goal), ordering constraints `[before, after]`, causal links and open preconditions.
 */
export interface PartialPlan {
  readonly steps: readonly StripsAction[]
  readonly orderings: readonly (readonly [number, number])[]
  readonly links: readonly CausalLink[]
  readonly open: readonly { readonly step: number; readonly fact: string }[]
}

/** A refinement of a partial plan: what was done, and the plan that results. */
export interface Refinement {
  readonly description: string
  readonly plan: PartialPlan
}

function before(plan: PartialPlan, a: number, b: number): boolean {
  // Whether a < b follows from the orderings (depth-first reachability).
  if (a === b) return false
  const next = new Map<number, number[]>()
  for (const [x, y] of plan.orderings) next.set(x, [...(next.get(x) ?? []), y])
  const seen = new Set<number>([a])
  const stack = [a]
  while (stack.length > 0) {
    const u = stack.pop() as number
    for (const v of next.get(u) ?? []) {
      if (v === b) return true
      if (!seen.has(v)) {
        seen.add(v)
        stack.push(v)
      }
    }
  }
  return false
}

function order(plan: PartialPlan, a: number, b: number): PartialPlan | null {
  if (a === b || before(plan, b, a)) return null
  if (plan.orderings.some(([x, y]) => x === a && y === b)) return plan
  return { ...plan, orderings: [...plan.orderings, [a, b]] }
}

/** The first threat: a step that deletes a link's fact and may fall between its ends. */
function threat(plan: PartialPlan): { link: CausalLink; step: number } | null {
  for (const link of plan.links) {
    for (let c = 0; c < plan.steps.length; c++) {
      if (c === link.from || c === link.to) continue
      if (!(plan.steps[c]?.del ?? []).includes(link.fact)) continue
      if (before(plan, c, link.from) || before(plan, link.to, c)) continue
      return { link, step: c }
    }
  }
  return null
}

function refinements(plan: PartialPlan, actions: readonly StripsAction[]): Refinement[] {
  const t = threat(plan)
  if (t !== null) {
    const out: Refinement[] = []
    const name = plan.steps[t.step]?.name ?? ''
    const promoted = order(plan, t.link.to, t.step)
    if (promoted !== null) out.push({ description: `promote ${name}`, plan: promoted })
    const demoted = order(plan, t.step, t.link.from)
    if (demoted !== null) out.push({ description: `demote ${name}`, plan: demoted })
    return out
  }
  const goal = plan.open[plan.open.length - 1]
  if (goal === undefined) return []
  const rest = plan.open.slice(0, -1)
  const out: Refinement[] = []
  // Reuse an existing step that achieves the fact and can come first.
  plan.steps.forEach((s, i) => {
    if (i === goal.step || !s.add.includes(goal.fact)) return
    const ordered = order(plan, i, goal.step)
    if (ordered === null) return
    out.push({
      description: `link ${s.name} -> ${goal.fact}`,
      plan: { ...ordered, links: [...ordered.links, { from: i, fact: goal.fact, to: goal.step }], open: rest },
    })
  })
  // Or add a new step for it.
  for (const a of actions) {
    if (!a.add.includes(goal.fact)) continue
    const i = plan.steps.length
    const grown: PartialPlan = {
      steps: [...plan.steps, a],
      orderings: [...plan.orderings, [0, i], [i, 1], ...(goal.step === 1 ? [] : ([[i, goal.step]] as const))],
      links: [...plan.links, { from: i, fact: goal.fact, to: goal.step }],
      open: [...rest, ...a.pre.map((fact) => ({ step: i, fact }))],
    }
    out.push({ description: `add ${a.name} -> ${goal.fact}`, plan: grown })
  }
  return out
}

function keyOf(plan: PartialPlan): string {
  return JSON.stringify([
    plan.steps.map((s) => s.name),
    [...plan.orderings].map(([a, b]) => `${a}<${b}`).sort(),
    [...plan.links].map((l) => `${l.from}:${l.fact}:${l.to}`).sort(),
    plan.open.map((o) => `${o.step}:${o.fact}`),
  ])
}

/**
 * Partial-order planning as a search problem over plans (AIMA 10.4.4, POP; McAllester and Rosenblitt 1991): the
 * actions of a plan are its refinements, resolving the first threat by promotion or demotion, or else the newest open
 * precondition by linking from an existing step or adding a new one. A plan with no flaws is a solution.
 *
 * @param problem - The STRIPS problem.
 * @returns The search problem; its heuristic counts open preconditions.
 * @concept partial-order-planning
 * @example
 * ```ts
 * import { partialOrderProblem, strips } from 'mdml-engine/search'
 * const p = strips([], ['done'], [{ name: 'do', pre: [], add: ['done'], del: [] }])
 * partialOrderProblem(p).initial.open // => [{ step: 1, fact: 'done' }]
 * ```
 */
export function partialOrderProblem(problem: StripsProblem): SearchProblem<PartialPlan, Refinement> {
  const start: StripsAction = { name: 'start', pre: [], add: problem.initial, del: [] }
  const finish: StripsAction = { name: 'finish', pre: problem.goal, add: [], del: [] }
  return {
    initial: {
      steps: [start, finish],
      orderings: [[0, 1]],
      links: [],
      open: problem.goal.map((fact) => ({ step: 1, fact })),
    },
    actions: (plan) => refinements(plan, problem.actions),
    result: (_plan, r) => r.plan,
    isGoal: (plan) => plan.open.length === 0 && threat(plan) === null,
    stepCost: (plan, r) => r.plan.steps.length - plan.steps.length,
    heuristic: (plan) => plan.open.length,
    key: keyOf,
    label: (plan) =>
      plan.steps
        .slice(2)
        .map((s) => s.name)
        .join(', ') || '(empty)',
  }
}

/**
 * A linearisation of a partial plan: its steps (Start and Finish dropped) in a topological order of the orderings
 * (graph's `topologicalSort`, smallest step first among the free).
 *
 * @param plan - The plan.
 * @returns Action names in an executable order.
 * @concept partial-order-planning
 * @example
 * ```ts
 * import { linearize } from 'mdml-engine/search'
 * const step = (name: string) => ({ name, pre: [], add: [], del: [] })
 * linearize({ steps: [step('start'), step('finish'), step('b'), step('a')], orderings: [[0, 1], [0, 2], [0, 3], [3, 2], [2, 1], [3, 1]], links: [], open: [] }) // => ['a', 'b']
 * ```
 */
export function linearize(plan: PartialPlan): string[] {
  const g = digraph(
    plan.orderings.map(([a, b]) => [a, b] as const),
    { nodes: plan.steps.length },
  )
  return topologicalSort(g)
    .filter((i) => i > 1)
    .map((i) => plan.steps[i]?.name ?? '')
}

/** The result of {@link partialOrderPlan}. */
export interface PartialOrderResult {
  readonly status: SearchResult<PartialPlan, Refinement>['status']
  readonly plan: PartialPlan | null
  /** A linearisation of the plan (empty without one). */
  readonly linearization: readonly string[]
  readonly search: SearchResult<PartialPlan, Refinement>
}

/**
 * Partial-order planning: best-first search over partial plans by the number of steps plus open preconditions, the
 * least-commitment planner that solves the Sussman anomaly without interleaving trouble.
 *
 * @param problem - The STRIPS problem.
 * @param options - The expansion budget (default 20000) and tracing.
 * @returns The plan with its orderings and causal links, a linearisation, and the search result.
 * @concept partial-order-planning
 * @example
 * ```ts
 * import { partialOrderPlan, strips } from 'mdml-engine/search'
 * // Socks and shoes (AIMA figure 10.12): each sock before its shoe, the two feet independent.
 * const p = strips([], ['right-shoe-on', 'left-shoe-on'], [
 *   { name: 'right-shoe', pre: ['right-sock-on'], add: ['right-shoe-on'], del: [] },
 *   { name: 'right-sock', pre: [], add: ['right-sock-on'], del: [] },
 *   { name: 'left-shoe', pre: ['left-sock-on'], add: ['left-shoe-on'], del: [] },
 *   { name: 'left-sock', pre: [], add: ['left-sock-on'], del: [] },
 * ])
 * const r = partialOrderPlan(p)
 * [r.plan?.steps.length, r.linearization.indexOf('left-sock') < r.linearization.indexOf('left-shoe')] // => [6, true]
 * ```
 */
export function partialOrderPlan(problem: StripsProblem, options: SearchOptions = {}): PartialOrderResult {
  const search = bestFirstSearch(partialOrderProblem(problem), (n) => n.state.steps.length + n.state.open.length, {
    maxExpansions: 20_000,
    ...options,
  })
  const plan = search.status === 'solved' ? (search.path[search.path.length - 1] ?? null) : null
  return { status: search.status, plan, linearization: plan === null ? [] : linearize(plan), search }
}
