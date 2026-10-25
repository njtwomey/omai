/**
 * STRIPS planning (Fikes and Nilsson): actions with preconditions, add and delete lists over ground facts. Problem
 * construction, action application and goal regression, plan validation, and forward (progression) and regression
 * planners that hand the problem to A*, greedy or other searches with the relaxation heuristics.
 */

import { MdmlError } from '../../base/index.ts'
import type { BestFirstOptions } from '../_best-first.ts'
import { aStarSearch, greedyBestFirstSearch, weightedAStarSearch } from '../informed.ts'
import type { SearchProblem, SearchResult } from '../problem.ts'
import { breadthFirstSearch } from '../uninformed.ts'
import { fromState, relaxation } from './_relaxed.ts'
import { mutexPairs } from './graphplan.ts'

/**
 * A STRIPS action (Fikes and Nilsson 1971): preconditions, add and delete lists of ground facts, and a cost
 * (default 1).
 *
 * @concept classical-planning
 */
export interface StripsAction {
  readonly name: string
  readonly pre: readonly string[]
  readonly add: readonly string[]
  readonly del: readonly string[]
  readonly cost?: number
}

/** A STRIPS planning problem: the facts true initially, the goal facts and the actions. */
export interface StripsProblem {
  readonly initial: readonly string[]
  readonly goal: readonly string[]
  readonly actions: readonly StripsAction[]
}

/** A planning state: the sorted list of facts that hold. */
export type PlanState = readonly string[]

const sorted = (facts: Iterable<string>): string[] => [...new Set(facts)].sort()

/**
 * Build a STRIPS problem, normalising fact lists (sorted, without duplicates).
 *
 * @param initial - The initial facts.
 * @param goal - The goal facts.
 * @param actions - The actions.
 * @returns The problem.
 * @throws MdmlError with code `value` when action names repeat or a cost is negative.
 * @concept classical-planning
 * @example
 * ```ts
 * import { strips } from 'mdml-engine/search'
 * const p = strips(['at-home'], ['at-shop'], [{ name: 'go', pre: ['at-home'], add: ['at-shop'], del: ['at-home'] }])
 * p.actions.length // => 1
 * ```
 */
export function strips(
  initial: readonly string[],
  goal: readonly string[],
  actions: readonly StripsAction[],
): StripsProblem {
  const names = new Set<string>()
  const normal = actions.map((a) => {
    if (names.has(a.name)) throw new MdmlError('value', `strips: action ${a.name} is repeated`)
    names.add(a.name)
    if (a.cost !== undefined && !(a.cost >= 0))
      throw new MdmlError('value', `strips: action ${a.name} has a negative cost`)
    return { ...a, pre: sorted(a.pre), add: sorted(a.add), del: sorted(a.del) }
  })
  return { initial: sorted(initial), goal: sorted(goal), actions: normal }
}

/**
 * Whether an action is applicable in a state (its preconditions hold).
 *
 * @param state - The facts that hold.
 * @param action - The action.
 * @returns True when every precondition holds.
 * @concept classical-planning
 * @example
 * ```ts
 * import { applicable } from 'mdml-engine/search'
 * applicable(['a', 'b'], { name: 'x', pre: ['a'], add: [], del: [] }) // => true
 * ```
 */
export function applicable(state: PlanState | ReadonlySet<string>, action: StripsAction): boolean {
  const has = state instanceof Set ? (f: string) => state.has(f) : (f: string) => (state as PlanState).includes(f)
  return action.pre.every(has)
}

/**
 * The state after an action: (state minus deletes) plus adds.
 *
 * @param state - The facts that hold.
 * @param action - An applicable action.
 * @returns The new state (sorted).
 * @concept classical-planning
 * @example
 * ```ts
 * import { applyAction } from 'mdml-engine/search'
 * applyAction(['at-home', 'hungry'], { name: 'go', pre: ['at-home'], add: ['at-shop'], del: ['at-home'] }) // => ['at-shop', 'hungry']
 * ```
 */
export function applyAction(state: PlanState, action: StripsAction): string[] {
  const del = new Set(action.del)
  return sorted([...state.filter((f) => !del.has(f)), ...action.add])
}

/**
 * Regress a goal through an action (AIMA 10.2.2): the action is relevant when it adds some goal fact and deletes
 * none; the regressed goal is (goal minus adds) plus preconditions.
 *
 * @param goal - The goal facts.
 * @param action - The action.
 * @returns The regressed goal, or null when the action is not relevant.
 * @concept classical-planning
 * @example
 * ```ts
 * import { regress } from 'mdml-engine/search'
 * regress(['at-shop', 'hungry'], { name: 'go', pre: ['at-home'], add: ['at-shop'], del: ['at-home'] }) // => ['at-home', 'hungry']
 * ```
 */
export function regress(goal: PlanState, action: StripsAction): string[] | null {
  if (!action.add.some((f) => goal.includes(f)) || action.del.some((f) => goal.includes(f))) return null
  const add = new Set(action.add)
  return sorted([...goal.filter((f) => !add.has(f)), ...action.pre])
}

/** The outcome of checking a plan. */
export interface PlanCheck {
  readonly valid: boolean
  /** The state reached (where the plan stopped, when invalid). */
  readonly state: PlanState
  readonly cost: number
  /** The index of the first inapplicable action, or -1. */
  readonly failedAt: number
}

/**
 * Check a plan: every action applicable in turn, and the goal holding at the end.
 *
 * @param problem - The problem.
 * @param plan - Action names in order.
 * @returns Validity, the state reached, the cost, and where the plan failed.
 * @throws MdmlError with code `value` when a name is not an action of the problem.
 * @concept classical-planning
 * @example
 * ```ts
 * import { strips, validatePlan } from 'mdml-engine/search'
 * const p = strips(['at-home'], ['at-shop'], [{ name: 'go', pre: ['at-home'], add: ['at-shop'], del: ['at-home'] }])
 * [validatePlan(p, ['go']).valid, validatePlan(p, ['go', 'go']).failedAt] // => [true, 1]
 * ```
 */
export function validatePlan(problem: StripsProblem, plan: readonly string[]): PlanCheck {
  const byName = new Map(problem.actions.map((a) => [a.name, a]))
  let state: PlanState = problem.initial
  let cost = 0
  for (const [i, name] of plan.entries()) {
    const a = byName.get(name)
    if (a === undefined) throw new MdmlError('value', `validatePlan: unknown action ${name}`)
    if (!applicable(state, a)) return { valid: false, state, cost, failedAt: i }
    state = applyAction(state, a)
    cost += a.cost ?? 1
  }
  return { valid: problem.goal.every((g) => state.includes(g)), state, cost, failedAt: -1 }
}

/** The heuristics available to the planners. */
export type PlanningHeuristic = 'h-add' | 'h-max' | 'h-ff' | 'goal-count' | 'blind'

/**
 * Forward (progression) planning as a search problem: states are fact sets, actions the applicable STRIPS actions,
 * with a heuristic over states.
 *
 * @param problem - The STRIPS problem.
 * @param heuristic - The heuristic (default blind).
 * @returns The search problem.
 * @concept classical-planning
 * @example
 * ```ts
 * import { breadthFirstSearch, forwardProblem, strips } from 'mdml-engine/search'
 * const p = strips(['at-home'], ['at-shop'], [{ name: 'go', pre: ['at-home'], add: ['at-shop'], del: ['at-home'] }])
 * breadthFirstSearch(forwardProblem(p)).actions.map((a) => a.name) // => ['go']
 * ```
 */
export function forwardProblem(
  problem: StripsProblem,
  heuristic: PlanningHeuristic = 'blind',
): SearchProblem<PlanState, StripsAction> {
  const relax = relaxation(problem)
  const h =
    heuristic === 'blind'
      ? () => 0
      : heuristic === 'goal-count'
        ? (s: PlanState) => problem.goal.filter((g) => !s.includes(g)).length
        : (s: PlanState) => relax(s, problem.goal, heuristic)
  return {
    initial: problem.initial,
    actions: (s) => {
      const set = new Set(s)
      return problem.actions.filter((a) => applicable(set, a))
    },
    result: applyAction,
    isGoal: (s) => problem.goal.every((g) => s.includes(g)),
    stepCost: (_s, a) => a.cost ?? 1,
    heuristic: h,
    key: (s) => s.join('\u0000'),
    label: (s) => s.join(' '),
  }
}

/**
 * Regression (backward) planning as a search problem: states are subgoal sets starting from the goal, actions the
 * relevant and consistent STRIPS actions, and a state is a goal when the initial state satisfies it. Heuristics
 * measure a subgoal set from the initial state (fact costs are computed once). Subgoal sets that no reachable state
 * can satisfy (a fact unreachable from the initial state, or an invariant mutex pair from {@link mutexPairs}) are
 * pruned, as HSPr does; without this, regression in domains such as blocks world drowns in spurious states.
 *
 * @param problem - The STRIPS problem.
 * @param heuristic - The heuristic (default blind).
 * @param options - Whether to prune by mutexes and reachability (default true).
 * @returns The search problem; reverse its action sequence to get the plan.
 * @concept classical-planning
 * @example
 * ```ts
 * import { breadthFirstSearch, regressionProblem, strips } from 'mdml-engine/search'
 * const p = strips(['at-home'], ['at-shop'], [{ name: 'go', pre: ['at-home'], add: ['at-shop'], del: ['at-home'] }])
 * breadthFirstSearch(regressionProblem(p)).actions.map((a) => a.name) // => ['go']
 * ```
 */
export function regressionProblem(
  problem: StripsProblem,
  heuristic: PlanningHeuristic = 'blind',
  options: { readonly prune?: boolean } = {},
): SearchProblem<PlanState, StripsAction> {
  const measure = fromState(problem, problem.initial)
  const init = new Set(problem.initial)
  const h =
    heuristic === 'blind'
      ? () => 0
      : heuristic === 'goal-count'
        ? (g: PlanState) => g.filter((f) => !init.has(f)).length
        : (g: PlanState) => measure(g, heuristic)
  const prune = options.prune ?? true
  const { reachable, mutexes } = prune ? mutexPairs(problem) : { reachable: [], mutexes: [] }
  const canReach = new Set(reachable)
  const exclusive = new Map<string, Set<string>>()
  for (const [a, b] of mutexes) {
    exclusive.set(a, (exclusive.get(a) ?? new Set()).add(b))
    exclusive.set(b, (exclusive.get(b) ?? new Set()).add(a))
  }
  const spurious = (g: PlanState): boolean =>
    g.some((f) => !canReach.has(f)) || g.some((f, i) => g.slice(i + 1).some((x) => exclusive.get(f)?.has(x) === true))
  return {
    initial: problem.goal,
    actions: (g) =>
      problem.actions.filter((a) => {
        const r = regress(g, a)
        return r !== null && !(prune && spurious(r))
      }),
    result: (g, a) => regress(g, a) ?? [...g],
    isGoal: (g) => g.every((f) => init.has(f)),
    stepCost: (_s, a) => a.cost ?? 1,
    heuristic: h,
    key: (g) => g.join('\u0000'),
    label: (g) => g.join(' '),
  }
}

/** Options of {@link forwardPlan} and {@link regressionPlan}. */
export interface PlanOptions extends BestFirstOptions<PlanState> {
  /** The heuristic (default h-add for forward, h-max for regression). */
  readonly heuristic?: PlanningHeuristic
  /** The search: A* (default), greedy best-first, weighted A* or breadth-first. */
  readonly search?: 'a-star' | 'greedy' | 'weighted-a-star' | 'breadth-first'
  /** The weight of weighted A* (default 2). */
  readonly weight?: number
}

/** The result of a planner: the plan as action names, its cost, and the underlying search result. */
export interface PlanResult {
  readonly status: SearchResult<PlanState, StripsAction>['status']
  readonly plan: readonly string[]
  readonly cost: number
  readonly search: SearchResult<PlanState, StripsAction>
}

function plan(p: SearchProblem<PlanState, StripsAction>, options: PlanOptions, backward: boolean): PlanResult {
  const { heuristic: _h, search = 'a-star', weight = 2, ...rest } = options
  const r =
    search === 'breadth-first'
      ? breadthFirstSearch(p, rest)
      : search === 'greedy'
        ? greedyBestFirstSearch(p, rest)
        : search === 'weighted-a-star'
          ? weightedAStarSearch(p, weight, rest)
          : aStarSearch(p, rest)
  const names = r.actions.map((a) => a.name)
  return { status: r.status, plan: backward ? names.reverse() : names, cost: r.cost, search: r }
}

/**
 * Forward state-space planning: progression search from the initial state with h-add (HSP, Bonet and Geffner 2001),
 * h-max, the FF heuristic (Hoffmann and Nebel 2001), goal counting or none, by A*, greedy best-first, weighted A* or
 * breadth-first search. A* with h-max (admissible) or breadth-first search gives optimal plans.
 *
 * @param problem - The STRIPS problem.
 * @param options - The heuristic (default h-add), the search (default A*), its weight, budgets and tracing.
 * @returns The plan, its cost and the search result.
 * @concept classical-planning
 * @example
 * ```ts
 * import { forwardPlan, strips } from 'mdml-engine/search'
 * // Have your cake and eat it too (AIMA figure 10.7).
 * const cake = strips(['have-cake'], ['have-cake', 'eaten-cake'], [
 *   { name: 'eat', pre: ['have-cake'], add: ['eaten-cake'], del: ['have-cake'] },
 *   { name: 'bake', pre: [], add: ['have-cake'], del: [] },
 * ])
 * forwardPlan(cake, { heuristic: 'h-max' }).plan // => ['eat', 'bake']
 * ```
 */
export function forwardPlan(problem: StripsProblem, options: PlanOptions = {}): PlanResult {
  return plan(forwardProblem(problem, options.heuristic ?? 'h-add'), options, false)
}

/**
 * Regression planning: backward search from the goal over subgoal sets (AIMA 10.2.2), with h-max (default), h-add,
 * FF, goal counting or none measured from the initial state.
 *
 * @param problem - The STRIPS problem.
 * @param options - The heuristic (default h-max), the search (default A*), its weight, budgets and tracing.
 * @returns The plan (in execution order), its cost and the search result (over subgoal sets).
 * @concept classical-planning
 * @example
 * ```ts
 * import { regressionPlan, strips } from 'mdml-engine/search'
 * const cake = strips(['have-cake'], ['have-cake', 'eaten-cake'], [
 *   { name: 'eat', pre: ['have-cake'], add: ['eaten-cake'], del: ['have-cake'] },
 *   { name: 'bake', pre: [], add: ['have-cake'], del: [] },
 * ])
 * regressionPlan(cake).plan // => ['eat', 'bake']
 * ```
 */
export function regressionPlan(problem: StripsProblem, options: PlanOptions = {}): PlanResult {
  return plan(regressionProblem(problem, options.heuristic ?? 'h-max'), options, true)
}
