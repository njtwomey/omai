/**
 * Delete-relaxation heuristics for classical planning, which estimate the distance to the goal by ignoring the facts
 * actions delete: the additive heuristic h-add and the admissible h-max (Bonet and Geffner, HSP), and the FF heuristic
 * with its relaxed plan (Hoffmann and Nebel).
 */

import { compile, extractPlan, relaxation, relaxedCosts } from './_relaxed.ts'
import type { PlanState, StripsProblem } from './strips.ts'

/**
 * The additive heuristic h-add (Bonet and Geffner 2001, HSP): the sum over goal facts of their relaxed costs, where a
 * fact's cost is the cheapest action adding it plus the sum of its preconditions' costs. Informative, not admissible.
 *
 * @param problem - The STRIPS problem.
 * @returns h-add of the problem's goal from a state (Infinity when unreachable even ignoring deletes).
 * @concept planning-heuristics
 * @example
 * ```ts
 * import { hAdd, strips } from 'mdml-engine/search'
 * const p = strips(['a'], ['c', 'd'], [
 *   { name: 'ab', pre: ['a'], add: ['b'], del: [] },
 *   { name: 'bc', pre: ['b'], add: ['c'], del: [] },
 *   { name: 'bd', pre: ['b'], add: ['d'], del: [] },
 * ])
 * hAdd(p)(['a']) // => 4
 * ```
 */
export function hAdd(problem: StripsProblem): (state: PlanState) => number {
  const r = relaxation(problem)
  return (s) => r(s, problem.goal, 'h-add')
}

/**
 * The max heuristic h-max (Bonet and Geffner 2001): like h-add with maxima in place of sums; admissible, so A* with
 * it plans optimally.
 *
 * @param problem - The STRIPS problem.
 * @returns h-max of the problem's goal from a state.
 * @concept planning-heuristics
 * @example
 * ```ts
 * import { hMax, strips } from 'mdml-engine/search'
 * const p = strips(['a'], ['c', 'd'], [
 *   { name: 'ab', pre: ['a'], add: ['b'], del: [] },
 *   { name: 'bc', pre: ['b'], add: ['c'], del: [] },
 *   { name: 'bd', pre: ['b'], add: ['d'], del: [] },
 * ])
 * hMax(p)(['a']) // => 2
 * ```
 */
export function hMax(problem: StripsProblem): (state: PlanState) => number {
  const r = relaxation(problem)
  return (s) => r(s, problem.goal, 'h-max')
}

/**
 * The FF heuristic (Hoffmann and Nebel 2001): the cost of a relaxed plan extracted backwards from the goal through
 * each fact's cheapest achiever under h-add (each action counted once). Between h-max and h-add.
 *
 * @param problem - The STRIPS problem.
 * @returns h-FF of the problem's goal from a state.
 * @concept planning-heuristics
 * @example
 * ```ts
 * import { hFF, strips } from 'mdml-engine/search'
 * const p = strips(['a'], ['c', 'd'], [
 *   { name: 'ab', pre: ['a'], add: ['b'], del: [] },
 *   { name: 'bc', pre: ['b'], add: ['c'], del: [] },
 *   { name: 'bd', pre: ['b'], add: ['d'], del: [] },
 * ])
 * hFF(p)(['a']) // => 3
 * ```
 */
export function hFF(problem: StripsProblem): (state: PlanState) => number {
  const r = relaxation(problem)
  return (s) => r(s, problem.goal, 'h-ff')
}

/**
 * FF's relaxed plan from a state: the actions (each once, in problem order) whose execution ignoring deletes reaches
 * the goal, chosen as cheapest achievers under h-add.
 *
 * @param problem - The STRIPS problem.
 * @param state - The state.
 * @returns The relaxed plan's action names, or null when the goal is unreachable even ignoring deletes.
 * @concept planning-heuristics
 * @example
 * ```ts
 * import { relaxedPlan, strips } from 'mdml-engine/search'
 * const p = strips(['a'], ['c'], [
 *   { name: 'ab', pre: ['a'], add: ['b'], del: ['a'] },
 *   { name: 'bc', pre: ['b'], add: ['c'], del: [] },
 * ])
 * relaxedPlan(p, ['a']) // => ['ab', 'bc']
 * ```
 */
export function relaxedPlan(problem: StripsProblem, state: PlanState): string[] | null {
  const c = compile(problem)
  const plan = extractPlan(c, relaxedCosts(c, state, 'sum'), problem.goal)
  return plan === null ? null : plan.map((a) => c.actions[a]?.name ?? '')
}
