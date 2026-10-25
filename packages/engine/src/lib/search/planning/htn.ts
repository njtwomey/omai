/**
 * Hierarchical task network planning: compound tasks are decomposed by methods into subtasks until only primitive
 * STRIPS actions remain. A SHOP-style total-order forward decomposition planner as a traced algorithm, and the
 * decomposition as a drawable tree.
 */

import { MdmlError } from '../../base/index.ts'
import { type Structure, type StructureNode, structure } from '../../scene/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import type { SearchRunOptions } from '../problem.ts'
import { applicable, applyAction, type PlanState, type StripsAction } from './strips.ts'

/** An HTN method: how to accomplish a compound task, when its preconditions hold, by an ordered list of subtasks. */
export interface HtnMethod {
  readonly name: string
  readonly task: string
  readonly pre?: readonly string[]
  readonly subtasks: readonly string[]
}

/** An HTN domain: primitive tasks are STRIPS actions (by name), compound tasks have methods. */
export interface HtnDomain {
  readonly actions: readonly StripsAction[]
  readonly methods: readonly HtnMethod[]
}

/** A step of a decomposition: the task, and the method (or `null` for a primitive action) that accomplished it. */
export interface Decomposition {
  readonly task: string
  readonly method: string | null
  /** The index of the decomposition that introduced this task (-1 for top-level tasks). */
  readonly parent: number
}

interface Node {
  readonly state: PlanState
  readonly tasks: readonly { readonly task: string; readonly parent: number }[]
  readonly plan: readonly string[]
  readonly tree: readonly Decomposition[]
}

/** The state of an HTN planner: the open choice points, the plan when found, and the effort. */
export interface HtnState {
  readonly status: 'running' | 'solved' | 'failure'
  readonly plan: readonly string[]
  readonly tree: readonly Decomposition[]
  readonly expansions: number
  readonly frontier: number
}

/** The result of {@link htnPlan}. */
export interface HtnResult {
  readonly status: 'solved' | 'failure' | 'budget'
  /** The primitive actions in order. */
  readonly plan: readonly string[]
  /** The decomposition, in the order tasks were accomplished. */
  readonly tree: readonly Decomposition[]
  readonly state: PlanState
  readonly expansions: number
}

interface Internal extends HtnState {
  status: 'running' | 'solved' | 'failure'
  plan: readonly string[]
  tree: readonly Decomposition[]
  expansions: number
  frontier: number
  final: PlanState
  readonly stack: Node[]
}

function htnEngine(domain: HtnDomain, initial: readonly string[], tasks: readonly string[]): Algorithm<Internal> {
  const actions = new Map(domain.actions.map((a) => [a.name, a]))
  const methods = new Map<string, HtnMethod[]>()
  for (const m of domain.methods) methods.set(m.task, [...(methods.get(m.task) ?? []), m])
  for (const t of tasks) {
    if (!actions.has(t) && !methods.has(t)) throw new MdmlError('value', `htnPlan: task ${t} has no action or method`)
  }
  return {
    name: 'htn',
    config: { tasks: [...tasks] },
    schema: channels({
      metrics: { expansions: { description: 'decompositions tried' }, frontier: { description: 'open choice points' } },
      messages: {
        decompose: { description: 'a task accomplished', payload: { task: 'task', method: 'method or null' } },
      },
    }),
    init: () => ({
      status: 'running',
      plan: [],
      tree: [],
      expansions: 0,
      frontier: 1,
      final: [...initial].sort(),
      stack: [
        { state: [...new Set(initial)].sort(), tasks: tasks.map((task) => ({ task, parent: -1 })), plan: [], tree: [] },
      ],
    }),
    step: (s) => {
      const node = s.stack.pop()
      if (node === undefined) {
        s.status = 'failure'
        return s
      }
      s.expansions += 1
      const [first, ...rest] = node.tasks
      if (first === undefined) {
        s.status = 'solved'
        s.plan = node.plan
        s.tree = node.tree
        s.final = node.state
        return s
      }
      const action = actions.get(first.task)
      const here = node.tree.length
      if (action !== undefined) {
        if (applicable(node.state, action)) {
          s.stack.push({
            state: applyAction(node.state, action),
            tasks: rest,
            plan: [...node.plan, action.name],
            tree: [...node.tree, { task: first.task, method: null, parent: first.parent }],
          })
        }
      } else {
        // Push the methods in reverse so the first applicable method is tried first (depth-first, as SHOP does).
        const options = (methods.get(first.task) ?? []).filter((m) =>
          (m.pre ?? []).every((f) => node.state.includes(f)),
        )
        for (const m of [...options].reverse()) {
          s.stack.push({
            state: node.state,
            tasks: [...m.subtasks.map((task) => ({ task, parent: here })), ...rest],
            plan: node.plan,
            tree: [...node.tree, { task: first.task, method: m.name, parent: first.parent }],
          })
        }
      }
      s.frontier = s.stack.length
      s.plan = node.plan
      s.tree = node.tree
      return s
    },
    done: (s) => (s.status === 'solved' ? 'converged' : s.status === 'failure' ? 'stopped' : undefined),
    observe: (s) => {
      const last = s.tree[s.tree.length - 1]
      return {
        metrics: { expansions: s.expansions, frontier: s.frontier },
        messages: last === undefined ? [] : [{ kind: 'decompose', payload: { task: last.task, method: last.method } }],
      }
    },
  }
}

/**
 * Simple hierarchical task network planning as a traced algorithm (total-order forward decomposition in the manner
 * of SHOP, Nau et al. 1999; AIMA 11.2): the first open task is either a primitive action, applied when applicable, or
 * a compound task, replaced by the subtasks of each method whose preconditions hold, depth-first with backtracking.
 * Each step expands one choice point.
 *
 * @param domain - The actions and methods.
 * @param initial - The initial facts.
 * @param tasks - The top-level tasks, in order.
 * @returns The algorithm.
 * @throws MdmlError with code `value` when a top-level task has neither an action nor a method.
 * @concept hierarchical-planning
 * @example
 * ```ts
 * import { htnPlanAlgorithm } from 'mdml-engine/search'
 * htnPlanAlgorithm({ actions: [{ name: 'wave', pre: [], add: [], del: [] }], methods: [] }, [], ['wave']).name // => 'htn'
 * ```
 */
export function htnPlanAlgorithm(
  domain: HtnDomain,
  initial: readonly string[],
  tasks: readonly string[],
): Algorithm<HtnState> {
  return htnEngine(domain, initial, tasks)
}

/**
 * Simple HTN planning: a plan of primitive actions accomplishing the tasks, with its decomposition.
 *
 * @param domain - The actions and methods.
 * @param initial - The initial facts.
 * @param tasks - The top-level tasks, in order.
 * @param options - The expansion budget (default 100000) and tracing.
 * @returns The plan, its decomposition and the final state.
 * @throws MdmlError with code `value` when a top-level task has neither an action nor a method.
 * @concept hierarchical-planning
 * @example
 * ```ts
 * import { htnPlan } from 'mdml-engine/search'
 * // Travel: take a taxi when the destination is near and cash is at hand, else walk.
 * const domain = {
 *   actions: [
 *     { name: 'call-taxi', pre: [], add: ['taxi'], del: [] },
 *     { name: 'ride-taxi', pre: ['taxi', 'cash'], add: ['at-park'], del: ['at-home', 'taxi', 'cash'] },
 *     { name: 'walk', pre: ['at-home'], add: ['at-park'], del: ['at-home'] },
 *   ],
 *   methods: [
 *     { name: 'by-taxi', task: 'travel', pre: ['cash'], subtasks: ['call-taxi', 'ride-taxi'] },
 *     { name: 'on-foot', task: 'travel', subtasks: ['walk'] },
 *   ],
 * }
 * htnPlan(domain, ['at-home', 'cash'], ['travel']).plan // => ['call-taxi', 'ride-taxi']
 * htnPlan(domain, ['at-home'], ['travel']).plan // => ['walk']
 * ```
 */
export function htnPlan(
  domain: HtnDomain,
  initial: readonly string[],
  tasks: readonly string[],
  options: SearchRunOptions & { readonly maxExpansions?: number } = {},
): HtnResult {
  const { maxExpansions = 100_000, ...run } = options
  const { state } = runAlgorithm(htnEngine(domain, initial, tasks), { ...run, maxSteps: maxExpansions })
  const status = state.status === 'running' ? 'budget' : state.status
  return {
    status,
    plan: status === 'solved' ? state.plan : [],
    tree: status === 'solved' ? state.tree : [],
    state: state.final,
    expansions: state.expansions,
  }
}

/**
 * A decomposition as a {@link Structure} tree: tasks as nodes (primitive actions as leaves) with edges from each task
 * to the subtasks its method introduced.
 *
 * @param result - A solved HTN result.
 * @returns The structure (type `htn`).
 * @concept hierarchical-planning
 * @example
 * ```ts
 * import { htnPlan, htnTree } from 'mdml-engine/search'
 * const domain = { actions: [{ name: 'a', pre: [], add: [], del: [] }], methods: [{ name: 'm', task: 't', subtasks: ['a', 'a'] }] }
 * htnTree(htnPlan(domain, [], ['t'])).edges.length // => 2
 * ```
 */
export function htnTree(result: HtnResult): Structure {
  const nodes: StructureNode[] = result.tree.map((d, i) => ({
    id: `t${i}`,
    label: d.task,
    role: d.method === null ? 'leaf' : 'internal',
    ...(d.method === null ? {} : { detail: d.method }),
  }))
  const edges = result.tree.flatMap((d, i) =>
    d.parent < 0 ? [] : [{ from: `t${d.parent}`, to: `t${i}`, directed: true }],
  )
  return structure('htn', nodes, edges, { direction: 'down' })
}
