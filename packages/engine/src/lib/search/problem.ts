import { WhiteboxError } from '../base/index.ts'
import { type Graph, type NodeId, nodeIndex } from '../graph/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import { type Structure, type StructureNode, structure } from '../scene/index.ts'
import type { RunOptions } from '../trace/index.ts'

/** The identity of a state in explored sets and frontiers: equal keys mean the same state. */
export type StateKey = string | number

/**
 * A search problem over an implicit state space (AIMA's `Problem`): an initial state, the actions available in a state,
 * the transition model, a goal test, a step cost (default 1) and a heuristic estimate of the cost to a goal (default
 * 0). States are compared by `key` (default: the state itself for strings and numbers, its JSON otherwise) and drawn by
 * `label` (default: the key as a string).
 *
 * @concept state-space-search
 */
export interface SearchProblem<S, A> {
  readonly initial: S
  actions(state: S): readonly A[]
  result(state: S, action: A): S
  isGoal(state: S): boolean
  stepCost?(state: S, action: A, next: S): number
  heuristic?(state: S): number
  key?(state: S): StateKey
  label?(state: S): string
}

/** A node of a search tree: a state, the node it was reached from, the action taken, the path cost and the depth. */
export interface SearchNode<S, A> {
  readonly state: S
  readonly parent: SearchNode<S, A> | null
  readonly action: A | undefined
  readonly pathCost: number
  readonly depth: number
}

/** How a search ended: a goal was found, the space was exhausted, a depth limit cut it off, or a budget ran out. */
export type SearchStatus = 'solved' | 'failure' | 'cutoff' | 'budget'

/**
 * Counts that describe a search's effort (AIMA's `InstrumentedProblem`): nodes expanded (successor generations),
 * nodes generated (transitions computed), goal tests, the largest frontier and the final explored set.
 */
export interface SearchStats {
  readonly expanded: number
  readonly generated: number
  readonly goalTests: number
  readonly maxFrontier: number
  readonly explored: number
}

/** The outcome of a path search: the goal node, its path and actions, its cost and depth, and the search's effort. */
export interface SearchResult<S, A> {
  readonly status: SearchStatus
  readonly node: SearchNode<S, A> | null
  readonly path: readonly S[]
  readonly actions: readonly A[]
  /** The path cost (Infinity when no goal was found). */
  readonly cost: number
  /** The solution depth (-1 when no goal was found). */
  readonly depth: number
  readonly stats: SearchStats
  /** The effective branching factor of the search (NaN when no goal was found at depth 1 or more). */
  readonly effectiveBranchingFactor: number
  /** Every generated node, root first, when the search was run with `recordTree`. */
  readonly tree?: readonly SearchNode<S, A>[]
}

/** A child produced by an expansion, and whether it was added to the frontier (false for duplicates and pruning). */
export interface GeneratedNode<S, A> {
  readonly node: SearchNode<S, A>
  readonly added: boolean
}

/**
 * The state of a traced path search, as seen by observers: the status, the goal node once found, the effort so far,
 * the sizes of the frontier and explored set, the node expanded in the last step and the children it generated, and
 * the current bound of an iterative deepening search. Searches advance their frontier in place, so read what you need
 * from a state before the next step (figures use the trace events instead).
 */
export interface SearchState<S, A> {
  readonly status: 'running' | SearchStatus
  readonly solution: SearchNode<S, A> | null
  readonly stats: SearchStats
  readonly frontierSize: number
  readonly exploredSize: number
  readonly current: SearchNode<S, A> | null
  readonly generated: readonly GeneratedNode<S, A>[]
  readonly bound?: number
}

/** Tracing options shared by every search (the tracer, policy, parent run, run id and seed or key). */
export type SearchRunOptions = Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId' | 'seed' | 'key'>>

/** Options shared by the path searches: an expansion budget, whether to keep the search tree, and tracing. */
export interface SearchOptions extends SearchRunOptions {
  /** Stop with status `budget` after this many expansions (default 100000). */
  readonly maxExpansions?: number
  /** Keep every generated node in the result's `tree` (for drawing the search tree). */
  readonly recordTree?: boolean
}

/**
 * The key function of a problem: its own `key`, else the state for strings and numbers and its JSON otherwise.
 *
 * @param problem - The problem (only `key` is read).
 * @returns A function from states to keys.
 * @example
 * ```ts
 * import { stateKeyOf } from 'whitebox-engine/search'
 * stateKeyOf({})([1, 2]) // => '[1,2]'
 * stateKeyOf({ key: (s: number[]) => s.length })([1, 2]) // => 2
 * ```
 */
export function stateKeyOf<S>(problem: { key?(state: S): StateKey }): (state: S) => StateKey {
  if (problem.key !== undefined) return (s) => (problem.key as (state: S) => StateKey)(s)
  return (s) => (typeof s === 'string' || typeof s === 'number' ? s : JSON.stringify(s))
}

/**
 * The root node of a problem's search tree.
 *
 * @param problem - The problem.
 * @returns A node for the initial state with cost 0 and depth 0.
 * @example
 * ```ts
 * import { rootNode } from 'whitebox-engine/search'
 * rootNode({ initial: 'a', actions: () => [], result: (s: string) => s, isGoal: () => false }).depth // => 0
 * ```
 */
export function rootNode<S, A>(problem: SearchProblem<S, A>): SearchNode<S, A> {
  return { state: problem.initial, parent: null, action: undefined, pathCost: 0, depth: 0 }
}

/**
 * The child of a node reached by an action (AIMA's `child_node`), with the step cost added to the path cost.
 *
 * @param problem - The problem.
 * @param parent - The node.
 * @param action - An action available in the node's state.
 * @returns The child node.
 * @throws WhiteboxError with code `value` when the step cost is negative or not a number.
 * @example
 * ```ts
 * import { childNode, rootNode } from 'whitebox-engine/search'
 * const p = { initial: 0, actions: () => [1, 2], result: (s: number, a: number) => s + a, isGoal: () => false }
 * childNode(p, rootNode(p), 2).state // => 2
 * ```
 */
export function childNode<S, A>(problem: SearchProblem<S, A>, parent: SearchNode<S, A>, action: A): SearchNode<S, A> {
  const state = problem.result(parent.state, action)
  const step = problem.stepCost === undefined ? 1 : problem.stepCost(parent.state, action, state)
  if (!(step >= 0)) throw new WhiteboxError('value', `search: step cost ${step} must be a non-negative number`)
  return { state, parent, action, pathCost: parent.pathCost + step, depth: parent.depth + 1 }
}

/**
 * The children of a node, one per available action in order.
 *
 * @param problem - The problem.
 * @param node - The node to expand.
 * @returns The child nodes.
 * @example
 * ```ts
 * import { expand, rootNode } from 'whitebox-engine/search'
 * const p = { initial: 0, actions: () => [1, 2], result: (s: number, a: number) => s + a, isGoal: () => false }
 * expand(p, rootNode(p)).map((n) => n.state) // => [1, 2]
 * ```
 */
export function expand<S, A>(problem: SearchProblem<S, A>, node: SearchNode<S, A>): SearchNode<S, A>[] {
  return problem.actions(node.state).map((a) => childNode(problem, node, a))
}

/**
 * The nodes from the root to a node.
 *
 * @param node - The node.
 * @returns The path of nodes, root first.
 * @example
 * ```ts
 * import { childNode, pathNodes, rootNode } from 'whitebox-engine/search'
 * const p = { initial: 0, actions: () => [1], result: (s: number, a: number) => s + a, isGoal: () => false }
 * pathNodes(childNode(p, rootNode(p), 1)).length // => 2
 * ```
 */
export function pathNodes<S, A>(node: SearchNode<S, A>): SearchNode<S, A>[] {
  const out: SearchNode<S, A>[] = []
  for (let n: SearchNode<S, A> | null = node; n !== null; n = n.parent) out.push(n)
  return out.reverse()
}

/**
 * The states from the initial state to a node's state.
 *
 * @param node - The node.
 * @returns The states, initial first.
 * @example
 * ```ts
 * import { childNode, rootNode, solutionPath } from 'whitebox-engine/search'
 * const p = { initial: 0, actions: () => [5], result: (s: number, a: number) => s + a, isGoal: () => false }
 * solutionPath(childNode(p, rootNode(p), 5)) // => [0, 5]
 * ```
 */
export function solutionPath<S, A>(node: SearchNode<S, A>): S[] {
  return pathNodes(node).map((n) => n.state)
}

/**
 * The actions from the initial state to a node.
 *
 * @param node - The node.
 * @returns The actions, first first.
 * @example
 * ```ts
 * import { childNode, rootNode, solutionActions } from 'whitebox-engine/search'
 * const p = { initial: 0, actions: () => [5], result: (s: number, a: number) => s + a, isGoal: () => false }
 * solutionActions(childNode(p, rootNode(p), 5)) // => [5]
 * ```
 */
export function solutionActions<S, A>(node: SearchNode<S, A>): A[] {
  return pathNodes(node)
    .slice(1)
    .map((n) => n.action as A)
}

/**
 * Replay a sequence of actions from the initial state and report where it ends and what it costs, checking that every
 * action is available where it is taken.
 *
 * @param problem - The problem.
 * @param actions - The actions.
 * @returns The final state, the path cost, and whether the final state is a goal.
 * @throws WhiteboxError with code `value` when an action is not available in the state it is applied to.
 * @example
 * ```ts
 * import { replay } from 'whitebox-engine/search'
 * const p = { initial: 0, actions: () => [1, 2], result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 3 }
 * replay(p, [1, 2]) // => { state: 3, cost: 2, goal: true }
 * ```
 */
export function replay<S, A>(
  problem: SearchProblem<S, A>,
  actions: readonly A[],
): { state: S; cost: number; goal: boolean } {
  let node = rootNode(problem)
  const same = (a: A, b: A): boolean => a === b || JSON.stringify(a) === JSON.stringify(b)
  for (const a of actions) {
    const available = problem.actions(node.state)
    const match = available.find((b) => same(a, b))
    if (match === undefined) throw new WhiteboxError('value', `replay: action ${JSON.stringify(a)} is not available`)
    node = childNode(problem, node, match)
  }
  return { state: node.state, cost: node.pathCost, goal: problem.isGoal(node.state) }
}

/**
 * The effective branching factor b* of a search (AIMA 3.6.1): the branching factor of a uniform tree of the solution
 * depth `d` that holds the `n` generated nodes, the root of `n + 1 = 1 + b + b^2 + ... + b^d`.
 *
 * @param generated - The number of nodes generated.
 * @param depth - The solution depth.
 * @returns b*, or NaN when the depth is not positive.
 * @example
 * ```ts
 * import { effectiveBranchingFactor } from 'whitebox-engine/search'
 * effectiveBranchingFactor(52, 5).toFixed(2) // => '1.92'
 * effectiveBranchingFactor(6, 2) // => 2
 * ```
 */
export function effectiveBranchingFactor(generated: number, depth: number): number {
  if (!(depth >= 1) || !(generated >= 0)) return Number.NaN
  const f = (b: number): number => {
    let s = 0
    let p = 1
    for (let i = 0; i <= depth; i++) {
      s += p
      p *= b
    }
    return s - (generated + 1)
  }
  if (f(0) === 0) return 0
  const r = brentq(f, 0, Math.max(generated, 1) + 1, { xtol: 1e-12 }).root
  const rounded = Math.round(r)
  return Math.abs(r - rounded) < 1e-9 ? rounded : r
}

/**
 * A search problem over an explicit graph (AIMA's `GraphProblem`): states are node indices, the actions in a state are
 * its successors in adjacency order, step costs are edge weights (the smallest of parallel edges), and the goal is
 * the target node.
 *
 * @param g - The graph (from `whitebox-engine/graph`).
 * @param source - The start node.
 * @param target - The goal node.
 * @param options - A heuristic on node indices (default 0).
 * @returns The problem; states are labelled by the graph's node labels.
 * @example
 * ```ts
 * import { graph } from 'whitebox-engine/graph'
 * import { graphProblem, uniformCostSearch } from 'whitebox-engine/search'
 * const g = graph([['a', 'b', 1], ['b', 'c', 1], ['a', 'c', 3]])
 * uniformCostSearch(graphProblem(g, 'a', 'c')).cost // => 2
 * ```
 */
export function graphProblem(
  g: Graph,
  source: NodeId,
  target: NodeId,
  options: { readonly heuristic?: (node: number) => number } = {},
): SearchProblem<number, number> {
  const s = nodeIndex(g, source)
  const t = nodeIndex(g, target)
  const a = g.out
  const weight = (u: number, v: number): number => {
    let best = Number.POSITIVE_INFINITY
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
      if (a.indices[k] === v) best = Math.min(best, a.weights[k] ?? 1)
    }
    return best
  }
  const heuristic = options.heuristic
  return {
    initial: s,
    actions: (u) => [...new Set(Array.from(a.indices.subarray(a.indptr[u] ?? 0, a.indptr[u + 1] ?? 0)))],
    result: (_u, v) => v,
    isGoal: (u) => u === t,
    stepCost: (u, _a, v) => weight(u, v),
    ...(heuristic === undefined ? {} : { heuristic }),
    key: (u) => u,
    label: (u) => g.labels[u] ?? String(u),
  }
}

/**
 * A search tree as a {@link Structure}: one node per generated search node (labelled by its state, with its path
 * cost and depth as badges) and an edge from each node to its children, labelled by the action. Nodes on the solution
 * path are highlighted.
 *
 * @param result - A search result run with `recordTree: true`.
 * @param options - How to label states (default `String`) and actions (default `JSON.stringify` unless strings or
 * numbers).
 * @returns The structure (type `search-tree`).
 * @throws WhiteboxError with code `value` when the result holds no tree.
 * @example
 * ```ts
 * import { breadthFirstSearch, searchTree } from 'whitebox-engine/search'
 * const p = { initial: 1, actions: (s: number) => (s < 4 ? [0, 1] : []), result: (s: number, a: number) => 2 * s + a, isGoal: (s: number) => s === 5 }
 * searchTree(breadthFirstSearch(p, { recordTree: true })).nodes.length // => 5
 * ```
 */
export function searchTree<S, A>(
  result: SearchResult<S, A>,
  options: { readonly label?: (state: S) => string; readonly actionLabel?: (action: A) => string } = {},
): Structure {
  const tree = result.tree
  if (tree === undefined) throw new WhiteboxError('value', 'searchTree: run the search with recordTree: true')
  const label = options.label ?? ((s: S) => (typeof s === 'string' ? s : JSON.stringify(s)))
  const actionLabel = options.actionLabel ?? ((a: A) => (typeof a === 'string' ? a : JSON.stringify(a)))
  const ids = new Map<SearchNode<S, A>, string>()
  tree.forEach((n, i) => {
    ids.set(n, `n${i}`)
  })
  const onPath = new Set(result.node === null ? [] : pathNodes(result.node))
  const nodes: StructureNode[] = tree.map((n, i) => ({
    id: `n${i}`,
    label: label(n.state),
    role: n === result.node ? 'output' : n.parent === null ? 'input' : 'state',
    badges: { g: n.pathCost, depth: n.depth },
  }))
  const edges = tree.flatMap((n) => {
    const from = n.parent === null ? undefined : ids.get(n.parent)
    const to = ids.get(n)
    if (from === undefined || to === undefined) return []
    return [{ from, to, directed: true, label: actionLabel(n.action as A) }]
  })
  const annotations = [...onPath].flatMap((n) => {
    const id = ids.get(n)
    return id === undefined ? [] : [{ target: id, text: 'solution path', kind: 'highlight' as const }]
  })
  return structure('search-tree', nodes, edges, { direction: 'down', annotations })
}
