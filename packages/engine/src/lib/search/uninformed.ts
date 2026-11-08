/**
 * Uninformed (blind) search, AIMA style: breadth-first, depth-first, depth-limited, iterative deepening, uniform-cost
 * and bidirectional search, each as a traced algorithm that records its frontier and explored set so it can be
 * animated, with a function form returning the solution path and effort.
 */

import { OmaiError } from '../base/index.ts'
import { BinaryHeap } from '../graph/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { type BestFirstOptions, bestFirstEngine } from './_best-first.ts'
import {
  beginStep,
  budgetOf,
  type Core,
  core,
  generatedChild,
  goalTest,
  noteFrontier,
  observer,
  PATH_SCHEMA,
  runSearch,
  solved,
  stopper,
} from './_engine.ts'
import {
  childNode,
  rootNode,
  type SearchNode,
  type SearchOptions,
  type SearchProblem,
  type SearchResult,
  type SearchState,
  type StateKey,
  stateKeyOf,
} from './problem.ts'

/** Options of breadth-first and depth-first search. */
export interface GraphSearchOptions extends SearchOptions {
  /** Graph search with an explored set (default true); false searches the tree, re-expanding repeated states. */
  readonly graph?: boolean
}

interface QueueState<S, A> extends Core<S, A> {
  readonly frontier: SearchNode<S, A>[]
  /** Index of the queue's head (breadth-first) in `frontier`. */
  head: number
  readonly onFrontier: Map<StateKey, number>
  readonly closed: Set<StateKey>
}

function queueSearch<S, A>(
  name: 'breadth-first' | 'depth-first',
  problem: SearchProblem<S, A>,
  options: GraphSearchOptions,
): Algorithm<SearchState<S, A> & QueueState<S, A>> {
  const key = stateKeyOf(problem)
  const graph = options.graph ?? true
  const fifo = name === 'breadth-first'
  const maxExpansions = budgetOf(options, name)
  const enter = (s: QueueState<S, A>, n: SearchNode<S, A>): void => {
    s.frontier.push(n)
    const k = key(n.state)
    s.onFrontier.set(k, (s.onFrontier.get(k) ?? 0) + 1)
  }
  const leave = (s: QueueState<S, A>, n: SearchNode<S, A>): void => {
    const k = key(n.state)
    const c = (s.onFrontier.get(k) ?? 1) - 1
    if (c === 0) s.onFrontier.delete(k)
    else s.onFrontier.set(k, c)
  }
  return {
    name,
    config: { graph },
    schema: PATH_SCHEMA,
    init: () => {
      const root = rootNode(problem)
      const s: QueueState<S, A> = {
        ...core<S, A>(options.recordTree === true),
        frontier: [],
        head: 0,
        onFrontier: new Map(),
        closed: new Set(),
      }
      s.tree?.push(root)
      // Breadth-first search tests for the goal when a node is generated, so the root is tested here.
      if (fifo && goalTest(problem, s, root.state)) solved(s, root)
      else enter(s, root)
      noteFrontier(s, s.frontier.length - s.head)
      return s
    },
    step: (s) => {
      beginStep(s)
      if (s.frontier.length - s.head === 0) {
        s.status = 'failure'
        return s
      }
      const node = (fifo ? s.frontier[s.head++] : s.frontier.pop()) as SearchNode<S, A>
      if (fifo && s.head > 1024 && s.head * 2 > s.frontier.length) {
        s.frontier.splice(0, s.head)
        s.head = 0
      }
      leave(s, node)
      s.current = node
      if (!fifo && goalTest(problem, s, node.state)) {
        solved(s, node)
        return s
      }
      if (graph) s.closed.add(key(node.state))
      s.stats.expanded += 1
      // All children are generated before any is tested, as aima-python's `node.expand` does.
      const children = problem.actions(node.state).map((a) => childNode(problem, node, a))
      for (const child of children) {
        const ck = key(child.state)
        if (s.status === 'solved' || (graph && (s.closed.has(ck) || s.onFrontier.has(ck)))) {
          generatedChild(s, child, false)
          continue
        }
        generatedChild(s, child, true)
        if (fifo && goalTest(problem, s, child.state)) solved(s, child)
        else enter(s, child)
      }
      if (s.status === 'solved') return s
      noteFrontier(s, s.frontier.length - s.head)
      s.exploredSize = s.closed.size
      s.stats.explored = s.closed.size
      return s
    },
    done: stopper(maxExpansions),
    observe: observer(problem),
  }
}

/**
 * Breadth-first search as a traced algorithm (aima-python's `breadth_first_graph_search`): a FIFO frontier, the goal
 * test applied when a node is generated, and (in graph search) children skipped when their state is explored or on the
 * frontier. Each step expands one node.
 *
 * @param problem - The problem.
 * @param options - Graph or tree search, the expansion budget and tree recording.
 * @returns The algorithm; run it with `runAlgorithm` or use {@link breadthFirstSearch}.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { breadthFirstSearchAlgorithm } from 'omai-engine/search'
 * const p = { initial: 0, actions: () => [1], result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 3 }
 * breadthFirstSearchAlgorithm(p).name // => 'breadth-first'
 * ```
 */
export function breadthFirstSearchAlgorithm<S, A>(
  problem: SearchProblem<S, A>,
  options: GraphSearchOptions = {},
): Algorithm<SearchState<S, A>> {
  return queueSearch('breadth-first', problem, options)
}

/**
 * Breadth-first search: the shallowest goal (fewest actions), optimal for unit step costs.
 *
 * @param problem - The problem.
 * @param options - Graph or tree search, the expansion budget, tree recording and tracing.
 * @returns The result; `status` is `failure` when the reachable space holds no goal and `budget` when the budget ran
 * out.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { breadthFirstSearch } from 'omai-engine/search'
 * const p = { initial: 1, actions: () => ['double', 'inc'], result: (s: number, a: string) => (a === 'double' ? 2 * s : s + 1), isGoal: (s: number) => s === 10 }
 * breadthFirstSearch(p).actions // => ['double', 'double', 'inc', 'double']
 * ```
 */
export function breadthFirstSearch<S, A>(
  problem: SearchProblem<S, A>,
  options: GraphSearchOptions = {},
): SearchResult<S, A> {
  return runSearch(queueSearch('breadth-first', problem, options), options)
}

/**
 * Depth-first search as a traced algorithm (aima-python's `depth_first_graph_search`): a LIFO frontier onto which all
 * children are pushed, so the last child is searched first; the goal test is applied when a node is popped. Each step
 * expands one node.
 *
 * @param problem - The problem.
 * @param options - Graph or tree search, the expansion budget and tree recording.
 * @returns The algorithm.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { depthFirstSearchAlgorithm } from 'omai-engine/search'
 * const p = { initial: 0, actions: () => [1], result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 3 }
 * depthFirstSearchAlgorithm(p).name // => 'depth-first'
 * ```
 */
export function depthFirstSearchAlgorithm<S, A>(
  problem: SearchProblem<S, A>,
  options: GraphSearchOptions = {},
): Algorithm<SearchState<S, A>> {
  return queueSearch('depth-first', problem, options)
}

/**
 * Depth-first search: memory linear in depth for tree search, complete in finite spaces as graph search, not optimal.
 * Tree search in a space with cycles is stopped by the expansion budget.
 *
 * @param problem - The problem.
 * @param options - Graph or tree search, the expansion budget, tree recording and tracing.
 * @returns The result.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { depthFirstSearch } from 'omai-engine/search'
 * const p = { initial: 0, actions: (s: number) => (s < 3 ? [1, 2] : []), result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 3 }
 * depthFirstSearch(p).actions // => [2, 1]
 * ```
 */
export function depthFirstSearch<S, A>(
  problem: SearchProblem<S, A>,
  options: GraphSearchOptions = {},
): SearchResult<S, A> {
  return runSearch(queueSearch('depth-first', problem, options), options)
}

/** Options of depth-limited and iterative deepening search. */
export interface DepthLimitedOptions extends SearchOptions {
  /** Skip children whose state already lies on the path from the root (default false, as in AIMA). */
  readonly avoidCycles?: boolean
}

interface Frame<S, A> {
  readonly node: SearchNode<S, A>
  children: SearchNode<S, A>[] | null
  next: number
  cutoff: boolean
}

interface DepthState<S, A> extends Core<S, A> {
  readonly stack: Frame<S, A>[]
  limit: number
  rootCutoff: boolean
}

function onPath<S, A>(node: SearchNode<S, A>, k: StateKey, key: (s: S) => StateKey): boolean {
  for (let n: SearchNode<S, A> | null = node; n !== null; n = n.parent) if (key(n.state) === k) return true
  return false
}

function depthSearch<S, A>(
  name: 'depth-limited' | 'iterative-deepening',
  problem: SearchProblem<S, A>,
  first: number,
  last: number,
  options: DepthLimitedOptions,
): Algorithm<SearchState<S, A> & DepthState<S, A>> {
  const key = stateKeyOf(problem)
  const maxExpansions = budgetOf(options, name)
  const start = (s: DepthState<S, A>): void => {
    const root = rootNode(problem)
    s.tree?.push(root)
    s.stack.push({ node: root, children: null, next: 0, cutoff: false })
    s.rootCutoff = false
    s.bound = s.limit
  }
  return {
    name,
    config: { first, last, avoidCycles: options.avoidCycles ?? false },
    schema: PATH_SCHEMA,
    init: () => {
      const s: DepthState<S, A> = {
        ...core<S, A>(options.recordTree === true),
        stack: [],
        limit: first,
        rootCutoff: false,
      }
      start(s)
      noteFrontier(s, 1)
      return s
    },
    step: (s) => {
      beginStep(s)
      // Advance the recursion of aima-python's recursive_dls until one node is expanded or the search ends.
      for (;;) {
        const top = s.stack[s.stack.length - 1]
        if (top === undefined) {
          if (s.rootCutoff && s.limit < last) {
            s.limit += 1
            start(s)
            return s
          }
          s.status = s.rootCutoff ? 'cutoff' : 'failure'
          return s
        }
        if (top.children === null) {
          if (goalTest(problem, s, top.node.state)) {
            solved(s, top.node)
            s.current = top.node
            return s
          }
          if (top.node.depth >= s.limit) {
            s.stack.pop()
            const parent = s.stack[s.stack.length - 1]
            if (parent === undefined) s.rootCutoff = true
            else parent.cutoff = true
            continue
          }
          s.stats.expanded += 1
          s.current = top.node
          const children: SearchNode<S, A>[] = []
          for (const a of problem.actions(top.node.state)) {
            const child = childNode(problem, top.node, a)
            const skip = options.avoidCycles === true && onPath(top.node, key(child.state), key)
            generatedChild(s, child, !skip)
            if (!skip) children.push(child)
          }
          top.children = children
          noteFrontier(
            s,
            s.stack.reduce((acc, f) => acc + (f.children === null ? 1 : f.children.length - f.next), 0),
          )
          return s
        }
        const child = top.children[top.next]
        if (child !== undefined) {
          top.next += 1
          s.stack.push({ node: child, children: null, next: 0, cutoff: false })
          continue
        }
        s.stack.pop()
        const parent = s.stack[s.stack.length - 1]
        if (parent === undefined) s.rootCutoff = top.cutoff
        else if (top.cutoff) parent.cutoff = true
      }
    },
    done: stopper(maxExpansions),
    observe: observer(problem),
  }
}

function checkLimit(limit: number, name: string): void {
  if (!(Number.isInteger(limit) && limit >= 0)) {
    throw new OmaiError('value', `${name}: the depth limit ${limit} must be a non-negative integer`)
  }
}

/**
 * Depth-limited search as a traced algorithm (aima-python's `depth_limited_search`, made iterative): depth-first tree
 * search that treats nodes at the limit as leaves and reports `cutoff` when the limit pruned some branch.
 *
 * @param problem - The problem.
 * @param limit - The depth limit.
 * @param options - Cycle checking along the path, the expansion budget and tree recording.
 * @returns The algorithm.
 * @throws OmaiError with code `value` when the limit is not a non-negative integer.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { depthLimitedSearchAlgorithm } from 'omai-engine/search'
 * const p = { initial: 0, actions: () => [1], result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 3 }
 * depthLimitedSearchAlgorithm(p, 2).name // => 'depth-limited'
 * ```
 */
export function depthLimitedSearchAlgorithm<S, A>(
  problem: SearchProblem<S, A>,
  limit: number,
  options: DepthLimitedOptions = {},
): Algorithm<SearchState<S, A>> {
  checkLimit(limit, 'depthLimitedSearch')
  return depthSearch('depth-limited', problem, limit, limit, options)
}

/**
 * Depth-limited search.
 *
 * @param problem - The problem.
 * @param limit - The depth limit.
 * @param options - Cycle checking, the expansion budget, tree recording and tracing.
 * @returns The result; `status` is `cutoff` when no goal lies within the limit but the limit pruned some branch.
 * @throws OmaiError with code `value` when the limit is not a non-negative integer.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { depthLimitedSearch } from 'omai-engine/search'
 * const p = { initial: 0, actions: () => [1], result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 3 }
 * depthLimitedSearch(p, 2).status // => 'cutoff'
 * depthLimitedSearch(p, 3).depth // => 3
 * ```
 */
export function depthLimitedSearch<S, A>(
  problem: SearchProblem<S, A>,
  limit: number,
  options: DepthLimitedOptions = {},
): SearchResult<S, A> {
  checkLimit(limit, 'depthLimitedSearch')
  return runSearch(depthSearch('depth-limited', problem, limit, limit, options), options)
}

/**
 * Iterative deepening search as a traced algorithm (aima-python's `iterative_deepening_search`): depth-limited search
 * with limits 0, 1, 2, ... until a search does not end in a cutoff. Effort accumulates over the iterations; a `bound`
 * message marks each new limit.
 *
 * @param problem - The problem.
 * @param options - The largest limit (default 1000), cycle checking, the expansion budget and tree recording.
 * @returns The algorithm.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { iterativeDeepeningSearchAlgorithm } from 'omai-engine/search'
 * const p = { initial: 0, actions: () => [1], result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 3 }
 * iterativeDeepeningSearchAlgorithm(p).name // => 'iterative-deepening'
 * ```
 */
export function iterativeDeepeningSearchAlgorithm<S, A>(
  problem: SearchProblem<S, A>,
  options: DepthLimitedOptions & { readonly maxDepth?: number } = {},
): Algorithm<SearchState<S, A>> {
  const maxDepth = options.maxDepth ?? 1000
  checkLimit(maxDepth, 'iterativeDeepeningSearch')
  return depthSearch('iterative-deepening', problem, 0, maxDepth, options)
}

/**
 * Iterative deepening search: breadth-first optimality for unit costs with depth-first memory.
 *
 * @param problem - The problem.
 * @param options - The largest limit (default 1000), cycle checking, the expansion budget, tree recording and tracing.
 * @returns The result; `cutoff` when no goal lies within `maxDepth`.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { iterativeDeepeningSearch } from 'omai-engine/search'
 * const p = { initial: 1, actions: () => ['double', 'inc'], result: (s: number, a: string) => (a === 'double' ? 2 * s : s + 1), isGoal: (s: number) => s === 10 }
 * iterativeDeepeningSearch(p).depth // => 4
 * ```
 */
export function iterativeDeepeningSearch<S, A>(
  problem: SearchProblem<S, A>,
  options: DepthLimitedOptions & { readonly maxDepth?: number } = {},
): SearchResult<S, A> {
  const maxDepth = options.maxDepth ?? 1000
  checkLimit(maxDepth, 'iterativeDeepeningSearch')
  return runSearch(depthSearch('iterative-deepening', problem, 0, maxDepth, options), options)
}

/**
 * Uniform-cost search as a traced algorithm (aima-python's `uniform_cost_search`): best-first graph search on the path
 * cost g, optimal for non-negative step costs.
 *
 * @param problem - The problem.
 * @param options - Graph or tree search, tie-breaking, the expansion budget and tree recording.
 * @returns The algorithm.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { uniformCostSearchAlgorithm } from 'omai-engine/search'
 * const p = { initial: 0, actions: () => [1], result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 3 }
 * uniformCostSearchAlgorithm(p).name // => 'uniform-cost'
 * ```
 */
export function uniformCostSearchAlgorithm<S, A>(
  problem: SearchProblem<S, A>,
  options: BestFirstOptions<S> = {},
): Algorithm<SearchState<S, A>> {
  return bestFirstEngine('uniform-cost', problem, (n) => n.pathCost, options, { graph: options.graph ?? true })
}

/**
 * Uniform-cost search (Dijkstra's algorithm on an implicit graph): the cheapest path to a goal.
 *
 * @param problem - The problem.
 * @param options - Graph or tree search, tie-breaking, the expansion budget, tree recording and tracing.
 * @returns The result.
 * @concept uninformed-search
 * @index function Uniform-cost search; BFS, DFS and iterative deepening beside it.
 * @example
 * ```ts
 * import { uniformCostSearch } from 'omai-engine/search'
 * const p = { initial: 1, actions: () => ['double', 'inc'], result: (s: number, a: string) => (a === 'double' ? 2 * s : s + 1), stepCost: (_s: number, a: string) => (a === 'double' ? 3 : 1), isGoal: (s: number) => s === 10 }
 * uniformCostSearch(p).cost // => 7
 * ```
 */
export function uniformCostSearch<S, A>(
  problem: SearchProblem<S, A>,
  options: BestFirstOptions<S> = {},
): SearchResult<S, A> {
  return runSearch(
    bestFirstEngine('uniform-cost', problem, (n) => n.pathCost, options, { graph: options.graph ?? true }),
    options,
  )
}

/**
 * A problem for bidirectional search: a search problem with an explicit goal state and, unless the actions are
 * reversible, the predecessors of a state (each with the action that leads from it to the state, and its cost).
 */
export interface BidirectionalProblem<S, A> extends SearchProblem<S, A> {
  readonly goal: S
  predecessors?(state: S): readonly { readonly state: S; readonly action: A; readonly cost?: number }[]
}

interface Side<S, A> {
  readonly heap: BinaryHeap<SearchNode<S, A>>
  readonly best: Map<StateKey, SearchNode<S, A>>
  readonly closed: Set<StateKey>
}

interface BidirectionalState<S, A> extends Core<S, A> {
  readonly forward: Side<S, A>
  readonly backward: Side<S, A>
  mu: number
  meet: { forward: SearchNode<S, A>; backward: SearchNode<S, A> } | null
}

function bidirectional<S, A>(
  problem: BidirectionalProblem<S, A>,
  options: SearchOptions,
): Algorithm<SearchState<S, A> & BidirectionalState<S, A>> {
  const key = stateKeyOf(problem)
  const maxExpansions = budgetOf(options, 'bidirectional')
  const cost = (p: S, a: A, s: S): number => problem.stepCost?.(p, a, s) ?? 1
  const predecessors =
    problem.predecessors?.bind(problem) ??
    ((s: S) =>
      problem.actions(s).flatMap((a) => {
        const p = problem.result(s, a)
        const ks = key(s)
        const back = problem.actions(p).find((b) => key(problem.result(p, b)) === ks)
        return back === undefined ? [] : [{ state: p, action: back, cost: cost(p, back, s) }]
      }))
  const side = (root: SearchNode<S, A>): Side<S, A> => {
    const heap = new BinaryHeap<SearchNode<S, A>>((a, b) => a.pathCost - b.pathCost)
    heap.push(root)
    return { heap, best: new Map([[key(root.state), root]]), closed: new Set() }
  }
  const top = (x: Side<S, A>): number => {
    for (let n = x.heap.peek(); n !== undefined; n = x.heap.peek()) {
      const k = key(n.state)
      if (!x.closed.has(k) && x.best.get(k) === n) return n.pathCost
      x.heap.pop()
    }
    return Number.POSITIVE_INFINITY
  }
  const join = (s: BidirectionalState<S, A>): SearchNode<S, A> | null => {
    if (s.meet === null) return null
    let node = s.meet.forward
    for (let b = s.meet.backward; b.parent !== null; b = b.parent) node = childNode(problem, node, b.action as A)
    return node
  }
  return {
    name: 'bidirectional',
    config: {},
    schema: PATH_SCHEMA,
    init: () => {
      const root = rootNode(problem)
      const goal: SearchNode<S, A> = { state: problem.goal, parent: null, action: undefined, pathCost: 0, depth: 0 }
      const s: BidirectionalState<S, A> = {
        ...core<S, A>(options.recordTree === true),
        forward: side(root),
        backward: side(goal),
        mu: Number.POSITIVE_INFINITY,
        meet: null,
      }
      s.tree?.push(root)
      if (key(root.state) === key(goal.state)) {
        s.mu = 0
        s.meet = { forward: root, backward: goal }
      }
      noteFrontier(s, 2)
      return s
    },
    step: (s) => {
      beginStep(s)
      const tf = top(s.forward)
      const tb = top(s.backward)
      if (tf + tb >= s.mu || (tf === Number.POSITIVE_INFINITY && tb === Number.POSITIVE_INFINITY)) {
        const node = join(s)
        if (node === null) s.status = 'failure'
        else solved(s, node)
        return s
      }
      const forward = tf <= tb
      const here = forward ? s.forward : s.backward
      const there = forward ? s.backward : s.forward
      const node = here.heap.pop() as SearchNode<S, A>
      const k = key(node.state)
      here.closed.add(k)
      s.stats.expanded += 1
      s.current = node
      const next: SearchNode<S, A>[] = forward
        ? problem.actions(node.state).map((a) => childNode(problem, node, a))
        : predecessors(node.state).map((p) => ({
            state: p.state,
            parent: node,
            action: p.action,
            pathCost: node.pathCost + (p.cost ?? cost(p.state, p.action, node.state)),
            depth: node.depth + 1,
          }))
      for (const child of next) {
        const ck = key(child.state)
        const old = here.best.get(ck)
        const better = !here.closed.has(ck) && (old === undefined || child.pathCost < old.pathCost)
        generatedChild(s, child, better)
        if (!better) continue
        here.best.set(ck, child)
        here.heap.push(child)
        const other = there.best.get(ck)
        if (other !== undefined && child.pathCost + other.pathCost < s.mu) {
          s.mu = child.pathCost + other.pathCost
          s.meet = forward ? { forward: child, backward: other } : { forward: other, backward: child }
        }
      }
      noteFrontier(s, s.forward.best.size - s.forward.closed.size + s.backward.best.size - s.backward.closed.size)
      s.exploredSize = s.forward.closed.size + s.backward.closed.size
      s.stats.explored = s.exploredSize
      return s
    },
    done: stopper(maxExpansions),
    observe: observer(problem),
  }
}

/**
 * Bidirectional search as a traced algorithm: uniform-cost searches forwards from the initial state and backwards
 * from the goal, each step expanding the side whose frontier minimum is smaller, stopped when the two minima sum to at
 * least the best meeting cost (Pohl's criterion), so the path found is optimal. With unit costs this is bidirectional
 * breadth-first search.
 *
 * @param problem - The problem, with its goal state and (unless the actions are reversible) predecessors.
 * @param options - The expansion budget and tree recording (the tree holds the forward nodes).
 * @returns The algorithm.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { bidirectionalSearchAlgorithm } from 'omai-engine/search'
 * const p = { initial: 0, goal: 4, actions: () => [-1, 1], result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 4 }
 * bidirectionalSearchAlgorithm(p).name // => 'bidirectional'
 * ```
 */
export function bidirectionalSearchAlgorithm<S, A>(
  problem: BidirectionalProblem<S, A>,
  options: SearchOptions = {},
): Algorithm<SearchState<S, A>> {
  return bidirectional(problem, options)
}

/**
 * Bidirectional search: an optimal path found by meeting in the middle, expanding about twice the square root of the
 * nodes of a one-sided search in uniform trees.
 *
 * @param problem - The problem, with its goal state and (unless the actions are reversible) predecessors.
 * @param options - The expansion budget, tree recording and tracing.
 * @returns The result.
 * @concept uninformed-search
 * @example
 * ```ts
 * import { bidirectionalSearch } from 'omai-engine/search'
 * const p = { initial: 0, goal: 4, actions: () => [-1, 1], result: (s: number, a: number) => s + a, isGoal: (s: number) => s === 4 }
 * bidirectionalSearch(p).path // => [0, 1, 2, 3, 4]
 * ```
 */
export function bidirectionalSearch<S, A>(
  problem: BidirectionalProblem<S, A>,
  options: SearchOptions = {},
): SearchResult<S, A> {
  return runSearch(bidirectional(problem, options), options)
}
