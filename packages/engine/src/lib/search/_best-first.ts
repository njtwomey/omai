import type { Json } from '../base/index.ts'
import { BinaryHeap } from '../graph/index.ts'
import type { Algorithm } from '../trace/index.ts'
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
  solved,
  stopper,
} from './_engine.ts'
import {
  childNode,
  rootNode,
  type SearchNode,
  type SearchOptions,
  type SearchProblem,
  type SearchState,
  type StateKey,
  stateKeyOf,
} from './problem.ts'

/**
 * How a best-first frontier orders nodes of equal priority: first in first out (default), last in first out, or a
 * comparison of states (aima-python's priority queue breaks ties by comparing states, so `(a, b) => (a < b ? -1 : a > b
 * ? 1 : 0)` reproduces its expansion order on states that compare as strings or tuples).
 */
export type TieBreak<S> = 'fifo' | 'lifo' | ((a: S, b: S) => number)

/** Options of a best-first search. */
export interface BestFirstOptions<S> extends SearchOptions {
  /** Graph search with an explored set (default true); false searches the tree, re-expanding repeated states. */
  readonly graph?: boolean
  readonly tieBreak?: TieBreak<S>
}

interface Entry<S, A> {
  readonly node: SearchNode<S, A>
  readonly f: number
  readonly seq: number
  alive: boolean
}

interface BestFirstState<S, A> extends Core<S, A> {
  readonly heap: BinaryHeap<Entry<S, A>>
  readonly open: Map<StateKey, Entry<S, A>>
  readonly closed: Set<StateKey>
  live: number
  seq: number
}

/**
 * The best-first engine (aima-python's `best_first_graph_search`): pop the node of lowest `f`, test it for the goal,
 * add its state to the explored set and push each child whose state is neither explored nor on the frontier; a child
 * whose state is on the frontier with a higher `f` replaces it. The heap is graph's `BinaryHeap` with lazy deletion.
 */
export function bestFirstEngine<S, A>(
  name: string,
  problem: SearchProblem<S, A>,
  f: (node: SearchNode<S, A>) => number,
  options: BestFirstOptions<S>,
  config: Json,
): Algorithm<SearchState<S, A> & BestFirstState<S, A>> {
  const key = stateKeyOf(problem)
  const graph = options.graph ?? true
  const tie = options.tieBreak ?? 'fifo'
  const breakTie =
    tie === 'fifo'
      ? (a: Entry<S, A>, b: Entry<S, A>) => a.seq - b.seq
      : tie === 'lifo'
        ? (a: Entry<S, A>, b: Entry<S, A>) => b.seq - a.seq
        : (a: Entry<S, A>, b: Entry<S, A>) => tie(a.node.state, b.node.state) || a.seq - b.seq
  const compare = (a: Entry<S, A>, b: Entry<S, A>): number => (a.f < b.f ? -1 : a.f > b.f ? 1 : breakTie(a, b))
  const maxExpansions = budgetOf(options, name)
  const push = (s: BestFirstState<S, A>, node: SearchNode<S, A>, fn: number): void => {
    const e: Entry<S, A> = { node, f: fn, seq: s.seq++, alive: true }
    s.heap.push(e)
    if (graph) s.open.set(key(node.state), e)
    s.live += 1
  }
  return {
    name,
    config,
    schema: PATH_SCHEMA,
    init: () => {
      const root = rootNode(problem)
      const s: BestFirstState<S, A> = {
        ...core<S, A>(options.recordTree === true),
        heap: new BinaryHeap(compare),
        open: new Map(),
        closed: new Set(),
        live: 0,
        seq: 0,
      }
      s.tree?.push(root)
      push(s, root, f(root))
      noteFrontier(s, s.live)
      return s
    },
    step: (s) => {
      beginStep(s)
      let e = s.heap.pop()
      while (e !== undefined && !e.alive) e = s.heap.pop()
      if (e === undefined) {
        s.status = 'failure'
        return s
      }
      s.live -= 1
      const node = e.node
      const k = key(node.state)
      if (graph) s.open.delete(k)
      s.current = node
      if (goalTest(problem, s, node.state)) {
        solved(s, node)
        noteFrontier(s, s.live)
        return s
      }
      if (graph) s.closed.add(k)
      s.stats.expanded += 1
      for (const a of problem.actions(node.state)) {
        const child = childNode(problem, node, a)
        const ck = key(child.state)
        const fc = f(child)
        if (!graph) {
          push(s, child, fc)
          generatedChild(s, child, true)
          continue
        }
        if (s.closed.has(ck)) {
          generatedChild(s, child, false)
          continue
        }
        const old = s.open.get(ck)
        if (old === undefined) {
          push(s, child, fc)
          generatedChild(s, child, true)
        } else if (fc < old.f) {
          old.alive = false
          s.live -= 1
          push(s, child, fc)
          generatedChild(s, child, true)
        } else {
          generatedChild(s, child, false)
        }
      }
      noteFrontier(s, s.live)
      s.exploredSize = s.closed.size
      s.stats.explored = s.closed.size
      return s
    },
    done: stopper(maxExpansions),
    observe: observer(problem),
  }
}
