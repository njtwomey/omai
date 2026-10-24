import { type Json, WhiteboxError } from '../base/index.ts'
import { type Algorithm, channels, type Observation, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import {
  effectiveBranchingFactor,
  type GeneratedNode,
  type SearchNode,
  type SearchOptions,
  type SearchProblem,
  type SearchResult,
  type SearchState,
  type SearchStatus,
  solutionActions,
  solutionPath,
  stateKeyOf,
} from './problem.ts'

/** Mutable effort counters shared by the path searches. */
export interface Counters {
  expanded: number
  generated: number
  goalTests: number
  maxFrontier: number
  explored: number
}

/** Fresh counters. */
export function counters(): Counters {
  return { expanded: 0, generated: 0, goalTests: 0, maxFrontier: 0, explored: 0 }
}

/** The mutable core every path-search state carries (the public fields of {@link SearchState}). */
export interface Core<S, A> {
  status: 'running' | SearchStatus
  solution: SearchNode<S, A> | null
  stats: Counters
  frontierSize: number
  exploredSize: number
  current: SearchNode<S, A> | null
  generated: GeneratedNode<S, A>[]
  bound?: number
  /** Every generated node, when recording the tree. */
  tree: SearchNode<S, A>[] | undefined
}

/** A fresh core. */
export function core<S, A>(record: boolean): Core<S, A> {
  return {
    status: 'running',
    solution: null,
    stats: counters(),
    frontierSize: 0,
    exploredSize: 0,
    current: null,
    generated: [],
    tree: record ? [] : undefined,
  }
}

/** The default expansion budget. */
export const DEFAULT_EXPANSIONS = 100_000

/** The trace channels of a path search. */
export const PATH_SCHEMA: TraceSchema = channels({
  metrics: {
    expanded: { description: 'nodes expanded so far' },
    generated: { description: 'nodes generated so far' },
    frontier: { description: 'nodes on the frontier' },
    explored: { description: 'states in the explored set' },
  },
  messages: {
    expand: {
      description: 'the node expanded in this step',
      payload: { node: 'state label', parent: 'parent label', g: 'path cost', h: 'heuristic', depth: 'depth' },
    },
    generate: { description: 'the children generated', payload: { nodes: 'children with labels, costs and added' } },
    bound: { description: 'a new depth or cost bound of an iterative deepening search', payload: { bound: 'bound' } },
    solution: { description: 'the solution found', payload: { path: 'state labels', cost: 'path cost' } },
  },
})

const finite = (x: number): number | null => (Number.isFinite(x) ? x : null)

/** The observer of a path search: metrics plus expand, generate, bound and solution messages. */
export function observer<S, A>(problem: SearchProblem<S, A>): (state: SearchState<S, A>) => Observation {
  const key = stateKeyOf(problem)
  const label = (s: S): string => problem.label?.(s) ?? String(key(s))
  const h = (s: S): number => problem.heuristic?.(s) ?? 0
  return (s) => {
    const messages: { kind: string; payload: Json }[] = []
    if (s.bound !== undefined && s.current === null) messages.push({ kind: 'bound', payload: { bound: s.bound } })
    const c = s.current
    if (c !== null) {
      messages.push({
        kind: 'expand',
        payload: {
          node: label(c.state),
          parent: c.parent === null ? null : label(c.parent.state),
          g: c.pathCost,
          h: finite(h(c.state)),
          depth: c.depth,
        },
      })
    }
    if (s.generated.length > 0) {
      messages.push({
        kind: 'generate',
        payload: {
          nodes: s.generated.map((x) => ({
            node: label(x.node.state),
            parent: x.node.parent === null ? null : label(x.node.parent.state),
            g: x.node.pathCost,
            h: finite(h(x.node.state)),
            added: x.added,
          })),
        },
      })
    }
    if (s.solution !== null && s.status === 'solved') {
      messages.push({
        kind: 'solution',
        payload: { path: solutionPath(s.solution).map(label), cost: s.solution.pathCost },
      })
    }
    return {
      metrics: {
        expanded: s.stats.expanded,
        generated: s.stats.generated,
        frontier: s.frontierSize,
        explored: s.exploredSize,
      },
      messages,
    }
  }
}

/** Stop rule shared by the path searches: finished, or out of expansions. */
export function stopper<S, A>(maxExpansions: number): (s: SearchState<S, A>) => 'converged' | 'stopped' | undefined {
  return (s) => {
    if (s.status === 'solved') return 'converged'
    if (s.status !== 'running') return 'stopped'
    return s.stats.expanded >= maxExpansions ? 'stopped' : undefined
  }
}

/** The result of a finished (or budget-stopped) path search. */
export function toResult<S, A>(
  s: SearchState<S, A>,
  tree: readonly SearchNode<S, A>[] | undefined,
): SearchResult<S, A> {
  const status: SearchStatus = s.status === 'running' ? 'budget' : s.status
  const node = status === 'solved' ? s.solution : null
  const stats = { ...s.stats }
  const depth = node === null ? -1 : node.depth
  return {
    status,
    node,
    path: node === null ? [] : solutionPath(node),
    actions: node === null ? [] : solutionActions(node),
    cost: node === null ? Number.POSITIVE_INFINITY : node.pathCost,
    depth,
    stats,
    effectiveBranchingFactor: node === null ? Number.NaN : effectiveBranchingFactor(stats.generated, depth),
    ...(tree === undefined ? {} : { tree }),
  }
}

/** Run a path-search algorithm to its end under the expansion budget and return its result. */
export function runSearch<S, A>(
  algorithm: Algorithm<SearchState<S, A> & { readonly tree: readonly SearchNode<S, A>[] | undefined }>,
  options: SearchOptions,
  stepsPerExpansion = 1,
): SearchResult<S, A> {
  const { maxExpansions = DEFAULT_EXPANSIONS, recordTree: _record, ...run } = options
  const { state } = runAlgorithm(algorithm, { ...run, maxSteps: stepsPerExpansion * maxExpansions + 16 })
  return toResult(state, state.tree)
}

/** Note a generated child: count it and keep it in the tree when recording. */
export function generatedChild<S, A>(c: Core<S, A>, node: SearchNode<S, A>, added: boolean): void {
  c.stats.generated += 1
  c.generated.push({ node, added })
  c.tree?.push(node)
}

/** Start a step: clear the per-step record. */
export function beginStep<S, A>(c: Core<S, A>): void {
  c.current = null
  c.generated = []
}

/** Track the largest frontier. */
export function noteFrontier<S, A>(c: Core<S, A>, size: number): void {
  c.frontierSize = size
  if (size > c.stats.maxFrontier) c.stats.maxFrontier = size
}

/** A goal test, counted. */
export function goalTest<S, A>(problem: SearchProblem<S, A>, c: Core<S, A>, state: S): boolean {
  c.stats.goalTests += 1
  return problem.isGoal(state)
}

/** Finish with a solution. */
export function solved<S, A>(c: Core<S, A>, node: SearchNode<S, A>): void {
  c.status = 'solved'
  c.solution = node
}

/** Validate an expansion budget. */
export function budgetOf(options: SearchOptions, name: string): number {
  const m = options.maxExpansions ?? DEFAULT_EXPANSIONS
  if (!(m >= 0)) throw new WhiteboxError('value', `${name}: maxExpansions must be non-negative`)
  return m
}
