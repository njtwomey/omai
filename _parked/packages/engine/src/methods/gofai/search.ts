/**
 * Path search on grid worlds, played expansion by expansion from the core's traced searches (`mdml-engine/search`):
 * breadth-first, depth-first, uniform-cost, greedy, A* and its weighted, iterative-deepening and bidirectional forms,
 * recording the frontier, the explored set, the node expanded and the path to it. `gridProblem` turns a grid world into
 * a search problem with an admissible heuristic.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { type BoardSteps, boardSteps, type Explained } from '../../lib/exposition/index.ts'
import {
  aStarSearchAlgorithm,
  type BidirectionalProblem,
  bestFirstSearchAlgorithm,
  bidirectionalSearchAlgorithm,
  breadthFirstSearchAlgorithm,
  depthFirstSearchAlgorithm,
  idaStarSearchAlgorithm,
  type SearchNode,
  type SearchState,
  type StateKey,
  stateKeyOf,
  uniformCostSearchAlgorithm,
} from '../../lib/search/index.ts'
import { type Algorithm, iterate } from '../../lib/trace/index.ts'
import { type CellMetric, cellCoords, exits, heuristicTo, terrainOf, type World } from '../../lib/worlds/index.ts'

/** The path searches a grid can be searched with. */
export type PathSearchName =
  | 'breadth-first'
  | 'depth-first'
  | 'uniform-cost'
  | 'greedy'
  | 'a-star'
  | 'weighted-a-star'
  | 'ida-star'
  | 'bidirectional'

/** Options of {@link gridProblem}. */
export interface GridProblemOptions {
  /** 4 (default) or 8 neighbours. */
  readonly connectivity?: 4 | 8
  /** The heuristic's cell distance (default manhattan, octile under 8-connectivity). */
  readonly metric?: CellMetric
}

/**
 * A grid world as a search problem over cells: moves to open neighbours, costing the terrain of the cell entered, with
 * the distance to the nearest goal as heuristic (admissible: no terrain costs less than 1) and the reverse moves for
 * bidirectional search.
 *
 * @param w - The world (its first start and its goals).
 * @param options - Connectivity and the heuristic's metric.
 * @returns The problem.
 * @throws MdmlError `value` when the world has no start or no goal.
 */
export function gridProblem(w: World, options: GridProblemOptions = {}): BidirectionalProblem<number, number> {
  const start = w.starts[0]
  const goal = w.goals[0]
  if (start === undefined || goal === undefined) throw new MdmlError('value', 'gridProblem: needs a start and a goal')
  const connectivity = options.connectivity ?? 4
  const move = { connectivity }
  const metric = options.metric ?? (connectivity === 8 ? 'octile' : 'manhattan')
  const h = heuristicTo(w, { metric })
  const goals = new Set(w.goals)
  const cost = (cell: number): number => terrainOf(w, cell).cost
  return {
    initial: start,
    goal,
    actions: (cell) => exits(w, cell, move).map(([d]) => d),
    result: (cell, d) => exits(w, cell, move).find(([e]) => e === d)?.[1] ?? cell,
    isGoal: (cell) => goals.has(cell),
    stepCost: (_cell, _d, next) => cost(next),
    heuristic: h,
    key: (cell) => cell,
    label: (cell) => (w.topology.kind === 'graph' ? String(cell) : `(${cellCoords(w, cell).join(', ')})`),
    predecessors: (cell) =>
      exits(w, cell, move).flatMap(([, from]) =>
        exits(w, from, move)
          .filter(([, to]) => to === cell)
          .map(([d]) => ({ state: from, action: d, cost: cost(cell) })),
      ),
  }
}

/** Options of {@link gridSearch}. */
export interface GridSearchOptions extends GridProblemOptions {
  /** The search (default `a-star`). */
  readonly algorithm?: PathSearchName
  /** The weight on the heuristic of weighted A* (default 2). */
  readonly weight?: number
  /** Stop after this many expansions (default 5000). */
  readonly maxExpansions?: number
  /** Text in each reached cell: its path cost `g`, heuristic `h`, `f = g + h`, or none (default `g` on small grids). */
  readonly labels?: 'none' | 'g' | 'h' | 'f'
  /** Record at most this many frames, evenly spaced, the last always kept (default 400). */
  readonly maxFrames?: number
}

/** What a grid search found. */
export interface GridSearchResult {
  readonly algorithm: PathSearchName
  readonly status: 'solved' | 'failure' | 'cutoff' | 'budget'
  /** The cells of the path from start to goal (empty when none was found). */
  readonly path: readonly number[]
  readonly cost: number
  readonly expanded: number
  readonly generated: number
  readonly maxFrontier: number
}

function algorithmOf(
  name: PathSearchName,
  problem: BidirectionalProblem<number, number>,
  options: { readonly maxExpansions: number; readonly weight: number },
): Algorithm<SearchState<number, number>> {
  const o = { maxExpansions: options.maxExpansions }
  const h = (n: SearchNode<number, number>): number => problem.heuristic?.(n.state) ?? 0
  switch (name) {
    case 'breadth-first':
      return breadthFirstSearchAlgorithm(problem, o)
    case 'depth-first':
      return depthFirstSearchAlgorithm(problem, o)
    case 'uniform-cost':
      return uniformCostSearchAlgorithm(problem, o)
    case 'greedy':
      return bestFirstSearchAlgorithm(problem, h, o)
    case 'a-star':
      return aStarSearchAlgorithm(problem, o)
    case 'weighted-a-star':
      return bestFirstSearchAlgorithm(problem, (n) => n.pathCost + options.weight * h(n), o)
    case 'ida-star':
      return idaStarSearchAlgorithm(problem, o)
    case 'bidirectional':
      return bidirectionalSearchAlgorithm(problem, o)
  }
}

const CATEGORIES = ['empty', 'wall', 'explored', 'frontier', 'path', 'current', 'start', 'goal'] as const
const C = Object.fromEntries(CATEGORIES.map((c, i) => [c, i])) as Record<(typeof CATEGORIES)[number], number>

function pathOf(node: SearchNode<number, number> | null): number[] {
  const out: number[] = []
  for (let n = node; n !== null; n = n.parent) out.push(n.state)
  return out.reverse()
}

/**
 * Search a grid world for a path from its start to a goal, recording every expansion for playback: the explored set,
 * the frontier, the node expanded and the path to it, the path costs reached, and the effort (expanded, frontier and
 * path cost per step). Breadth-first and depth-first search ignore terrain costs; uniform-cost search, A* and IDA*
 * find cheapest paths; greedy best-first and weighted A* trade optimality for speed.
 *
 * @param w - The world (square grid; its start and goals).
 * @param options - The search, connectivity, heuristic metric and weight, budget, labels and frames.
 * @returns The path, its cost and the effort, with a `boardSteps` record of the search.
 */
export function gridSearch(w: World, options: GridSearchOptions = {}): Explained<GridSearchResult> {
  const t = w.topology
  if (t.kind !== 'square') throw new MdmlError('value', 'gridSearch: square grids only')
  const name = options.algorithm ?? 'a-star'
  const problem = gridProblem(w, options)
  const maxExpansions = options.maxExpansions ?? 5000
  const algorithm = algorithmOf(name, problem, { maxExpansions, weight: options.weight ?? 2 })
  const key = stateKeyOf(problem)
  const h = (cell: number): number => problem.heuristic?.(cell) ?? 0
  const labelMode = options.labels ?? (w.n <= 160 ? 'g' : 'none')
  // The terrain of each cell: walls, open cells, and costly ones, which keep their own category (and their cost as text
  // when no other text is shown) until the frontier or the path reaches them.
  const categories: string[] = [...CATEGORIES]
  const costly = new Map<string, number>()
  const base = Array.from({ length: w.n }, (_, c) => {
    const tc = terrainOf(w, c)
    if (!tc.passable) return C.wall
    if (tc.cost === 1) return C.empty
    if (!costly.has(tc.name)) {
      costly.set(tc.name, categories.length)
      categories.push(tc.name)
    }
    return costly.get(tc.name) as number
  })
  const costText = Array.from({ length: w.n }, (_, c) => {
    const tc = terrainOf(w, c)
    return tc.passable && tc.cost !== 1 ? String(tc.cost) : ''
  })
  const start = problem.initial
  const goals = new Set(w.goals)
  const explored = new Set<StateKey>()
  const added = new Map<StateKey, number>()
  const best = new Map<number, number>()
  type Frame = { cells: number[]; labels: string[]; caption: string; expanded: number; frontier: number; g: number }
  const frames: Frame[] = []
  let bound: number | undefined
  let final: SearchState<number, number> | undefined
  const frame = (s: SearchState<number, number>, at: number): Frame => {
    const cells = [...base]
    for (const c of explored) if (cells[c as number] === C.empty) cells[c as number] = C.explored
    for (const c of added.keys()) if (!explored.has(c)) cells[c as number] = C.frontier
    const node = s.status === 'solved' && s.solution !== null ? s.solution : s.current
    for (const c of pathOf(node)) cells[c] = C.path
    if (s.current !== null && s.status !== 'solved') cells[s.current.state] = C.current
    cells[start] = C.start
    for (const g of goals) cells[g] = C.goal
    const labels = costText.map((text, c) => {
      if (c === start) return 'S'
      if (goals.has(c)) return 'G'
      const g = best.get(c)
      if (labelMode === 'none' || g === undefined) return text
      const v = labelMode === 'g' ? g : labelMode === 'h' ? h(c) : g + h(c)
      return Number.isInteger(v) ? String(v) : v.toFixed(1)
    })
    const c = s.current
    const caption =
      s.status === 'solved' && s.solution !== null
        ? `goal reached: cost ${s.solution.pathCost}, ${s.stats.expanded} expanded`
        : c === null
          ? bound === undefined
            ? `step ${at}`
            : `new bound ${bound}`
          : `expand ${problem.label?.(c.state) ?? c.state}: g = ${c.pathCost}, h = ${h(c.state)}`
    return {
      cells,
      labels,
      caption,
      expanded: s.stats.expanded,
      frontier: s.frontierSize,
      g: c?.pathCost ?? s.solution?.pathCost ?? 0,
    }
  }
  for (const { t: at, state: s } of iterate(algorithm, { maxSteps: 4 * maxExpansions + 16 })) {
    if (s.bound !== undefined && s.bound !== bound) {
      // A new iteration of a deepening search starts from scratch.
      bound = s.bound
      explored.clear()
      added.clear()
      best.clear()
    }
    if (at === 0) {
      added.set(key(start), 0)
      best.set(start, 0)
    }
    if (s.current !== null) {
      explored.add(key(s.current.state))
      best.set(s.current.state, s.current.pathCost)
    }
    for (const g of s.generated)
      if (g.added) {
        added.set(key(g.node.state), g.node.pathCost)
        const prev = best.get(g.node.state)
        if (prev === undefined || g.node.pathCost < prev) best.set(g.node.state, g.node.pathCost)
      }
    frames.push(frame(s, at))
    final = s
  }
  if (final === undefined) throw new MdmlError('value', 'gridSearch: the search did not start')
  const status = final.status === 'running' ? 'budget' : final.status
  const solution = status === 'solved' ? final.solution : null
  const path = pathOf(solution)
  const cap = options.maxFrames ?? 400
  const kept =
    frames.length <= cap
      ? frames
      : Array.from({ length: cap }, (_, i) => frames[Math.round((i * (frames.length - 1)) / (cap - 1))] as Frame)
  const record: BoardSteps = boardSteps({
    method: name,
    layout: { type: 'grid', rows: t.rows, columns: t.cols },
    categories,
    steps: kept.map((f) => ({ cells: f.cells, labels: f.labels, caption: f.caption })),
    metrics: [
      { name: 'expanded', values: kept.map((f) => f.expanded) },
      { name: 'frontier', values: kept.map((f) => f.frontier) },
    ],
    stepName: 'step',
    summary:
      solution === null
        ? `no path (${status}) after ${final.stats.expanded} expansions`
        : `path cost ${solution.pathCost} after ${final.stats.expanded} expansions`,
  })
  return {
    algorithm: name,
    status,
    path,
    cost: solution?.pathCost ?? Number.POSITIVE_INFINITY,
    expanded: final.stats.expanded,
    generated: final.stats.generated,
    maxFrontier: final.stats.maxFrontier,
    exposition: [record],
  }
}
