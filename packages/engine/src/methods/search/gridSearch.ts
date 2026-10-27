/**
 * Path search on a grid world as a method of nothing but its configuration: breadth-first, depth-first, uniform-cost,
 * greedy, A* and weighted A* on one of the library's hand-drawn maps, one expansion per step. Every step puts a board of
 * cell categories on the tape (walls, explored cells, the frontier, the node expanded, the path to it) beside the
 * frontier size, the cells explored and the path cost, so a page can play the search as it floods the map; the
 * `board` projection reads those boards back as a `boardSteps` record for the board view.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { boardSteps } from '../../lib/exposition/index.ts'
import { gridProblem, SEARCH_MAPS, type SearchMapName, searchMap } from '../../lib/problems/index.ts'
import {
  aStarSearchAlgorithm,
  bestFirstSearchAlgorithm,
  breadthFirstSearchAlgorithm,
  depthFirstSearchAlgorithm,
  type SearchNode,
  type SearchState,
  uniformCostSearchAlgorithm,
} from '../../lib/search/index.ts'
import { choice, dict, real } from '../../lib/space/index.ts'
import {
  type Algorithm,
  type Config,
  defineMethod,
  Log,
  type Method,
  metricSeries,
  type Run,
  scalar,
  tensor,
  tensorSeries,
} from '../../lib/trace/index.ts'
import { terrainOf, type World } from '../../lib/worlds/index.ts'

/** The path searches {@link gridSearch} can run. */
export const GRID_SEARCHES = [
  'breadth-first',
  'depth-first',
  'uniform-cost',
  'greedy',
  'a-star',
  'weighted-a-star',
] as const

/** A path search of {@link gridSearch}. */
export type GridSearchName = (typeof GRID_SEARCHES)[number]

/**
 * What a cell of the board is: free, a wall, explored (expanded), on the frontier (generated, waiting), the node being
 * expanded now, on the path to that node (the solution once found), the start, the goal, or costly terrain (water,
 * forest) no search has reached yet.
 */
export const BOARD_CELLS = {
  free: 0,
  wall: 1,
  explored: 2,
  frontier: 3,
  current: 4,
  path: 5,
  start: 6,
  goal: 7,
  costly: 8,
} as const

/** The state of a grid search: the search's own state beside the cells it has expanded and generated so far. */
// The search's frontier and explored sets live in its closures and maps, so this state has no JSON form and no
// `serialise`: the board and scalars on the tape are what a page shows, and nothing here is probed or sought.
interface GridState {
  readonly search: SearchState<number, number>
  readonly explored: Set<number>
  readonly reached: Set<number>
}

type Cell = number

function searchOf(
  name: GridSearchName,
  problem: ReturnType<typeof gridProblem>,
  weight: number,
): Algorithm<SearchState<Cell, Cell>> {
  const options = { maxExpansions: 5000 }
  const h = (n: SearchNode<Cell, Cell>): number => problem.heuristic?.(n.state) ?? 0
  switch (name) {
    case 'breadth-first':
      return breadthFirstSearchAlgorithm(problem, options)
    case 'depth-first':
      return depthFirstSearchAlgorithm(problem, options)
    case 'uniform-cost':
      return uniformCostSearchAlgorithm(problem, options)
    case 'greedy':
      return bestFirstSearchAlgorithm(problem, h, options)
    case 'a-star':
      return aStarSearchAlgorithm(problem, options)
    case 'weighted-a-star':
      return bestFirstSearchAlgorithm(problem, (n) => n.pathCost + weight * h(n), options)
  }
}

// The terrain of each cell as a board category: walls, costly terrain, free cells.
function terrainCells(w: World): number[] {
  return Array.from({ length: w.n }, (_, c) => {
    const terrain = terrainOf(w, c)
    return !terrain.passable ? BOARD_CELLS.wall : terrain.cost === 1 ? BOARD_CELLS.free : BOARD_CELLS.costly
  })
}

function shapeOf(name: SearchMapName): readonly [number, number] {
  const t = searchMap(name).topology
  if (t.kind !== 'square') throw new MdmlError('value', `gridSearch: the map ${name} is not a square grid`)
  return [t.rows, t.cols]
}

/**
 * Path search on a grid world, expansion by expansion. Each step the board has the walls, the cells explored, the cells
 * generated and waiting on the frontier, the node expanded now and the path to it (the solution once found); the
 * scalars `frontier`, `explored` and `pathCost` follow the effort. Breadth-first and depth-first search ignore terrain
 * costs; uniform-cost search, A* and weighted A* find cheapest paths (weighted A* within the weight of the optimum);
 * greedy best-first trades optimality for speed. Water costs 5 to enter and forest 2, as the maps say.
 *
 * Knobs: `algorithm`, `map` (`open`, `detour`, `river`, `rooms`) and `weight` (the weight on the heuristic of weighted
 * A*, ignored by the others). The run ends when the goal is reached or the frontier is empty.
 *
 * @example
 * ```ts
 * import { gridSearch } from 'mdml-engine/methods/search'
 * import { session } from 'mdml-engine/trace'
 * const s = session(gridSearch, { mode: 'eager', steps: 500 })
 * s.status // => 'converged'
 * ```
 */
export const gridSearch: Method<GridState, Config> = defineMethod<GridState, Config>({
  name: 'search/grid-search',
  config: dict({
    algorithm: choice(GRID_SEARCHES, { default: 'a-star', label: 'search' }),
    map: choice(Object.keys(SEARCH_MAPS) as SearchMapName[], { default: 'detour', label: 'map' }),
    weight: real(1, 10, { default: 2, label: 'weight (weighted A*)' }),
  }),
  make: (c) => {
    const name = c['algorithm'] as GridSearchName
    const world = searchMap(c['map'] as SearchMapName)
    const problem = gridProblem(world)
    const inner = searchOf(name, problem, c['weight'] as number)
    const terrain = terrainCells(world)
    const [rows, cols] = shapeOf(c['map'] as SearchMapName)
    const start = problem.initial
    return {
      name: `search/grid-search/${name}`,
      config: { algorithm: name, map: c['map'] as string, weight: c['weight'] as number },
      init: (key) => ({ search: inner.init(key), explored: new Set<number>(), reached: new Set<number>([start]) }),
      step: (s, ctx) => {
        const search = inner.step(s.search, ctx)
        if (search.current !== null) s.explored.add(search.current.state)
        for (const g of search.generated) if (g.added) s.reached.add(g.node.state)
        const node = search.status === 'solved' ? search.solution : search.current
        if (ctx.wants('board')) {
          const cells = [...terrain]
          for (const cell of s.explored) cells[cell] = BOARD_CELLS.explored
          for (const cell of s.reached) if (!s.explored.has(cell)) cells[cell] = BOARD_CELLS.frontier
          for (let n = node; n !== null; n = n.parent) cells[n.state] = BOARD_CELLS.path
          if (search.current !== null && search.status !== 'solved') cells[search.current.state] = BOARD_CELLS.current
          cells[start] = BOARD_CELLS.start
          for (const goal of world.goals) cells[goal] = BOARD_CELLS.goal
          ctx.log(
            'board',
            Array.from({ length: rows }, (_, r) => cells.slice(r * cols, (r + 1) * cols)),
          )
        }
        ctx.log('frontier', search.frontierSize)
        ctx.log('explored', s.explored.size)
        ctx.log('pathCost', node?.pathCost ?? 0)
        return { search, explored: s.explored, reached: s.reached }
      },
      done: (s, t) => inner.done?.(s.search, t),
    }
  },
  // The boards on the tape as one record, in the order they were logged, with the effort metrics beside them.
  projections: {
    board: (run: Run) => {
      const boards = tensorSeries(run, 'board').filter((e) => e.values !== undefined && e.indices === undefined)
      const first = boards[0]
      if (first === undefined) return undefined
      const [rows, columns] = [first.shape[0] ?? 0, first.shape[1] ?? 0]
      const metrics = ['frontier', 'explored', 'pathCost'].flatMap((name) => {
        const values = metricSeries(run, name).values
        return values.length === boards.length ? [{ name, values }] : []
      })
      return boardSteps({
        method: 'grid search',
        layout: { type: 'grid', rows, columns },
        categories: ['empty', 'wall', 'explored', 'frontier', 'current', 'path', 'start', 'goal', 'terrain'],
        steps: boards.map((e) => ({ cells: [...(e.values ?? [])] })),
        metrics,
        stepName: 'expansion',
      })
    },
  },
  contract: (c) => ({
    board: tensor({
      role: 'state',
      shape: [...shapeOf(c['map'] as SearchMapName)],
      description:
        'cell categories: 0 free, 1 wall, 2 explored, 3 frontier, 4 expanded now, 5 path to it or the solution, 6 start, 7 goal, 8 costly terrain not yet reached',
    }),
    frontier: scalar({ description: 'nodes waiting on the frontier' }),
    explored: scalar({ description: 'cells expanded so far' }),
    pathCost: scalar({ description: 'the cost of the path to the node expanded now (the solution once found)' }),
  }),
  // The board is drawn cell by cell, so it is kept in full (`all` keeps summaries).
  defaults: { board: Log({ detail: 'full' }), frontier: Log(), explored: Log(), pathCost: Log() },
})
