// One world, every application: the same maze solved by breadth-first search and A* (lib/search), the same textbook
// world as a tabular environment solved by value iteration (lib/rl), and the same map's cell graph handed to graph
// algorithms (lib/graph). Tests may use higher layers; these are the adapters the applications need.
import { describe, expect, it } from 'vitest'
import { connectedComponents, dijkstra, graph } from '../graph/index.ts'
import { key } from '../random/index.ts'
import { mdpFromEnv, tabularEnv, valueIteration } from '../rl/index.ts'
import { aStarSearch, breadthFirstSearch, type SearchProblem } from '../search/index.ts'
import {
  type AgentState,
  deterministicModel,
  distanceField,
  frozenLakeWorld,
  generateMaze,
  heuristicTo,
  regions,
  rules,
  terrainCost,
  toAdjacency,
  toScene,
  transitionModel,
  type World,
} from './index.ts'

const problem = (w: World, withCosts = false): SearchProblem<AgentState, number> => {
  const m = deterministicModel(w, withCosts ? rules(terrainCost()) : [])
  const h = heuristicTo(w)
  return { ...m, heuristic: (s: AgentState) => h(s.cell) }
}

describe('worlds feed search, reinforcement learning and graph algorithms', () => {
  it('breadth-first search and A* solve a generated maze along its unique path', () => {
    for (let seed = 0; seed < 5; seed++) {
      const w = generateMaze(key(seed), { rows: 6, cols: 7, algorithm: 'wilson' })
      const goal = w.goals[0] ?? 0
      const hops = distanceField(w, w.starts[0] ?? 0)[goal] ?? 0
      const bfs = breadthFirstSearch(problem(w))
      const astar = aStarSearch(problem(w))
      expect([bfs.status, bfs.depth, astar.cost]).toEqual(['solved', hops, hops])
      expect(astar.path.map((s) => s.cell)).toEqual(bfs.path.map((s) => s.cell))
    }
  })

  it('A* with terrain costs matches Dijkstra on the adjacency the graph domain builds', () => {
    const w = generateMaze(key(3), { rows: 5, cols: 5, algorithm: 'prim' })
    const adj = toAdjacency(w, { weighted: true })
    const g = graph(adj.edges, { nodes: adj.labels, directed: adj.directed })
    expect(connectedComponents(g).count).toBe(regions(w).count)
    const start = adj.cells.indexOf(w.starts[0] ?? 0)
    const goal = adj.cells.indexOf(w.goals[0] ?? 0)
    expect(aStarSearch(problem(w, true)).cost).toBeCloseTo(dijkstra(g, start).distance[goal] ?? 0, 12)
  })

  it('a textbook world becomes a tabular environment that value iteration solves', () => {
    const t = frozenLakeWorld()
    const m = transitionModel(t.world, t.rules, t.options)
    const env = tabularEnv({
      id: 'FrozenLakeWorld-v0',
      description: 'FrozenLake built on worlds',
      nStates: m.nStates,
      nActions: m.nActions,
      transitions: m.outcomes,
      initial: Array.from(m.initial),
      render: (s) => toScene(t.world, { agent: m.state(s).cell }),
    })
    const solution = valueIteration(mdpFromEnv(env, 0.99))
    // The optimal success probability from the start of the slippery 4 x 4 lake, discounted by 0.99.
    expect(solution.V[0]).toBeCloseTo(0.542, 3)
    expect(env.render(0).width).toBe(128)
  })
})
