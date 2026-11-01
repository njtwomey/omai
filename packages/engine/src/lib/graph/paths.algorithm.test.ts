import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { defaults } from '../space/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { gridGraph, pathGraph } from './generators.ts'
import { digraph, graph } from './graph.ts'
import { bellmanFord, dijkstra, floydWarshall } from './paths.ts'
import { pathHeuristics, shortestPathMethods } from './paths-options.ts'
import {
  aStarAlgorithm,
  bellmanFordAlgorithm,
  dijkstraAlgorithm,
  floydWarshallAlgorithm,
  shortestPathContract,
} from './shortest-paths.ts'

const diamond = graph([
  [0, 1, 4],
  [0, 2, 1],
  [2, 1, 2],
  [1, 3, 5],
])
// Node 4 is not reached from 0, so distances hold Infinity (which JSON cannot carry).
const withIsland = digraph([
  [0, 1, 4],
  [0, 2, 1],
  [2, 1, 2],
  [1, 3, 5],
  [4, 3, 1],
])
const negative = digraph([
  [0, 1, 4],
  [0, 2, 5],
  [2, 1, -3],
  [1, 3, 2],
])

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

describe('dijkstraAlgorithm', () => {
  it('run to the end gives the numbers the function gave before', () => {
    const run = runAlgorithm(dijkstraAlgorithm(diamond, 0), { maxSteps: 10 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(4)
    expect(run.state.expanded).toEqual([0, 2, 1, 3])
    expect(Array.from(run.state.distance)).toEqual([0, 3, 1, 8])
    expect(Array.from(run.state.predecessor)).toEqual([-1, 2, 0, 1])
    expect(Array.from(dijkstra(diamond, 0).distance)).toEqual([0, 3, 1, 8])
  })

  it('stops once the target is settled', () => {
    const run = runAlgorithm(dijkstraAlgorithm(diamond, 0, { target: 2 }), { maxSteps: 10 })
    expect(run.state.expanded).toEqual([0, 2])
    expect(run.status).toBe('converged')
  })

  it('resumes from a JSON checkpoint with infinite distances to the same end', () => {
    const alg = dijkstraAlgorithm(withIsland, 0)
    const whole = runAlgorithm(alg, { maxSteps: 10 }).state
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 8, start: { state: stored, step: 2, serialised: true } })
    expect(rest.state.expanded).toEqual(whole.expanded)
    expect(Array.from(rest.state.distance)).toEqual(Array.from(whole.distance))
    expect(Array.from(rest.state.predecessor)).toEqual(Array.from(whole.predecessor))
    expect(whole.distance[4]).toBe(Number.POSITIVE_INFINITY)
  })

  it('logs the settled node, the frontier in settling order, and the arrays', () => {
    const rec = recorder()
    runAlgorithm(dijkstraAlgorithm(diamond, 0), {
      maxSteps: 10,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('sp/settled')).toEqual([0, 2, 1, 3])
    expect(metric('sp/priority')).toEqual([0, 1, 3, 8])
    expect(metric('sp/relaxed')).toEqual([2, 1, 1, 0])
    const frontier = rec.events.filter((e) => e.type === 'tensor' && e.name === 'sp/frontier')
    expect(frontier).toHaveLength(4)
    const distance = rec.events.filter((e) => e.type === 'tensor' && e.name === 'sp/distance')
    expect(distance.map((e) => (e.type === 'tensor' ? e.shape[0] : 0))).toEqual([4, 4, 4, 4])
    expect(Object.keys(shortestPathContract)).toContain('sp/negative-cycle')
  })

  it('settles nodes in non-decreasing distance and agrees with Bellman-Ford on random graphs', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.nat(7), fc.nat(7), fc.integer({ min: 0, max: 9 })), { minLength: 1, maxLength: 24 }),
        (edges) => {
          const g = digraph(edges.map(([u, v, w]) => [u, v, w] as [number, number, number]))
          const run = runAlgorithm(dijkstraAlgorithm(g, 0), { maxSteps: g.n + 1 })
          const order = run.state.expanded.map((v) => run.state.distance[v] ?? 0)
          const monotone = order.every((d, i) => i === 0 || d >= (order[i - 1] ?? 0))
          const bf = bellmanFord(g, 0).distance
          return monotone && Array.from(bf).every((d, v) => d === run.state.distance[v])
        },
      ),
    )
  })

  it('throws on a negative weight it reaches', () => {
    const bad = digraph([
      [0, 1, 1],
      [1, 2, -1],
    ])
    expect(() => runAlgorithm(dijkstraAlgorithm(bad, 0), { maxSteps: 10 })).toThrow(/negative/)
  })
})

describe('aStarAlgorithm', () => {
  const g = gridGraph(5, 5)
  const manhattan = (v: number) => Math.abs(Math.floor(v / 5) - 1) + Math.abs((v % 5) - 1)

  it('with a heuristic expands fewer nodes and finds the same distance', () => {
    const informed = runAlgorithm(aStarAlgorithm(g, 0, 6, { heuristic: manhattan }), { maxSteps: 40 })
    const blind = runAlgorithm(aStarAlgorithm(g, 0, 6), { maxSteps: 40 })
    expect(informed.state.distance[6]).toBe(2)
    expect(blind.state.distance[6]).toBe(2)
    expect(informed.state.expanded.length).toBeLessThan(blind.state.expanded.length)
    expect(informed.state.expanded.at(-1)).toBe(6)
  })

  it('resumes from a checkpoint', () => {
    const alg = aStarAlgorithm(g, 0, 24, { heuristic: (v) => 8 - Math.floor(v / 5) - (v % 5) })
    const whole = runAlgorithm(alg, { maxSteps: 40 }).state
    const half = runAlgorithm(alg, { maxSteps: 3 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 40, start: { state: stored, step: 3, serialised: true } }).state
    expect(rest.expanded).toEqual(whole.expanded)
    expect(Array.from(rest.distance)).toEqual(Array.from(whole.distance))
  })

  it('rejects a heuristic weight below 1', () => {
    expect(() => aStarAlgorithm(g, 0, 15, { weight: 0.5 })).toThrow(/at least 1/)
  })
})

describe('bellmanFordAlgorithm', () => {
  it('relaxes edges in node order within a round, and is done when a round changes nothing', () => {
    const rec = runAlgorithm(bellmanFordAlgorithm(negative, 0), { maxSteps: 1 })
    expect(Array.from(rec.state.distance)).toEqual([0, 2, 5, 6])
    const end = runAlgorithm(bellmanFordAlgorithm(negative, 0), { maxSteps: 10 })
    expect(end.status).toBe('converged')
    expect(Array.from(end.state.distance)).toEqual([0, 2, 5, 4])
    expect(Array.from(bellmanFord(negative, 0).distance)).toEqual([0, 2, 5, 4])
  })

  it('flags a negative cycle and the function throws', () => {
    const cyc = digraph([
      [0, 1, 1],
      [1, 2, -3],
      [2, 1, 1],
    ])
    const run = runAlgorithm(bellmanFordAlgorithm(cyc, 0), { maxSteps: 10 })
    expect(run.status).toBe('diverged')
    expect(run.state.negativeCycle).toBe(true)
    expect(() => bellmanFord(cyc, 0)).toThrow(/negative cycle/)
  })

  it('resumes from a JSON checkpoint', () => {
    const alg = bellmanFordAlgorithm(withIsland, 0)
    const whole = runAlgorithm(alg, { maxSteps: 10 }).state
    const half = runAlgorithm(alg, { maxSteps: 1 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 10, start: { state: stored, step: 1, serialised: true } }).state
    expect(Array.from(rest.distance)).toEqual(Array.from(whole.distance))
    expect(rest.rounds).toBe(whole.rounds)
  })

  it('logs the round, the edges improved and the arrays', () => {
    const rec = recorder()
    runAlgorithm(bellmanFordAlgorithm(negative, 0), {
      maxSteps: 10,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('sp/round')).toEqual([1, 2, 3])
    expect(metric('sp/relaxed').at(-1)).toBe(0)
  })
})

describe('floydWarshallAlgorithm', () => {
  const g = digraph([
    [0, 1, 3],
    [1, 2, 1],
    [2, 0, 2],
    [0, 3, 10],
    [2, 3, 1],
  ])

  it('after pivot k is exact for paths through the first k nodes only', () => {
    const n = g.n
    const alg = floydWarshallAlgorithm(g)
    for (let k = 0; k <= n; k++) {
      const { distance } = runAlgorithm(alg, { maxSteps: k }).state
      // Brute force: the cheapest walk from i to j whose intermediate nodes are all below k.
      const brute = new Float64Array(n * n).fill(Number.POSITIVE_INFINITY)
      for (let i = 0; i < n; i++) brute[i * n + i] = 0
      for (const e of g.edges) brute[e.source * n + e.target] = Math.min(brute[e.source * n + e.target] ?? 0, e.weight)
      for (let m = 0; m < k; m++)
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++)
            brute[i * n + j] = Math.min(brute[i * n + j] ?? 0, (brute[i * n + m] ?? 0) + (brute[m * n + j] ?? 0))
      expect(Array.from(distance)).toEqual(Array.from(brute))
    }
  })

  it('run to the end is floydWarshall', () => {
    const run = runAlgorithm(floydWarshallAlgorithm(g), { maxSteps: 10 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(4)
    expect(Array.from(floydWarshall(g).distance.data)).toEqual(Array.from(run.state.distance))
    expect(run.state.distance[3]).toBe(5) // 0 -> 1 -> 2 -> 3
  })

  it('resumes from a JSON checkpoint, infinities included', () => {
    const alg = floydWarshallAlgorithm(withIsland)
    const whole = runAlgorithm(alg, { maxSteps: 10 }).state
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 10, start: { state: stored, step: 2, serialised: true } }).state
    expect(Array.from(rest.distance)).toEqual(Array.from(whole.distance))
    expect(Array.from(rest.predecessor)).toEqual(Array.from(whole.predecessor))
  })

  it('stops with status diverged on a negative cycle', () => {
    const cyc = digraph([
      [0, 1, 1],
      [1, 0, -3],
    ])
    expect(runAlgorithm(floydWarshallAlgorithm(cyc), { maxSteps: 5 }).status).toBe('diverged')
    expect(() => floydWarshall(cyc)).toThrow(/negative cycle/)
  })

  it('logs the pivot and the improved pairs', () => {
    const rec = recorder()
    runAlgorithm(floydWarshallAlgorithm(pathGraph(4)), {
      maxSteps: 10,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('sp/pivot')).toEqual([0, 1, 2, 3])
    expect(metric('sp/improved')).toHaveLength(4)
  })
})

describe('the option tables', () => {
  const g = gridGraph(4, 4)

  it('every heuristic is admissible: it never exceeds the distance left', () => {
    for (const kind of pathHeuristics.names) {
      const h = pathHeuristics.make({ kind })(g, 15, true)
      const left = bellmanFord(g, 15).distance // the grid is undirected, so distances are symmetric
      for (let v = 0; v < g.n; v++) expect(h(v)).toBeLessThanOrEqual((left[v] ?? 0) + 1e-12)
    }
  })

  it('the landmark heuristic is admissible on a directed graph with weights', () => {
    const d = digraph([
      [0, 1, 2],
      [1, 2, 3],
      [2, 3, 1],
      [0, 3, 9],
      [3, 0, 1],
      [1, 3, 5],
    ])
    const h = pathHeuristics.make({ kind: 'landmarks', count: 3 })(d, 3, true)
    for (let v = 0; v < d.n; v++) {
      const truth = dijkstra(d, v).distance[3] ?? 0
      expect(h(v)).toBeLessThanOrEqual(truth + 1e-12)
    }
  })

  it('every method option builds an algorithm that reaches the same distances', () => {
    for (const kind of shortestPathMethods.names) {
      const alg = shortestPathMethods.make({ ...(defaults(shortestPathMethods.space) as object), kind })(g, 0, {
        target: 15,
      })
      const run = runAlgorithm(alg, { maxSteps: 100 })
      expect(run.state.distance[15]).toBe(6)
    }
  })

  it('each heuristic runs inside a-star and finds the shortest path', () => {
    for (const heuristic of pathHeuristics.names) {
      const alg = shortestPathMethods.make({ kind: 'a-star', heuristic, landmarks: 2, weight: 1 })(g, 0, { target: 15 })
      expect(runAlgorithm(alg, { maxSteps: 100 }).state.distance[15]).toBe(6)
    }
  })

  it('a-star without a target is refused', () => {
    expect(() => shortestPathMethods.make({ kind: 'a-star' })(g, 0)).toThrow(/needs a target/)
  })
})
