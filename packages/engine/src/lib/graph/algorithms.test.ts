import { describe, expect, it } from 'vitest'
import { key as makeKey } from '../random/index.ts'
import { defaults } from '../space/index.ts'
import { type Algorithm, policies, recorder, runAlgorithm } from '../trace/index.ts'
import { greedyColouring, isProperColouring, maximalCliques, maximalIndependentSet } from './colouring.ts'
import {
  bronKerboschAlgorithm,
  type ColouringStrategy,
  cliqueContract,
  colouringContract,
  colouringStrategies,
  greedyColouringAlgorithm,
  independentSetRules,
  maximalIndependentSetAlgorithm,
  pivotRules,
} from './colouring-algorithms.ts'
import {
  biconnectivityAlgorithm,
  componentMethods,
  kosarajuAlgorithm,
  tarjanAlgorithm,
} from './component-algorithms.ts'
import { articulationPoints, biconnectedComponents, bridges, stronglyConnectedComponents } from './components.ts'
import { flowContract, maxFlowAlgorithm, maxFlowMethods, minCostFlowAlgorithm } from './flow-algorithms.ts'
import { bipartiteMatching, maxFlow, minCostFlow, minCut } from './flows.ts'
import { barbellGraph, completeGraph, cycleGraph, gridGraph, petersenGraph, starGraph } from './generators.ts'
import { digraph, graph } from './graph.ts'
import { hopcroftKarpAlgorithm, matchingContract } from './hopcroft-karp.ts'
import {
  boruvkaAlgorithm,
  kruskalAlgorithm,
  primAlgorithm,
  spanningTreeContract,
  spanningTreeMethods,
} from './spanning-trees.ts'
import { readyRules, topologicalSort, topologicalSortAlgorithm } from './traversal.ts'
import { minimumSpanningTree } from './trees.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

/** Run to the end, and again from a JSON checkpoint cut at `cut` steps; both must end the same. */
function expectResumes<S>(alg: Algorithm<S>, cut: number, project: (s: S) => unknown): S {
  const whole = runAlgorithm(alg, { maxSteps: 10_000 })
  const half = runAlgorithm(alg, { maxSteps: cut })
  const stored = viaJson(alg.serialise?.(half.state) ?? null)
  const rest = runAlgorithm(alg, { maxSteps: 10_000, start: { state: stored, step: half.steps, serialised: true } })
  expect(project(rest.state)).toEqual(project(whole.state))
  expect(rest.status).toBe(whole.status)
  expect(rest.steps).toBe(whole.steps)
  return whole.state
}

/** The values logged to scalar and tensor paths of a run. */
function logged<S>(alg: Algorithm<S>, maxSteps = 10_000): Map<string, unknown[]> {
  const rec = recorder()
  runAlgorithm(alg, {
    maxSteps,
    tracer: rec,
    policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
  })
  const out = new Map<string, unknown[]>()
  for (const e of rec.events) {
    if (e.type === 'metric') out.set(e.name, [...(out.get(e.name) ?? []), e.value])
    if (e.type === 'tensor') out.set(e.name, [...(out.get(e.name) ?? []), Array.from(e.values as ArrayLike<number>)])
  }
  return out
}

const weighted = graph([
  [0, 1, 4],
  [0, 2, 1],
  [1, 2, 2],
  [1, 3, 5],
  [2, 3, 8],
  [4, 5, 3],
])

describe('spanning-tree algorithms', () => {
  const builders = [kruskalAlgorithm, primAlgorithm, boruvkaAlgorithm]
  const names = ['kruskal', 'prim', 'boruvka'] as const

  it('all three find the tree minimumSpanningTree gives, for minimum and maximum', () => {
    for (const maximum of [false, true]) {
      for (const [i, build] of builders.entries()) {
        const run = runAlgorithm(build(weighted, { maximum }), { maxSteps: 50 })
        const want = minimumSpanningTree(weighted, { algorithm: names[i] ?? 'kruskal', maximum })
        expect(run.status).toBe('converged')
        expect([...run.state.chosen].sort((a, b) => a - b)).toEqual(
          want.edges
            .map((e) => weighted.edges.findIndex((f) => f.source === e.source && f.target === e.target))
            .sort((a, b) => a - b),
        )
        expect(run.state.weight).toBeCloseTo(want.weight, 9)
      }
    }
    expect(minimumSpanningTree(weighted).weight).toBe(11)
  })

  it('Kruskal considers edges in weight order and rejects the cycle edges', () => {
    const l = logged(kruskalAlgorithm(weighted))
    expect(l.get('mst/edge')).toEqual([1, 2, 5, 0, 3])
    expect(l.get('mst/accepted')).toEqual([1, 1, 1, 0, 1])
    expect(l.get('mst/weight')).toEqual([1, 3, 6, 6, 11])
    expect(l.get('mst/components')).toEqual([5, 4, 3, 3, 2])
    expect(Object.keys(spanningTreeContract)).toContain('mst/chosen')
  })

  it('Prim starts a new tree at edge -1 for each component', () => {
    const l = logged(primAlgorithm(weighted))
    expect(l.get('mst/edge')?.filter((e) => e === -1)).toHaveLength(2)
    expect(l.get('mst/components')?.at(-1)).toBe(2)
  })

  it('resumes from a JSON checkpoint', () => {
    for (const build of builders) {
      const s = expectResumes(build(gridGraph(4, 4)), 5, (x) => ({ c: x.chosen, w: x.weight }))
      expect(s.chosen).toHaveLength(15)
    }
  })

  it('a forest completes with one tree per component and n - components edges', () => {
    const g = graph(
      [
        [0, 1],
        [2, 3],
      ],
      { nodes: 6 },
    )
    for (const build of builders) expect(runAlgorithm(build(g), { maxSteps: 50 }).state.chosen).toHaveLength(2)
  })

  it('rejects directed graphs, and the option table builds each', () => {
    expect(() => kruskalAlgorithm(graph([[0, 1]], { directed: true }))).toThrow(/undirected/)
    expect(spanningTreeMethods.names).toEqual(['kruskal', 'prim', 'boruvka'])
    expect(spanningTreeMethods.make(defaults(spanningTreeMethods.space))(completeGraph(3)).name).toBe('graph/kruskal')
  })
})

const network = digraph([
  [0, 1, 3],
  [0, 2, 2],
  [1, 2, 1],
  [1, 3, 2],
  [2, 3, 3],
])

describe('maxFlowAlgorithm', () => {
  const methods = ['dinic', 'edmonds-karp'] as const

  it('one augmentation per step, the flow value rising to the old function value', () => {
    for (const method of methods) {
      const run = runAlgorithm(maxFlowAlgorithm(network, 0, 3, { method }), { maxSteps: 20 })
      expect(run.status).toBe('converged')
      expect(run.state.value).toBe(5)
      expect(run.steps).toBe(run.state.augmentations)
      expect(maxFlow(network, 0, 3, { method })).toMatchObject({ value: 5, augmentations: run.steps })
      const l = logged(maxFlowAlgorithm(network, 0, 3, { method }))
      const values = l.get('flow/value') as number[]
      expect(values.at(-1)).toBe(5)
      expect(values.every((v, i) => i === 0 || v > (values[i - 1] ?? 0))).toBe(true)
      expect((l.get('flow/residual') ?? [])[0]).toHaveLength(2 * network.edges.length)
      expect((l.get('flow/cut') ?? []).at(-1)).toEqual([1, 0, 0, 0])
    }
    expect(Object.keys(flowContract)).toContain('flow/potential')
  })

  it('ends with a cut whose capacity is the flow value, and conserved flow', () => {
    const run = runAlgorithm(maxFlowAlgorithm(network, 0, 3), { maxSteps: 20 })
    const c = minCut(network, 0, 3)
    expect(c.value).toBe(run.state.value)
    expect(
      c.cutEdges.reduce(
        (acc, [u, v]) => acc + (network.edges.find((e) => e.source === u && e.target === v)?.weight ?? 0),
        0,
      ),
    ).toBe(5)
    const f = maxFlow(network, 0, 3).flow
    const net = [0, 0, 0, 0]
    for (const [e, edge] of network.edges.entries()) {
      net[edge.source] = (net[edge.source] ?? 0) - (f[e] ?? 0)
      net[edge.target] = (net[edge.target] ?? 0) + (f[e] ?? 0)
    }
    expect(net).toEqual([-5, 0, 0, 5])
  })

  it('resumes from a JSON checkpoint, with infinite capacities', () => {
    const big = digraph([
      [0, 1, 1],
      [0, 2, 1],
      [1, 3, 1],
      [2, 3, 1],
      [1, 2, 1],
      [3, 4, 5],
    ])
    for (const method of methods) {
      const alg = maxFlowAlgorithm(big, 0, 4, { method, capacity: [1, 1, 1, 1, 1, Number.POSITIVE_INFINITY] })
      const s = expectResumes(alg, 1, (x) => ({ v: x.value, r: Array.from(x.residual, String), p: x.pending }))
      expect(s.value).toBe(2)
    }
  })

  it('refuses source equal to sink, and the unbounded flow throws when stepping', () => {
    expect(() => maxFlowAlgorithm(network, 1, 1)).toThrow(/same node/)
    const unbounded = digraph([[0, 1]])
    const alg = maxFlowAlgorithm(unbounded, 0, 1, { capacity: [Number.POSITIVE_INFINITY] })
    expect(() => runAlgorithm(alg, { maxSteps: 5 })).toThrow(/unbounded/)
  })

  it('the option table builds each method', () => {
    expect(maxFlowMethods.names).toEqual(['dinic', 'edmonds-karp'])
    expect(maxFlowMethods.make({ kind: 'dinic' })(network, 0, 3).name).toBe('graph/max-flow-dinic')
  })
})

describe('minCostFlowAlgorithm', () => {
  const g = digraph([
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
  ])
  const options = { capacity: [4, 2, 3, 5], cost: [2, 1, 1, 3] }

  it('routes supply path by path to the cost the function returns', () => {
    const alg = minCostFlowAlgorithm(g, [-4, 0, 0, 4], options)
    const run = runAlgorithm(alg, { maxSteps: 20 })
    expect(run.status).toBe('converged')
    expect(run.state.cost).toBeCloseTo(13, 9)
    expect(Array.from(run.state.flow)).toEqual([3, 1, 3, 1])
    expect(minCostFlow(g, [-4, 0, 0, 4], options).cost).toBeCloseTo(13, 9)
    const l = logged(alg)
    const routed = l.get('flow/value') as number[]
    expect(routed.at(-1)).toBe(4)
    // Successive shortest paths cost non-decreasing amounts per unit.
    const costs = l.get('flow/cost') as number[]
    const unit = costs.map((c, i) => (c - (costs[i - 1] ?? 0)) / ((l.get('flow/augment') as number[])[i] ?? 1))
    expect(unit.every((u, i) => i === 0 || u >= (unit[i - 1] ?? 0) - 1e-9)).toBe(true)
  })

  it('saturates negative-cost edges in init and resumes from JSON', () => {
    const neg = digraph([
      [0, 1],
      [1, 2],
      [0, 2],
    ])
    const alg = minCostFlowAlgorithm(neg, [-2, 0, 2], { capacity: [2, 2, 3], cost: [-1, 1, 5] })
    const s = expectResumes(alg, 1, (x) => ({ c: x.cost, f: Array.from(x.flow), p: Array.from(x.potential) }))
    expect(s.cost).toBeCloseTo(0, 9)
    expect(() => runAlgorithm(minCostFlowAlgorithm(g, [-9, 0, 0, 9], options), { maxSteps: 20 })).toThrow(/infeasible/)
  })

  it('rejects undirected graphs and unbalanced demands', () => {
    expect(() => minCostFlowAlgorithm(graph([[0, 1]]), [-1, 1])).toThrow(/directed/)
    expect(() => minCostFlowAlgorithm(g, [-1, 0, 0, 2])).toThrow(/sum to zero/)
  })
})

describe('hopcroftKarpAlgorithm', () => {
  const b = graph([
    [0, 3],
    [0, 4],
    [1, 3],
    [2, 5],
    [2, 4],
    [1, 5],
  ])

  it('one phase per step, the matching growing to the function size', () => {
    const alg = hopcroftKarpAlgorithm(b)
    const run = runAlgorithm(alg, { maxSteps: 10 })
    expect(run.status).toBe('converged')
    expect(run.state.size).toBe(3)
    expect(bipartiteMatching(b).size).toBe(3)
    const l = logged(alg)
    const sizes = l.get('match/size') as number[]
    expect(sizes.every((v, i) => i === 0 || v >= (sizes[i - 1] ?? 0))).toBe(true)
    expect(sizes.at(-1)).toBe(3)
    expect((l.get('match/phase') as number[])[0]).toBe(1)
    expect((l.get('match/augmenting') as number[]).reduce((a, c) => a + c, 0)).toBe(3)
    expect(Object.keys(matchingContract)).toContain('match/layers')
  })

  it('matches both ends of every pair and resumes from JSON with infinite layers', () => {
    const s = expectResumes(hopcroftKarpAlgorithm(gridGraph(3, 4)), 1, (x) => ({
      m: Array.from(x.mate),
      n: x.size,
      f: String(x.found),
    }))
    expect(s.size).toBe(6)
    for (let u = 0; u < s.mate.length; u++) {
      const v = s.mate[u] ?? -1
      if (v !== -1) expect(s.mate[v]).toBe(u)
    }
  })

  it('rejects an odd cycle and an edge inside one side', () => {
    expect(() => hopcroftKarpAlgorithm(completeGraph(3))).toThrow(/not bipartite/)
    expect(() => hopcroftKarpAlgorithm(graph([[0, 1]]), { left: [0, 1] })).toThrow(/within one side/)
  })
})

describe('strong-component algorithms', () => {
  const g = digraph([
    [0, 1],
    [1, 2],
    [2, 0],
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 4],
  ])

  it('Tarjan and Kosaraju find the components stronglyConnectedComponents gives', () => {
    const want = stronglyConnectedComponents(g).components
    expect(want).toEqual([[0, 1, 2], [3, 4], [5]])
    for (const build of [tarjanAlgorithm, kosarajuAlgorithm]) {
      const run = runAlgorithm(build(g), { maxSteps: 200 })
      expect(run.status).toBe('converged')
      expect(run.state.count).toBe(3)
      const groups = new Map<number, number[]>()
      for (const [v, c] of run.state.component.entries()) groups.set(c, [...(groups.get(c) ?? []), v])
      expect([...groups.values()].sort((p, q) => (p[0] ?? 0) - (q[0] ?? 0))).toEqual(want)
    }
  })

  it('Tarjan completes sink components first and the lowlink never exceeds the discovery time', () => {
    const l = logged(tarjanAlgorithm(g))
    const last = (l.get('dfs/component') ?? []).at(-1) as number[]
    expect(last[3]).toBeLessThan(last[0] ?? 0)
    const disc = (l.get('dfs/discover') ?? []).at(-1) as number[]
    const low = (l.get('dfs/low') ?? []).at(-1) as number[]
    expect(low.every((x, v) => x <= (disc[v] ?? 0))).toBe(true)
    expect((l.get('dfs/count') ?? []).at(-1)).toBe(3)
  })

  it('Kosaraju logs two passes', () => {
    const phases = logged(kosarajuAlgorithm(g)).get('dfs/phase') as number[]
    expect(phases[0]).toBe(1)
    expect(phases.at(-1)).toBe(2)
    expect(phases.every((p, i) => i === 0 || p >= (phases[i - 1] ?? 0))).toBe(true)
  })

  it('resume from a JSON checkpoint', () => {
    for (const build of [tarjanAlgorithm, kosarajuAlgorithm]) {
      for (const cut of [1, 5, 12]) {
        expectResumes(build(g), cut, (s) => ({ c: Array.from(s.component), n: s.count, f: s.finished }))
      }
    }
  })

  it('handles the empty graph and an undirected graph', () => {
    for (const build of [tarjanAlgorithm, kosarajuAlgorithm]) {
      expect(runAlgorithm(build(digraph([], { nodes: 0 })), { maxSteps: 5 }).steps).toBe(0)
      expect(
        runAlgorithm(
          build(
            graph([
              [0, 1],
              [2, 3],
            ]),
          ),
          { maxSteps: 50 },
        ).state.count,
      ).toBe(2)
    }
    expect(componentMethods.names).toEqual(['tarjan', 'kosaraju'])
  })
})

describe('biconnectivityAlgorithm', () => {
  const barbell = barbellGraph(3, 1)

  it('finds the articulation points, bridges and blocks the functions return', () => {
    const run = runAlgorithm(biconnectivityAlgorithm(barbell), { maxSteps: 300 })
    expect(run.status).toBe('converged')
    expect(
      Array.from(run.state.cut.entries())
        .filter(([, c]) => c === 1)
        .map(([v]) => v),
    ).toEqual(articulationPoints(barbell))
    expect(run.state.bridges.length).toBe(bridges(barbell).length)
    expect(run.state.blocks).toHaveLength(biconnectedComponents(barbell).length)
    expect(articulationPoints(barbell)).toEqual([2, 3, 4])
  })

  it('logs lowpoints no later than discovery and a growing block count', () => {
    const l = logged(biconnectivityAlgorithm(barbell))
    const counts = l.get('dfs/count') as number[]
    expect(counts.every((c, i) => i === 0 || c >= (counts[i - 1] ?? 0))).toBe(true)
    const disc = (l.get('dfs/discover') ?? []).at(-1) as number[]
    const low = (l.get('dfs/low') ?? []).at(-1) as number[]
    expect(low.every((x, v) => x <= (disc[v] ?? 0))).toBe(true)
    expect((l.get('dfs/bridges') ?? []).at(-1)).toBe(2)
  })

  it('resumes from a JSON checkpoint at any step', () => {
    for (const cut of [1, 4, 9, 20]) {
      expectResumes(biconnectivityAlgorithm(barbell), cut, (s) => ({
        c: Array.from(s.cut),
        b: s.bridges,
        k: s.blocks,
      }))
    }
  })

  it('rejects a directed graph', () => {
    expect(() => biconnectivityAlgorithm(digraph([[0, 1]]))).toThrow(/undirected/)
  })
})

describe('topologicalSortAlgorithm', () => {
  const dag = digraph([
    [5, 2],
    [5, 0],
    [4, 0],
    [4, 1],
    [2, 3],
    [3, 1],
  ])

  it('places one ready node per step and gives the order topologicalSort gives', () => {
    const run = runAlgorithm(topologicalSortAlgorithm(dag), { maxSteps: 20 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(6)
    expect(run.state.order).toEqual([4, 5, 0, 2, 3, 1])
    expect(topologicalSort(dag)).toEqual([4, 5, 0, 2, 3, 1])
  })

  it('every rule gives a valid order, and the largest rule differs from the smallest', () => {
    for (const rule of ['smallest', 'largest', 'fifo'] as const) {
      const { order } = runAlgorithm(topologicalSortAlgorithm(dag, { rule }), { maxSteps: 20 }).state
      const at = new Map(order.map((v, i) => [v, i]))
      expect(order).toHaveLength(6)
      for (const e of dag.edges) expect(at.get(e.source) ?? 99).toBeLessThan(at.get(e.target) ?? -1)
    }
    const largest = runAlgorithm(topologicalSortAlgorithm(dag, { rule: 'largest' }), { maxSteps: 20 }).state.order
    expect(largest[0]).toBe(5)
    expect(readyRules.names).toEqual(['smallest', 'largest', 'fifo'])
  })

  it('logs the ready set and the indegrees, and ends diverged on a cycle', () => {
    const l = logged(topologicalSortAlgorithm(dag))
    expect(l.get('order/node')).toEqual([4, 5, 0, 2, 3, 1])
    expect(l.get('order/ready')?.[0]).toEqual([5])
    expect((l.get('order/indegree') ?? []).at(-1)).toEqual([0, 0, 0, 0, 0, 0])
    const cyc = digraph([
      [0, 1],
      [1, 2],
      [2, 1],
    ])
    expect(runAlgorithm(topologicalSortAlgorithm(cyc), { maxSteps: 10 }).status).toBe('diverged')
    expect(() => topologicalSort(cyc)).toThrow(/cycle/)
  })

  it('resumes from a JSON checkpoint', () => {
    for (const rule of ['smallest', 'fifo'] as const) {
      expectResumes(topologicalSortAlgorithm(dag, { rule }), 3, (s) => ({
        o: s.order,
        r: s.ready,
        i: Array.from(s.indegree),
      }))
    }
  })
})

describe('greedyColouringAlgorithm', () => {
  const petersen = petersenGraph()

  it('colours one node per step, properly, and matches greedyColouring for every strategy', () => {
    for (const strategy of colouringStrategies.names as ColouringStrategy[]) {
      const key = makeKey(3)
      const alg = greedyColouringAlgorithm(petersen, { strategy, key })
      const run = runAlgorithm(alg, { maxSteps: 20 })
      expect(run.steps).toBe(10)
      expect(isProperColouring(petersen, run.state.colours)).toBe(true)
      expect(Array.from(run.state.colours)).toEqual(greedyColouring(petersen, { strategy, key }))
    }
  })

  it('DSATUR picks the most saturated node and logs saturation', () => {
    const l = logged(greedyColouringAlgorithm(cycleGraph(5), { strategy: 'dsatur' }))
    expect(l.get('colour/assign')).toEqual([0, 1, 0, 1, 2])
    const sat = l.get('colour/saturation') as number[][]
    expect(sat[0]?.every((x) => x === 0 || x === 1)).toBe(true)
    expect((l.get('colour/count') ?? []).at(-1)).toBe(3)
    expect(Object.keys(colouringContract)).toContain('colour/colours')
  })

  it('resumes from a JSON checkpoint, for a fixed order and for DSATUR', () => {
    for (const strategy of ['largest-first', 'dsatur', 'independent-set'] as const) {
      expectResumes(greedyColouringAlgorithm(petersen, { strategy }), 4, (s) => ({
        c: Array.from(s.colours),
        p: s.position,
        s: s.saturation,
      }))
    }
  })

  it('random-sequential needs a key', () => {
    expect(() => greedyColouringAlgorithm(petersen, { strategy: 'random-sequential' })).toThrow(/needs a key/)
  })
})

describe('maximalIndependentSetAlgorithm', () => {
  it('adds one node per step to a set that is independent and maximal', () => {
    const g = petersenGraph()
    for (const key of [undefined, makeKey(5)]) {
      const run = runAlgorithm(maximalIndependentSetAlgorithm(g, key === undefined ? {} : { key }), { maxSteps: 20 })
      const set = new Set(run.state.chosen)
      expect(run.steps).toBe(set.size)
      for (const e of g.edges) expect(set.has(e.source) && set.has(e.target)).toBe(false)
      for (let v = 0; v < g.n; v++) {
        if (!set.has(v))
          expect(
            g.edges.some((e) => (e.source === v && set.has(e.target)) || (e.target === v && set.has(e.source))),
          ).toBe(true)
      }
      expect([...run.state.chosen]).toEqual(maximalIndependentSet(g, key === undefined ? {} : { key }))
    }
  })

  it('keeps forced nodes, logs the shrinking free set, and resumes', () => {
    const g = starGraph(4)
    const l = logged(maximalIndependentSetAlgorithm(g, { nodes: [0] }))
    expect(l.get('mis/node') ?? []).toEqual([])
    const l2 = logged(maximalIndependentSetAlgorithm(g))
    expect(l2.get('mis/size')).toEqual([1, 2, 3, 4])
    expect(l2.get('mis/free')?.[0]).toEqual([2, 3, 4])
    expectResumes(maximalIndependentSetAlgorithm(petersenGraph(), { key: makeKey(1) }), 2, (s) => ({
      c: s.chosen,
      f: s.free,
      k: s.cursor,
    }))
    expect(() => maximalIndependentSetAlgorithm(g, { nodes: [0, 1] })).toThrow(/not independent/)
    expect(independentSetRules.names).toEqual(['min-degree', 'random'])
  })
})

describe('bronKerboschAlgorithm', () => {
  const g = graph([
    [0, 1],
    [1, 2],
    [2, 0],
    [2, 3],
    [3, 4],
    [4, 2],
    [4, 5],
  ])

  it('finds the maximal cliques of maximalCliques with every pivot rule', () => {
    const want = maximalCliques(g)
    expect(want).toEqual([
      [0, 1, 2],
      [2, 3, 4],
      [4, 5],
    ])
    for (const pivot of ['tomita', 'p-only', 'none'] as const) {
      const run = runAlgorithm(bronKerboschAlgorithm(g, { pivot }), { maxSteps: 500 })
      expect(run.status).toBe('converged')
      expect(
        run.state.cliques.map((c) => [...c]).sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0) || a.length - b.length),
      ).toEqual(want)
    }
  })

  it('without a pivot makes at least as many calls, and logs R, P and X of each call', () => {
    const calls = (pivot: 'tomita' | 'none'): number =>
      runAlgorithm(bronKerboschAlgorithm(g, { pivot }), { maxSteps: 500 }).steps
    expect(calls('none')).toBeGreaterThanOrEqual(calls('tomita'))
    const l = logged(bronKerboschAlgorithm(g))
    const r = l.get('clique/R') as number[][]
    expect(r[0]).toHaveLength(1)
    expect((l.get('clique/found') ?? []).at(-1)).toBe(3)
    expect(l.get('clique/P')?.length).toBe(r.length)
    expect(l.get('clique/X')?.length).toBe(r.length)
    expect(Object.keys(cliqueContract)).toContain('clique/pivot')
    expect(pivotRules.names).toEqual(['tomita', 'p-only', 'none'])
  })

  it('resumes from a JSON checkpoint at several cut points', () => {
    for (const cut of [1, 3, 6]) expectResumes(bronKerboschAlgorithm(g), cut, (s) => ({ k: s.cliques, s: s.stack }))
  })

  it('rejects a directed graph', () => {
    expect(() => bronKerboschAlgorithm(digraph([[0, 1]]))).toThrow(/undirected/)
  })
})
