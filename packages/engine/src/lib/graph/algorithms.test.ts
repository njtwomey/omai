import { describe, expect, it } from 'vitest'
import { defaults } from '../space/index.ts'
import { type Algorithm, policies, recorder, runAlgorithm } from '../trace/index.ts'
import { flowContract, maxFlowAlgorithm, maxFlowMethods, minCostFlowAlgorithm } from './flow-algorithms.ts'
import { bipartiteMatching, maxFlow, minCostFlow, minCut } from './flows.ts'
import { completeGraph, gridGraph } from './generators.ts'
import { digraph, graph } from './graph.ts'
import { hopcroftKarpAlgorithm, matchingContract } from './hopcroft-karp.ts'
import {
  boruvkaAlgorithm,
  kruskalAlgorithm,
  primAlgorithm,
  spanningTreeContract,
  spanningTreeMethods,
} from './spanning-trees.ts'
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
