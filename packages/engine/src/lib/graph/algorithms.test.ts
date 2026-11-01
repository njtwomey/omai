import { describe, expect, it } from 'vitest'
import { defaults } from '../space/index.ts'
import { type Algorithm, policies, recorder, runAlgorithm } from '../trace/index.ts'
import { completeGraph, gridGraph } from './generators.ts'
import { graph } from './graph.ts'
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
