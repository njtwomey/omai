import { describe, expect, it } from 'vitest'
import * as P from '../index.ts'

const tree = (a: number, b: number): P.GraphInput => P.treeInput([{ f: a }, { f: b }], { parents: [-1, 0] })
const treeGraphs = [tree(1, 0), tree(1, 1), tree(0, 0), tree(0, 1), tree(1, 0), tree(0, 1)]
const treeLabels = [
  ['a', 'b'],
  ['a', 'a'],
  ['b', 'b'],
  ['b', 'a'],
  ['a', 'b'],
  ['b', 'a'],
]

/** A 3 x 3 grid whose label is the sign of the cell's observation, with one flipped observation. */
function grid(seed: readonly number[], noisy: number): { x: P.GraphInput; y: string[] } {
  const y = seed.map((v) => (v > 0 ? 'p' : 'n'))
  const nodes = seed.map((v, i) => ({ obs: i === noisy ? -v : v }))
  return { x: P.gridInput(nodes, 3, 3), y }
}

describe('the graph CRF estimator', () => {
  it('fits trees through the structured slot and predicts labels and marginals', () => {
    const fitted = P.graphCrfEstimator({ structure: P.treeCrf, c2: 0.1 }).fit(P.graphCrfData(treeGraphs, treeLabels))
    expect(fitted.classes).toEqual(['a', 'b'])
    expect(fitted.predict(treeGraphs)).toEqual(treeLabels)
    expect(fitted.score(treeGraphs, treeLabels)).toBe(1)
    expect(fitted.model.edgeTypes).toEqual(['parent'])
    const m = fitted.predictMarginalsSingle(tree(1, 0))
    expect(m).toHaveLength(2)
    expect((m[0]?.['a'] ?? 0) + (m[0]?.['b'] ?? 0)).toBeCloseTo(1, 12)
    expect(m[0]?.['a']).toBeGreaterThan(0.5)
    expect(Object.keys(fitted.weights()).some((n) => n.startsWith('node:f|'))).toBe(true)
    expect(fitted.params.converged).toBe(true)
  })

  it('matches trainStructured on the same oracle and penalty', () => {
    const model = P.treeCrf(['a', 'b'], ['f'])
    const direct = P.trainStructured(
      {
        oracle: P.graphOracle(model),
        data: treeGraphs.map((x, i) => ({ x, y: treeLabels[i] ?? [] })),
        penalty: { c2: 0.1 },
        start: model.weights,
      },
      { tol: 1e-6 },
    )
    const fitted = P.graphCrfEstimator({ structure: P.treeCrf, c2: 0.1 }).fit(P.graphCrfData(treeGraphs, treeLabels))
    expect(fitted.params.weights).toEqual(Array.from(direct.weights))
    expect(fitted.params.objective).toBe(direct.objective)
  })

  it('builds the default structure from the edge types of the graphs, and fits grids and skip chains', () => {
    const cases = [
      grid([1, 1, -1, 1, -1, -1, 1, 1, -1], 4),
      grid([-1, -1, 1, -1, 1, 1, -1, -1, 1], 0),
      grid([1, 1, 1, -1, -1, -1, 1, 1, 1], 8),
    ]
    const fitted = P.graphCrfEstimator({ c2: 0.5 }).fit(
      P.graphCrfData(
        cases.map((c) => c.x),
        cases.map((c) => c.y),
      ),
    )
    expect(fitted.model.edgeTypes).toEqual(['grid'])
    expect(
      fitted.score(
        cases.map((c) => c.x),
        cases.map((c) => c.y),
      ),
    ).toBeGreaterThan(0.7)
    const chain = (v: readonly number[], keys: readonly (string | undefined)[]): P.GraphInput =>
      P.skipChainInput(
        v.map((o) => ({ o })),
        { keys },
      )
    const skips = [
      chain([1, 0.1, 1], ['x', undefined, 'x']),
      chain([-1, 0.1, -1], ['y', undefined, 'y']),
      chain([1, -0.1, 1], ['x', undefined, 'x']),
    ]
    const skipLabels = [
      ['u', 'u', 'u'],
      ['v', 'v', 'v'],
      ['u', 'u', 'u'],
    ]
    const skipFitted = P.graphCrfEstimator({ structure: P.skipChainCrf, c2: 0.1 }).fit(
      P.graphCrfData(skips, skipLabels),
    )
    expect(skipFitted.model.edgeTypes).toEqual(['chain', 'skip'])
    expect(skipFitted.predict(skips)).toEqual(skipLabels)
  })

  it('serialises, revives, clones, and rejects bad data', () => {
    const estimator = P.graphCrfEstimator({ structure: P.treeCrf, c2: 0.1 })
    const fitted = estimator.fit(P.graphCrfData(treeGraphs, treeLabels))
    const json = JSON.parse(JSON.stringify(fitted.serialise())) as { estimator: string; params: P.GraphCrfParams }
    const revived = P.graphCrfEstimator({ structure: P.treeCrf }).revive(json.params)
    expect(revived.params).toEqual(fitted.params)
    expect(revived.predict(treeGraphs)).toEqual(fitted.predict(treeGraphs))
    expect(estimator.with({ c2: 3 }).hyper.c2).toBe(3)
    expect(estimator.hyper.c2).toBe(0.1)
    expect(fitted.predictSingle({ nodes: [], edges: [] })).toEqual([])
    expect(() => estimator.fit({ x: [0] } as unknown as P.GraphCrfData)).toThrow(/needs graphs/)
    expect(() => P.graphCrfData(treeGraphs, [])).toThrow(/label lists/)
    expect(() => estimator.fit(P.graphCrfData([tree(1, 0)], [['a']]))).toThrow(/nodes and/)
    expect(() => estimator.fit(P.graphCrfData([{ nodes: [], edges: [] }], [[]]))).toThrow(/no non-empty/)
  })

  it('trains with another solver of the slot', () => {
    const fitted = P.graphCrfEstimator({
      structure: P.treeCrf,
      c2: 0.1,
      solver: { kind: 'passive-aggressive' },
      maxIterations: 50,
    }).fit(P.graphCrfData(treeGraphs, treeLabels))
    expect(fitted.score(treeGraphs, treeLabels)).toBe(1)
  })
})
