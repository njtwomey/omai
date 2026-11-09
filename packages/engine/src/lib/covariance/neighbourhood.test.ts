import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { gaussianGraph } from '../datasets/index.ts'
import { linalg } from '../numerics/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { key } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { item, toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { edgeRecovery, graphEdges } from './evaluation.ts'
import {
  clime,
  climeAlgorithm,
  meinshausenBuhlmannPenalty,
  neighbourhoodSelection,
  neighbourhoodSelectionAlgorithm,
} from './neighbourhood.ts'

const planted = (n: number, p = 6, seed = 1) => {
  const { data, truth } = generate(gaussianGraph, {
    key: key(seed),
    knobs: { n, p, structure: 'band', strength: 0.4 },
  })
  return { x: toArray(data.x) as number[][], adjacency: truth.extra?.['adjacency'] as boolean[][] }
}

describe('Meinshausen-Buhlmann neighbourhood selection', () => {
  it('has the penalty Phi^-1(1 - level / (2 p^2)) / sqrt(n)', () => {
    expect(meinshausenBuhlmannPenalty(100, 10, 0.05)).toBeCloseTo(item(ndtri(1 - 0.05 / 200)) / 10, 14)
    expect(meinshausenBuhlmannPenalty(400, 10)).toBeCloseTo(meinshausenBuhlmannPenalty(100, 10) / 2, 14)
  })

  it('recovers the chain better as the sample grows', () => {
    const f1 = [60, 600, 6000].map((n) => {
      const { x, adjacency } = planted(n, 8)
      return edgeRecovery(neighbourhoodSelection(x).or, adjacency).f1
    })
    expect(f1[2]).toBe(1)
    expect(f1[2]).toBeGreaterThanOrEqual(f1[0] ?? 1)
    expect(f1[1]).toBeGreaterThan(f1[0] ?? 1)
  })

  it('a huge penalty selects nothing and none selects every pair; the diagonal is never a neighbour', () => {
    const { x } = planted(200, 5)
    expect(graphEdges(neighbourhoodSelection(x, { penalty: 10 }).or).length).toBe(0)
    const dense = neighbourhoodSelection(x, { penalty: 0, tol: 1e-10 })
    expect(graphEdges(dense.and).length).toBe(10)
    for (let a = 0; a < 5; a++) expect(dense.coefficients[a]?.[a]).toBe(0)
  })

  it('"and" is a subgraph of "or" and `rule` picks which one is the graph', () => {
    const { x } = planted(100, 6)
    const nb = neighbourhoodSelection(x, { penalty: 0.1 })
    expect(graphEdges(nb.and).every(([a, b]) => nb.or[a]?.[b] === true)).toBe(true)
    expect(neighbourhoodSelection(x, { penalty: 0.1, rule: 'and' }).graph).toEqual(nb.and)
    expect(nb.graph).toEqual(nb.or)
    expect(nb.penalty).toBe(0.1)
    expect(nb.or).toEqual(nb.or.map((r, a) => r.map((_, b) => nb.or[b]?.[a])))
  })

  it('traces a step per node, the neighbours accumulating, and restores from its serialised state', () => {
    const { x } = planted(100, 4)
    const alg = neighbourhoodSelectionAlgorithm(x, { penalty: 0.05 })
    const run = runAlgorithm(alg, { maxSteps: 10 })
    expect(run.steps).toBe(4)
    expect(run.state.node).toBe(4)
    expect(alg.deserialise?.(alg.serialise?.(run.state) ?? null)).toEqual(run.state)
    expect(() =>
      neighbourhoodSelection([
        [1, 2],
        [2, 3],
      ]),
    ).toThrow(/at least 3 rows/)
    expect(() => neighbourhoodSelection(x, { penalty: -1 })).toThrow(/non-negative/)
  })

  it('standardising matters: without it a column with a large scale gets fewer neighbours', () => {
    const { x } = planted(150, 4)
    const scaled = x.map((r) => [r[0] as number, ...(r.slice(1) as number[]).map((v) => v * 10)])
    expect(neighbourhoodSelection(scaled, { penalty: 0.1 }).or).toEqual(neighbourhoodSelection(x, { penalty: 0.1 }).or)
  })
})

describe('CLIME', () => {
  const cov = (n: number, p = 5) => {
    const { x } = planted(n, p)
    return toArray(empiricalCovariance(x)) as number[][]
  }

  it('with a zero level is the inverse of the covariance', () => {
    const s = cov(100)
    const inverse = toArray(linalg.inv(s)) as number[][]
    const out = clime(s, { lambda: 0 })
    for (let a = 0; a < 5; a++)
      for (let b = 0; b < 5; b++) expect(out.precision[a]?.[b]).toBeCloseTo(inverse[a]?.[b] ?? 0, 7)
  })

  it('satisfies its constraint column by column, is symmetric and gets sparser as the level grows', () => {
    const s = cov(150)
    let previous = Number.POSITIVE_INFINITY
    for (const lambda of [0.02, 0.1, 0.3, 0.6]) {
      const out = clime(s, { lambda })
      for (let j = 0; j < 5; j++) {
        const column = out.columns[j] as number[]
        for (let i = 0; i < 5; i++) {
          const residual = s[i]?.reduce((t, v, k) => t + v * (column[k] ?? 0), 0) ?? 0
          expect(Math.abs(residual - (i === j ? 1 : 0))).toBeLessThan(lambda + 1e-8)
        }
      }
      expect(out.precision).toEqual(out.precision.map((r, a) => r.map((_, b) => out.precision[b]?.[a])))
      const edges = graphEdges(out.graph).length
      expect(edges).toBeLessThanOrEqual(previous)
      previous = edges
    }
    expect(previous).toBe(0)
  })

  it('finds the chain from a moderate sample', () => {
    const { x, adjacency } = planted(1000, 6)
    const out = clime(toArray(empiricalCovariance(x)) as number[][], { lambda: 0.08 })
    expect(edgeRecovery(out.graph, adjacency).recall).toBe(1)
    expect(edgeRecovery(out.graph, adjacency).precision).toBeGreaterThan(0.7)
  })

  it('traces a step per column, and an infeasible column is an error', () => {
    const run = runAlgorithm(
      climeAlgorithm(
        [
          [2, 0],
          [0, 4],
        ],
        { lambda: 0 },
      ),
      { maxSteps: 5 },
    )
    expect(run.steps).toBe(2)
    expect(run.state.columns).toEqual([
      [0.5, 0],
      [0, 0.25],
    ])
    expect(() =>
      clime(
        [
          [1, 1],
          [1, 1],
        ],
        { lambda: 0 },
      ),
    ).toThrow(/column/)
    expect(() =>
      clime(
        [
          [1, 0],
          [0, 1],
        ],
        { lambda: -1 },
      ),
    ).toThrow(/non-negative/)
  })
})
