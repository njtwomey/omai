import { describe, expect, it } from 'vitest'
import { smoothSolvers, solverConfigOf } from '../optim/index.ts'
import { key, normal } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import {
  largeMarginNearestNeighbour,
  lmnnSolvers,
  ncaValueAndGradient,
  neighbourhoodComponentsAnalysis,
} from './metric-learning.ts'
import { kNeighboursClassifier } from './neighbours.ts'
import { matrixView } from './validation.ts'

/** Two classes separated along the first axis, with two noisy axes that carry nothing and dominate the scale. */
function data(): { readonly x: number[][]; readonly y: number[] } {
  const noise = toFlat(normal(key(4), [40, 2]))
  const x = Array.from({ length: 40 }, (_, i) => [
    (i % 2) * 1.2 + 0.1 * (noise[i * 2] ?? 0),
    6 * (noise[i * 2 + 1] ?? 0),
    6 * (noise[(i * 2 + 7) % 80] ?? 0),
  ])
  return { x, y: Array.from({ length: 40 }, (_, i) => i % 2) }
}

describe('neighbourhood components analysis', () => {
  it('has a gradient that matches finite differences', () => {
    const { x, y } = data()
    const view = matrixView(x.slice(0, 12), 'test')
    const codes = y.slice(0, 12)
    const a = [0.3, -0.2, 0.5, 0.1, 0.4, -0.3]
    const r = ncaValueAndGradient(a, 2, view, codes)
    for (let i = 0; i < a.length; i++) {
      const up = ncaValueAndGradient(
        a.map((v, j) => (j === i ? v + 1e-6 : v)),
        2,
        view,
        codes,
      ).value
      const down = ncaValueAndGradient(
        a.map((v, j) => (j === i ? v - 1e-6 : v)),
        2,
        view,
        codes,
      ).value
      expect(r.gradient[i]).toBeCloseTo((up - down) / 2e-6, 6)
    }
    expect(r.probabilities.every((p) => p >= 0 && p <= 1)).toBe(true)
  })

  it('learns to ignore the noisy features, which a neighbour classifier then shows', () => {
    const { x, y } = data()
    const train = { x: x.slice(0, 28), y: y.slice(0, 28) }
    const test = { x: x.slice(28), y: y.slice(28) }
    const nca = neighbourhoodComponentsAnalysis({ nComponents: 1, init: 'identity', maxIter: 100 })
    const fitted = nca.fit(train)
    const knn = (map: (v: number[][]) => number[][]): number =>
      kNeighboursClassifier({ nNeighbors: 3 })
        .fit({ x: map(train.x), y: train.y })
        .score({ x: map(test.x), y: test.y })
    const before = knn((v) => v)
    const after = knn((v) => toArray(fitted.transform(v)) as number[][])
    expect(after).toBeGreaterThan(before)
    expect(fitted.params.objective).toBeGreaterThan(24)
  })

  it('logs the objective rising, and the map at each step of a session', () => {
    const { x, y } = data()
    const nca = neighbourhoodComponentsAnalysis({ nComponents: 2, init: 'pca', maxIter: 30 })
    const s = nca.session({ x, y }, { key: key(0), logs: 'all' })
    s.advance({ steps: 6 })
    const objective = metricSeries(s.run(), 'solver/objective').values
    expect(objective.length).toBeGreaterThan(3)
    expect(objective[objective.length - 1]).toBeGreaterThan(objective[0] ?? 0)
    expect((s.fittedAt(0).params.components[0] ?? []).length).toBe(3)
    expect(s.fittedAt(5).params.objective).toBeGreaterThan(s.fittedAt(0).params.objective)
    expect(s.probe(3, 'model/embedding') as unknown[]).toHaveLength(40)
  })

  it('starts from a given map, and refuses one of the wrong shape', () => {
    const { x, y } = data()
    const f = neighbourhoodComponentsAnalysis({ init: [[1, 0, 0]], nComponents: 1, maxIter: 1 }).fit({ x, y })
    expect(f.params.components).toHaveLength(1)
    expect(() => neighbourhoodComponentsAnalysis({ init: [[1, 0]], nComponents: 1 }).fit({ x, y })).toThrow(
      /initial map/,
    )
  })
})

describe('large margin nearest neighbour', () => {
  it('lowers its loss and its impostors, step by step', () => {
    const { x, y } = data()
    const est = largeMarginNearestNeighbour({ nNeighbors: 2, maxIter: 80, learningRate: 1e-4 })
    const s = est.session({ x, y }, { key: key(0), logs: 'all' })
    s.advance()
    const loss = metricSeries(s.run(), 'solver/objective').values
    const impostors = metricSeries(s.run(), 'solver/impostors').values
    expect(loss[loss.length - 1]).toBeLessThan(loss[0] ?? 0)
    expect(impostors[impostors.length - 1]).toBeLessThan(impostors[0] ?? 0)
    const fitted = s.fittedAt(s.t)
    expect(toArray(fitted.transform(x.slice(0, 2)))).toHaveLength(2)
  })
})

describe('the solver slot of metric learning', () => {
  it('NCA with the default solver reproduces the L-BFGS fit it had before the slot', () => {
    const { x, y } = data()
    const f = neighbourhoodComponentsAnalysis({ nComponents: 2, init: 'pca', maxIter: 30 }).fit({ x, y })
    // The values of the direct L-BFGS fit, before the solver became a slot.
    expect(f.params.objective).toBeCloseTo(37.99999990618548, 5)
    expect(f.params.components[0]?.[0]).toBeCloseTo(-4.3901388701, 5)
    expect(f.params.components[1]?.[1]).toBeCloseTo(2.471370887, 5)
  })

  it('NCA reaches about the same objective with BFGS and with conjugate gradient', () => {
    const { x, y } = data()
    const hyper = { nComponents: 1, init: 'identity' as const, maxIter: 200, gradTol: 1e-7 }
    const reference = neighbourhoodComponentsAnalysis(hyper).fit({ x, y }).params.objective
    for (const solver of [
      { kind: 'bfgs', gradTol: 1e-7 },
      { kind: 'conjugateGradient', gradTol: 1e-7 },
    ]) {
      const f = neighbourhoodComponentsAnalysis({ ...hyper, solver: solverConfigOf(smoothSolvers, solver) }).fit({
        x,
        y,
      })
      expect(f.params.objective, solver.kind).toBeGreaterThan(reference - 0.5)
    }
  }, 60_000)

  it('LMNN with the default solver reproduces the adaptive descent it had before the slot', () => {
    const { x, y } = data()
    const f = largeMarginNearestNeighbour({ nNeighbors: 2, maxIter: 80, learningRate: 1e-4 }).fit({ x, y })
    // The values of the hand-rolled descent, before the solver became a slot.
    expect(f.params.objective).toBeCloseTo(2.110746465429699, 6)
    expect(f.params.components[0]?.[0]).toBeCloseTo(1.1433612424, 6)
    expect(f.params.components[2]?.[2]).toBeCloseTo(0.0172372152, 6)
  })

  it('LMNN lowers its loss from the start with L-BFGS and with the stochastic gradient mode', () => {
    const { x, y } = data()
    const hyper = { nNeighbors: 2, maxIter: 60, learningRate: 1e-4 }
    const start = largeMarginNearestNeighbour({ ...hyper, maxIter: 0 }).fit({ x, y }).params.objective
    const adaptive = largeMarginNearestNeighbour(hyper).fit({ x, y }).params.objective
    for (const solver of [
      { kind: 'lbfgs', gradTol: 1e-7 },
      { kind: 'gradient', optimiser: { kind: 'adam' }, learningRate: 1e-3, gradTol: 1e-9 },
    ]) {
      const f = largeMarginNearestNeighbour({ ...hyper, solver: solverConfigOf(lmnnSolvers, solver) }).fit({ x, y })
      expect(f.params.objective, solver.kind).toBeLessThan(start)
    }
    expect(adaptive).toBeLessThan(start)
  }, 60_000)
})
