import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { Log, metricSeries } from '../trace/index.ts'
import { type KanFitted, kanClassifier, kanRegressor } from './kan.ts'

const predicted = (f: KanFitted, x: TensorLike): number[] => Array.from(toFlat(f.predict?.(x) as Tensor))

const grid = (n: number, lo = -1, hi = 1): number[] =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
  Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - (b[i] ?? 0))), 0)

const curve = {
  x: grid(60).map((v) => [v]),
  y: grid(60).map((v) => Math.sin(Math.PI * v) + 0.5 * v * v),
}

describe('kanRegressor', () => {
  const est = kanRegressor.with({ hidden: [], grid: 8, epochs: 300, learningRate: 0.05 })

  it('fits a one-dimensional function through the trainer base: score, predictions and the tape', () => {
    const fit = est.fit(curve, { key: key(1) })
    expect(fit.network.widths).toEqual([1, 1])
    expect(fit.score?.(curve)).toBeGreaterThan(0.99)
    expect(predicted(fit, curve.x).length).toBe(60)
    expect(metricSeries(fit.run, 'train/loss').values.length).toBe(300)
    const e = fit.edgeCurve(0, 0, 0)
    expect(e.shape).toEqual([41])
  })

  it('revives from its parameters to the same predictions', () => {
    const fit = est.fit(curve, { key: key(1), explain: false })
    const again = est.revive(fit.params)
    expect(maxAbs(predicted(again, curve.x), predicted(fit, curve.x))).toBeLessThan(1e-12)
  })

  it('takes every edge family, the knots and the optimiser from their slots', () => {
    for (const changes of [
      { basis: { kind: 'rbf' } },
      { basis: { kind: 'chebyshev' } },
      { basis: { kind: 'fourier' } },
      { knots: { kind: 'learnable' } },
      { knots: { kind: 'quantile' } },
      { base: { kind: 'none' } },
      { optimiser: { kind: 'sgd' }, learningRate: 0.01 },
    ] as const) {
      const fit = est.with({ ...changes, epochs: 40 }).fit(curve, { key: key(2), explain: false })
      for (const v of predicted(fit, curve.x)) expect(Number.isFinite(v)).toBe(true)
    }
  })

  it('shrinks and sparsifies with the paper’s penalties, and logs them', () => {
    const data = {
      x: grid(40).map((v, i) => [v, Math.cos(i), Math.sin(2.7 * i)]),
      y: grid(40).map((v) => 2 * v),
    }
    const plain = kanRegressor.with({ hidden: [3], epochs: 80, learningRate: 0.03 })
    const sparse = plain.with({ lambdaL1: 0.02, lambdaEntropy: 0.02 })
    const a = plain.fit(data, { key: key(3), explain: false })
    const s = sparse.session(data, {
      key: key(3),
      logs: {
        'train/loss': Log(),
        'kan/penalty': Log({ every: 20 }),
        'kan/l1': Log({ every: 20 }),
        'kan/sparsity': Log({ every: 40 }),
      },
    })
    s.advance()
    const b = s.fittedAt(s.t)
    const total = (f: Pick<KanFitted, 'edgeScales'>): number =>
      f.edgeScales().reduce((acc, m) => acc + toFlat(m).reduce((u, v) => u + v, 0), 0)
    expect(total(b)).toBeLessThan(total(a))
    expect(b.sparsity(0.05).fraction).toBeGreaterThan(a.sparsity(0.05).fraction)
    expect(b.score?.(data)).toBeGreaterThan(0.8)
    const penalty = metricSeries(s.run(), 'kan/penalty')
    expect(penalty.steps).toEqual([0, 20, 40, 60, 80])
    expect(Math.min(...penalty.values)).toBeGreaterThan(0)
    expect(metricSeries(s.run(), 'kan/sparsity').values).toHaveLength(3)
  })
})

describe('the KAN session', () => {
  const est = kanRegressor.with({ hidden: [2], grid: 5, epochs: 120, learningRate: 0.03 })
  const data = {
    x: grid(50).map((v, i) => [v, Math.sin(i)]),
    y: grid(50).map((v, i) => v * v + 0.5 * Math.sin(i)),
  }

  it('plays the edges: curves at logged steps change as the network learns, and any step can be probed', () => {
    const s = est.session(data, { key: key(4), logs: { 'train/loss': Log(), 'kan/curves': Log({ every: 40 }) } })
    s.advance()
    expect(s.status).toBe('completed')
    const early = s.probe(0, 'kan/curves') as {
      widths: number[]
      edges: { values: number[]; layer: number; scale: number }[]
    }
    const late = s.probe(120, 'kan/curves') as typeof early
    expect(early.widths).toEqual([2, 2, 1])
    expect(early.edges).toHaveLength(2 * 2 + 2 * 1)
    expect(maxAbs(early.edges[0]?.values ?? [], late.edges[0]?.values ?? [])).toBeGreaterThan(0.01)
    expect(metricSeries(s.run(), 'kan/curves').steps ?? []).toBeDefined()
    const one = s.probe(120, 'kan/edge/1/0/0') as { values: number[] } | null
    expect(one?.values).toHaveLength(41)
    const grids = s.probe(120, 'kan/grids') as { layer: number; knots: number[][] }[]
    expect(grids).toHaveLength(2)
    expect(grids[0]?.knots[0]).toHaveLength(4)
  })

  it('moves free knots, which the grids path reports', () => {
    const s = est
      .with({ knots: { kind: 'learnable' }, hidden: [] })
      .session(data, { key: key(5), logs: { 'train/loss': Log() } })
    s.advance()
    const before = (s.probe(0, 'kan/grids') as { knots: number[][] }[])[0]?.knots[0] ?? []
    const after = (s.probe(120, 'kan/grids') as { knots: number[][] }[])[0]?.knots[0] ?? []
    expect(maxAbs(before, after)).toBeGreaterThan(1e-4)
  })
})

describe('kanClassifier', () => {
  it('separates two classes by an additive boundary and exposes the symbolic reading', () => {
    const x = grid(120).map((v, i) => [v, Math.sin(1.7 * i)])
    const y = x.map(([a, b]) => ((a ?? 0) + 0.5 * (b ?? 0) > 0.1 ? 1 : 0))
    const fit = kanClassifier
      .with({ hidden: [], epochs: 150, learningRate: 0.05, grid: 4 })
      .fit({ x, y }, { key: key(6) })
    expect(fit.score?.({ x, y })).toBeGreaterThan(0.95)
    expect(fit.network.widths).toEqual([2, 2])
    expect(Object.keys(fit.symbolic().layers[0]?.edges ?? [])).toHaveLength(2)
  })
})
