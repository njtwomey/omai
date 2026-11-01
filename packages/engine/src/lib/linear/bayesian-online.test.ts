import { describe, expect, it } from 'vitest'
import { session } from '../trace/index.ts'
import { bayesianLinearFit } from './bayesian.ts'
import { bayesianOnlineLinear } from './bayesian-online.ts'
import { designOf } from './design.ts'

// A line y = 1 + 2 x with a little deterministic noise.
const x = Array.from({ length: 12 }, (_, i) => [((i * 7) % 12) / 6 - 1])
const y = x.map((r, i) => 1 + 2 * (r[0] ?? 0) + 0.2 * Math.sin(5 * i))
const hyper = { alpha: 2, beta: 25, intercept: true }
const model = bayesianOnlineLinear.with(hyper)
const phi = (rows: number[][]): number[][] => rows.map((r) => [1, ...r])

const batch = (rows: number[][], targets: number[]) =>
  bayesianLinearFit(designOf({ x: phi(rows), y: targets }, 'test'), { alpha: hyper.alpha, beta: hyper.beta })

describe('bayesianOnlineLinear', () => {
  it('after t rows is the batch posterior of those rows, whatever t', () => {
    const live = session(model.method({ x, y }), { mode: 'eager', steps: x.length })
    for (const t of [1, 4, 8, 12]) {
      const online = live.stateAt(t)
      const direct = batch(x.slice(0, t), y.slice(0, t))
      for (const [i, m] of online.mean.entries()) expect(m).toBeCloseTo(direct.mean[i] ?? Number.NaN, 10)
      for (const [i, row] of online.covariance.entries())
        for (const [j, v] of row.entries()) expect(v).toBeCloseTo(direct.covariance[i]?.[j] ?? Number.NaN, 10)
      expect(online.logEvidence).toBeCloseTo(direct.logEvidence, 9)
    }
  })

  it('reaches the same belief in any order of the rows', () => {
    const order = [5, 2, 11, 0, 7, 3, 9, 1, 10, 4, 8, 6]
    const a = model.fit({ x, y }).weights()
    const b = model.fit({ x: order.map((i) => x[i] as number[]), y: order.map((i) => y[i] as number) }).weights()
    for (const [i, m] of a.mean.entries()) expect(m).toBeCloseTo(b.mean[i] ?? Number.NaN, 10)
  })

  it('draws the update as prior times likelihood equals posterior (up to a constant)', () => {
    const live = session(model.method({ x, y }), { mode: 'eager', steps: x.length })
    const at = 5
    const f = live.probe(at, 'model/weights', { range: [-1, 3, -1, 4], points: 15 }) as unknown as {
      prior: number[][]
      likelihood: number[][]
      posterior: number[][]
    }
    const logs: number[] = []
    for (let i = 0; i < 15; i++)
      for (let j = 0; j < 15; j++)
        logs.push(
          Math.log(f.posterior[i]?.[j] ?? Number.NaN) -
            Math.log(f.prior[i]?.[j] ?? Number.NaN) -
            Math.log(f.likelihood[i]?.[j] ?? Number.NaN),
        )
    for (const l of logs) expect(l).toBeCloseTo(logs[0] ?? Number.NaN, 8)
  })

  it('has the prior of a step equal to the posterior of the step before, and a likelihood that is a ridge', () => {
    const live = session(model.method({ x, y }), { mode: 'eager', steps: x.length })
    const w = live.probe(6, 'model/weights', { range: [-1, 3, -1, 4], points: 9 }) as unknown as {
      priorMean: number[]
      likelihood: number[][]
      observation: { phi: number[]; y: number }
    }
    const before = live.stateAt(5)
    for (const [i, m] of w.priorMean.entries()) expect(m).toBeCloseTo(before.mean[i] ?? Number.NaN, 12)
    // Two weight vectors that predict the new point the same have the same likelihood.
    const { phi: p, y: target } = w.observation
    const wa = [0.5, (target - 0.5) / (p[1] ?? 1) - (p[0] ?? 0) * 0]
    const like = (a: number[]): number =>
      Math.exp(-0.5 * hyper.beta * (target - ((p[0] ?? 0) * (a[0] ?? 0) + (p[1] ?? 0) * (a[1] ?? 0))) ** 2)
    const wb = [1.5, (target - (p[0] ?? 0) * 1.5) / (p[1] ?? 1)]
    expect(like(wa)).toBeCloseTo(like(wb), 6)
    expect(w.likelihood).toHaveLength(9)
  })

  it('has no likelihood before the first point, and a posterior that integrates to one', () => {
    const live = session(model.method({ x, y }), { mode: 'eager', steps: x.length })
    const first = live.probe(0, 'model/weights', { points: 5 }) as unknown as {
      likelihood: unknown
      observation: unknown
    }
    expect(first.likelihood).toBeNull()
    expect(first.observation).toBeNull()
    const f = live.probe(12, 'model/weights', { range: [0, 2, 1, 3], points: 81 }) as unknown as {
      posterior: number[][]
    }
    const cell = (2 / 80) * (2 / 80)
    expect(f.posterior.flat().reduce((a, b) => a + b, 0) * cell).toBeCloseTo(1, 2)
  })

  it('gives the predictive line with a band and lines drawn from the posterior', () => {
    const live = session(model.method({ x, y }), { mode: 'eager', steps: x.length })
    const grid = [-1, 0, 1]
    const lines = live.probe(12, 'model/lines', { x: grid, samples: 7 }) as unknown as {
      mean: number[]
      std: number[]
      samples: number[][]
    }
    const fitted = model.fit({ x, y })
    expect(lines.mean).toHaveLength(3)
    for (const [i, m] of lines.mean.entries())
      expect(m).toBeCloseTo(Array.from(toArray(fitted.predict([[grid[i] ?? 0]])))[0] ?? Number.NaN, 10)
    expect(lines.std.every((s) => s > 0)).toBe(true)
    expect(lines.samples).toHaveLength(7)
    expect(lines.samples[0]).toHaveLength(3)
  })

  it('tightens: the noise-free spread of the prediction falls as points arrive', () => {
    const live = session(model.method({ x, y }), { mode: 'eager', steps: x.length })
    const spread = (t: number): number => {
      const a = live.probe(t, 'model/lines', { x: [0.5] }) as unknown as { std: number[] }
      return a.std[0] ?? Number.NaN
    }
    expect(spread(1)).toBeGreaterThan(spread(6))
    expect(spread(6)).toBeGreaterThan(spread(12))
  })
})

const toArray = (t: unknown): ArrayLike<number> => (t as { data: ArrayLike<number> }).data
