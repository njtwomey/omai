import { describe, expect, it } from 'vitest'
import { expectClose } from '../../../testing/close.ts'
import { minimise } from '../../../testing/optimise.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { add, matmul, mean, square, sub, sum, tensor, toArray, toFlat, type Value } from '../../tensor/index.ts'
import { NBeats, NBeatsBlock, seasonalityBasis, trendBasis } from './forecasting.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

describe('N-BEATS bases', () => {
  it('the trend basis is the polynomials of normalised time, so coefficients are polynomial coefficients', () => {
    expect(toArray(trendBasis(2, 4))).toEqual([
      [1, 1, 1, 1],
      [0, 0.25, 0.5, 0.75],
      [0, 0.0625, 0.25, 0.5625],
    ])
    // theta @ T evaluates 1 + 2 s - s^2 + 0.5 s^3 at s = t / H exactly.
    const theta = tensor([[1, 2, -1, 0.5]])
    const H = 6
    const want = Array.from({ length: H }, (_, t) => 1 + 2 * (t / H) - (t / H) ** 2 + 0.5 * (t / H) ** 3)
    expectClose(matmul(theta, trendBasis(3, H)), tensor([want]), 1e-14)
  })

  it('the seasonality basis holds cosines of 0 .. K - 1 cycles and sines of 1 .. K cycles', () => {
    const s = toArray(seasonalityBasis(2, 8)) as number[][]
    expect(s).toHaveLength(4)
    expect(s[0]).toEqual(Array(8).fill(1))
    expect(s[1]?.[2]).toBeCloseTo(0, 12)
    expect(s[2]?.[1]).toBeCloseTo(Math.sin(Math.PI / 4), 14)
    expect(s[3]?.[1]).toBeCloseTo(Math.sin(Math.PI / 2), 14)
    // Distinct harmonics are orthogonal over a whole number of periods.
    const dot = (a: number[], b: number[]): number => a.reduce((t, v, i) => t + v * (b[i] as number), 0)
    expect(dot(s[1] as number[], s[2] as number[])).toBeCloseTo(0, 12)
    expect(dot(s[2] as number[], s[3] as number[])).toBeCloseTo(0, 12)
  })
})

describe('NBeatsBlock', () => {
  it('a trend or seasonality block is its coefficients expanded over the fixed bases', () => {
    for (const basis of ['trend', 'seasonality'] as const) {
      const block = new NBeatsBlock(10, 4, { basis, width: 8, layers: 2, key: key(1) })
      const x = n(2, [3, 10])
      const theta = block.coefficients(block.params, x as Value)
      const out = block.forward(x)
      expectClose(out.forecast, matmul(theta.forecast, block.forecastBasis as Value))
      expectClose(out.backcast, matmul(theta.backcast, block.backcastBasis as Value))
      expect(out.forecast.shape).toEqual([3, 4])
      expect(out.backcast.shape).toEqual([3, 10])
    }
    expect(new NBeatsBlock(10, 4, { basis: 'trend', size: 2, width: 4, layers: 1 }).forecastBasis?.shape).toEqual([
      3, 4,
    ])
  })

  it('a generic block is two learned linear expansions, and a block passes the gradient check', () => {
    const block = new NBeatsBlock(6, 3, { width: 5, layers: 1, size: 4, key: key(3) })
    expect(Object.keys(block.stateDict())).toContain('backcast.weight')
    const x = n(4, [2, 6])
    expect(checkPrimitive((v: Value) => block.apply(block.params, v).forecast, [x], { rtol: 1e-5 }).failures).toEqual(
      [],
    )
    const trend = new NBeatsBlock(6, 3, { basis: 'trend', width: 5, layers: 1, key: key(5) })
    expect(checkPrimitive((v: Value) => trend.apply(trend.params, v).backcast, [x], { rtol: 1e-5 }).failures).toEqual(
      [],
    )
  })
})

describe('NBeats', () => {
  const model = new NBeats(12, 4, {
    stacks: [
      { basis: 'trend', blocks: 2, width: 8, layers: 2 },
      { basis: 'seasonality', blocks: 1, width: 8, layers: 2 },
    ],
    key: key(10),
  })

  it('is doubly residual: the forecast sums the blocks and the residual is the window minus the backcasts', () => {
    const x = n(11, [3, 12])
    const out = model.forward(x)
    expect(model.blockCount).toBe(3)
    expect(out.contributions.shape).toEqual([3, 3, 4])
    expect(out.backcasts.shape).toEqual([3, 3, 12])
    expectClose(out.forecast, sum(out.contributions, 0), 1e-12)
    expectClose(out.residual, sub(x, sum(out.backcasts, 0)), 1e-12)
    // Each block reads the residual left by the ones before it.
    const first = model.part<NBeatsBlock>('block0').forward(x as Value)
    const second = model.part<NBeatsBlock>('block1').forward(sub(x, first.backcast))
    expectClose(toFlat(out.contributions).slice(12, 24), second.forecast, 1e-12)
  })

  it('a trend stack trained on polynomial series reproduces the polynomial forecast', () => {
    const lookback = 12
    const horizon = 4
    const trend = new NBeats(lookback, horizon, {
      stacks: [{ basis: 'trend', blocks: 1, size: 3, width: 16, layers: 2 }],
      key: key(20),
    })
    // Random cubics a + b s + c s^2 + d s^3 observed on s = t / lookback; the continuation is read at the same s grid beyond.
    const make = (seed: number): { window: number[]; future: number[] } => {
      const [a, b, c, d] = toFlat(n(seed, [4])).map((v) => 0.5 * v) as [number, number, number, number]
      const f = (t: number): number => a + b * (t / lookback) + c * (t / lookback) ** 2 + d * (t / lookback) ** 3
      return {
        window: Array.from({ length: lookback }, (_, t) => f(t)),
        future: Array.from({ length: horizon }, (_, t) => f(lookback + t)),
      }
    }
    const train = Array.from({ length: 120 }, (_, i) => make(300 + i))
    const x = tensor(train.map((s) => s.window))
    const y = tensor(train.map((s) => s.future))
    const fit = minimise(trend.params, (p) => mean(square(sub(trend.apply(p, x).forecast, y))), 500, 0.02)
    const test = Array.from({ length: 16 }, (_, i) => make(900 + i))
    const predicted = trend.apply(fit.params, tensor(test.map((s) => s.window))).forecast
    const err = toFlat(sub(predicted, tensor(test.map((s) => s.future)))).map(Math.abs)
    expect(fit.losses[0] as number).toBeGreaterThan(0.5)
    const naive = test.flatMap((s) => s.future.map((v) => Math.abs(v - (s.window.at(-1) as number))))
    const meanOf = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
    // An order of magnitude below repeating the last observation.
    expect(meanOf(err)).toBeLessThan(0.15 * meanOf(naive))
    expect(add(predicted, 0).shape).toEqual([16, 4])
  })

  it('validates its stacks and passes the conformance check', () => {
    expect(() => new NBeats(4, 2, { stacks: [] })).toThrow(/at least one stack/)
    const small = new NBeats(5, 2, {
      stacks: [{ basis: 'generic', blocks: 2, width: 4, layers: 1, size: 3 }],
      key: key(30),
    })
    expect(
      checkPrimitive((v: Value) => small.apply(small.params, v).forecast, [n(31, [2, 5])], { rtol: 1e-5 }).failures,
    ).toEqual([])
  })
})
