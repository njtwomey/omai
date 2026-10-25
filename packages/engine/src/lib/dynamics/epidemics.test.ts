import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { epidemicAlgorithm, seirCounts, seirCurves } from './epidemics.ts'

describe('SEIR equations', () => {
  it('conserve the population, and spread only when R0 > 1', () => {
    const big = seirCurves({ beta: 0.6, sigma: 0.25, gamma: 0.15 }, { days: 200 })
    for (let k = 0; k < big.t.length; k += 20)
      expect((big.s[k] ?? 0) + (big.e[k] ?? 0) + (big.i[k] ?? 0) + (big.r[k] ?? 0)).toBeCloseTo(1, 7)
    expect(big.r.at(-1)).toBeGreaterThan(0.9)
    const small = seirCurves({ beta: 0.1, sigma: 0.25, gamma: 0.15 }, { days: 200 })
    expect(Math.max(...small.i)).toBeLessThanOrEqual(0.0100001)
    // The final size solves s = exp(-R0 (1 - s)) for a well-mixed epidemic seeded small.
    const sInf = big.s.at(-1) ?? 0
    expect(sInf).toBeCloseTo(Math.exp(-big.r0 * (1 - sInf)), 2)
  })
})

describe('epidemic agents', () => {
  const run = (o: Parameters<typeof epidemicAlgorithm>[0], steps = 600) =>
    runAlgorithm(epidemicAlgorithm({ people: 200, radius: 4, ...o }), { key: key(2), maxSteps: steps }).state

  it('keeps everyone in the room, and nobody is lost between states', () => {
    const s = run({})
    expect(seirCounts(s).reduce((a, b) => a + b, 0)).toBe(200)
    for (let i = 0; i < 200; i++) {
      expect(s.x[i]).toBeGreaterThanOrEqual(0)
      expect(s.x[i]).toBeLessThanOrEqual(100)
    }
  })

  it('flattens the curve when most people stay put', () => {
    const free = seirCounts(run({}))
    const distanced = seirCounts(run({ distancing: 0.8 }))
    expect(distanced[0]).toBeGreaterThan(free[0])
    expect(() => epidemicAlgorithm({ distancing: 2 })).toThrow()
  })
})
