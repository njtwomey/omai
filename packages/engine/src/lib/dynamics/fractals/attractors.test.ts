import { describe, expect, it } from 'vitest'
import { jacfwd } from '../../autodiff/index.ts'
import { mul, reshape, tensor, toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { fieldDivergence } from '../fields.ts'
import { cliffordMap, deJongMap, henonMap, ikedaMap, lyapunovExponents } from '../maps.ts'
import { rosslerField, thomasField } from '../named-systems.ts'
import * as f from './index.ts'

describe('planar maps for attractors', () => {
  it('give the hand-worked images of the origin and of a second point', () => {
    expect(toFlat(deJongMap()(tensor([0, 0])))).toEqual([-1, -1])
    expect(toFlat(cliffordMap()(tensor([0, 0])))).toEqual([1, 0.7])
    expect(toFlat(ikedaMap()(tensor([0, 0])))).toEqual([1, 0])
    const [x, y] = toFlat(deJongMap(1, 2, 3, 4)(tensor([0.5, 0.25]))) as [number, number]
    expect(x).toBeCloseTo(Math.sin(0.25) - Math.cos(1), 14)
    expect(y).toBeCloseTo(Math.sin(1.5) - Math.cos(1), 14)
  })
  it('the Ikeda map contracts area by exactly u^2 everywhere', () => {
    for (const u of [0.6, 0.9]) {
      const J = toFlat(jacfwd((s) => reshape(ikedaMap(u)(s), [2]))(tensor([0.7, -0.4])))
      expect((J[0] ?? 0) * (J[3] ?? 0) - (J[1] ?? 0) * (J[2] ?? 0)).toBeCloseTo(u * u, 12)
    }
  })
  it('the Henon cloud has a positive Lyapunov exponent: it is chaotic, and stays bounded', () => {
    const s = runAlgorithm(f.mapCloud(henonMap(), { batch: 200 }), { seed: 0, maxSteps: 10 }).state
    expect(s.cloud.length).toBe(4000)
    for (let i = 0; i < s.cloud.length; i += 2) {
      expect(Math.abs(s.cloud[i] ?? 9)).toBeLessThan(1.5)
      expect(Math.abs(s.cloud[i + 1] ?? 9)).toBeLessThan(0.5)
    }
    expect(lyapunovExponents(henonMap(), [0.1, 0.1], { n: 3000, count: 1 })[0]).toBeGreaterThan(0.3)
  })
  it('the Henon attractor has a box-counting dimension near the known 1.26', () => {
    const s = runAlgorithm(f.mapCloud(henonMap(), { batch: 2000 }), { seed: 0, maxSteps: 10 }).state
    const d = f.boxCountingDimension(s.cloud, { minLevel: 3, maxLevel: 7 }).dimension
    expect(d).toBeGreaterThan(1.1)
    expect(d).toBeLessThan(1.45)
  })
  it('the cloud algorithm stops, as diverged, when the orbit escapes', () => {
    const run = runAlgorithm(
      f.mapCloud((x) => mul(x, 10), { start: [1], batch: 400 }),
      { seed: 0, maxSteps: 5 },
    )
    expect(run.status).toBe('diverged')
    expect(run.state.diverged).toBe(true)
  })
  it('counts the switches between the lobes of a series', () => {
    expect(f.lobeSwitches([1, 2, -1, -3, 4, 5, -2])).toBe(3)
    expect(f.lobeSwitches([1, 1, 1])).toBe(0)
  })
})

describe('flows for attractors', () => {
  it("Rossler's divergence is a + x - c and Thomas' is the constant -3b", () => {
    expect(fieldDivergence(rosslerField(), [2, 0.5, 1])).toBeCloseTo(0.2 + 2 - 5.7, 9)
    expect(fieldDivergence(thomasField({ b: 0.3 }), [1, -2, 0.5])).toBeCloseTo(-0.9, 9)
  })
  it('Thomas has the origin as an equilibrium', () => {
    expect(toFlat(thomasField()(0, tensor([0, 0, 0])))).toEqual([0, 0, 0])
  })
})
