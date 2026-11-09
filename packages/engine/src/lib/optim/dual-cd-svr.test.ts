import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { type DualCdSvrLoss, linearSvrDualCd, linearSvrDualCdAlgorithm } from './dual-cd-svr.ts'

const data = (seed: number, n: number): { x: number[][]; y: number[] } => {
  const z = toFlat(normal(key(seed), [n * 3]))
  const x = Array.from({ length: n }, (_, i) => [z[i * 3] ?? 0, z[i * 3 + 1] ?? 0])
  return { x, y: x.map((r, i) => 3 * (r[0] ?? 0) - 2 * (r[1] ?? 0) + 0.5 + 0.2 * (z[i * 3 + 2] ?? 0)) }
}

describe('linearSvrDualCd', () => {
  for (const loss of ['l1', 'l2'] as DualCdSvrLoss[])
    it(`${loss}: the gap closes from above, the dual rises, and w is the sum of beta x`, () => {
      const { x, y } = data(1, 80)
      const alg = linearSvrDualCdAlgorithm(x, y, 5, { loss, epsilon: 0.1, tol: 1e-8 })
      let s = alg.init(key(0))
      let lastDual = s.dual
      for (let t = 0; t < 3000 && !s.converged; t++) {
        s = alg.step(s, { t, key: key(t + 1) } as never)
        expect(s.primal - s.dual).toBeGreaterThanOrEqual(-1e-9)
        expect(s.dual).toBeGreaterThanOrEqual(lastDual - 1e-9)
        lastDual = s.dual
      }
      expect(s.converged).toBe(true)
      expect(s.primal - s.dual).toBeLessThan(1e-6)
      const w = [0, 0, 0]
      for (const [i, b] of Array.from(s.beta).entries()) {
        w[0] = (w[0] ?? 0) + b * (x[i]?.[0] ?? 0)
        w[1] = (w[1] ?? 0) + b * (x[i]?.[1] ?? 0)
        w[2] = (w[2] ?? 0) + b
      }
      for (const [j, v] of w.entries()) expect(v).toBeCloseTo(s.w[j] ?? 0, 9)
      if (loss === 'l1') for (const b of s.beta) expect(Math.abs(b)).toBeLessThanOrEqual(5 + 1e-12)
    })

  it('recovers the planted line, and a tube wider than the targets gives zero weights', () => {
    const { x, y } = data(2, 200)
    const r = linearSvrDualCd(x, y, 1000, { loss: 'l2', epsilon: 0, tol: 1e-9, maxSweeps: 5000 })
    expect(Math.abs((r.coef[0] ?? 0) - 3)).toBeLessThan(0.15)
    expect(Math.abs((r.coef[1] ?? 0) + 2)).toBeLessThan(0.15)
    expect(Math.abs(r.intercept - 0.5)).toBeLessThan(0.15)
    const wide = linearSvrDualCd(x, y, 1, { epsilon: 100, tol: 1e-12 })
    expect(Array.from(wide.coef).every((v) => v === 0)).toBe(true)
    expect(wide.converged).toBe(true)
  })

  it('row weights multiply C: weight 2 equals a repeated row; resuming from a checkpoint reproduces the run', () => {
    const { x, y } = data(3, 30)
    const w = y.map((_, i) => (i % 4 === 0 ? 2 : 1))
    const opts = { loss: 'l2', epsilon: 0.05, tol: 1e-11, maxSweeps: 5000 } as const
    const weighted = linearSvrDualCd(x, y, 2, { ...opts, weights: w })
    const repeated = linearSvrDualCd(
      x.flatMap((row, i) => (i % 4 === 0 ? [row, row] : [row])),
      y.flatMap((v, i) => (i % 4 === 0 ? [v, v] : [v])),
      2,
      opts,
    )
    for (const [j, v] of Array.from(weighted.coef).entries()) expect(v).toBeCloseTo(repeated.coef[j] ?? 0, 6)
    const alg = linearSvrDualCdAlgorithm(x, y, 2, { tol: 1e-8 })
    const full = runAlgorithm(alg, { maxSteps: 30, key: key(7) })
    const half = runAlgorithm(alg, { maxSteps: 4, key: key(7) })
    const restored = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(half.state))))
    const rest = runAlgorithm(alg, { maxSteps: 26, key: key(7), start: { state: restored, step: 4 } })
    expect(Array.from(rest.state.w)).toEqual(Array.from(full.state.w))
  })

  it('names its errors', () => {
    const { x, y } = data(4, 10)
    expect(() => linearSvrDualCd(x, y, 0)).toThrow(/C must be positive/)
    expect(() => linearSvrDualCd(x, y, 1, { epsilon: -1 })).toThrow(/epsilon/)
    expect(() => linearSvrDualCd(x, [1, 2], 1)).toThrow(/differ/)
    expect(() => linearSvrDualCd(x, y, 1, { weights: y.map(() => 0) })).toThrow(/positive/)
  })
})
