import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { kalmanEm } from './em.ts'

function simulate(T: number, q: number, r: number, seed: number): number[][] {
  const e = toFlat(normal(key(seed), [T, 2]))
  let x = 0
  return Array.from({ length: T }, (_, t) => {
    x = 0.9 * x + Math.sqrt(q) * (e[2 * t] ?? 0)
    return [x + Math.sqrt(r) * (e[2 * t + 1] ?? 0)]
  })
}

describe('EM for linear-Gaussian state-space models', () => {
  it('never decreases the log-likelihood and recovers the noise variances', () => {
    const zs = simulate(150, 0.5, 0.2, 11)
    const r = kalmanEm({ F: [[0.5]], H: [[1]], Q: [[1]], R: [[1]], x0: [0], P0: [[1]] }, zs, {
      maxIterations: 80,
      learn: { H: false },
      tol: 1e-10,
    })
    r.history.forEach((v, i) => {
      if (i > 0) expect(v).toBeGreaterThanOrEqual((r.history[i - 1] ?? 0) - 1e-8)
    })
    expect((r.history.at(-1) ?? 0) - (r.history[0] ?? 0)).toBeGreaterThan(10)
    // F moves from 0.5 towards the true 0.9 and the total noise variance is recovered.
    expect(r.model.F.data[0] ?? 0).toBeGreaterThan(0.75)
    const q = r.model.Q.data[0] ?? 0
    const rr = r.model.R.data[0] ?? 0
    expect(Math.abs(q / (1 - 0.9 ** 2) + rr - (0.5 / (1 - 0.81) + 0.2))).toBeLessThan(1.2)
  })

  it('rejects missing observations', () => {
    expect(() =>
      kalmanEm({ F: [[1]], H: [[1]], Q: [[1]], R: [[1]], x0: [0], P0: [[1]] }, [[1], [Number.NaN]]),
    ).toThrow()
  })
})
