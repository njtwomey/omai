import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key, normal } from '../../lib/random/index.ts'
import { linearGaussianStateSpace } from './latent-gaussian.ts'

// A random-walk level (variance 0.1) seen through noise (variance 1).
const steps = Array.from(normal(key(1), [120]).data)
const noise = Array.from(normal(key(2), [120]).data)
const level: number[] = []
for (const [t, s] of steps.entries()) level.push((level[t - 1] ?? 0) + Math.sqrt(0.1) * s)
const series = level.map((l, t) => l + (noise[t] ?? 0))

describe('linear-Gaussian latent models', () => {
  it('a local level smooths the noise away and EM never lowers the likelihood', () => {
    const r = linearGaussianStateSpace(series)
    const err = (m: readonly number[]) => m.reduce((a, v, t) => a + (v - (level[t] ?? 0)) ** 2, 0) / m.length
    expect(err(r.smoothed.mean)).toBeLessThan(err(series) / 2)
    expect(err(r.smoothed.mean)).toBeLessThanOrEqual(err(r.filtered.mean) + 1e-9)
    for (let i = 1; i < r.history.length; i++)
      expect(r.history[i] ?? 0).toBeGreaterThanOrEqual((r.history[i - 1] ?? 0) - 1e-6)
    const grids = recordsOf(r.exposition, 'posteriorGrid')
    expect(grids.length).toBe(2)
    expect(grids[0]?.observed?.targets.length).toBe(120)
    expect(recordsOf(r.exposition, 'coefficientTable')[0]?.names).toEqual(['observation variance', 'level variance'])
  })

  it('a local linear trend runs with missing values', () => {
    const gappy = series.map((v, t) => (t % 10 === 5 ? Number.NaN : v))
    const r = linearGaussianStateSpace(gappy, { model: 'local-linear-trend', maxIter: 10 })
    expect(r.smoothed.mean.every(Number.isFinite)).toBe(true)
    expect(r.model.Q.length).toBe(2)
  })
})
