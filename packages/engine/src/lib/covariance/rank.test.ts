import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { gaussianGraph } from '../datasets/index.ts'
import { linalg } from '../numerics/index.ts'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { edgeRecovery, precisionGraph } from './evaluation.ts'
import { graphicalLasso } from './graphical-lasso.ts'
import {
  glassoNonparanormal,
  glassoNonparanormalAlgorithm,
  nonparanormalCorrelation,
  nonparanormalTransform,
  winsorisationLevel,
} from './rank.ts'

const sample = (marginal: 'gaussian' | 'cubic' | 'exp', n: number, p = 6, seed = 1) => {
  const { data, truth } = generate(gaussianGraph, {
    key: key(seed),
    knobs: { n, p, structure: 'band', strength: 0.4, marginal },
  })
  return {
    x: toArray(data.x) as number[][],
    latent: toArray(data.y) as number[][],
    truth: truth.extra?.['covariance'] as number[][],
    adjacency: truth.extra?.['adjacency'] as boolean[][],
  }
}

describe('the nonparanormal transform', () => {
  it('gives each column zero mean and unit variance, or its own mean and spread with scale: original', () => {
    const { x } = sample('exp', 80, 3)
    const z = toArray(nonparanormalTransform(x)) as number[][]
    for (let j = 0; j < 3; j++) {
      const col = z.map((r) => r[j] ?? 0)
      const mean = col.reduce((a, v) => a + v, 0) / col.length
      expect(mean).toBeCloseTo(0, 12)
      expect(col.reduce((a, v) => a + (v - mean) ** 2, 0) / col.length).toBeCloseTo(1, 12)
    }
    const kept = toArray(nonparanormalTransform(x, { scale: 'original' })) as number[][]
    const raw = x.map((r) => r[0] ?? 0)
    const m = raw.reduce((a, v) => a + v, 0) / raw.length
    expect(kept.reduce((a, r) => a + (r[0] ?? 0), 0) / kept.length).toBeCloseTo(m, 10)
  })

  it('is monotone in each variable, gives ties equal scores and leaves Gaussian data nearly unchanged', () => {
    const z = toFlat(nonparanormalTransform([[3], [1], [2], [2], [5], [4]], { winsorise: false }))
    expect(z[2]).toBe(z[3])
    expect(z[1] ?? 0).toBeLessThan(z[2] ?? 0)
    expect(z[2] ?? 0).toBeLessThan(z[0] ?? 0)
    expect(z[0] ?? 0).toBeLessThan(z[5] ?? 0)
    const { x, latent } = sample('gaussian', 2000, 2)
    const t = toArray(nonparanormalTransform(x)) as number[][]
    const dev = t.reduce((a, r, i) => a + Math.abs((r[0] ?? 0) - (latent[i]?.[0] ?? 0)), 0) / t.length
    expect(dev).toBeLessThan(0.06)
  })

  it('maps a monotone transform of a Gaussian back to (almost) the Gaussian', () => {
    const { x, latent } = sample('exp', 4000, 2)
    const t = toArray(nonparanormalTransform(x)) as number[][]
    const dev = t.reduce((a, r, i) => a + Math.abs((r[1] ?? 0) - (latent[i]?.[1] ?? 0)), 0) / t.length
    expect(dev).toBeLessThan(0.06)
  })

  it('truncates at the level of Liu, Lafferty and Wasserman', () => {
    expect(winsorisationLevel(100)).toBeCloseTo(1 / (4 * 100 ** 0.25 * Math.sqrt(Math.PI * Math.log(100))), 14)
    const wide = toFlat(
      nonparanormalTransform(
        Array.from({ length: 50 }, (_, i) => [i]),
        { scale: 'original' },
      ),
    )
    const shrunk = toFlat(
      nonparanormalTransform(
        Array.from({ length: 50 }, (_, i) => [i]),
        { winsorise: false, scale: 'original' },
      ),
    )
    expect(wide.length).toBe(shrunk.length)
    expect(() => nonparanormalTransform([[1], [2]])).toThrow(/at least 3/)
  })
})

describe('the nonparanormal correlation', () => {
  it('is consistent for the latent correlation under every estimator, also for heavy-tailed or skewed marginals', () => {
    for (const marginal of ['gaussian', 'cubic', 'exp'] as const) {
      const { x, truth } = sample(marginal, 3000)
      for (const method of ['spearman', 'kendall', 'normal-scores'] as const) {
        const r = nonparanormalCorrelation(x, { method })
        for (let a = 0; a < 6; a++)
          for (let b = 0; b < 6; b++) expect(Math.abs((r[a]?.[b] ?? 0) - (truth[a]?.[b] ?? 0))).toBeLessThan(0.06)
      }
    }
  })

  it('beats the Pearson correlation of the observed variables when the marginals are heavy tailed', () => {
    const { x, truth } = sample('cubic', 1500)
    const spearman = nonparanormalCorrelation(x)
    const err = (m: number[][]): number =>
      m.reduce((s, r, a) => s + r.reduce((t, v, b) => t + Math.abs(v - (truth[a]?.[b] ?? 0)), 0), 0)
    const centred = x.map((r) => r.map((v, j) => v - x.reduce((a, q) => a + (q[j] ?? 0), 0) / x.length))
    const pearson = centred[0]?.map((_, a) =>
      centred[0]?.map((__, b) => {
        const dot = (i: number, j: number): number => centred.reduce((s, r) => s + (r[i] ?? 0) * (r[j] ?? 0), 0)
        return dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b))
      }),
    ) as number[][]
    expect(err(spearman)).toBeLessThan(err(pearson))
  })

  it('has a unit diagonal and is repaired to be positive definite when the sine transform is not', () => {
    const x = [
      [1, 2, 1],
      [2, 1, 2],
      [3, 3, 1],
      [4, 5, 5],
      [5, 4, 3],
    ]
    const raw = nonparanormalCorrelation(x, { method: 'kendall', positiveDefinite: false })
    const fixed = nonparanormalCorrelation(x, { method: 'kendall' })
    for (let a = 0; a < 3; a++) expect(fixed[a]?.[a]).toBe(1)
    expect(Math.min(...toFlat(linalg.eigvalsh(fixed)))).toBeGreaterThan(1e-7)
    expect(Math.min(...toFlat(linalg.eigvalsh(raw)))).toBeLessThan(Math.min(...toFlat(linalg.eigvalsh(fixed))) + 1e-9)
    expect(() => nonparanormalCorrelation(x, { method: 'pearson' as 'kendall' })).toThrow(/unknown method/)
  })

  it('is the Pearson correlation of the normal scores for method normal-scores', () => {
    const { x } = sample('exp', 200, 3)
    const direct = nonparanormalCorrelation(x, { method: 'normal-scores' })
    const z = toArray(nonparanormalTransform(x)) as number[][]
    const dot = (i: number, j: number): number => z.reduce((s, r) => s + (r[i] ?? 0) * (r[j] ?? 0), 0) / z.length
    expect(direct[0]?.[1]).toBeCloseTo(dot(0, 1), 12)
  })
})

describe('the nonparanormal graphical lasso', () => {
  it('is the graphical lasso of the rank correlation, and finds the graph of skewed data', () => {
    const { x, adjacency } = sample('exp', 800, 6)
    const out = glassoNonparanormal(x, 0.05, { tol: 1e-8 })
    const direct = graphicalLasso(nonparanormalCorrelation(x), 0.05, { tol: 1e-8 })
    expect(out.precision).toEqual(direct.precision)
    expect(out.correlation.length).toBe(6)
    const found = edgeRecovery(precisionGraph(out.precision), adjacency)
    expect(found.recall).toBe(1)
    expect(found.f1).toBeGreaterThan(0.7)
  })

  it('is the graphical lasso algorithm of the same correlation, traced', () => {
    const { x } = sample('gaussian', 100, 3)
    expect(glassoNonparanormalAlgorithm(x, 0.1, { method: 'kendall' }).name).toBe('graphical-lasso')
  })
})
