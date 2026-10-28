import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder } from '../trace/index.ts'
import * as V from './index.ts'

const step = (n: number): number[][] =>
  Array.from({ length: n }, () => Array.from({ length: n }, (_, j) => (j < n / 2 ? 0.2 : 0.8)))

const noisy = (seed: number, n = 16, s = 0.05): number[][] => {
  const e = toFlat(normal(key(seed), [n * n]))
  return step(n).map((r, i) => r.map((v, j) => v + s * (e[i * n + j] ?? 0)))
}

const mse = (a: readonly number[], b: readonly number[]): number =>
  a.reduce((s, v, i) => s + (v - (b[i] ?? 0)) ** 2, 0) / a.length

describe('denoising', () => {
  it('reduces noise on a step edge while keeping constants', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1000 }), (seed) => {
        const x = noisy(seed)
        const clean = step(16).flat()
        const before = mse(x.flat(), clean)
        for (const y of [
          V.nonLocalMeans(x, { patchSize: 3, searchRadius: 3, h: 0.1 }),
          V.denoiseTvChambolle(x, { weight: 0.05 }),
          V.anisotropicDiffusion(x, { steps: 10, kappa: 0.1 }),
          V.guidedFilter(x, x, { radius: 2, eps: 0.01 }),
        ])
          expect(mse(toFlat(y), clean)).toBeLessThan(before)
      }),
      { numRuns: 5 },
    )
    const flat = Array.from({ length: 6 }, () => new Array<number>(6).fill(0.4))
    for (const y of [V.denoiseTvChambolle(flat), V.anisotropicDiffusion(flat), V.guidedFilter(flat, flat)])
      for (const v of toFlat(y)) expect(v).toBeCloseTo(0.4, 12)
    expect(() => V.anisotropicDiffusionAlgorithm(flat, { lambda: 0.5 })).toThrow()
  })

  it('inpaints holes harmonically and traces the diffusion', () => {
    fc.assert(
      fc.property(fc.integer({ min: 2, max: 6 }), (len) => {
        // A linear ramp is harmonic: a gap in it is filled exactly.
        const row = Array.from({ length: 10 }, (_, j) => j / 9)
        const mask = row.map((_, j) => (j >= 2 && j < 2 + len ? 1 : 0))
        const filled = toFlat(V.inpaint([row], [mask], { maxIter: 2000 }))
        for (const [j, v] of filled.entries()) expect(v).toBeCloseTo(row[j] ?? 0, 4)
      }),
      { numRuns: RUNS },
    )
    const tracer = recorder()
    V.anisotropicDiffusion(step(8), { steps: 4, tracer, policy: policies.minimal })
    expect(tracer.events.filter((e) => e.type === 'metric' && e.name === 'change').length).toBeGreaterThan(0)
  })
})

describe('pyramids and stereo', () => {
  it('reconstructs Laplacian pyramids exactly and blends across a seam', () => {
    fc.assert(
      fc.property(fc.integer({ min: 5, max: 20 }), fc.integer({ min: 5, max: 20 }), (h, w) => {
        const x = Array.from({ length: h }, (_, i) =>
          Array.from({ length: w }, (_, j) => Math.sin(i * 0.7) + Math.cos(j * 1.3)),
        )
        const back = toFlat(V.reconstructLaplacian(V.laplacianPyramid(x, 2)))
        for (const [k, v] of x.flat().entries()) expect(back[k]).toBeCloseTo(v, 10)
      }),
      { numRuns: RUNS },
    )
    const a = Array.from({ length: 16 }, () => new Array<number>(16).fill(1))
    const b = Array.from({ length: 16 }, () => new Array<number>(16).fill(0))
    const m = Array.from({ length: 16 }, () => Array.from({ length: 16 }, (_, j) => (j < 8 ? 1 : 0)))
    const blend = toFlat(V.pyramidBlend(a, b, m, 2))
    // The seam is smoothed at every scale: the row falls monotonically from (near) 1 to (near) 0.
    const row = blend.slice(16 * 8, 16 * 9)
    for (let j = 1; j < 16; j++) expect(row[j] ?? 0).toBeLessThanOrEqual((row[j - 1] ?? 0) + 1e-9)
    expect(row[0] ?? 0).toBeGreaterThan(0.7)
    expect(row[15] ?? 1).toBeLessThan(0.3)
  })

  it('recovers constant disparities and converts them to depth', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 5 }), (d) => {
        const tex = (x: number, y: number): number => Math.sin(x * 1.3) * Math.cos(y * 0.7) + Math.sin(x * 0.4 + y)
        const L = Array.from({ length: 10 }, (_, y) => Array.from({ length: 24 }, (_, x) => tex(x, y)))
        const R = Array.from({ length: 10 }, (_, y) => Array.from({ length: 24 }, (_, x) => tex(x + d, y)))
        for (const disp of [
          V.blockMatching(L, R, { numDisparities: 8 }),
          V.blockMatching(L, R, { numDisparities: 8, cost: 'ncc', lrCheck: false }),
          V.semiGlobalMatching(L, R, { numDisparities: 8, paths: 4 }),
        ]) {
          const v = toFlat(disp)
          expect(Math.round(v[5 * 24 + 14] ?? -1)).toBe(d)
        }
      }),
      { numRuns: 5 },
    )
    expect(toFlat(V.disparityToDepth([[0, 10]], 50, 0.2))).toEqual([0, 1])
  })
})
