import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import {
  expandGroups,
  gridSuperpixels,
  maskedPredictor,
  slicSuperpixels,
  timeWindows,
  tokenise,
  tokenPredictor,
} from './modalities.ts'
import { featureAblation, morrisEffects, occlusion, rise } from './perturbation.ts'
import { kernelShap } from './shapley.ts'
import { limeMasked } from './surrogates.ts'

const linear =
  (w: readonly number[], c = 0) =>
  (x: Tensor): Tensor => {
    const d = w.length
    const n = x.shape[0] ?? 0
    const v = toFlat(x)
    return fromStorage(
      Float64Array.from({ length: n }, (_, i) => w.reduce((s, wj, j) => s + wj * (v[i * d + j] ?? 0), c)),
      [n],
      'float64',
    )
  }

describe('perturbation attributions', () => {
  it('ablation of a linear model is w (x - baseline), groups summing their members', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: -5, max: 5 }), { minLength: 4, maxLength: 4 }),
        fc.array(fc.integer({ min: -5, max: 5 }), { minLength: 4, maxLength: 4 }),
        (w, x) => {
          const f = linear(w, 1)
          const a = featureAblation(f, [x], { baseline: 1 })[0] ?? []
          for (const [j, v] of a.entries()) expect(v).toBeCloseTo((w[j] ?? 0) * ((x[j] ?? 0) - 1), 10)
          const g = featureAblation(f, [x], { baseline: 1, featureMask: [0, 0, 1, 1] })[0] ?? []
          expect(g[0]).toBeCloseTo((a[0] ?? 0) + (a[1] ?? 0), 10)
          expect(g[1]).toBe(g[0])
          // Occlusion by single elements is ablation.
          const o = occlusion(f, [x], { shape: [4], window: [1], baseline: 1 })[0] ?? []
          for (const [j, v] of o.entries()) expect(v).toBeCloseTo(a[j] ?? 0, 10)
        },
      ),
      { numRuns: 30 },
    )
  })

  it('occlusion of an image averages over the windows covering each pixel', () => {
    const f = linear([1, 1, 1, 1])
    const o = occlusion(f, [[1, 2, 3, 4]], { shape: [2, 2], window: [2, 1] })[0] ?? []
    // Columns: windows remove 1 + 3 and 2 + 4.
    expect(o).toEqual([4, 6, 4, 6])
    expect(() => occlusion(f, [[1, 2, 3, 4]], { shape: [3, 2], window: [1, 1] })).toThrow(/does not hold/)
  })

  it('RISE is reproducible and finds the pixel the model reads', () => {
    const f = (x: Tensor): Tensor => {
      const n = x.shape[0] ?? 0
      const v = toFlat(x)
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => v[i * 36 + 21] ?? 0),
        [n],
        'float64',
      )
    }
    const x = [new Array(36).fill(1)]
    const s = rise(f, x, { shape: [6, 6], nMasks: 300, cells: 3, key: key(4) })[0] ?? []
    expect(s.indexOf(Math.max(...s))).toBe(21)
    expect(rise(f, x, { shape: [6, 6], nMasks: 20, key: key(4) })).toEqual(
      rise(f, x, { shape: [6, 6], nMasks: 20, key: key(4) }),
    )
  })

  it('Morris effects are the scaled coefficients of a linear model and spread for a product', () => {
    const r = morrisEffects(
      linear([2, -1, 0]),
      [
        [0, 1],
        [0, 3],
        [5, 6],
      ],
      { nTrajectories: 6, key: key(1) },
    )
    expect(r.mu[0]).toBeCloseTo(2, 10)
    expect(r.mu[1]).toBeCloseTo(-3, 10)
    expect(r.muStar[2]).toBeCloseTo(0, 12)
    expect(r.sigma[0]).toBeCloseTo(0, 10)
    const product = (x: Tensor): Tensor => {
      const v = toFlat(x)
      const n = x.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => (v[2 * i] ?? 0) * (v[2 * i + 1] ?? 0)),
        [n],
        'float64',
      )
    }
    const p = morrisEffects(
      product,
      [
        [0, 1],
        [0, 1],
      ],
      { nTrajectories: 30, key: key(2) },
    )
    expect(p.sigma[0] ?? 0).toBeGreaterThan(0.1)
  })
})

describe('modality helpers', () => {
  it('builds superpixels and windows that cover every feature', () => {
    const grid = gridSuperpixels([5, 5], 2)
    expect(new Set(grid).size).toBe(9)
    const image = Array.from({ length: 64 }, (_, p) => (p % 8 < 4 ? 0.1 : 0.9))
    const slic = slicSuperpixels(image, [8, 8], { nSegments: 4 })
    expect(slic.length).toBe(64)
    // No segment straddles the edge between the halves.
    for (const s of new Set(slic)) {
      const cols = slic.flatMap((v, p) => (v === s ? [p % 8 < 4] : []))
      expect(new Set(cols).size).toBe(1)
    }
    expect(timeWindows(7, 3)).toEqual([0, 0, 0, 1, 1, 1, 2])
  })

  it('Shapley values over groups add up from the baseline to the input', () => {
    const f = linear([1, 2, 3, 4, 5, 6])
    const x = [1, 1, 1, 1, 1, 1]
    const { predict, groups } = maskedPredictor(f, x, timeWindows(6, 2))
    const shap = kernelShap(predict, [[1, 1, 1]], [[0, 0, 0]])
    expect(shap.values[0]).toEqual([3, 7, 11].map((v) => expect.closeTo(v, 9)))
    expect(expandGroups(shap.values[0] ?? [], groups, 6)[5]).toBeCloseTo(11, 9)
  })

  it('LIME explains a text through its tokens', () => {
    const { tokens } = tokenise('a good film, not a bad one')
    const score = (texts: readonly string[]) =>
      texts.map((t) => (t.includes('good') ? 1 : 0) - (t.includes('bad') ? 0.5 : 0))
    const r = limeMasked(tokenPredictor(tokens, score), tokens.length, { nSamples: 400, key: key(3), nFeatures: 2 })
    const top = r.weights.map(([j]) => tokens[j])
    expect(top).toContain('good')
    expect(r.weights.find(([j]) => tokens[j] === 'bad')?.[1] ?? 0).toBeLessThan(0)
  })
})
