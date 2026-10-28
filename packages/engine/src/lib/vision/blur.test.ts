import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

describe('smoothing filters', () => {
  it('builds normalised, symmetric Gaussian kernels', () => {
    for (const [k, s] of [
      [3, 0],
      [9, 0],
      [11, 0],
      [6, 1.1],
    ] as const) {
      const g = Array.from(V.getGaussianKernel(k, s))
      expect(g.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14)
      for (let i = 0; i < k; i++) expect(g[i]).toBeCloseTo(g[k - 1 - i] ?? 0, 15)
    }
    expect(() => V.getGaussianKernel(0)).toThrow()
  })

  it('preserves constants and the mean of every channel under the box and Gaussian filters', () => {
    const x = Array.from({ length: 6 }, () => Array.from({ length: 5 }, () => [0.25, 0.5, 0.75]))
    for (const y of [V.blur(x, [3, 3]), V.gaussianBlur(x, [5, 5], 1), V.medianBlur(x, 3)])
      expect(toFlat(y).map((v) => Math.round(v * 1e12) / 1e12)).toEqual(x.flat(2))
    expect(V.gaussianBlur(x, [0, 0], 1).shape).toEqual([6, 5, 3])
    expect(() => V.gaussianBlur(x, [4, 4], 1)).toThrow()
    expect(() => V.medianBlur(x, 4)).toThrow()
  })

  it('correlates with a kernel under each border', () => {
    expect(toFlat(V.filter2D([[1, 2, 3]], [[0, 1, 0]]))).toEqual([1, 2, 3])
    expect(toFlat(V.filter2D([[1, 2, 3]], [[1, 0, 0]], { border: 'constant' }))).toEqual([0, 1, 2])
    expect(V.borderMode()).toBe('mirror')
  })
})
