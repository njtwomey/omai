import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { eigvalsh } from '../numerics/linalg/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import {
  makeBiclusters,
  makeCheckerboard,
  makeSparseCodedSignal,
  makeSparseSpdMatrix,
  makeSpdMatrix,
  sCurve,
  severedSphere,
  swissRoll,
  swissRollEmbedding,
  torus,
} from './index.ts'

describe('manifolds', () => {
  it('Swiss roll points are the embedding of their latents, with noise on top', () => {
    const { data, truth } = generate(swissRoll, { key: key(0), knobs: { n: 200, noise: 0 } })
    const t = truth.latents?.['t'] as Tensor
    const h = truth.latents?.['height'] as Tensor
    expect(toFlat(swissRollEmbedding(t, h))).toEqual(toFlat(data.x))
    expect(toFlat(data.y)).toEqual(toFlat(t))
    const tt = toFlat(t)
    expect(tt.every((v) => v >= 1.5 * Math.PI && v <= 4.5 * Math.PI)).toBe(true)
    const noisy = generate(swissRoll, { key: key(0), knobs: { n: 2000, noise: 0.5 } })
    const clean = toFlat(noisy.truth.latents?.['clean'] as Tensor)
    const dev = toFlat(noisy.data.x).map((v, i) => v - (clean[i] ?? 0))
    expect(Math.sqrt(dev.reduce((a, b) => a + b * b, 0) / dev.length)).toBeCloseTo(0.5, 1)
  })
  it('the Swiss roll with a hole leaves its middle empty', () => {
    const { truth } = generate(swissRoll, { key: key(1), knobs: { n: 2000, hole: true } })
    const t = toFlat(truth.latents?.['t'] as Tensor)
    const h = toFlat(truth.latents?.['height'] as Tensor)
    const inHole = t.filter((v, i) => v > 2.5 * Math.PI && v < 3.5 * Math.PI && (h[i] ?? 0) > 7 && (h[i] ?? 0) < 14)
    expect(inHole.length).toBe(0)
  })
  it('S-curve, severed sphere and torus lie on their surfaces', () => {
    const s = generate(sCurve, { key: key(2), knobs: { n: 100 } })
    const x = toFlat(s.data.x)
    const t = toFlat(s.truth.latents?.['t'] as Tensor)
    t.forEach((v, i) => {
      expect(x[3 * i]).toBeCloseTo(Math.sin(v), 12)
    })
    const sphere = generate(severedSphere, { key: key(3), knobs: { n: 300 } })
    const p = toFlat(sphere.data.x)
    for (let i = 0; i < 300; i++)
      expect(Math.hypot(p[3 * i] ?? 0, p[3 * i + 1] ?? 0, p[3 * i + 2] ?? 0)).toBeCloseTo(1, 12)
    const theta = toFlat(sphere.truth.latents?.['theta'] as Tensor)
    expect(theta.every((v) => v > Math.PI / 8 && v < Math.PI - Math.PI / 8)).toBe(true)
    const tor = generate(torus, { key: key(4), knobs: { n: 300, majorRadius: 3, minorRadius: 1 } })
    const q = toFlat(tor.data.x)
    for (let i = 0; i < 300; i++) {
      const ring = Math.hypot(q[3 * i] ?? 0, q[3 * i + 1] ?? 0) - 3
      expect(Math.hypot(ring, q[3 * i + 2] ?? 0)).toBeCloseTo(1, 12)
    }
  })
})

describe('matrices', () => {
  it('sparse coded signals are exact sparse combinations of unit atoms', () => {
    const s = makeSparseCodedSignal(key(5), { nSamples: 10, nComponents: 8, nFeatures: 6, nNonzeroCoefs: 3 })
    const dict = toFlat(s.dictionary)
    for (let c = 0; c < 8; c++) {
      expect(Math.hypot(...dict.slice(c * 6, (c + 1) * 6))).toBeCloseTo(1, 12)
    }
    const code = toFlat(s.code)
    for (let i = 0; i < 10; i++) expect(code.slice(i * 8, (i + 1) * 8).filter((v) => v !== 0).length).toBe(3)
    expect(() =>
      makeSparseCodedSignal(key(0), { nSamples: 1, nComponents: 2, nFeatures: 2, nNonzeroCoefs: 3 }),
    ).toThrow()
  })
  it('SPD matrices have the stated eigenvalues', () => {
    const { x, eigenvalues } = makeSpdMatrix(key(6), 5)
    const ev = toFlat(eigvalsh(x)).sort((a, b) => b - a)
    ev.forEach((v, i) => {
      expect(v).toBeCloseTo(eigenvalues[i] ?? 0, 10)
    })
    const sparse = makeSparseSpdMatrix(key(7), 6, { alpha: 0.7, normDiag: true })
    expect(toFlat(eigvalsh(sparse.precision)).every((v) => v > 0)).toBe(true)
    const prec = toFlat(sparse.precision)
    for (let i = 0; i < 6; i++) expect(prec[i * 7]).toBe(1)
  })
  it('biclusters and checkerboards plant constant blocks', () => {
    const b = makeBiclusters(key(8), [12, 9], 3)
    const x = toFlat(b.x)
    for (let i = 0; i < 12; i++)
      for (let j = 0; j < 9; j++) {
        const same = b.rowLabels[i] === b.columnLabels[j]
        if (!same) expect(x[i * 9 + j]).toBe(0)
        else expect(x[i * 9 + j]).toBeGreaterThanOrEqual(10)
      }
    expect(b.rows.map((r) => r.filter(Boolean).length).reduce((a, c) => a + c, 0)).toBe(12)
    const c = makeCheckerboard(key(9), [8, 6], [2, 3], { shuffle: false })
    const y = toFlat(c.x)
    for (let i = 0; i < 8; i++)
      for (let j = 0; j < 6; j++) {
        const block = (c.rowLabels[i] ?? 0) * 3 + (c.columnLabels[j] ?? 0)
        const first = c.rows[block]?.indexOf(true) ?? 0
        const firstCol = c.columns[block]?.indexOf(true) ?? 0
        expect(y[i * 6 + j]).toBe(y[first * 6 + firstCol])
      }
  })
})
