import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { armax, arx, dmd, edmd, n4sid, polynomialLibrary, stlsq } from './identification.ts'

function excitation(T: number, seed: number): number[] {
  return toFlat(normal(key(seed), [T]))
}

describe('system identification', () => {
  it('ARX recovers the coefficients of noise-free data (property)', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -0.9, max: 0.9, noNaN: true }),
        fc.double({ min: -0.5, max: 0.5, noNaN: true }),
        fc.double({ min: -2, max: 2, noNaN: true }),
        fc.double({ min: -2, max: 2, noNaN: true }),
        (a1, a2, b1, b2) => {
          // Keep the AR part stable: roots of z^2 + a1 z + a2 inside the unit circle.
          fc.pre(Math.abs(a2) < 1 && Math.abs(a1) < 1 + a2)
          // Identifiable only when the model is minimal: B nonzero and no root shared with A.
          const root = Math.abs(b1) > 0.1 ? -b2 / b1 : Number.NaN
          fc.pre(Number.isFinite(root) && Math.abs(root * root + a1 * root + a2) > 0.05)
          const u = excitation(60, 3)
          const y = [0, 0]
          for (let t = 2; t < 60; t++) {
            y.push(-a1 * (y[t - 1] ?? 0) - a2 * (y[t - 2] ?? 0) + b1 * (u[t - 1] ?? 0) + b2 * (u[t - 2] ?? 0))
          }
          const m = arx(y, u, { na: 2, nb: 2 })
          return [...m.a, ...m.b].every((v, i) => Math.abs(v - ([a1, a2, b1, b2][i] ?? 0)) < 1e-8)
        },
      ),
      { numRuns: 25 },
    )
    expect(() => arx([1, 2], [1], { na: 1, nb: 1 })).toThrow()
  })

  it('ARMAX removes the bias ARX has under coloured noise', () => {
    const T = 300
    const u = excitation(T, 1)
    const e = excitation(T, 2).map((v) => 0.3 * v)
    const y = [0]
    for (let t = 1; t < T; t++) y.push(0.7 * (y[t - 1] ?? 0) + (u[t - 1] ?? 0) + (e[t] ?? 0) + 0.8 * (e[t - 1] ?? 0))
    const m = armax(y, u, { na: 1, nb: 1, nc: 1 })
    expect(Math.abs((m.a[0] ?? 0) + 0.7)).toBeLessThan(0.05)
    expect(Math.abs((m.b[0] ?? 0) - 1)).toBeLessThan(0.05)
    expect(Math.abs((m.c[0] ?? 0) - 0.8)).toBeLessThan(0.15)
    expect(Math.abs(m.noiseVariance - 0.09)).toBeLessThan(0.03)
  })

  it('N4SID recovers the poles and the input-output map of a second-order system', () => {
    const T = 300
    const u = excitation(T, 4)
    // Poles 0.9 and 0.5, x = [x1, x2], y = x1 + x2.
    let x1 = 0
    let x2 = 0
    const y: number[] = []
    for (let t = 0; t < T; t++) {
      y.push(x1 + x2)
      x1 = 0.9 * x1 + (u[t] ?? 0)
      x2 = 0.5 * x2 - 0.5 * (u[t] ?? 0)
    }
    const r = n4sid(y, u, { order: 2 })
    const a = toFlat(r.A)
    const tr = (a[0] ?? 0) + (a[3] ?? 0)
    const det = (a[0] ?? 0) * (a[3] ?? 0) - (a[1] ?? 0) * (a[2] ?? 0)
    expect(Math.abs(tr - 1.4)).toBeLessThan(1e-6)
    expect(Math.abs(det - 0.45)).toBeLessThan(1e-6)
    // Markov parameters D and C B (similarity invariants): 0 and 1 - 0.5.
    const cb = toFlat(r.C).reduce((acc, c, i) => acc + c * (toFlat(r.B)[i] ?? 0), 0)
    expect(Math.abs(toFlat(r.D)[0] ?? 1)).toBeLessThan(1e-6)
    expect(Math.abs(cb - 0.5)).toBeLessThan(1e-6)
    expect((r.singularValues[2] ?? 1) / (r.singularValues[0] ?? 1)).toBeLessThan(1e-8)
  })

  it('DMD amplitudes and modes reconstruct the first snapshot', () => {
    const cols: number[][] = []
    for (let k = 0; k < 8; k++) cols.push([0.9 ** k + 0.5 ** k, 0.9 ** k - 0.5 ** k, 2 * 0.9 ** k])
    const X = [0, 1, 2].map((i) => cols.map((c) => c[i] ?? 0))
    const r = dmd(X)
    const n = 3
    const rank = r.amplitudes.re.length
    for (let i = 0; i < n; i++) {
      let v = 0
      for (let j = 0; j < rank; j++) {
        v +=
          (r.modes.re.data[i * rank + j] ?? 0) * (r.amplitudes.re[j] ?? 0) -
          (r.modes.im.data[i * rank + j] ?? 0) * (r.amplitudes.im[j] ?? 0)
      }
      expect(Math.abs(v - (X[i]?.[0] ?? 0))).toBeLessThan(1e-10)
    }
  })

  it('EDMD finds the Koopman eigenvalues of a polynomial map', () => {
    // x' = 0.8 x, y' = 0.5 y + x^2 is closed on {x, y, x^2}: eigenvalues 0.8, 0.5, 0.64.
    const pts = toFlat(normal(key(9), [30, 2]))
    const xs = Array.from({ length: 30 }, (_, i) => [pts[2 * i] ?? 0, pts[2 * i + 1] ?? 0])
    const ys = xs.map(([a, b]) => [0.8 * (a ?? 0), 0.5 * (b ?? 0) + (a ?? 0) ** 2])
    const lib = polynomialLibrary(2, 2)
    const ev = edmd(xs, ys, lib.features).eigenvalues.re.map((v) => Math.round(v * 1e8) / 1e8)
    for (const target of [1, 0.8, 0.5, 0.64]) expect(ev).toContain(target)
  })

  it('STLSQ zeroes every coefficient for a huge threshold', () => {
    const c = stlsq(
      [
        [1, 2],
        [3, 4],
        [5, 7],
      ],
      [[1], [2], [3]],
      { threshold: 100 },
    )
    expect(Array.from(c.data)).toEqual([0, 0])
  })
})
