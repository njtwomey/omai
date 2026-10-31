import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as L from './index.ts'

const diag = (d: number[]): number[][] => d.map((v, i) => d.map((_, j) => (i === j ? v : 0)))

describe('subspace iteration', () => {
  const spectrum = [9, 5, 3, 1, 0.5, 0.2]

  it('converges to the dominant eigenvalues and an invariant subspace', () => {
    const r = L.subspaceIteration(diag(spectrum), 3, { key: key(1), tol: 1e-12 })
    expect(r.converged).toBe(true)
    r.values.forEach((v, i) => {
      expect(v).toBeCloseTo(spectrum[i] ?? 0, 9)
    })
    // The basis spans e1, e2, e3: its rows 3..5 vanish.
    const q = toFlat(r.basis)
    for (let row = 3; row < 6; row++)
      for (let c = 0; c < 3; c++) expect(Math.abs(q[row * 3 + c] ?? 1)).toBeLessThan(1e-6)
  })

  it('settles the leading Ritz value faster than the trailing ones, at the squared rate (lambda_{k+1} / lambda_i)', () => {
    const r = L.subspaceIteration(diag(spectrum), 3, { key: key(2), tol: 1e-13 })
    const errs = (steps: number): number[] => {
      const alg = L.subspaceIterationAlgorithm(diag(spectrum), 3, { tol: 0 })
      const s = runAlgorithm(alg, { maxSteps: steps, key: key(2) }).state
      return s.ritzValues.map((v, i) => Math.abs(v - (spectrum[i] ?? 0)))
    }
    const e = errs(6)
    expect(e[0]).toBeLessThan(e[2] ?? 0)
    expect(r.residualHistory[r.residualHistory.length - 1] ?? 1).toBeLessThan(r.residualHistory[0] ?? 0)
    // The vector error of the third direction shrinks by lambda_4 / lambda_3 = 1/3 a step; the Ritz value error by its square.
    const ratio = (errs(8)[2] ?? 1) / (errs(7)[2] ?? 1)
    expect(ratio).toBeGreaterThan(0.08)
    expect(ratio).toBeLessThan(0.15)
  })

  it('works on an implicit operator, is deterministic in the key, and serialises', () => {
    const matvec = (x: Float64Array): Float64Array => Float64Array.from(x, (v, i) => v * (spectrum[i] ?? 0))
    const alg = L.subspaceIterationAlgorithm({ n: 6, matvec }, 2, { tol: 0 })
    const a = runAlgorithm(alg, { maxSteps: 10, key: key(5) })
    const b = runAlgorithm(alg, { maxSteps: 10, key: key(5) })
    expect(a.state.ritzValues).toEqual(b.state.ritzValues)
    const half = runAlgorithm(alg, { maxSteps: 4, key: key(5) })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const resumed = runAlgorithm(alg, { maxSteps: 6, key: key(5), start: { state: stored, step: 4, serialised: true } })
    expect(resumed.state.ritzValues).toEqual(a.state.ritzValues)
  })

  it('rejects a block larger than the matrix', () => {
    expect(() => L.subspaceIteration(diag(spectrum), 7)).toThrow(/block size/)
  })
})

describe('one-sided Jacobi SVD', () => {
  it('orthogonalises the columns: the off-diagonal measure falls quadratically at the end', () => {
    const a = [
      [4, 1, 2],
      [1, 3, 0],
      [0, 2, 5],
      [1, 1, 1],
    ]
    const r = L.jacobiSvd(a)
    expect(r.converged).toBe(true)
    const h = r.history.filter((v) => v > 0)
    expect(h[0]).toBeGreaterThan(0.3)
    // Near the end each sweep roughly squares the previous measure.
    for (let i = 1; i < h.length - 1; i++)
      if ((h[i - 1] ?? 0) < 1e-3) expect(h[i] ?? 1).toBeLessThan((h[i - 1] ?? 0) ** 1.5 + 1e-15)
  })

  it('is orthogonal in U and V, and reconstructs the matrix', () => {
    const a = [
      [2, -1],
      [0.5, 3],
      [1, 1],
    ]
    const { u, s, vt } = L.jacobiSvd(a)
    const un = toFlat(u)
    const vn = toFlat(vt)
    const dotCols = (i: number, j: number): number =>
      [0, 1, 2].reduce((t, r) => t + (un[r * 2 + i] ?? 0) * (un[r * 2 + j] ?? 0), 0)
    expect(dotCols(0, 0)).toBeCloseTo(1, 12)
    expect(dotCols(0, 1)).toBeCloseTo(0, 12)
    expect((vn[0] ?? 0) * (vn[2] ?? 0) + (vn[1] ?? 0) * (vn[3] ?? 0)).toBeCloseTo(0, 12)
    expect(s[0]).toBeGreaterThan(s[1] ?? 0)
  })

  it('keeps tiny singular values accurate to their own size', () => {
    // Q diag(d): the singular values are d, down to 1e-15 of the largest.
    const d = [1, 1e-5, 1e-10, 1e-15]
    const q = [
      [0.5, 0.5, 0.5, 0.5],
      [0.5, -0.5, 0.5, -0.5],
      [0.5, 0.5, -0.5, -0.5],
      [0.5, -0.5, -0.5, 0.5],
    ]
    const a = q.map((row) => row.map((v, j) => v * (d[j] ?? 0)))
    const r = L.jacobiSvd(a)
    r.s.forEach((v, i) => {
      expect(Math.abs(v - (d[i] ?? 0)) / (d[i] ?? 1)).toBeLessThan(1e-9)
    })
  })

  it('a step can be one rotation or one sweep and both reach the same singular values', () => {
    const a = [
      [3, 1],
      [1, 3],
      [0, 1],
    ]
    const sweep = L.jacobiSvd(a, { granularity: 'sweep' })
    const rotation = L.jacobiSvd(a, { granularity: 'rotation' })
    expect(rotation.history.length).toBeGreaterThanOrEqual(sweep.history.length)
    sweep.s.forEach((v, i) => {
      expect(v).toBeCloseTo(rotation.s[i] ?? 0, 12)
    })
  })

  it('serialises its state and rejects wide matrices', () => {
    const alg = L.jacobiSvdAlgorithm([
      [1, 2, 0],
      [0, 1, 1],
      [1, 0, 3],
      [2, 1, 1],
    ])
    const straight = runAlgorithm(alg, { maxSteps: 3, seed: 0 })
    const first = runAlgorithm(alg, { maxSteps: 1, seed: 0 })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(first.state)))
    const resumed = runAlgorithm(alg, { maxSteps: 2, seed: 0, start: { state: stored, step: 1, serialised: true } })
    expect(resumed.state.u).toEqual(straight.state.u)
    expect(() => L.jacobiSvdAlgorithm([[1, 2, 3]])).toThrow(/rows/)
  })
})
