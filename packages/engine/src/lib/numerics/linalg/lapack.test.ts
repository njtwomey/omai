import { describe, expect, it } from 'vitest'
import { dasum, ddot, dgemm, dgemv, idamax } from './blas.ts'
import { dgeqrf, dgesv, dgesvd, dpotrf, dpotrs, dsyev, dtrsv } from './lapack.ts'

const A = [4, 1, 0.5, 1, 3, -0.5, 0.5, -0.5, 2]

describe('number-array LAPACK routines', () => {
  it('dpotrf factors SPD matrices and reports the failing minor', () => {
    const { factor, info } = dpotrf(A, 3)
    expect(info).toBe(0)
    // U^T U = A
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        let s = 0
        for (let k = 0; k < 3; k++) s += (factor[k * 3 + i] ?? 0) * (factor[k * 3 + j] ?? 0)
        expect(s).toBeCloseTo(A[i * 3 + j] ?? 0, 13)
      }
    expect(dpotrf([1, 0, 0, 0, -1, 0, 0, 0, 1], 3).info).toBe(2)
    expect(dpotrf([0, 0, 0, 1], 2).info).toBe(1)
  })
  it('dpotrs, dgesv and dtrsv solve their systems', () => {
    const b = [1, -2, 0.5]
    const x1 = dpotrs(dpotrf(A, 3).factor, b, 3)
    const x2 = dgesv(A, b, 3)
    const back = dgemv(A, 3, 3, x1)
    for (let i = 0; i < 3; i++) {
      expect(back[i]).toBeCloseTo(b[i] ?? 0, 13)
      expect(x2[i]).toBeCloseTo(x1[i] ?? 0, 13)
    }
    expect(Array.from(dtrsv([2, 0, 1, 4], [2, 6], 2, { lower: true }))).toEqual([1, 1.25])
    expect(Array.from(dtrsv([2, 1, 0, 4], [2, 6], 2, { transpose: true }))).toEqual([1, 1.25])
    expect(() => dgesv([1, 2, 2, 4], [1, 1], 2)).toThrow()
  })
  it('dsyev, dgeqrf and dgesvd decompose', () => {
    const { w, v } = dsyev(A, 3)
    expect((w[0] ?? 0) <= (w[1] ?? 0) && (w[1] ?? 0) <= (w[2] ?? 0)).toBe(true)
    for (let k = 0; k < 3; k++) {
      const col = [0, 1, 2].map((i) => v[i * 3 + k] ?? 0)
      const av = dgemv(A, 3, 3, col)
      for (let i = 0; i < 3; i++) expect(av[i]).toBeCloseTo((w[k] ?? 0) * (col[i] ?? 0), 12)
    }
    const { q, r } = dgeqrf([1, 2, 3, 4, 5, 6], 3, 2)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 2; j++) {
        const qr = (q[i * 2] ?? 0) * (r[j] ?? 0) + (q[i * 2 + 1] ?? 0) * (r[2 + j] ?? 0)
        expect(qr).toBeCloseTo(i * 2 + j + 1, 12)
      }
    const { s } = dgesvd([3, 0, 0, 4], 2, 2)
    expect(Array.from(s)).toEqual([4, 3])
  })
  it('idamax and dasum', () => {
    expect(idamax([])).toBe(-1)
    expect(idamax([1, -5, 5])).toBe(1)
    expect(dasum([1, -2, 3])).toBe(6)
    expect(ddot([1, 2], [3, 4])).toBe(11)
  })
  it('dgemm multiplies with either factor transposed', () => {
    const a = [1, 2, 3, 4, 5, 6] // 2 x 3
    const b = [1, 0, 2, 1, 0, 3] // 3 x 2
    expect(Array.from(dgemm(a, b, 2, 3, 2))).toEqual([5, 11, 14, 23])
    // A^T A (3 x 3) and A A^T (2 x 2).
    expect(Array.from(dgemm(a, a, 3, 2, 3, { transposeA: true }))).toEqual([17, 22, 27, 22, 29, 36, 27, 36, 45])
    expect(Array.from(dgemm(a, a, 2, 3, 2, { transposeB: true }))).toEqual([14, 32, 32, 77])
    for (let i = 0; i < 2; i++) expect(dgemm(a, b, 2, 3, 2)[i * 2]).toBe(dgemv(a, 2, 3, [1, 2, 0])[i])
  })
})
