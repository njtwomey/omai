import { describe, expect, it } from 'vitest'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

describe('Routh and Jury', () => {
  it('handle a zero first entry and a zero row', () => {
    const eps = S.routh([1, 2, 2, 4, 11, 10])
    expect(eps.epsilonRows.length).toBeGreaterThan(0)
    expect(eps.signChanges).toBe(2)
    const zero = S.routh([1, 2, 24, 48, -25, -50])
    expect(zero.zeroRows.length).toBe(1)
    expect(zero.signChanges).toBe(1)
    expect(zero.stable).toBe(false)
    // s^2 + 1: roots on the imaginary axis, no right half-plane roots, not stable.
    const marginal = S.routh([1, 0, 1])
    expect(marginal.signChanges).toBe(0)
    expect(marginal.stable).toBe(false)
  })

  it('list the Jury conditions', () => {
    const r = S.jury([1, -2.1, 1.2])
    expect(r.stable).toBe(false)
    expect(r.conditions.map((c) => c.satisfied)).toContain(false)
    expect(S.jury([-1, 0.2]).stable).toBe(true)
  })
})

describe('poles, zeros and root loci', () => {
  it('find invariant zeros of a square system with invertible D as eig(A - B D^-1 C)', () => {
    const a = tensor([
      [-1, 0.5, 0],
      [0, -2, 1],
      [0.3, 0, -3],
    ])
    const b = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
    ])
    const c = tensor([
      [1, 0, 0],
      [0, 1, 1],
    ])
    const d = tensor([
      [1, 0.2],
      [0, 2],
    ])
    const z = S.zeros(S.ss(a, b, c, d))
    // A - B D^{-1} C, computed by hand: D^{-1} = [[1, -0.1], [0, 0.5]].
    const want = S.poles(
      S.ss(
        [
          [-2, 0.6, 0.1],
          [0, -2.5, 0.5],
          [-0.7, -0.4, -3.4],
        ],
        [[1], [1], [1]],
        [[1, 1, 1]],
      ),
    )
    for (const [i, v] of z.re.entries()) expect(v).toBeCloseTo(want.re[i] ?? Number.NaN, 8)
    expect(() => S.zeros(S.ss(a, b, [[1, 0, 0]]))).toThrow()
  })

  it('root loci start at the poles and end at the zeros or along the asymptotes', () => {
    const r = S.rootLocus(S.tf([1, 3], [1, 2, 5, 0]))
    expect(r.asymptotes.angles.length).toBe(2)
    expect(r.asymptotes.centroid).toBeCloseTo((-2 - 0 + 3) / 2, 12)
    const first = r.roots[0] as { re: number[] }
    expect([...first.re].sort((x, y) => x - y)[2]).toBeCloseTo(0, 12)
    const last = r.roots[r.roots.length - 1] as { re: number[]; im: number[] }
    // One branch ends at the zero -3; the other two head off along the asymptotes.
    const nearest = last.re.reduce(
      (acc, x, i) => Math.min(acc, Math.hypot(x + 3, last.im[i] ?? 0)),
      Number.POSITIVE_INFINITY,
    )
    expect(nearest).toBeLessThan(1e-2)
    const g = S.rootLocus(S.tf([1], [1, 3, 2]))
    expect(g.breakaway.points[0]).toBeCloseTo(-1.5, 12)
    expect(g.breakaway.gains[0]).toBeCloseTo(0.25, 12)
  })

  it('evaluate the closed loop poles at a gain', () => {
    const r = S.rootLocus(S.tf([1], [1, 3, 2]), [2])
    const want = S.poles(S.feedback(S.tf([2], [1, 3, 2])))
    const got = r.roots[0] as { re: number[]; im: number[] }
    const sortRe = (x: readonly number[]): number[] => [...x].sort((p, q) => p - q)
    expect(sortRe(got.re).map((v) => Math.round(v * 1e10))).toEqual(sortRe(want.re).map((v) => Math.round(v * 1e10)))
  })
})
