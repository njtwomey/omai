import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { item, tensor, toFlat } from '../tensor/index.ts'
import * as I from './index.ts'

const weights = (n: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: 0.01, max: 1, noNaN: true }), { minLength: n, maxLength: n })
const prob = (n: number): fc.Arbitrary<number[]> =>
  weights(n).map((w) => {
    const s = w.reduce((a, b) => a + b, 0)
    return w.map((v) => v / s)
  })
/** A stochastic matrix [rows, cols]. */
const channel = (rows: number, cols: number): fc.Arbitrary<number[][]> =>
  fc.array(prob(cols), { minLength: rows, maxLength: rows })
const table = (r: number, c: number): fc.Arbitrary<number[][]> => fc.array(weights(c), { minLength: r, maxLength: r })

const times = (px: readonly number[], w: readonly (readonly number[])[]): number[][] =>
  px.map((a, i) => (w[i] ?? []).map((v) => a * v))
const compose = (a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number[][] =>
  a.map((row) => (b[0] ?? []).map((_, j) => row.reduce((s, v, k) => s + v * (b[k]?.[j] ?? 0), 0)))

describe('information inequalities and identities (property)', () => {
  it('Gibbs: KL(p || q) >= 0, with equality at p = q', () => {
    fc.assert(
      fc.property(prob(5), prob(5), (p, q) => {
        expect(item(I.relativeEntropy(p, q))).toBeGreaterThanOrEqual(-1e-15)
        expect(Math.abs(item(I.relativeEntropy(p, p)))).toBeLessThan(1e-15)
      }),
      { numRuns: RUNS },
    )
  })

  it('Pinsker: TV(p, q) <= sqrt(KL(p || q) / 2); JS <= log 2', () => {
    fc.assert(
      fc.property(prob(4), prob(4), (p, q) => {
        expect(item(I.totalVariation(p, q))).toBeLessThanOrEqual(Math.sqrt(item(I.relativeEntropy(p, q)) / 2) + 1e-12)
        expect(item(I.jensenShannonDivergence(p, q))).toBeLessThanOrEqual(Math.LN2 + 1e-12)
      }),
      { numRuns: RUNS },
    )
  })

  it('0 <= H(p) <= log n', () => {
    fc.assert(
      fc.property(weights(6), (w) => {
        const h = item(I.entropy(w))
        expect(h).toBeGreaterThanOrEqual(-1e-15)
        expect(h).toBeLessThanOrEqual(Math.log(6) + 1e-12)
      }),
      { numRuns: RUNS },
    )
  })

  it('chain rule: H(X, Y) = H(Y) + H(X | Y) and I(X; Y) = H(X) - H(X | Y), symmetric', () => {
    fc.assert(
      fc.property(table(3, 4), (t) => {
        const total = t.flat().reduce((a, b) => a + b, 0)
        const px = t.map((r) => r.reduce((a, b) => a + b, 0) / total)
        const py = (t[0] ?? []).map((_, j) => t.reduce((s, r) => s + (r[j] ?? 0), 0) / total)
        const hxy = item(I.jointEntropy(t))
        const hxGivenY = item(I.conditionalEntropy(t, 1))
        expect(hxy).toBeCloseTo(item(I.entropy(py)) + hxGivenY, 12)
        const mi = item(I.mutualInformation(t))
        expect(mi).toBeCloseTo(item(I.entropy(px)) - hxGivenY, 12)
        const transposed = (t[0] ?? []).map((_, j) => t.map((r) => r[j] ?? 0))
        expect(item(I.mutualInformation(transposed))).toBeCloseTo(mi, 12)
        expect(mi).toBeGreaterThanOrEqual(-1e-14)
      }),
      { numRuns: RUNS },
    )
  })

  it('chain rule for mutual information: I(X; Y, Z) = I(X; Z) + I(X; Y | Z)', () => {
    fc.assert(
      fc.property(fc.array(table(2, 3), { minLength: 2, maxLength: 2 }), (t) => {
        // t is [x, y, z]; I(X; (Y, Z)) treats (y, z) as one variable.
        const flat = t.map((plane) => plane.flat())
        const xz = t.map((plane) => (plane[0] ?? []).map((_, z) => plane.reduce((s, r) => s + (r[z] ?? 0), 0)))
        const lhs = item(I.mutualInformation(flat))
        const rhs = item(I.mutualInformation(xz)) + item(I.conditionalMutualInformation(t))
        expect(lhs).toBeCloseTo(rhs, 12)
      }),
      { numRuns: RUNS },
    )
  })

  it('data processing: for X -> Y -> Z, I(X; Z) <= I(X; Y)', () => {
    fc.assert(
      fc.property(prob(3), channel(3, 4), channel(4, 3), (px, w1, w2) => {
        const xy = times(px, w1)
        const xz = times(px, compose(w1, w2))
        expect(item(I.mutualInformation(xz))).toBeLessThanOrEqual(item(I.mutualInformation(xy)) + 1e-12)
        // Processing also cannot increase divergence: KL(p W || q W) <= KL(p || q).
        const q = [0.2, 0.5, 0.3]
        const pw = compose([px], w1)[0] ?? []
        const qw = compose([q], w1)[0] ?? []
        expect(item(I.relativeEntropy(pw, qw))).toBeLessThanOrEqual(item(I.relativeEntropy(px, q)) + 1e-12)
      }),
      { numRuns: RUNS },
    )
  })

  it('interaction information of three variables equals I(X; Y) - I(X; Y | Z)', () => {
    fc.assert(
      fc.property(fc.array(table(2, 2), { minLength: 2, maxLength: 2 }), (t) => {
        const xy = t.map((plane) => plane.map((r) => r.reduce((a, b) => a + b, 0)))
        const lhs = item(I.interactionInformation(t))
        expect(lhs).toBeCloseTo(item(I.mutualInformation(xy)) - item(I.conditionalMutualInformation(t)), 12)
      }),
      { numRuns: RUNS },
    )
  })

  it('NPMI lies in [-1, 1]', () => {
    fc.assert(
      fc.property(table(3, 3), (t) => {
        for (const v of toFlat(I.normalisedPointwiseMutualInformation(tensor(t)))) {
          expect(v).toBeGreaterThanOrEqual(-1 - 1e-12)
          expect(v).toBeLessThanOrEqual(1 + 1e-12)
        }
      }),
      { numRuns: RUNS },
    )
  })
})
