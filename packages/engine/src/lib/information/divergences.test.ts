import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { checkPrimitive } from '../autodiff/index.ts'
import { item, log, mul, neg, square, sub, sum, tensor, toArray, type Value } from '../tensor/index.ts'
import * as I from './index.ts'

const p = tensor([0.1, 0.2, 0.3, 0.4])
const q = tensor([0.25, 0.25, 0.3, 0.2])
const prob = (n: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: 0.01, max: 1, noNaN: true }), { minLength: n, maxLength: n }).map((w) => {
    const s = w.reduce((a, b) => a + b, 0)
    return w.map((v) => v / s)
  })

describe('divergence families', () => {
  it('named f-divergences agree with custom generators and with the discrete functions', () => {
    expect(item(I.fDivergence(p, q, 'kl'))).toBeCloseTo(item(I.relativeEntropy(p, q)), 14)
    expect(item(I.fDivergence(p, q, (t: Value) => mul(t, log(t))))).toBeCloseTo(item(I.relativeEntropy(p, q)), 14)
    expect(item(I.fDivergence(p, q, (t: Value) => neg(log(t))))).toBeCloseTo(item(I.relativeEntropy(q, p)), 14)
    expect(item(I.fDivergence(p, q, (t: Value) => square(sub(t, 1))))).toBeCloseTo(
      item(I.fDivergence(p, q, 'pearson')),
      14,
    )
    expect(item(I.fDivergence(p, q, 'jensenShannon'))).toBeCloseTo(item(I.jensenShannonDivergence(p, q)), 14)
    expect(item(I.fDivergence(p, q, 'totalVariation'))).toBeCloseTo(item(I.totalVariation(p, q)), 14)
    expect(item(I.fDivergence([1, 0], [0, 1], 'pearson'))).toBe(Number.POSITIVE_INFINITY)
    expect(item(I.fDivergence([0, 1], [0, 1], 'triangular'))).toBe(0)
  })

  it('squared Hellinger is one minus the Bhattacharyya coefficient', () => {
    const bc = [0.1, 0.2, 0.3, 0.4].reduce((s, v, i) => s + Math.sqrt(v * ([0.25, 0.25, 0.3, 0.2][i] ?? 0)), 0)
    expect(item(I.fDivergence(p, q, 'squaredHellinger'))).toBeCloseTo(1 - bc, 14)
  })

  it('the Renyi family has the right limits and is monotone in the order', () => {
    expect(item(I.renyiDivergence(p, q, 1))).toBeCloseTo(item(I.relativeEntropy(p, q)), 14)
    expect(item(I.renyiDivergence(p, q, 1 + 1e-7))).toBeCloseTo(item(I.relativeEntropy(p, q)), 6)
    expect(item(I.renyiEntropy(p, 1))).toBeCloseTo(item(I.entropy(p)), 14)
    expect(item(I.renyiEntropy([0.5, 0.5, 0], 0, { base: 2 }))).toBeCloseTo(1, 14)
    expect(item(I.renyiEntropy(p, Number.POSITIVE_INFINITY))).toBeCloseTo(-Math.log(0.4), 14)
    expect(() => I.renyiDivergence(p, q, -1)).toThrow()
    expect(() => I.renyiEntropy(p, -1)).toThrow()
    fc.assert(
      fc.property(prob(4), prob(4), (a, b) => {
        const orders = [0, 0.5, 1, 2, 5, Number.POSITIVE_INFINITY]
        const d = orders.map((o) => item(I.renyiDivergence(a, b, o)))
        const h = orders.map((o) => item(I.renyiEntropy(a, o)))
        for (let i = 1; i < orders.length; i++) {
          expect(d[i] ?? 0).toBeGreaterThanOrEqual((d[i - 1] ?? 0) - 1e-12)
          expect(h[i] ?? 0).toBeLessThanOrEqual((h[i - 1] ?? 0) + 1e-12)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('alpha and beta divergences are non-negative and reduce to known cases', () => {
    const m1 = tensor([0.5, 1.5, 2])
    const m2 = tensor([1, 1, 3])
    expect(item(I.alphaDivergence(m1, m2, 0.5))).toBeCloseTo(
      2 * [0.5, 1.5, 2].reduce((s, v, i) => s + (Math.sqrt(v) - Math.sqrt([1, 1, 3][i] ?? 0)) ** 2, 0),
      14,
    )
    expect(item(I.alphaDivergence(m1, m2, 1 + 1e-7))).toBeCloseTo(item(I.alphaDivergence(m1, m2, 1)), 5)
    expect(item(I.betaDivergence(m1, m2, 1 + 1e-7))).toBeCloseTo(item(I.betaDivergence(m1, m2, 1)), 5)
    expect(item(I.betaDivergence(m1, m2, 1e-7))).toBeCloseTo(item(I.betaDivergence(m1, m2, 0)), 5)
    expect(item(I.betaDivergence(m1, m2, 0))).toBeCloseTo(item(I.bregmanDivergence(m1, m2, 'itakuraSaito')), 14)
    fc.assert(
      fc.property(prob(3), prob(3), fc.double({ min: -2, max: 3, noNaN: true }), (a, b, order) => {
        expect(item(I.alphaDivergence(a, b, order))).toBeGreaterThanOrEqual(-1e-12)
        expect(item(I.betaDivergence(a, b, order))).toBeGreaterThanOrEqual(-1e-12)
      }),
      { numRuns: RUNS },
    )
  })

  it('a custom Bregman generator matches the named ones', () => {
    const m1 = tensor([0.5, 1.5, 2])
    const m2 = tensor([1, 1, 3])
    const negEntropy = (x: Value): Value => sum(mul(x, log(x)))
    expect(item(I.bregmanDivergence(m1, m2, negEntropy))).toBeCloseTo(
      item(I.bregmanDivergence(m1, m2, 'negativeEntropy')),
      13,
    )
    const burg = (x: Value): Value => neg(sum(log(x)))
    expect(item(I.bregmanDivergence(m1, m2, burg))).toBeCloseTo(item(I.bregmanDivergence(m1, m2, 'itakuraSaito')), 13)
  })

  it('grid options integrate by the trapezoidal rule along any axis', () => {
    const d = tensor([
      [1, 2, 3],
      [2, 2, 2],
    ])
    const e = tensor([
      [1, 1, 1],
      [1, 1, 1],
    ])
    expect(toArray(I.fDivergence(d, e, (t: Value) => t, { dx: 2 }))).toEqual([8, 8])
    expect(toArray(I.fDivergence(d, e, (t: Value) => t, { x: [0, 1, 3] }))).toEqual([1.5 + 5, 6])
    expect(toArray(I.fDivergence(d, e, (t: Value) => t, { axis: 0, dx: 1 }))).toEqual([1.5, 2, 2.5])
    expect(() => I.fDivergence([1], [1], 'kl', { dx: 1 })).toThrow()
    expect(() => I.fDivergence([1, 2], [1, 2], 'kl', { x: [1, 0] })).toThrow()
    expect(() => I.fDivergence([1, 2], [1, 2], 'kl', { x: [0, 1, 2] })).toThrow()
  })

  for (const [name, f] of [
    ['kl', (a: Value, b: Value) => I.fDivergence(a, b, 'kl')],
    ['squaredHellinger', (a: Value, b: Value) => I.fDivergence(a, b, 'squaredHellinger')],
    ['pearson', (a: Value, b: Value) => I.fDivergence(a, b, 'pearson')],
    ['jensenShannon', (a: Value, b: Value) => I.fDivergence(a, b, 'jensenShannon')],
    ['triangular', (a: Value, b: Value) => I.fDivergence(a, b, 'triangular')],
    ['renyi 2', (a: Value, b: Value) => I.renyiDivergence(a, b, 2)],
    ['renyi 0.5', (a: Value, b: Value) => I.renyiDivergence(a, b, 0.5)],
    ['renyiEntropy 3', (a: Value) => I.renyiEntropy(a, 3)],
    ['alpha -1', (a: Value, b: Value) => I.alphaDivergence(a, b, -1)],
    ['beta 1.5', (a: Value, b: Value) => I.betaDivergence(a, b, 1.5)],
    ['beta 0', (a: Value, b: Value) => I.betaDivergence(a, b, 0)],
    ['bregman custom', (a: Value, b: Value) => I.bregmanDivergence(a, b, (x: Value) => sum(mul(x, log(x))))],
    ['kl on a grid', (a: Value, b: Value) => I.fDivergence(a, b, 'kl', { dx: 0.5 })],
  ] as const) {
    it(`${name} passes the derivative conformance check`, () => {
      expect(checkPrimitive(f as (...xs: Value[]) => Value, [p, q], { vmap: false }).failures).toEqual([])
    })
  }
})
