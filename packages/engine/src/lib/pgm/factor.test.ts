import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { maxOf } from '../base/index.ts'
import * as P from './index.ts'

/** A random positive factor over a subset of four binary/ternary variables. */
const VARS = ['a', 'b', 'c', 'd']
const CARD: Record<string, number> = { a: 2, b: 3, c: 2, d: 3 }
const factorArb = fc.subarray(VARS, { minLength: 1 }).chain((vs) =>
  fc
    .array(fc.double({ min: 0.05, max: 3, noNaN: true }), {
      minLength: vs.reduce((n, v) => n * (CARD[v] ?? 1), 1),
      maxLength: vs.reduce((n, v) => n * (CARD[v] ?? 1), 1),
    })
    .map((values) =>
      P.factor(
        vs,
        vs.map((v) => CARD[v] ?? 1),
        values,
      ),
    ),
)

/** The value of a factor at an assignment of all four variables. */
const at = (f: P.Factor, a: Record<string, number>): number => P.factorAt(f, a)
const everything = P.allAssignments(
  VARS,
  VARS.map((v) => CARD[v] ?? 1),
)

describe('factor algebra', () => {
  it('products multiply pointwise and commute up to variable order', () => {
    fc.assert(
      fc.property(factorArb, factorArb, (f, g) => {
        const fg = P.factorProduct(f, g)
        const gf = P.factorProduct(g, f)
        for (const a of everything) {
          expect(at(fg, a)).toBeCloseTo(at(f, a) * at(g, a), 12)
          expect(at(gf, a)).toBeCloseTo(at(fg, a), 12)
        }
      }),
      { numRuns: 40 },
    )
  })

  it('sum and max marginals agree with brute force', () => {
    fc.assert(
      fc.property(factorArb, (f) => {
        const out = f.variables.slice(0, 1)
        const m = P.marginalise(f, out)
        const x = P.maxMarginalise(f, out)
        for (const a of P.allAssignments(m.variables, m.cardinality)) {
          const matching = P.allAssignments(f.variables, f.cardinality).filter((b) =>
            m.variables.every((v) => b[v] === a[v]),
          )
          expect(P.factorAt(m, a)).toBeCloseTo(
            matching.reduce((s, b) => s + P.factorAt(f, b), 0),
            12,
          )
          expect(P.factorAt(x, a)).toBeCloseTo(maxOf(matching.map((b) => P.factorAt(f, b))), 12)
        }
      }),
      { numRuns: 40 },
    )
  })

  it('division undoes multiplication', () => {
    fc.assert(
      fc.property(factorArb, factorArb, (f, g) => {
        const back = P.factorDivide(P.factorProduct(f, g), g)
        for (const a of everything) expect(at(back, a)).toBeCloseTo(at(f, a), 10)
      }),
      { numRuns: 40 },
    )
  })

  it('permutation, reduction and marginals onto a scope keep the values', () => {
    const f = P.factor(['a', 'b', 'c'], [2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    const p = P.permuteFactor(f, ['c', 'a', 'b'])
    for (const a of P.allAssignments(f.variables, f.cardinality))
      expect(P.factorAt(p, a)).toBeCloseTo(P.factorAt(f, a), 12)
    const r = P.reduceFactor(f, { b: 2, c: 1 })
    expect(r.variables).toEqual(['a'])
    expect(P.factorValues(r).map(Math.round)).toEqual([6, 12])
    expect(P.factorValues(P.factorMarginal(f, ['c', 'a'])).map(Math.round)).toEqual([9, 27, 12, 30])
    expect(() => P.reduceFactor(f, { b: 3 })).toThrow(/no state/)
    expect(() => P.factorDivide(P.factor(['a'], [2], [1, 1]), f)).toThrow(/numerator/)
  })

  it('normalise, powers, entropy and argmax', () => {
    const f = P.factor(['a', 'b'], [2, 2], [1, 3, 0, 4])
    const n = P.normaliseFactor(f)
    expect(P.factorValues(n).reduce((s, v) => s + v)).toBeCloseTo(1, 14)
    expect(P.factorValues(P.factorPower(f, 2)).map(Math.round)).toEqual([1, 9, 0, 16])
    expect(P.factorEntropy(n)).toBeCloseTo(-[1, 3, 4].reduce((s, v) => s + (v / 8) * Math.log(v / 8), 0), 14)
    expect(P.factorArgmax(f)).toEqual({ a: 1, b: 1 })
    expect(() => P.normaliseFactor(P.factor(['a'], [2], [0, 0]))).toThrow(/zero everywhere/)
    expect(P.factorProductAll([]).logValues).toEqual(new Float64Array([0]))
  })

  it('validate scopes', () => {
    expect(() => P.factor(['a', 'a'], [2, 2], [1, 1, 1, 1])).toThrow(/distinct/)
    expect(() => P.factor(['a'], [2], [1])).toThrow(/assignments/)
    expect(() => P.factorProduct(P.factor(['a'], [2], [1, 1]), P.factor(['a'], [3], [1, 1, 1]))).toThrow(/cardinality/)
  })
})
