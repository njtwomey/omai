// The cell algebra of the model compiler: affine forms, lookups of a discrete variable, scaled and reciprocal gamma variables.
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  type Aff,
  affOf,
  type Cell,
  combine,
  constAff,
  constCell,
  isConstCell,
  logName,
  lookup,
  lookupArithmetic,
  lookupOfForms,
  prod,
  recipArithmetic,
  scaleAff,
  scaledArithmetic,
  scaledOf,
  unflatten,
} from './_cells.ts'
import { Refusal } from './_refusal.ts'

const aff = (offset: number, terms: Record<string, number> = {}): Aff => ({
  terms: new Map(Object.entries(terms)),
  offset,
})
const form = (a: Aff): Cell => ({ kind: 'aff', aff: a })
const gamma = (variable: string): Cell => ({ kind: 'rv', variable, family: 'gamma' })
const discrete = (variable: string): Cell => ({ kind: 'rv', variable, family: 'discrete' })
const states = (): number => 3
const refusal = (f: () => unknown): Refusal | undefined => {
  try {
    f()
  } catch (e) {
    if (e instanceof Refusal) return e
    throw e
  }
  return undefined
}

describe('affine forms', () => {
  it('combines with a coefficient and drops the terms that cancel', () => {
    const sum = combine(aff(1, { x: 2, y: 1 }), aff(3, { x: 2 }), -1)
    expect([...sum.terms]).toEqual([['y', 1]])
    expect(sum.offset).toBe(-2)
  })

  it('scales, and scaling by zero is the constant zero', () => {
    const s = scaleAff(aff(1, { x: 2 }), 3)
    expect([...s.terms]).toEqual([['x', 6]])
    expect(s.offset).toBe(3)
    expect(scaleAff(aff(5, { x: 1 }), 0)).toEqual(constAff(0))
  })

  it('tells a constant cell from a random one', () => {
    expect(isConstCell(constCell(2))).toBe(true)
    expect(isConstCell(form(aff(0, { x: 1 })))).toBe(false)
    expect(isConstCell(gamma('t'))).toBe(false)
  })

  it('combine is linear (property)', () => {
    const coef = fc.integer({ min: -4, max: 4 })
    const form3 = fc.record({ x: coef, y: coef, z: coef, c: coef })
    const make = (r: { x: number; y: number; z: number; c: number }): Aff => aff(r.c, { x: r.x, y: r.y, z: r.z })
    const at = (a: Aff, p: Record<string, number>): number =>
      a.offset + [...a.terms].reduce((s, [k, c]) => s + c * (p[k] ?? 0), 0)
    fc.assert(
      fc.property(form3, form3, coef, coef, coef, coef, (a, b, k, x, y, z) => {
        const p = { x, y, z }
        expect(at(combine(make(a), make(b), k), p)).toBeCloseTo(at(make(a), p) + k * at(make(b), p), 9)
      }),
    )
  })
})

describe('affOf refusals', () => {
  it('returns the form of an affine cell', () => {
    const a = aff(1, { x: 1 })
    expect(affOf(form(a))).toBe(a)
  })

  it('refuses arithmetic on every other kind of cell, with the kind in the reason', () => {
    const lookup: Cell = { kind: 'lookup', variable: 'z', values: [1, 2] }
    const scaled: Cell = { kind: 'scaled', variable: 't', scale: 2, offset: 1 }
    const recip: Cell = { kind: 'recip', variable: 't', scale: 2 }
    const sticks: Cell = { kind: 'sticks', variables: ['v0'] }
    expect(refusal(() => affOf(lookup))?.message).toContain("looked up by the discrete variable 'z'")
    expect(refusal(() => affOf(scaled))?.message).toContain("scaled precision of 't'")
    expect(refusal(() => affOf(recip))?.message).toContain("inverse-gamma variable behind 't'")
    expect(refusal(() => affOf(sticks))?.message).toContain('stick-breaking weights')
    const rv = refusal(() => affOf(gamma('t')))
    expect(rv?.kind).toBe('operator')
    expect(rv?.message).toBe("arithmetic on the gamma variable 't': only Gaussian values add up")
  })
})

describe('scaled and reciprocal gamma variables', () => {
  it('reads a gamma variable or a scaled one as scale tau + offset', () => {
    expect(scaledOf(gamma('t'))).toEqual({ variable: 't', scale: 1, offset: 0 })
    const s: Cell = { kind: 'scaled', variable: 't', scale: 2, offset: 1 }
    expect(scaledOf(s)).toBe(s)
    expect(scaledOf(constCell(1))).toBeUndefined()
  })

  it('keeps the multiples and offsets of a precision', () => {
    const t = gamma('t')
    expect(scaledArithmetic('mul', t, constCell(3))).toEqual({ kind: 'scaled', variable: 't', scale: 3, offset: 0 })
    expect(scaledArithmetic('add', constCell(2), scaledArithmetic('mul', t, constCell(3)) as Cell)).toEqual({
      kind: 'scaled',
      variable: 't',
      scale: 3,
      offset: 2,
    })
    expect(scaledArithmetic('div', t, constCell(4))).toEqual({ kind: 'scaled', variable: 't', scale: 0.25, offset: 0 })
    expect(scaledArithmetic('sub', t, constCell(1))).toEqual({ kind: 'scaled', variable: 't', scale: 1, offset: -1 })
  })

  it('is undefined when neither operand is gamma based, and refuses a known value over a precision', () => {
    expect(scaledArithmetic('add', constCell(1), constCell(2))).toBeUndefined()
    expect(refusal(() => scaledArithmetic('div', constCell(1), gamma('t')))?.message).toContain('has no message rule')
    expect(refusal(() => scaledArithmetic('sub', constCell(1), gamma('t')))?.kind).toBe('operator')
    expect(refusal(() => scaledArithmetic('add', gamma('t'), form(aff(0, { x: 1 }))))?.kind).toBe('operator')
  })

  it('multiplies and divides an inverse-gamma variance, and a known value over it is a precision', () => {
    const r: Cell = { kind: 'recip', variable: 't', scale: 2 }
    expect(recipArithmetic('mul', r, constCell(3))).toEqual({ kind: 'recip', variable: 't', scale: 6 })
    expect(recipArithmetic('div', r, constCell(4))).toEqual({ kind: 'recip', variable: 't', scale: 0.5 })
    expect(recipArithmetic('div', constCell(6), r)).toEqual({ kind: 'scaled', variable: 't', scale: 3, offset: 0 })
    expect(recipArithmetic('add', constCell(1), constCell(2))).toBeUndefined()
    expect(refusal(() => recipArithmetic('add', r, constCell(1)))?.message).toContain(
      "'add' of an inverse-gamma variance",
    )
    expect(refusal(() => recipArithmetic('mul', r, constCell(-1)))?.message).toContain('non-positive')
  })
})

describe('lookups of a discrete variable', () => {
  const look = (values: number[]): Cell => ({ kind: 'lookup', variable: 'z', values })

  it('does arithmetic state by state with a known value or another lookup of the same variable', () => {
    expect(lookupArithmetic('mul', look([1, 2, 3]), constCell(2), states)).toEqual(look([2, 4, 6]))
    expect(lookupArithmetic('sub', constCell(10), look([1, 2, 3]), states)).toEqual(look([9, 8, 7]))
    expect(lookupArithmetic('add', look([1, 2]), look([10, 20]), states)).toEqual(look([11, 22]))
  })

  it('reads a discrete variable as its states and a Bernoulli one as two', () => {
    expect(lookupArithmetic('mul', discrete('z'), constCell(3), states)).toEqual(look([0, 3, 6]))
    const bern: Cell = { kind: 'rv', variable: 'z', family: 'bernoulli' }
    expect(lookupArithmetic('add', bern, constCell(1), states)).toEqual(look([1, 2]))
  })

  it('is undefined when no operand is one, and refuses what has no rule', () => {
    expect(lookupArithmetic('add', constCell(1), constCell(2), states)).toBeUndefined()
    expect(refusal(() => lookupArithmetic('pow', look([1]), constCell(2), states))?.message).toBe(
      "'pow' of a discrete variable has no message rule",
    )
    expect(
      refusal(() => lookupArithmetic('add', look([1, 2]), { kind: 'lookup', variable: 'w', values: [1, 2] }, states))
        ?.message,
    ).toContain("two different discrete variables ('z', 'w')")
    expect(refusal(() => lookupArithmetic('add', look([1, 2]), form(aff(0, { x: 1 })), states))?.message).toBe(
      "'add' of a discrete variable and a random value has no message rule",
    )
  })
})

describe('lookup constructors', () => {
  it('builds a lookup with only the parts it is given', () => {
    expect(lookup('z', [1, 2])).toEqual({ kind: 'lookup', variable: 'z', values: [1, 2] })
    expect('items' in lookup('z', [1, 2])).toBe(false)
    const precisions = [{ variable: 't', scale: 1, offset: 0 }, undefined]
    expect(lookup('z', [0, 5], { precisions })).toEqual({ kind: 'lookup', variable: 'z', values: [0, 5], precisions })
  })

  it('reads known forms as numbers and keeps the random ones in items', () => {
    const x = aff(0, { x: 1 })
    expect(lookupOfForms('z', [constCell(1), constCell(3)])).toEqual({ kind: 'lookup', variable: 'z', values: [1, 3] })
    const mixed = lookupOfForms('z', [constCell(1), form(x)])
    expect(mixed).toEqual({ kind: 'lookup', variable: 'z', values: [1, 0], items: [undefined, x] })
  })
})

describe('index helpers', () => {
  it('multiplies shapes and unflattens row-major (property)', () => {
    expect(prod([])).toBe(1)
    expect(prod([2, 3, 4])).toBe(24)
    expect(unflatten(5, [2, 3])).toEqual([1, 2])
    expect(logName('tau')).toBe('tau#log')
    fc.assert(
      fc.property(fc.array(fc.integer({ min: 1, max: 4 }), { minLength: 1, maxLength: 4 }), fc.nat(), (shape, k) => {
        const flat = k % prod(shape)
        const idx = unflatten(flat, shape)
        expect(idx.reduce((acc, i, d) => acc * (shape[d] as number) + i, 0)).toBe(flat)
      }),
    )
  })
})
