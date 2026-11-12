// The constants pass of the model compiler: which nodes are known before the walk, with what values, and from where.
import { describe, expect, it } from 'vitest'
import { fromStorage, toFlat, type Value } from '../../../tensor/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops } from '../index.ts'
import { defaultOperators, type NamedValue } from '../operators.ts'
import { isKnownExpr, Known, knownConstants } from './_known.ts'

const gram = (args: readonly NamedValue[]): Value => {
  const x = Array.from(toFlat((args[0] as NamedValue).value))
  return fromStorage(Float64Array.from(x.map((v) => v * 10)), [x.length], 'float64')
}
const values = (known: Known, name: string): number[] => Array.from(toFlat((known.get(name) as NamedValue).value))

const model = defineModel('known', (m) => {
  const n = m.range('n')
  const x = m.data('x', { dims: [n], optional: true })
  const twice = m.derived('twice', ops.mul(x, 2))
  const z = m.latent('z', d.normal(0, 1))
  const zero = m.derived('zero', ops.mul(z, 0))
  const mixed = m.derived('mixed', ops.add(twice, z))
  m.derived('tenfold', m.custom('tenfold', gram, [twice], { dims: [n] }))
  m.observe('y', d.normal(mixed, 1), m.data('y', { dims: [n] }))
  void zero
})

describe('knownConstants', () => {
  const bound = model.bind({ x: [1, 2], y: [0, 0] })
  const { known, customsCalled } = knownConstants(bound, defaultOperators())

  it('knows bound data and the derived nodes built from known things only, with their values', () => {
    expect(known.has('x')).toBe(true)
    expect(known.has('twice')).toBe(true)
    expect(values(known, 'twice')).toEqual([2, 4])
  })

  it('does not know a latent or a node that reads one (the pass is syntactic: `z * 0` is left to the walk)', () => {
    for (const name of ['z', 'zero', 'mixed']) expect(known.has(name)).toBe(false)
  })

  it('evaluates a supplied custom function of known values and says which it called', () => {
    expect(values(known, 'tenfold')).toEqual([20, 40])
    expect(customsCalled.get('tenfold')).toEqual(['tenfold'])
    expect(customsCalled.get('twice')).toEqual([])
  })

  it('knows a node only from the position it is declared at', () => {
    expect(known.has('twice', 1)).toBe(false)
    expect(known.has('twice', 2)).toBe(true)
    expect(known.has('x', 1)).toBe(true)
  })

  it('does not know data that is not bound, nor what is built from it', () => {
    const unbound = knownConstants(model.bind({ y: [0, 0] }), defaultOperators()).known
    expect(unbound.has('x')).toBe(false)
    expect(unbound.has('twice')).toBe(false)
  })
})

describe('isKnownExpr', () => {
  it('is true for constants, known references and operators of them, false once a reference is not known', () => {
    const known = new Known()
    const one = { kind: 'const', value: 1 } as const
    known.declare('a', { value: fromStorage(Float64Array.of(3), [], 'float64'), dims: [] }, 0)
    const ref = (name: string) => ({ kind: 'ref', name }) as const
    expect(isKnownExpr(ref('a'), known, {})).toBe(true)
    expect(isKnownExpr({ kind: 'op', op: 'add', args: [ref('a'), one] }, known, {})).toBe(true)
    expect(isKnownExpr({ kind: 'op', op: 'add', args: [ref('a'), ref('b')] }, known, {})).toBe(false)
    expect(isKnownExpr(ref('a'), known, {}, 0)).toBe(false)
  })

  it('treats a custom function as known only when it is supplied and its arguments are known', () => {
    const known = new Known()
    const call = { kind: 'custom', name: 'f', args: [{ kind: 'const', value: 1 }], dims: [], eventRank: 0 } as const
    expect(isKnownExpr(call, known, {})).toBe(false)
    expect(isKnownExpr(call, known, { f: () => fromStorage(Float64Array.of(1), [], 'float64') })).toBe(true)
  })
})
