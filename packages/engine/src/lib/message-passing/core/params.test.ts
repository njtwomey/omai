import { describe, expect, it } from 'vitest'
import { ParameterError } from './errors.ts'
import { exampleParams, mergeParams, namedParams, param, validateParams } from './params.ts'

const message = (run: () => void): string => {
  try {
    run()
  } catch (e) {
    expect(e).toBeInstanceOf(ParameterError)
    return (e as Error).message
  }
  return 'no error'
}

describe('validateParams', () => {
  const spec = {
    precision: param.positive(),
    offset: param.number({ optional: true }),
    n: param.integer({ min: 1 }),
    a: param.vector(),
    kind: param.text({ choices: ['x', 'y'] }),
    on: param.flag({ optional: true }),
    free: param.any({ optional: true }),
  }
  const ok = { precision: 2, n: 3, a: [1, 2], kind: 'x' }

  it('accepts a valid set, with or without the optional ones', () => {
    expect(() => validateParams('op', spec, ok)).not.toThrow()
    expect(() => validateParams('op', spec, { ...ok, offset: -1, on: false, free: { any: ['thing'] } })).not.toThrow()
  })

  it('names the operator and the parameter for a missing one', () => {
    expect(message(() => validateParams('op', spec, { ...ok, precision: undefined as never }))).toBe(
      "op: the parameter 'precision' is required (a number greater than 0)",
    )
  })

  it('names the operator and the parameter for an unknown one, and lists what the operator takes', () => {
    expect(message(() => validateParams('op', spec, { ...ok, variance: 2 }))).toBe(
      "op: unknown parameter 'variance' (it takes: precision, offset, n, a, kind, on, free)",
    )
    expect(message(() => validateParams('op', {}, { a: 1 }))).toBe("op: unknown parameter 'a' (it takes no parameters)")
  })

  it('rejects each kind of wrong value', () => {
    expect(message(() => validateParams('op', spec, { ...ok, precision: -2 }))).toContain(
      "'precision' must be a number greater than 0",
    )
    expect(message(() => validateParams('op', spec, { ...ok, precision: '2' }))).toContain('got "2"')
    expect(message(() => validateParams('op', spec, { ...ok, precision: Number.NaN }))).toContain("'precision'")
    expect(message(() => validateParams('op', spec, { ...ok, precision: Number.POSITIVE_INFINITY }))).toContain(
      "'precision'",
    )
    expect(message(() => validateParams('op', spec, { ...ok, n: 1.5 }))).toContain(
      "'n' must be an integer of at least 1",
    )
    expect(message(() => validateParams('op', spec, { ...ok, n: 0 }))).toContain("'n'")
    expect(message(() => validateParams('op', spec, { ...ok, a: [1, 'b'] }))).toContain(
      "'a' must be an array of numbers",
    )
    expect(message(() => validateParams('op', spec, { ...ok, a: 3 }))).toContain("'a'")
    expect(message(() => validateParams('op', spec, { ...ok, kind: 'z' }))).toContain("one of 'x', 'y'")
    expect(message(() => validateParams('op', spec, { ...ok, on: 1 }))).toContain("'on' must be a boolean")
  })

  it('lets a number be infinite only when it says so', () => {
    const bound = { high: param.number({ infinite: true }), low: param.number() }
    expect(() => validateParams('op', bound, { high: Number.POSITIVE_INFINITY, low: 0 })).not.toThrow()
    expect(message(() => validateParams('op', bound, { high: 1, low: Number.NEGATIVE_INFINITY }))).toContain("'low'")
  })

  it('takes a spec that is a function of the parameters', () => {
    const dependent = (p: Readonly<Record<string, unknown>>) =>
      p['kind'] === 'a' ? { x: param.number() } : { y: param.number() }
    expect(() => validateParams('op', dependent, { kind: 'a', x: 1 })).toThrow(/unknown parameter 'kind'/)
    const full = (p: Readonly<Record<string, unknown>>) => ({ kind: param.text(), ...dependent(p) })
    expect(() => validateParams('op', full, { kind: 'a', x: 1 })).not.toThrow()
    expect(() => validateParams('op', full, { kind: 'a', y: 1 })).toThrow(/unknown parameter 'y'/)
  })
})

describe('exampleParams, namedParams, mergeParams', () => {
  it('builds valid parameters from a spec', () => {
    const spec = {
      p: param.positive(),
      q: param.integer({ min: 4 }),
      r: param.vector(),
      s: param.number({ optional: true }),
    }
    expect(exampleParams(spec)).toEqual({ p: 1, q: 4, r: [1, 2] })
    expect(Object.keys(exampleParams(spec, true))).toEqual(['p', 'q', 'r', 's'])
    expect(() => validateParams('op', spec, exampleParams(spec, true))).not.toThrow()
  })

  it('declares a list of names, some optional and some vectors', () => {
    const spec = namedParams(['a', 'b', 'cuts'], { optional: ['b'], vectors: ['cuts'] })
    expect([spec['a']?.optional, spec['b']?.optional, spec['cuts']?.kind]).toEqual([false, true, 'vector'])
  })

  it('merges a lift with its inner operator, and gives nothing when the inner one declares nothing', () => {
    const own = { a: param.vector() }
    expect(Object.keys(mergeParams(own, { noise: param.positive() }) ?? {})).toEqual(['a', 'noise'])
    const fn = mergeParams(own, () => ({ y: param.number() }))
    expect(typeof fn === 'function' ? Object.keys(fn({})) : []).toEqual(['a', 'y'])
    expect(mergeParams(own, undefined)).toBeUndefined()
  })
})
