import { describe, expect, it } from 'vitest'
import { OmaiError } from '../../base/index.ts'
import { arg, assertDamping, errorText, evidenceOrNaN, isFlat, keyOf, memoised, perPower } from './_messages.ts'
import { messageArg } from './args.ts'
import { ImproperMessageError } from './errors.ts'
import { gaussian, gaussianEta } from './families.ts'

describe('the shared conventions of operators', () => {
  it('reads the argument of a port', () => {
    const a = messageArg(gaussian, gaussianEta(0, 1))
    expect(arg({ mode: 'ep', to: 0, args: [a], params: {}, power: 1 }, 0)).toBe(a)
  })

  it('knows a flat message from one with information', () => {
    expect(isFlat([0, 0])).toBe(true)
    expect(isFlat([0, 1e-12])).toBe(false)
    expect(isFlat(undefined)).toBe(false)
  })

  it('divides a site by the power and leaves the unpowered one as it is', () => {
    const site = [1, 2]
    expect(perPower(site, 1)).toBe(site)
    expect(perPower(site, 2)).toEqual([0.5, 1])
  })

  it('gives NaN evidence for an improper message and lets every other error through', () => {
    expect(evidenceOrNaN(() => 3)).toBe(3)
    expect(
      evidenceOrNaN(() => {
        throw new ImproperMessageError('no density')
      }),
    ).toBeNaN()
    expect(() =>
      evidenceOrNaN(() => {
        throw new OmaiError('value', 'a parameter')
      }),
    ).toThrow('a parameter')
  })

  it('says what a caught value is', () => {
    expect(errorText(new Error('boom'))).toBe('boom')
    expect(errorText('text')).toBe('text')
  })

  it('refuses a damping outside [0, 1) in the words of the algorithm', () => {
    expect(() => assertDamping(0.5, 'x')).not.toThrow()
    expect(() => assertDamping(1, 'stochastic EP')).toThrow('stochastic EP: damping must be in [0, 1), got 1')
    expect(() => assertDamping(Number.NaN, 'AMP')).toThrow('AMP: damping must be in [0, 1)')
  })
})

describe('keyOf and memoised', () => {
  it('writes every number in full: NaN, Infinity and -Infinity stay distinct (JSON writes all three as null)', () => {
    const keys = [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, null].map((v) => keyOf([v]))
    expect(new Set(keys).size).toBe(4)
    expect(keyOf({ b: 1, a: [2, Float64Array.of(3)] })).toBe(keyOf({ a: [2, Float64Array.of(3)], b: 1 }))
    expect(keyOf([0.1 + 0.2])).not.toBe(keyOf([0.3]))
  })

  it('keeps a value in the context cache under its key, computes it every time without one, and keeps no throw', () => {
    let runs = 0
    const compute = (): number => ++runs
    const c = { mode: 'ep', to: 0, args: [], params: {}, power: 1, memo: new Map() } as const
    expect([memoised(c, 'k', compute), memoised(c, 'k', compute), memoised(c, 'other', compute)]).toEqual([1, 1, 2])
    const bare = { mode: 'ep', to: 0, args: [], params: {}, power: 1 } as const
    expect([memoised(bare, 'k', compute), memoised(bare, 'k', compute)]).toEqual([3, 4])
    expect(() =>
      memoised(c, 'bad', () => {
        throw new Error('no')
      }),
    ).toThrow('no')
    expect(c.memo.has('bad')).toBe(false)
  })
})
