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

  describe('memoised', () => {
    const at = (memo: Map<string, unknown> | undefined) => ({
      mode: 'ep' as const,
      to: 0,
      args: [],
      params: {},
      power: 1,
      ...(memo === undefined ? {} : { memo }),
    })

    it('computes a value once for a key and again for another', () => {
      const memo = new Map<string, unknown>()
      let calls = 0
      const compute = (): number => ++calls
      expect([
        memoised(at(memo), 'a', compute),
        memoised(at(memo), 'a', compute),
        memoised(at(memo), 'b', compute),
      ]).toEqual([1, 1, 2])
    })

    it('computes every time when the context has no cache', () => {
      let calls = 0
      memoised(at(undefined), 'a', () => ++calls)
      memoised(at(undefined), 'a', () => ++calls)
      expect(calls).toBe(2)
    })

    it('runs a computation that throws an ImproperMessageError once for every port of the factor, and throws it again', () => {
      const memo = new Map<string, unknown>()
      let calls = 0
      const tilt = (): number => {
        calls++
        throw new ImproperMessageError('the tilt has no mass')
      }
      for (let port = 0; port < 61; port++)
        expect(() => memoised(at(memo), 'chain', tilt)).toThrow(ImproperMessageError)
      expect(calls).toBe(1)
      expect(() => memoised(at(memo), 'chain', tilt)).toThrow('the tilt has no mass')
      // another key is its own computation, and clearing the cache (the end of a sweep) forgets the failure
      expect(() => memoised(at(memo), 'other', tilt)).toThrow(ImproperMessageError)
      expect(calls).toBe(2)
      memo.clear()
      expect(memoised(at(memo), 'chain', () => 7)).toBe(7)
    })

    it('does not keep an error that is not an improper message', () => {
      const memo = new Map<string, unknown>()
      let calls = 0
      const bug = (): number => {
        calls++
        throw new OmaiError('value', 'a parameter')
      }
      expect(() => memoised(at(memo), 'k', bug)).toThrow('a parameter')
      expect(() => memoised(at(memo), 'k', bug)).toThrow('a parameter')
      expect(calls).toBe(2)
    })
  })
})
