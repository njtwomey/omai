import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  bool,
  box,
  choice,
  contains,
  decode,
  defaults,
  dict,
  discrete,
  encode,
  flatten,
  grid,
  int,
  multiBinary,
  multiDiscrete,
  real,
  type Space,
  sample,
  tuple,
  unflatten,
} from './index.ts'

const spaces: readonly (readonly [string, Space])[] = [
  ['real', real(-1, 3)],
  ['log real', real(1e-5, 1, { log: true })],
  ['stepped real', real(0, 1, { step: 0.25 })],
  ['int', int(2, 9)],
  ['log int', int(1, 1000, { log: true })],
  ['choice', choice(['a', 'b', 'c'])],
  ['bool', bool()],
  ['box', box([-1, 0], [1, 5], [2])],
  ['discrete', discrete(4, { start: 1 })],
  ['multiDiscrete', multiDiscrete([2, 3])],
  ['multiBinary', multiBinary(3)],
  ['tuple', tuple([discrete(2), real(0, 1)])],
  [
    'conditional dict',
    dict({
      optimiser: choice(['sgd', 'adam']),
      momentum: real(0, 0.99, { when: { key: 'optimiser', in: ['sgd'] } }),
      lr: real(1e-4, 1, { log: true }),
    }),
  ],
]

describe('spaces', () => {
  for (const [name, space] of spaces) {
    it(`${name}: samples, defaults and flattening stay in the space`, () => {
      fc.assert(
        fc.property(fc.integer({ min: 0, max: 1_000_000 }), (seed) => {
          const v = sample(space, key(seed))
          expect(contains(space, v)).toBe(true)
          expect(unflatten(space, flatten(space, v))).toEqual(v)
        }),
        { numRuns: 50 },
      )
      expect(contains(space, defaults(space))).toBe(true)
    })
  }

  it('encodes to the unit cube and back', () => {
    const s = dict({ lr: real(1e-4, 1, { log: true }), k: int(1, 9), act: choice(['relu', 'tanh']), flag: bool() })
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1_000_000 }), (seed) => {
        const v = sample(s, key(seed))
        const u = encode(s, v)
        expect(u.every((x) => x >= 0 && x <= 1)).toBe(true)
        const back = decode(s, u) as Record<string, number | string | boolean>
        const orig = v as Record<string, number | string | boolean>
        expect(back['k']).toBe(orig['k'])
        expect(back['act']).toBe(orig['act'])
        expect(back['lr'] as number).toBeCloseTo(orig['lr'] as number, 10)
      }),
      { numRuns: 50 },
    )
  })

  it('drops inactive conditional keys', () => {
    const s = dict({ opt: choice(['sgd', 'adam']), momentum: real(0, 1, { when: { key: 'opt', in: ['sgd'] } }) })
    expect(decode(s, [1, 0.5])).toEqual({ opt: 'adam' })
    expect(grid(s, 3)).toHaveLength(4)
  })

  it('rejects invalid intervals', () => {
    expect(() => real(1, 0)).toThrow(/high >= low/)
    expect(() => real(0, 1, { log: true })).toThrow(/log scale/)
  })
})
