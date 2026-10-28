import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  activeDims,
  bool,
  box,
  choice,
  clip,
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
  union,
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
  [
    'union of modes',
    dict({
      penalty: choice(['none', 'l2']),
      solver: union(
        {
          closedForm: dict({}),
          irls: dict({ tol: real(1e-10, 1e-2, { log: true }), maxIter: int(1, 200) }),
          sgd: dict({
            batch: int(1, 256, { log: true }),
            optimiser: union({ plain: dict({}), momentum: dict({ beta: real(0, 0.99) }) }, { tag: 'name' }),
          }),
        },
        { default: 'irls' },
      ),
    }),
  ],
]

const solver = union(
  { irls: dict({ tol: real(1e-10, 1e-2, { log: true }) }), sgd: dict({ batch: int(1, 256) }) },
  { default: 'irls' },
)

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

  it('a union names its option under the tag and carries that option’s knobs beside it', () => {
    expect(defaults(solver)).toEqual({ kind: 'irls', tol: 1e-6 })
    expect(contains(solver, { kind: 'sgd', batch: 32 })).toBe(true)
    expect(contains(solver, { kind: 'sgd', tol: 1e-6 })).toBe(false)
    expect(contains(solver, { kind: 'newton' })).toBe(false)
    expect(contains(solver, 'irls')).toBe(false)
    expect(() => clip(solver, { kind: 'newton' })).toThrow(/not an option/)
    expect(clip(solver, { kind: 'sgd', batch: 1000 })).toEqual({ kind: 'sgd', batch: 256 })
    expect(activeDims(dict({ solver }), { solver: { kind: 'sgd', batch: 8 } })).toEqual([
      'solver.kind',
      'solver.sgd.batch',
    ])
    // The flat layout: the tag, then every option's numbers (inactive ones NaN), so unions nest and tune leaf by leaf.
    expect(flatten(solver, { kind: 'sgd', batch: 8 })).toEqual([1, Number.NaN, 8])
    expect(unflatten(solver, [0, 1e-4, Number.NaN])).toEqual({ kind: 'irls', tol: 1e-4 })
    expect(() => union({})).toThrow(/at least one/)
    expect(() => union({ a: dict({}) }, { default: 'b' })).toThrow(/not an option/)
    expect(() => union({ a: dict({ kind: bool() }) })).toThrow(/named like the tag/)
  })
})
