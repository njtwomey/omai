import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  activeDims,
  bool,
  box,
  choice,
  contains,
  decode,
  dict,
  discrete,
  distance,
  encode,
  encodedDim,
  flatten,
  int,
  multiBinary,
  multiDiscrete,
  neighbours,
  real,
  type Space,
  sample,
  tuple,
  union,
} from './index.ts'

const spaces: readonly Space[] = [
  real(-1, 3),
  real(1e-5, 1, { log: true }),
  real(0, 1, { step: 0.25 }),
  int(2, 9),
  int(1, 1000, { log: true }),
  choice(['a', 'b', 'c']),
  bool(),
  box([-1, 0], [1, 5], [2]),
  discrete(4, { start: 1 }),
  multiDiscrete([2, 3]),
  multiBinary(3),
  tuple([discrete(2), real(0, 1)]),
  dict({
    model: choice(['linear', 'tree', 'net']),
    alpha: real(1e-3, 10, { log: true, when: { key: 'model', in: ['linear'] } }),
    depth: int(1, 12, { when: { key: 'model', in: ['tree'] } }),
    pruning: bool({ when: { key: 'model', in: ['tree'] } }),
    // Chained: active only when pruning (itself conditional) is on.
    ccp: real(0, 0.1, { when: { key: 'pruning', in: [true] } }),
    net: dict(
      { width: int(4, 64, { log: true }), act: choice(['relu', 'tanh']) },
      { when: { key: 'model', in: ['net'] } },
    ),
  }),
  dict({
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
]

const draw = fc.tuple(fc.integer({ min: 0, max: spaces.length - 1 }), fc.integer({ min: 0, max: 1_000_000 }))

describe('space properties', () => {
  it('neighbours stay inside the space, differ from the value, and change one active dimension', () => {
    fc.assert(
      fc.property(draw, fc.integer({ min: 2, max: 21 }), ([i, seed], points) => {
        const s = spaces[i] as Space
        const v = sample(s, key(seed))
        const dims = activeDims(s, v)
        for (const n of neighbours(s, v, { points })) {
          expect(contains(s, n)).toBe(true)
          expect(JSON.stringify(n)).not.toBe(JSON.stringify(v))
          const before = flatten(s, v)
          const after = flatten(s, n)
          const moved = new Set(
            activeDims(s, n).filter((d) => {
              const pick = (value: unknown): unknown =>
                d === '' ? value : d.split('.').reduce((o: unknown, p) => (o as Record<string, unknown>)?.[p], value)
              return JSON.stringify(pick(n)) !== JSON.stringify(pick(v))
            }),
          )
          // One dimension moved; any others that differ were switched on or off by it.
          const changedShared = [...moved].filter((d) => dims.includes(d))
          expect(changedShared.length).toBeLessThanOrEqual(1)
          expect(before.length).toBe(after.length)
        }
      }),
    )
  })

  it('neighbourhoods are symmetric for integer, discrete and boolean spaces', () => {
    const symmetric: readonly Space[] = [
      int(-3, 3),
      discrete(5, { start: 2 }),
      bool(),
      multiBinary(4),
      multiDiscrete([3, 2]),
    ]
    fc.assert(
      fc.property(fc.integer({ min: 0, max: symmetric.length - 1 }), fc.integer({ min: 0, max: 10_000 }), (i, seed) => {
        const s = symmetric[i] as Space
        const v = sample(s, key(seed))
        for (const n of neighbours(s, v)) {
          expect(neighbours(s, n).some((m) => JSON.stringify(m) === JSON.stringify(v))).toBe(true)
        }
      }),
    )
  })

  it('encode lands in the unit cube with a fixed dimension, and decode inverts it', () => {
    fc.assert(
      fc.property(draw, ([i, seed]) => {
        const s = spaces[i] as Space
        const v = sample(s, key(seed))
        const u = encode(s, v)
        expect(u).toHaveLength(encodedDim(s))
        for (const x of u) {
          expect(x).toBeGreaterThanOrEqual(-1e-12)
          expect(x).toBeLessThanOrEqual(1 + 1e-12)
        }
        const back = decode(s, u)
        expect(contains(s, back)).toBe(true)
        expect(distance(s, back, v)).toBeLessThan(1e-9 + (s.kind === 'int' || s.kind === 'dict' ? 1e-3 : 0))
      }),
    )
  })

  it('active dimensions are exactly the keys present in sampled configurations', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1_000_000 }), (seed) => {
        const s = spaces[spaces.length - 1] as Space
        const value = sample(s, key(seed))
        const v = value as Readonly<Record<string, unknown>>
        const dims = activeDims(s, value)
        const top = [...new Set(dims.map((d) => d.split('.')[0]))]
        expect(top).toEqual(Object.keys(v))
        if (v['pruning'] === true) expect(dims).toContain('ccp')
        else expect(dims).not.toContain('ccp')
      }),
    )
  })
})
