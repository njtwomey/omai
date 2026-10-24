import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  activeDims,
  bool,
  box,
  choice,
  clip,
  contains,
  defaults,
  dict,
  discrete,
  int,
  multiBinary,
  multiDiscrete,
  real,
  sample,
  tuple,
  unflatten,
} from './index.ts'

const chained = dict({
  pruning: bool({ when: { key: 'model', in: ['tree'] }, default: true }),
  ccp: real(0, 0.1, { when: { key: 'pruning', in: [true] } }),
  model: choice(['linear', 'tree'], { default: 'tree' }),
})

describe('operations', () => {
  it('lists active dimensions through dicts and tuples', () => {
    const s = dict({ opt: dict({ lr: real(0, 1), betas: tuple([real(0, 1), real(0, 1)]) }), seed: int(0, 9) })
    expect(activeDims(s, defaults(s))).toEqual(['opt.lr', 'opt.betas.0', 'opt.betas.1', 'seed'])
    expect(activeDims(real(0, 1), 0.5)).toEqual([''])
    expect(activeDims(box(0, 1, [3]), [0, 0, 0])).toEqual([''])
  })

  it('resolves conditions in any key order and through chains', () => {
    expect(defaults(chained)).toEqual({ pruning: true, ccp: 0.05, model: 'tree' })
    expect(activeDims(chained, { model: 'linear' })).toEqual(['model'])
    expect(contains(chained, { model: 'tree', pruning: true })).toBe(false)
    expect(contains(chained, { model: 'tree', pruning: false })).toBe(true)
    expect(contains(chained, { model: 'linear', pruning: 3 } as never)).toBe(true)
    for (let seed = 0; seed < 20; seed++) expect(contains(chained, sample(chained, key(seed)))).toBe(true)
  })

  it('uses declared defaults for every kind and snaps stepped reals', () => {
    expect(defaults(box(0, 1, [2], { default: [0.5, 0.5] }))).toEqual([0.5, 0.5])
    expect(defaults(discrete(4, { start: 2, default: 3 }))).toBe(3)
    expect(defaults(multiDiscrete([3, 3], { default: [1, 2] }))).toEqual([1, 2])
    expect(defaults(multiBinary(2, { default: [1, 0] }))).toEqual([1, 0])
    expect(defaults(real(0, 1, { step: 0.3 }))).toBeCloseTo(0.6, 12)
  })

  it('clips every kind of value', () => {
    expect(clip(real(0, 1, { step: 0.25 }), 0.4)).toBe(0.5)
    expect(clip(multiDiscrete([3, 2]), [5.2, -1])).toEqual([2, 0])
    expect(clip(multiBinary(3), [0.7, 0.2, 1])).toEqual([1, 0, 1])
    expect(clip(bool(), true)).toBe(true)
    expect(clip(chained, { model: 'linear', pruning: true, ccp: 3 })).toEqual({ model: 'linear' })
    expect(() => clip(choice(['a']), 'b')).toThrow()
  })

  it('unflattens inactive vector dimensions as absent', () => {
    const s = dict({ on: bool(), v: box(0, 1, [2], { when: { key: 'on', in: [true] } }) })
    expect(unflatten(s, [0, Number.NaN, Number.NaN])).toEqual({ on: false })
  })
})
