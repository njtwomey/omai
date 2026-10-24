import { describe, expect, it } from 'vitest'
import {
  bool,
  box,
  choice,
  decode,
  dict,
  discrete,
  encode,
  grid,
  int,
  multiBinary,
  multiDiscrete,
  neighbours,
  real,
  tuple,
} from './index.ts'

const chained = dict({
  model: choice(['linear', 'tree']),
  depth: int(1, 4, { when: { key: 'model', in: ['tree'] } }),
  pruning: bool({ when: { key: 'model', in: ['tree'] } }),
  ccp: real(0, 0.1, { when: { key: 'pruning', in: [true] }, default: 0.01 }),
})

describe('encoding', () => {
  it('encodes vector spaces element by element and inactive keys as 0.5 per coordinate', () => {
    expect(encode(box([0, -1], [2, 1], [2]), [1, 1])).toEqual([0.5, 1])
    expect(encode(multiDiscrete([3, 1]), [2, 0])).toEqual([1, 0.5])
    expect(encode(multiBinary(2), [1, 0])).toEqual([1, 0])
    expect(encode(dict({ on: bool(), v: box(0, 1, [3], { when: { key: 'on', in: [true] } }) }), { on: false })).toEqual(
      [0, 0.5, 0.5, 0.5],
    )
    expect(decode(tuple([multiBinary(2), multiDiscrete([4])]), [0.2, 0.9, 0.5])).toEqual([[0, 1], [2]])
    expect(decode(box(0, 4, [2]), [0.25, 2])).toEqual([1, 4])
  })

  it('decodes chained conditions', () => {
    expect(decode(chained, [1, 1, 1, 0.5])).toEqual({ model: 'tree', depth: 4, pruning: true, ccp: 0.05 })
    expect(decode(chained, [1, 0, 0, 0.5])).toEqual({ model: 'tree', depth: 1, pruning: false })
    expect(decode(chained, [0, 1, 1, 1])).toEqual({ model: 'linear' })
  })

  it('grids follow chained conditions, nested dicts and tuples', () => {
    const configs = grid(chained, 2)
    // linear; tree x depth (2 grid values) x pruning off; tree x depth x pruning on x ccp (2 values).
    expect(configs).toHaveLength(1 + 2 + 4)
    expect(configs[0]).toEqual({ model: 'linear' })
    expect(configs).toContainEqual({ model: 'tree', depth: 4, pruning: true, ccp: 0.1 })
    expect(configs).toContainEqual({ model: 'tree', depth: 1, pruning: false })
    expect(grid(tuple([bool(), discrete(2)]))).toEqual([
      [false, 0],
      [false, 1],
      [true, 0],
      [true, 1],
    ])
    expect(grid(dict({ inner: dict({ a: bool() }) }))).toEqual([{ inner: { a: false } }, { inner: { a: true } }])
    expect(grid(int(1, 3), 9)).toEqual([1, 2, 3])
    const logGrid = grid(real(1, 100, { log: true }), 3) as number[]
    expect(logGrid.map((x) => Math.round(x * 1e9) / 1e9)).toEqual([1, 10, 100])
    expect(() => grid(box(0, 1, [2]))).toThrow()
  })
})

describe('neighbours', () => {
  it('steps reals on a grid or by their step, clipped to the bounds', () => {
    expect(neighbours(real(0, 1), 0.5, { points: 5 })).toEqual([0.25, 0.75])
    expect(neighbours(real(0, 1), 1, { points: 5 })).toEqual([0.75])
    expect(neighbours(real(0, 1, { step: 0.25 }), 0)).toEqual([0.25])
    const logs = neighbours(real(1, 100, { log: true }), 10, { points: 3 }) as number[]
    expect(logs[0]).toBeCloseTo(1, 12)
    expect(logs[1]).toBeCloseTo(100, 12)
    expect(() => neighbours(real(0, 1), 0.5, { points: 1 })).toThrow()
  })

  it('moves every kind of dimension once', () => {
    expect(neighbours(choice(['a', 'b', 'c']), 'b')).toEqual(['a', 'c'])
    expect(neighbours(discrete(3, { start: 1 }), 1)).toEqual([2])
    expect(neighbours(multiBinary(2), [0, 1])).toEqual([
      [1, 1],
      [0, 0],
    ])
    expect(neighbours(multiDiscrete([2, 3]), [1, 0])).toEqual([
      [0, 0],
      [1, 1],
    ])
    expect(neighbours(box(0, 1, [2]), [0, 1], { points: 3 })).toEqual([
      [0.5, 1],
      [0, 0.5],
    ])
    expect(neighbours(tuple([bool(), int(0, 1)]), [true, 0])).toEqual([
      [false, 0],
      [true, 1],
    ])
  })

  it('repairs conditional dimensions when their parent changes', () => {
    expect(neighbours(chained, { model: 'linear' })).toEqual([{ model: 'tree', depth: 3, pruning: false }])
    const fromTree = neighbours(chained, { model: 'tree', depth: 1, pruning: false })
    expect(fromTree).toEqual([
      { model: 'linear' },
      { model: 'tree', depth: 2, pruning: false },
      { model: 'tree', depth: 1, pruning: true, ccp: 0.01 },
    ])
  })
})

describe('decoding end points', () => {
  it('stays inside the bounds when the end points round outside them', () => {
    // exp(log(1e-5)) is 9.999999999999999e-6 and 0.1 + (0.3 - 0.1) is 0.30000000000000004.
    expect(Math.exp(Math.log(1e-5))).toBeLessThan(1e-5)
    expect(decode(real(1e-5, 1, { log: true }), [0])).toBe(1e-5)
    expect(decode(real(1e-5, 1e-2, { log: true }), [1])).toBe(1e-2)
    expect(decode(box(0.1, 0.3, [2]), [0, 1])).toEqual([0.1, 0.3])
    for (const v of grid(real(1e-5, 3e-3, { log: true }), 7) as number[]) {
      expect(v).toBeGreaterThanOrEqual(1e-5)
      expect(v).toBeLessThanOrEqual(3e-3)
    }
  })
})
