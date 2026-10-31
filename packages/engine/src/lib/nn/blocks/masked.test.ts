import { describe, expect, it } from 'vitest'
import { expectClose } from '../../../testing/close.ts'
import { checkPrimitive, grad } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { asTensor, reshape, sum, toArray, toFlat, type Value, zeros } from '../../tensor/index.ts'
import { Conv1d } from '../module/conv.ts'
import { CausalConv1d, GatedMaskedConv, MaskedConv2d, pixelCnnMask } from './masked.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

const flatIndex = (shape: readonly number[], idx: readonly number[]): number =>
  idx.reduce((o, i, d) => o * (shape[d] as number) + i, 0)

/** `x` plus one at pixel `at` of every channel (batch 0). */
function bumped(x: Value, at: readonly [number, number]): Value {
  const data = toFlat(x)
  const [channels, h, w] = [x.shape[1] as number, x.shape[2] as number, x.shape[3] as number]
  for (let c = 0; c < channels; c++)
    data[(c * h + at[0]) * w + at[1]] = (data[(c * h + at[0]) * w + at[1]] as number) + 1
  return reshape(asTensor(data), x.shape)
}

/** Whether output unit `out` (any channel) changes when the input pixel `at` is perturbed. */
function dependsOn(
  f: (x: Value) => Value,
  x: Value,
  out: readonly [number, number],
  at: readonly [number, number],
): boolean {
  const a = f(x)
  const b = f(bumped(x, at))
  const [fa, fb] = [toFlat(a), toFlat(b)]
  for (let c = 0; c < (a.shape[1] as number); c++) {
    const k = flatIndex(a.shape, [0, c, out[0], out[1]])
    if (Math.abs((fa[k] as number) - (fb[k] as number)) > 1e-12) return true
  }
  return false
}

const earlier = (r: number, c: number, i: number, j: number): boolean => r < i || (r === i && c < j)

describe('pixelCnnMask', () => {
  it('opens the taps before the centre in raster order, and the centre only for mask B', () => {
    expect(toArray(pixelCnnMask(1, 1, 3, 'A'))).toEqual([
      [
        [
          [1, 1, 1],
          [1, 0, 0],
          [0, 0, 0],
        ],
      ],
    ])
    expect(toArray(pixelCnnMask(1, 1, [3, 5], 'B'))).toEqual([
      [
        [
          [1, 1, 1, 1, 1],
          [1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0],
        ],
      ],
    ])
  })

  it('orders colour groups at the centre: A connects an output group to earlier groups, B up to its own', () => {
    const centre = (type: 'A' | 'B'): number[][] => {
      const m = toFlat(pixelCnnMask(3, 3, 3, type, 3))
      return [0, 1, 2].map((o) => [0, 1, 2].map((i) => m[flatIndex([3, 3, 3, 3], [o, i, 1, 1])] as number))
    }
    expect(centre('A')).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
    ])
    expect(centre('B')).toEqual([
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
    ])
  })
})

describe('MaskedConv2d', () => {
  const grid = 5
  const x = n(1, [1, 1, grid, grid])

  it('a stack of mask A then mask B layers sees only earlier pixels, and the first layer excludes the pixel itself', () => {
    const a = new MaskedConv2d(1, 2, 3, 'A', { key: key(2) })
    const b = new MaskedConv2d(2, 2, 3, 'B', { key: key(3) })
    const c = new MaskedConv2d(2, 1, 3, 'B', { key: key(4) })
    const stack = (v: Value): Value => c.forward(b.forward(a.forward(v)))
    let seen = 0
    for (let i = 0; i < grid; i++)
      for (let j = 0; j < grid; j++)
        for (let r = 0; r < grid; r++)
          for (let q = 0; q < grid; q++) {
            const dep = dependsOn(stack, x, [i, j], [r, q])
            if (!earlier(r, q, i, j)) expect(dep, `output (${i},${j}) on input (${r},${q})`).toBe(false)
            else if (r === i && q === j - 1) expect(dep).toBe(true)
            seen += dep ? 1 : 0
          }
    expect(seen).toBeGreaterThan(30)
    // The first layer alone: the pixel itself is excluded, its left neighbour is not.
    expect(dependsOn((v) => a.forward(v), x, [2, 2], [2, 2])).toBe(false)
    expect(dependsOn((v) => a.forward(v), x, [2, 2], [2, 1])).toBe(true)
    expect(dependsOn((v) => b.forward(a.forward(v)), x, [2, 2], [2, 2])).toBe(false)
  })

  it('the masked weights receive no gradient and the layer passes the conformance check', () => {
    const m = new MaskedConv2d(2, 2, 3, 'A', { key: key(5) })
    const g = grad((p: Record<string, Value>) => sum(m.apply(p, x2())))(m.params as Record<string, Value>) as {
      weight: Value
    }
    const mask = toFlat(m.mask)
    for (const [i, v] of toFlat(g.weight).entries()) if (mask[i] === 0) expect(Math.abs(v)).toBe(0)
    expect(checkPrimitive((v) => m.apply(m.params, v), [n(6, [1, 2, 4, 4])], { rtol: 1e-5 }).failures).toEqual([])
    function x2(): Value {
      return n(6, [1, 2, 4, 4])
    }
  })
})

describe('Gated PixelCNN layers', () => {
  const grid = 6
  const x = n(10, [1, 3, grid, grid])
  const first = new GatedMaskedConv(3, 3, { first: true, key: key(11) })
  const second = new GatedMaskedConv(3, 3, { key: key(12) })
  const gated = (v: Value): Value => {
    const a = first.forward({ v, h: v })
    return second.forward(a).h
  }

  it('the horizontal stack sees every earlier pixel in its cone and nothing at or after the pixel', () => {
    for (let i = 0; i < grid; i++)
      for (let j = 0; j < grid; j++)
        for (let r = 0; r < grid; r++)
          for (let q = 0; q < grid; q++)
            if (!earlier(r, q, i, j))
              expect(dependsOn(gated, x, [i, j], [r, q]), `output (${i},${j}) on input (${r},${q})`).toBe(false)
    expect(dependsOn(gated, x, [3, 2], [3, 1])).toBe(true)
    expect(dependsOn(gated, x, [3, 2], [2, 3])).toBe(true)
  })

  it('removes the blind spot that a stack of masked convolutions leaves to the upper right', () => {
    const a = new MaskedConv2d(3, 3, 3, 'A', { key: key(13) })
    const b = new MaskedConv2d(3, 3, 3, 'B', { key: key(14) })
    const naive = (v: Value): Value => b.forward(a.forward(v))
    // Output (3, 2) against input (2, 4): two columns right of the pixel, one row above.
    expect(dependsOn(naive, x, [3, 2], [2, 4])).toBe(false)
    expect(dependsOn(gated, x, [3, 2], [2, 4])).toBe(true)
  })

  it('has the conformance property and a residual in later layers only', () => {
    expect(
      checkPrimitive((v) => second.apply(second.params, { v, h: v }).h, [n(15, [1, 3, 4, 4])], { rtol: 1e-5 }).failures,
    ).toEqual([])
    expect(Object.keys(second.stateDict())).toContain('horizontal.weight')
  })
})

describe('CausalConv1d', () => {
  it('equals a plain convolution on the left-padded input, keeps the length and is causal', () => {
    const m = new CausalConv1d(2, 3, 3, { dilation: 2, key: key(20) })
    const x = n(21, [1, 2, 12])
    const y = m.forward(x)
    expect(y.shape).toEqual([1, 3, 12])
    const plain = new Conv1d(2, 3, 3, { dilation: 2, key: key(20) })
    plain.setParams(m.params)
    const xs = toFlat(x)
    const padded = [[0, 1].map((c) => [0, 0, 0, 0, ...xs.slice(c * 12, (c + 1) * 12)])]
    expectClose(plain.forward(padded), y, 1e-12)
    // The output at t depends on the inputs at t, t-2 and t-4 only: bumping input 8 changes outputs 8, 10 and 12.
    const bump = xs.map((v, i) => (i % 12 === 8 ? v + 1 : v))
    const [before, after] = [toFlat(y), toFlat(m.forward(reshape(asTensor(bump), x.shape)))]
    const changed = (t: number): boolean =>
      [0, 1, 2].some((c) => Math.abs((before[c * 12 + t] as number) - (after[c * 12 + t] as number)) > 1e-12)
    for (let t = 0; t < 12; t++) expect(changed(t), `output ${t}`).toBe(t === 8 || t === 10)
    expect(checkPrimitive((v) => m.apply(m.params, v), [n(22, [1, 2, 6])], { rtol: 1e-5 }).failures).toEqual([])
  })
})
