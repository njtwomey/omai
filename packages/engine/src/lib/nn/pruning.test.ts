import { describe, expect, it } from 'vitest'
import { maxOf, minOf } from '../base/index.ts'
import { key, normal } from '../random/index.ts'
import { mul, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import {
  applyMasks,
  constantSparsity,
  globalMagnitudeMasks,
  linearSparsity,
  magnitudeMask,
  nmMask,
  polynomialSparsity,
  rewind,
  sparsityOf,
  sparsityReport,
  structuredMask,
} from './pruning.ts'
import { iterativeMagnitudePruning } from './pruning-iterative.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

describe('pruning', () => {
  it('magnitude masks reach the asked sparsity and keep the largest weights', () => {
    const x = n(0, [10, 10])
    const m = magnitudeMask(x, 0.3)
    expect(sparsityOf(m)).toBe(0.3)
    const xs = toFlat(x)
    const kept = toFlat(m)
    const keptMin = minOf(xs.filter((_, i) => kept[i] === 1).map(Math.abs))
    const prunedMax = maxOf(xs.filter((_, i) => kept[i] === 0).map(Math.abs))
    expect(keptMin).toBeGreaterThanOrEqual(prunedMax)
    expect(() => magnitudeMask(x, 1.5)).toThrow()
  })

  it('global masks pool layers; existing masks stay pruned; vectors are left alone', () => {
    const params = { a: tensor([[1, 2, 3, 4]]), b: tensor([[10, 20, 30, 40]]), bias: tensor([0.1, 0.2]) }
    const m = globalMagnitudeMasks(params, 0.5)
    expect(toArray(m['a'] as Value)).toEqual([[0, 0, 0, 0]])
    expect(toArray(m['b'] as Value)).toEqual([[1, 1, 1, 1]])
    expect(toArray(m['bias'] as Value)).toEqual([1, 1])
    // The four pruned entries of a come first even though a's values are now larger than b's first.
    const again = globalMagnitudeMasks({ a: tensor([[5, 6, 7, 8]]), b: tensor([[0.01, 20, 30, 40]]) }, 5 / 8, {
      masks: m,
    })
    expect(toArray(again['a'] as Value)).toEqual([[0, 0, 0, 0]])
    expect(toArray(again['b'] as Value)).toEqual([[0, 1, 1, 1]])
  })

  it('structured and N:M masks', () => {
    const w = n(1, [6, 4])
    const rows = toArray(structuredMask(w, 0.5, { norm: 1 })) as number[][]
    expect(rows.filter((r) => r.every((v) => v === 0)).length).toBe(3)
    expect(rows.every((r) => r.every((v) => v === r[0]))).toBe(true)
    const cols = toArray(structuredMask(w, 0.25, { axis: 1 })) as number[][]
    expect((cols[0] ?? []).filter((v) => v === 0).length).toBe(1)
    const nm = toFlat(nmMask(n(2, [3, 8]), 2, 4))
    for (let g = 0; g < nm.length; g += 4) expect(nm.slice(g, g + 4).reduce((a, b) => a + b, 0)).toBe(2)
    expect(() => nmMask(n(2, [3, 6]), 2, 4)).toThrow()
  })

  it('applies masks, reports sparsity and rewinds', () => {
    const p = {
      w: tensor([
        [1, -2],
        [3, 0.1],
      ]),
    }
    const m = { w: magnitudeMask(p.w, 0.25) }
    const pruned = applyMasks(p, m)
    expect(toArray(pruned['w'] as Value)).toEqual([
      [1, -2],
      [3, 0],
    ])
    const report = sparsityReport(pruned)
    expect(report.tensors['w']).toEqual({ zeros: 1, size: 4, sparsity: 0.25 })
    expect(report.global).toBe(0.25)
    expect(
      toArray(
        rewind(
          {
            w: tensor([
              [9, 8],
              [7, 6],
            ]),
          },
          m,
        )['w'] as Value,
      ),
    ).toEqual([
      [9, 8],
      [7, 0],
    ])
  })

  it('iterative magnitude pruning reaches the final sparsity and rewinds to the initial weights', () => {
    const initial = { w: n(3, [4, 5]) }
    // A stand-in for training: scale the weights (so the trained values differ from the initial ones).
    const r = iterativeMagnitudePruning(initial, (q) => ({ w: mul(q['w'] as Value, 2) }), {
      finalSparsity: 0.75,
      rounds: 3,
    })
    expect(r.rounds.map((x) => Math.round(x.sparsity * 1000) / 1000)).toEqual([0.37, 0.603, 0.75])
    expect(sparsityOf(r.masks['w'] as Value)).toBe(0.75)
    const ticket = toFlat(r.ticket['w'] as Value)
    const init = toFlat(initial.w)
    for (const [i, v] of ticket.entries()) expect(v === 0 || v === init[i]).toBe(true)
  })

  it('sparsity schedules hit their endpoints', () => {
    expect([constantSparsity(0.5, 3)(2), constantSparsity(0.5, 3)(3)]).toEqual([0, 0.5])
    const lin = linearSparsity(0.1, 0.9, 10, 20)
    expect([lin(0), lin(15), lin(30)].map((v) => Math.round(v * 1e9) / 1e9)).toEqual([0.1, 0.5, 0.9])
    const cubic = polynomialSparsity(0, 0.8, 0, 100)
    expect([cubic(0), cubic(100), cubic(200)]).toEqual([0, 0.8, 0.8])
    expect(cubic(50)).toBeCloseTo(0.8 - 0.8 * 0.125, 12)
    expect(() => polynomialSparsity(0, 0.5, 5, 5)).toThrow()
  })
})
